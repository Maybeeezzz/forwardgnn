"""Stage attribution and exclusive storage categories for diagnostic snapshots."""
import ast
import csv
import json
from pathlib import Path
import re

CATEGORIES = ('parameters', 'gradients', 'optimizer_state', 'checkpoint', 'cache',
              'graph_data', 'logits_loss', 'activations', 'other_visible')
LABELS = ('参数', '梯度', '优化器状态', '早停副本', '缓存', '图数据', 'logits/loss', '激活', '其他可见')


class StageResolver:
    def __init__(self):
        self.statements = {}
        self.training_lines = {}
        self.validation_lines = {}

    def statement(self, filename, line):
        if filename not in self.statements:
            mapping = {}
            training, validation = set(), set()
            try:
                source = Path(filename).read_text()
                tree = ast.parse(source)
                # Only simple statements: a surrounding loop/if must not swallow its body.
                for node in ast.walk(tree):
                    if isinstance(node, (ast.For, ast.AsyncFor)) and any(
                            isinstance(n, ast.Name) and n.id == 'epoch' for n in ast.walk(node.target)):
                        training.update(range(node.lineno, node.end_lineno + 1))
                    if isinstance(node, ast.Assign) and any(
                            isinstance(n, ast.Name) and n.id.startswith(('val_', 'test_'))
                            for target in node.targets for n in ast.walk(target)):
                        validation.update(range(node.lineno, node.end_lineno + 1))
                    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.Expr, ast.Return)):
                        text = ast.get_source_segment(source, node) or ''
                        for index in range(node.lineno, node.end_lineno + 1):
                            mapping[index] = text
            except (OSError, SyntaxError, UnicodeError):
                pass
            self.statements[filename] = mapping
            self.training_lines[filename] = training
            self.validation_lines[filename] = validation
        return self.statements[filename].get(line, '')

    def resolve(self, frame, location, layers):
        """Best-effort generic Python/PyTorch stages; explicit scopes take priority."""
        context = {'layer': location.get('layer'), 'epoch_zero_based': location.get('epoch_zero_based')}
        frames = []
        ancestor = frame
        while ancestor is not None:
            owner = ancestor.f_locals.get('self')
            if context['layer'] is None and id(owner) in layers:
                context['layer'] = layers[id(owner)]
            if context['layer'] in (None, 'model'):
                for local_name in ('layer', 'conv'):
                    candidate = layers.get(id(ancestor.f_locals.get(local_name)))
                    if candidate:
                        context['layer'] = candidate
                        break
            if context['epoch_zero_based'] is None and isinstance(ancestor.f_locals.get('epoch'), int):
                context['epoch_zero_based'] = ancestor.f_locals['epoch']
            frames.append(ancestor)
            ancestor = ancestor.f_back
        phase = location.get('phase', 'other')
        application_frames = [f for f in frames if self.selected(f.f_code.co_filename)]
        for f in application_frames:
            self.statement(f.f_code.co_filename, f.f_lineno)
        validation = any(f.f_lineno in self.validation_lines.get(f.f_code.co_filename, ())
                         for f in application_frames)
        if validation or any(f.f_code.co_name in ('eval_model', 'evaluate', 'test', 'validation_step') for f in frames):
            phase = 'evaluation'
        elif phase != 'backward':
            for current in frames:
                function = current.f_code.co_name
                if function in ('save_checkpoint', 'load_checkpoint'):
                    phase = 'checkpoint'
                    break
                if function == 'forward':
                    owner = current.f_locals.get('self')
                    is_loss = any(base.__name__ == '_Loss' and base.__module__.startswith('torch.nn')
                                  for base in type(owner).__mro__)
                    phase = 'loss_build' if is_loss else 'forward'
                    break
                # Only application/model code participates in source inference.
                if not self.selected(current.f_code.co_filename):
                    continue
                statement = self.statement(current.f_code.co_filename, current.f_lineno)
                if '.backward(' in statement:
                    phase = 'backward'
                elif '.zero_grad(' in statement:
                    phase = 'zero_grad'
                elif '.step(' in statement:
                    phase = 'optimizer_step' if self.optimizer_call(current, statement) else 'other'
                elif 'load_checkpoint(' in statement or 'save_checkpoint(' in statement:
                    phase = 'checkpoint'
                elif statement:
                    phase = self.module_call(current, statement) or 'other'
                else:
                    continue
                if phase != 'other':
                    break
        context['phase'] = phase
        in_epoch_loop = any(f.f_lineno in self.training_lines.get(f.f_code.co_filename, ())
                            for f in application_frames)
        if phase == 'evaluation':
            context['period'] = 'evaluation'
        elif phase == 'checkpoint':
            context['period'] = 'other'
        elif phase == 'backward' or in_epoch_loop:
            context['period'] = 'training'
        else:
            # Includes initialization and SF layer-transition forward outside
            # the epoch loop. A lingering local epoch is not training evidence.
            context['period'] = 'other'
        return context

    def selected(self, filename):
        return self.select_file(filename) if hasattr(self, 'select_file') else True

    @staticmethod
    def resolve_object(node, locals_):
        if isinstance(node, ast.Name):
            return locals_.get(node.id)
        if isinstance(node, ast.Attribute):
            parent = StageResolver.resolve_object(node.value, locals_)
            attrs = getattr(parent, '__dict__', {})
            if node.attr in attrs:
                return attrs[node.attr]
            return attrs.get('_modules', {}).get(node.attr)
        return None

    def calls(self, statement):
        try:
            return [node for node in ast.walk(ast.parse(statement)) if isinstance(node, ast.Call)]
        except SyntaxError:
            return []

    def optimizer_call(self, frame, statement):
        for call in self.calls(statement):
            if isinstance(call.func, ast.Attribute) and call.func.attr == 'step':
                owner = self.resolve_object(call.func.value, frame.f_locals)
                if any(base.__name__ == 'Optimizer' and base.__module__.startswith('torch.optim')
                       for base in type(owner).__mro__):
                    return True
        return False

    def module_call(self, frame, statement):
        for call in self.calls(statement):
            owner = self.resolve_object(call.func, frame.f_locals)
            if isinstance(call.func, ast.Attribute) and call.func.attr == 'forward':
                return 'forward'
            bases = type(owner).__mro__
            if any(base.__name__ == '_Loss' and base.__module__.startswith('torch.nn') for base in bases):
                return 'loss_build'
            if any(base.__name__ == 'Module' and base.__module__.startswith('torch.nn') for base in bases):
                return 'forward'
        return None


def category_for(variable):
    path = variable['path']
    if path.endswith('.grad') or path == 'incoming_gradient':
        return 'gradients'
    if variable.get('is_parameter') or '._parameters[' in path or '.param_groups[' in path:
        return 'parameters'
    if '.state[' in path and ('optimizer' in path or "['exp_avg" in path):
        return 'optimizer_state'
    if 'best_model_state_dict' in path:
        return 'checkpoint'
    if '_cached_' in path:
        return 'cache'
    if re.search(r'\.(data|train_data|val_data|test_data|aug_graph|graph_augmenter)(\.|\[|$)', path):
        return 'graph_data'
    name = path.rsplit('.', 1)[-1].split('[')[0]
    if name in ('edge_index', 'edge_weight', 'edge_type', 'train_mask', 'eval_mask', 'train_y'):
        return 'graph_data'
    if name in ('logits', 'loss', 'prob', 'pred', 'accumulated_probs'):
        return 'logits_loss'
    if name in ('x', 'node_feats', 'node_emb', 'train_node_emb', 'virtual_node_emb',
                'eval_node_feats', 'virtual_node_feats', 'out', 'return_value', 'l2_norm') or variable.get('grad_fn'):
        return 'activations'
    return 'other_visible'


def categorize(details):
    # A storage can have aliases in several categories. Choose one by explicit priority.
    priority = {name: index for index, name in enumerate(
        ('gradients', 'parameters', 'optimizer_state', 'checkpoint', 'cache',
         'graph_data', 'logits_loss', 'activations', 'other_visible'))}
    assignments = {}
    for variable in details['variables']:
        key, category = variable['storage_id'], category_for(variable)
        if key not in assignments or priority[category] < priority[assignments[key]]:
            assignments[key] = category
    totals = dict.fromkeys(CATEGORIES, 0)
    classified = []
    for storage in details['storages']:
        category = assignments.get(storage['storage_id'], 'other_visible')
        totals[category] += storage['bytes']
        classified.append(dict(storage, category=category))
    return totals, classified


class StageMemorySummary:
    def __init__(self):
        self.groups = {}

    def observe(self, event, details, context):
        key = (context.get('layer'), context['phase'])
        totals, storages = categorize(details)
        group = self.groups.setdefault(key, {'samples': 0, 'interval_peak_allocated_bytes': 0})
        group['samples'] += 1
        group['interval_peak_allocated_bytes'] = max(group['interval_peak_allocated_bytes'], event['interval_peak_allocated_bytes'])
        if 'snapshot' not in group or event['allocated_bytes'] > group['snapshot']['allocated_bytes']:
            group['snapshot'] = dict(context, sequence=event['sequence'],
                                     allocated_bytes=event['allocated_bytes'], reserved_bytes=event['reserved_bytes'],
                                     categories_bytes=totals,
                                     visible_storage_bytes=sum(totals.values()),
                                     unaccounted_bytes=event['allocated_bytes'] - sum(totals.values()),
                                     inspection_error_count=len(details.get('inspection_errors', [])),
                                     storages=storages)

    def write(self, directory):
        directory = Path(directory)
        rows = []
        for (layer, phase), group in sorted(self.groups.items(), key=lambda item: (str(item[0][0] or ''), item[0][1])):
            snapshot = group['snapshot']
            row = dict(layer=layer, phase=phase, epoch_zero_based=snapshot.get('epoch_zero_based'),
                       sequence=snapshot['sequence'], samples=group['samples'])
            for name in ('allocated', 'reserved', 'visible_storage', 'unaccounted'):
                row[name + '_mib'] = snapshot[name + '_bytes'] / 2**20
            row['interval_peak_allocated_mib'] = group['interval_peak_allocated_bytes'] / 2**20
            row.update({name + '_mib': snapshot['categories_bytes'][name] / 2**20 for name in CATEGORIES})
            row['inspection_error_count'] = snapshot['inspection_error_count']
            rows.append(row)
        with (directory / 'memory-stage-summary.csv').open('w', newline='') as stream:
            if rows:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        (directory / 'memory-stage-summary.json').write_text(json.dumps(list(self.groups.values()), indent=2) + '\n')
        lines = ['# 按层与阶段统计显存（MiB）', '',
                 '每行取该层/阶段所有 epoch 中 allocated 最高的已观测边界；所有分类来自同一快照并按 storage 去重。',
                 '这是该时刻整个进程的占用，包含其他层常驻状态，不是该层独占或该阶段新增量。',
                 '区间峰值单独列出，不能用边界变量分解解释区间内部瞬时峰值。', '',
                 '| 层 | 阶段 | epoch(0起) | 事件 | allocated | ' + ' | '.join(LABELS) + ' | 未归因差额 | 区间峰值 |',
                 '|---|' + '---|' * (6 + len(CATEGORIES))]
        for row in rows:
            cells = [str(row['layer'] or '—'), row['phase'], str(row['epoch_zero_based']), str(row['sequence']),
                     '{:.3f}'.format(row['allocated_mib'])]
            cells += ['{:.3f}'.format(row[name + '_mib']) for name in CATEGORIES]
            cells += ['{:.3f}'.format(row['unaccounted_mib']), '{:.3f}'.format(row['interval_peak_allocated_mib'])]
            lines.append('| ' + ' | '.join(cells) + ' |')
        lines += ['', '分类规则和别名优先级见 docs/layer_memory_profile.md。JSON 保留选中快照每块 storage 的分类和变量别名。',
                  '分类是可见变量的语义推断；未归因差额保留原符号，不自动解释为激活或泄漏。']
        (directory / 'memory-stage-summary.md').write_text('\n'.join(lines) + '\n')


def main():
    import argparse
    from itertools import zip_longest
    parser = argparse.ArgumentParser(description='Regenerate stage tables from a worker trace; no PyTorch required.')
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    summary = StageMemorySummary()
    with (args.directory / 'memory-events.jsonl').open() as events, (args.directory / 'memory-variables.jsonl').open() as variables:
        for event_line, variable_line in zip_longest(events, variables):
            if event_line is None or variable_line is None:
                parser.error('Event and variable logs have different lengths; incomplete trace.')
            event, details = json.loads(event_line), json.loads(variable_line)
            if event['sequence'] != details['sequence']:
                parser.error('Event and variable sequence mismatch.')
            if 'stage_context' not in event:
                parser.error('This trace predates stage attribution; rerun the diagnostic with the updated tracer.')
            summary.observe(event, details, event['stage_context'])
    summary.write(args.directory)
    print(args.directory / 'memory-stage-summary.md')


if __name__ == '__main__':
    main()
