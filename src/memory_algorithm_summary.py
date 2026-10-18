"""Storage-level training-state attribution; usable offline without PyTorch.

Reported peaks are observed retained-state peaks, not exact counterfactual peaks.
Native temporary workspaces are deliberately not charged to either side.
"""
import csv
import fnmatch
import json
from pathlib import Path
import re

ALGORITHM = ('gradients', 'optimizer_state', 'saved_for_backward', 'algorithm_data',
             'algorithm_cache', 'objective', 'checkpoint')


def layer_name(path):
    path = re.sub(r"\._modules\[['\"]([^'\"]+)['\"]\]", r".\1", path)
    match = re.search(r"(?:layers|convs)\.(\d+)(?:\.|$)", path)
    return 'layer{}'.format(int(match.group(1)) + 1) if match else 'shared'


def classify(details, rules):
    variables = {}
    for v in details['variables']:
        variables.setdefault(v['storage_id'], []).append(v)
    result = []
    for storage in details['storages']:
        vs = variables.get(storage['storage_id'], [])
        choices = []
        owners = set()
        for v in vs:
            path = v['path']
            owner = v.get('memory_owner') or v.get('saved_owner') or layer_name(path)
            if owner != 'shared':
                owners.add(owner)
            # Parameters/buffers/raw inputs win over autograd aliases: saving a
            # reference to an already resident input does not allocate another copy.
            if not path.endswith('.grad') and (v.get('is_parameter') or '._parameters[' in path or '.param_groups[' in path or '._buffers[' in path):
                choices.append((0, 'model_baseline', 'parameter/buffer'))
            if v.get('baseline_storage'):
                choices.append((0, 'model_baseline', 'previously observed baseline storage'))
            for rule in rules:
                if fnmatch.fnmatchcase(path, rule['pattern']):
                    category = rule['category']
                    priority = {'model_baseline': 0, 'gradients': 1, 'optimizer_state': 1,
                                'checkpoint': 1, 'algorithm_data': 2, 'algorithm_cache': 3,
                                'objective': 4, 'saved_for_backward': 5}
                    choices.append((priority[category], category, rule['pattern']))
            if path.endswith('.grad') or path == 'incoming_gradient':
                choices.append((1, 'gradients', 'gradient'))
            if '.state[' in path and ('optimizer' in path or "['exp_avg" in path):
                choices.append((1, 'optimizer_state', 'optimizer state'))
            if 'best_model_state_dict' in path:
                choices.append((1, 'checkpoint', 'early stopping copy'))
            if v.get('saved_for_backward'):
                choices.append((5, 'saved_for_backward', 'live autograd saved-tensor handle'))
        if not choices and details.get('stage_context', {}).get('phase') in ('forward', 'evaluation'):
            from memory_stage_summary import category_for
            if any(category_for(v) in ('activations', 'logits_loss') for v in vs):
                choices.append((8, 'model_forward_visible', 'ordinary visible forward tensor; no training retention evidence'))
        _, category, reason = min(choices) if choices else (9, 'unattributed_visible', 'no explicit evidence')
        result.append(dict(storage, category=category, reason=reason,
                           owner=next(iter(owners)) if len(owners) == 1 else 'shared'))
    return result


class AlgorithmMemorySummary:
    def __init__(self, rules):
        self.rules = rules
        self.groups = {}
        self.global_peak = None

    def observe(self, event, details, context):
        entries = classify(details, self.rules)
        totals = {c: sum(x['bytes'] for x in entries if x['category'] == c)
                  for c in ALGORITHM + ('model_baseline', 'model_forward_visible', 'unattributed_visible')}
        total = sum(totals[c] for c in ALGORITHM)
        snapshot = dict(sequence=event['sequence'], phase=context['phase'],
                        active_layer=layer_name(str(context.get('layer') or '')),
                        epoch_zero_based=context.get('epoch_zero_based'),
                        algorithm_bytes=total, categories_bytes=totals,
                        allocator_bytes=event['allocated_bytes'],
                        allocator_minus_visible_bytes=event['allocated_bytes']-sum(x['bytes'] for x in entries),
                        interval_peak_allocated_bytes=event['interval_peak_allocated_bytes'],
                        storages=entries)
        if self.global_peak is None or total > self.global_peak['algorithm_bytes']:
            self.global_peak = snapshot
        # Each row is a simultaneous owner-specific snapshot, not the total
        # process charged to whichever layer happens to be executing.
        owners = {'all', 'shared'} | {x['owner'] for x in entries}
        for owner in owners:
            subset = entries if owner == 'all' else [x for x in entries if x['owner'] == owner]
            values = {c: sum(x['bytes'] for x in subset if x['category'] == c) for c in totals}
            size = sum(values[c] for c in ALGORITHM)
            key = (snapshot['active_layer'], context['phase'], owner)
            old = self.groups.get(key)
            if old is None or size > old['algorithm_bytes']:
                self.groups[key] = dict(snapshot, owner=owner, algorithm_bytes=size,
                                        categories_bytes=values, storages=subset)
        return snapshot

    def write(self, directory):
        directory = Path(directory)
        rows = []
        for key, snap in sorted(self.groups.items()):
            row = {k: snap[k] for k in ('active_layer', 'phase', 'owner', 'sequence', 'epoch_zero_based')}
            row['algorithm_observed_peak_mib'] = snap['algorithm_bytes']/2**20
            row.update({c+'_mib': n/2**20 for c, n in snap['categories_bytes'].items()})
            row['allocator_minus_visible_mib'] = snap['allocator_minus_visible_bytes']/2**20
            row['process_interval_peak_mib'] = snap['interval_peak_allocated_bytes']/2**20
            rows.append(row)
        with (directory/'algorithm-memory-stages.csv').open('w', newline='') as f:
            if rows:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        payload = dict(schema_version=1, rules=self.rules, global_peak=self.global_peak,
                       stage_peaks=list(self.groups.values()),
                       metric='observed unique CUDA storage retained for training; not exact causal extra allocation')
        (directory/'algorithm-memory-summary.json').write_text(json.dumps(payload, indent=2)+'\n')
        lines = ['# 算法保留显存（MiB）', '',
                 '按 storage 去重；参数/原始图与算法专属数据共享 storage 时基线优先。',
                 'saved_for_backward 表示非基线的 autograd 保留存储；不能证明其每个时刻都比无梯度 forward 多占同等字节。',
                 '峰值是已观测存储峰值，未包含无法观测的 backward/optimizer 临时空间；不是完整因果额外峰值。',
                 'owner=all 是该时刻全模型，其他 owner 为存储归属层；shared 无法唯一分层。各行峰值不能相加。',
                 '未归因差额含模型算子 workspace、未观测训练临时空间和分配器取整，不归给任一方。', '',
                 '|执行层|阶段|存储归属|事件|算法峰值|梯度|优化器|autograd保留|专属数据|缓存|目标|早停|',
                 '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for row in rows:
            cells = [str(row[k]) for k in ('active_layer','phase','owner','sequence')]
            cells += ['{:.3f}'.format(row[k]) for k in ['algorithm_observed_peak_mib']+[c+'_mib' for c in ALGORITHM]]
            lines.append('| '+' | '.join(cells)+' |')
        (directory/'algorithm-memory-stages.md').write_text('\n'.join(lines)+'\n')


def load_rules(path):
    rules = json.loads(Path(path).read_text()) if path else []
    for rule in rules:
        if rule['category'] not in ALGORITHM + ('model_baseline',):
            raise ValueError('Invalid memory attribution category: '+rule['category'])
    return rules


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Compare completed algorithm-memory runs; no torch required.')
    parser.add_argument('--sf', type=Path, required=True)
    parser.add_argument('--bp', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows, sources = [], {}
    for label, path in [('SF', args.sf), ('BP', args.bp)]:
        candidates = sorted(path.glob('run*/algorithm-memory-summary.json'))
        file = path/'algorithm-memory-summary.json'
        if not file.exists():
            if not candidates: parser.error('No algorithm-memory-summary.json under '+str(path))
            file = candidates[-1]
        payload = json.loads(file.read_text())
        if not payload.get('complete', False): parser.error('Incomplete run: '+str(file))
        sources[label] = str(file)
        peak = payload['global_peak']
        print('{}: observed global algorithm peak {:.3f} MiB, event {}, phase {}'.format(
            label, peak['algorithm_bytes']/2**20, peak['sequence'], peak['phase']))
        with file.with_name('algorithm-memory-stages.csv').open() as f:
            rows.extend(dict(algorithm=label, **r) for r in csv.DictReader(f))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'comparison.csv').open('w', newline='') as f:
        if rows:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (args.output/'sources.json').write_text(json.dumps(sources,indent=2)+'\n')
    lines = ['# SF / BP 算法保留显存对比', '',
             '每行使用该执行层/阶段/存储归属自己的观测峰值；不能将各行相加。',
             '原生临时空间未归因；autograd 保存量不是精确的无梯度反事实差值。', '',
             '|算法|执行层|阶段|存储归属|算法峰值 MiB|', '|---|---|---|---|---:|']
    for row in rows:
        lines.append('| {} | {} | {} | {} | {:.3f} |'.format(
            row['algorithm'], row['active_layer'], row['phase'], row['owner'],
            float(row['algorithm_observed_peak_mib'])))
    (args.output/'comparison.md').write_text('\n'.join(lines)+'\n')
    print(args.output/'comparison.csv')
    print(args.output/'comparison.md')


if __name__ == '__main__':
    main()
