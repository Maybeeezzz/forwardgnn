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
             'algorithm_cache', 'objective', 'checkpoint', 'local_heads', 'feedback_state',
             'perturbation_state', 'parameter_backup', 'update_buffer')


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
            explicit = v.get('algorithm_category')
            if explicit is not None:
                if explicit not in ALGORITHM:
                    raise ValueError('Invalid explicit algorithm category: '+str(explicit))
                choices.append((-1, explicit, 'explicit algorithm-owned tensor'))
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
                    # An explicit override is needed for algorithm-owned Parameters/buffers.
                    rank = -1 if rule.get('override_baseline') else priority.get(category, 2)
                    choices.append((rank, category, rule['pattern']))
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
        self.period_peaks = dict.fromkeys(('training', 'evaluation', 'other', 'unclassified'))

    def observe(self, event, details, context):
        entries = classify(details, self.rules)
        totals = {c: sum(x['bytes'] for x in entries if x['category'] == c)
                  for c in ALGORITHM + ('model_baseline', 'model_forward_visible', 'unattributed_visible')}
        total = sum(totals[c] for c in ALGORITHM)
        period = context.get('period', 'unclassified')
        if period == 'warmup':
            self.period_peaks.setdefault('warmup', None)
        if period not in self.period_peaks:
            raise ValueError('Unknown memory period: '+str(period))
        snapshot = dict(sequence=event['sequence'], phase=context['phase'], period=period,
                        active_layer=layer_name(str(context.get('layer') or '')),
                        epoch_zero_based=context.get('epoch_zero_based'),
                        algorithm_bytes=total, categories_bytes=totals,
                        allocator_bytes=event['allocated_bytes'],
                        allocator_minus_visible_bytes=event['allocated_bytes']-sum(x['bytes'] for x in entries),
                        interval_peak_allocated_bytes=event['interval_peak_allocated_bytes'],
                        storages=entries)
        if self.global_peak is None or total > self.global_peak['algorithm_bytes']:
            self.global_peak = snapshot
        old_peak = self.period_peaks[period]
        if old_peak is None or total > old_peak['algorithm_bytes']:
            self.period_peaks[period] = snapshot
        # Each row is a simultaneous owner-specific snapshot, not the total
        # process charged to whichever layer happens to be executing.
        owners = {'all', 'shared'} | {x['owner'] for x in entries}
        for owner in owners:
            subset = entries if owner == 'all' else [x for x in entries if x['owner'] == owner]
            values = {c: sum(x['bytes'] for x in subset if x['category'] == c) for c in totals}
            size = sum(values[c] for c in ALGORITHM)
            key = (period, snapshot['active_layer'], context['phase'], owner)
            old = self.groups.get(key)
            if old is None or size > old['algorithm_bytes']:
                self.groups[key] = dict(snapshot, owner=owner, algorithm_bytes=size,
                                        categories_bytes=values, storages=subset)
        return snapshot

    def write(self, directory):
        directory = Path(directory)
        rows = []
        for key, snap in sorted(self.groups.items()):
            row = {k: snap[k] for k in ('period', 'active_layer', 'phase', 'owner', 'sequence', 'epoch_zero_based')}
            row['algorithm_observed_peak_mib'] = snap['algorithm_bytes']/2**20
            row.update({c+'_mib': n/2**20 for c, n in snap['categories_bytes'].items()})
            row['allocator_minus_visible_mib'] = snap['allocator_minus_visible_bytes']/2**20
            row['process_interval_peak_mib'] = snap['interval_peak_allocated_bytes']/2**20
            rows.append(row)
        with (directory/'algorithm-memory-stages.csv').open('w', newline='') as f:
            if rows:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        payload = dict(schema_version=2, rules=self.rules, global_peak=self.global_peak,
                       period_peaks=self.period_peaks,
                       stage_peaks=list(self.groups.values()),
                       metric='observed unique CUDA storage retained for training; not exact causal extra allocation')
        (directory/'algorithm-memory-summary.json').write_text(json.dumps(payload, indent=2)+'\n')
        period_rows = []
        for period, peak in self.period_peaks.items():
            period_rows.append(dict(period=period, observed=peak is not None,
                                    algorithm_observed_peak_mib=peak['algorithm_bytes']/2**20 if peak else None,
                                    sequence=peak['sequence'] if peak else None,
                                    active_layer=peak['active_layer'] if peak else None,
                                    phase=peak['phase'] if peak else None))
        with (directory/'algorithm-memory-periods.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(period_rows[0]))
            writer.writeheader(); writer.writerows(period_rows)
        lines = ['# 算法保留显存（MiB）', '',
                 '训练、评估、其他（初始化/层间转换等）分别取峰值；未观测不表示零。', '',
                 '|统计区间|算法峰值 MiB|事件|', '|---|---:|---:|']
        for r in period_rows:
            lines.append('| {} | {} | {} |'.format(r['period'],
                         '{:.3f}'.format(r['algorithm_observed_peak_mib']) if r['observed'] else '未观测',
                         r['sequence'] if r['observed'] else '—'))
        lines += ['',
                 '按 storage 去重；参数/原始图与算法专属数据共享 storage 时基线优先。',
                 'saved_for_backward 表示非基线的 autograd 保留存储；不能证明其每个时刻都比无梯度 forward 多占同等字节。',
                 '峰值是已观测存储峰值，未包含无法观测的 backward/optimizer 临时空间；不是完整因果额外峰值。',
                 'owner=all 是该时刻全模型，其他 owner 为存储归属层；shared 无法唯一分层。各行峰值不能相加。',
                 '未归因差额含模型算子 workspace、未观测训练临时空间和分配器取整，不归给任一方。', '',
                 '|统计区间|执行层|阶段|存储归属|事件|算法峰值|'+'|'.join(ALGORITHM)+'|',
                 '|---|---|---|---|---:|---:|'+'|'.join(['---:']*len(ALGORITHM))+'|']
        for row in rows:
            cells = [str(row[k]) for k in ('period','active_layer','phase','owner','sequence')]
            cells += ['{:.3f}'.format(row[k]) for k in ['algorithm_observed_peak_mib']+[c+'_mib' for c in ALGORITHM]]
            lines.append('| '+' | '.join(cells)+' |')
        (directory/'algorithm-memory-stages.md').write_text('\n'.join(lines)+'\n')


def load_rules(path):
    rules = json.loads(Path(path).read_text()) if path else []
    for rule in rules:
        if rule['category'] not in ALGORITHM + ('model_baseline',):
            raise ValueError('Invalid memory attribution category: '+rule['category'])
    return rules


def peak_period_row(label, period, peak):
    """All composition fields come from this one period-maximum snapshot."""
    categories = {k: v/2**20 for k, v in (peak or {}).get('categories_bytes', {}).items()
                  if k in ALGORITHM}
    dominant = max(categories, key=categories.get) if any(categories.values()) else None
    storages = [s for s in (peak or {}).get('storages', []) if s.get('category') in ALGORITHM]
    owners = {}
    for storage in storages:
        owner = storage.get('owner', 'shared')
        owners[owner] = owners.get(owner, 0)+storage['bytes']/2**20
    top = [dict(mib=s['bytes']/2**20, category=s['category'], owner=s.get('owner', 'shared'),
                aliases=s.get('aliases', [])[:3])
           for s in sorted(storages, key=lambda s:s['bytes'], reverse=True)[:3]]
    return dict(algorithm=label, period=period, observed=peak is not None,
                algorithm_observed_peak_mib=peak['algorithm_bytes']/2**20 if peak else None,
                sequence=peak.get('sequence') if peak else None,
                active_layer=peak.get('active_layer') if peak else None,
                phase=peak.get('phase') if peak else None,
                epoch_zero_based=peak.get('epoch_zero_based') if peak else None,
                dominant_category=dominant, dominant_category_mib=categories.get(dominant),
                peak_categories_json=json.dumps(categories, sort_keys=True),
                peak_owners_json=json.dumps(owners, sort_keys=True),
                top_storages_json=json.dumps(top, ensure_ascii=False))


def peak_location(row, prefix):
    def display(key):
        value = row.get(prefix+'_'+key)
        return '未知' if value is None or value == '' else str(value)
    return '{} / {} / epoch={} / event={}'.format(
        display('layer'), display('phase'), display('epoch_zero_based'), display('sequence'))


def peak_comparisons(period_rows):
    """Pair independent SF/BP period maxima, never add layer/stage maxima."""
    indexed = {(r['algorithm'], r['period']): r for r in period_rows}
    result = []
    for period in ('training', 'evaluation'):
        sf, bp = indexed.get(('SF', period), {}), indexed.get(('BP', period), {})
        def value(row):
            return (float(row['algorithm_observed_peak_mib'])
                    if row.get('observed') in (True, 'True') else None)
        s, b = value(sf), value(bp)
        both = s is not None and b is not None
        row = dict(period=period, sf_peak_mib=s, bp_peak_mib=b,
                           saved_mib=b-s if both else None,
                           saving_percent=100*(b-s)/b if both and b > 0 else None,
                           sf_to_bp_ratio=s/b if both and b > 0 else None,
                           sf_layer=sf.get('active_layer', ''), sf_phase=sf.get('phase', ''),
                           bp_layer=bp.get('active_layer', ''), bp_phase=bp.get('phase', ''))
        for prefix, source in (('sf',sf), ('bp',bp)):
            for key in ('epoch_zero_based','sequence','dominant_category','dominant_category_mib',
                        'peak_categories_json','peak_owners_json','top_storages_json'):
                row[prefix+'_'+key] = source.get(key)
        for key in ('layer','phase'):
            left, right = row['sf_'+key], row['bp_'+key]
            known = both and left not in (None,'','shared','other','unclassified') and right not in (None,'','shared','other','unclassified')
            row[key+'_comparison'] = ('same' if left == right else 'different') if known else 'unknown'
        result.append(row)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Compare completed algorithm-memory runs; no torch required.')
    parser.add_argument('--sf', type=Path, required=True)
    parser.add_argument('--bp', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows, sources, period_rows = [], {}, []
    for label, path in [('SF', args.sf), ('BP', args.bp)]:
        candidates = sorted(path.glob('run*/algorithm-memory-summary.json'))
        file = path/'algorithm-memory-summary.json'
        if not file.exists():
            if not candidates: parser.error('No algorithm-memory-summary.json under '+str(path))
            file = candidates[-1]
        payload = json.loads(file.read_text())
        if not payload.get('complete', False): parser.error('Incomplete run: '+str(file))
        if payload.get('schema_version', 0) < 2:
            parser.error('Legacy trace lacks reliable period attribution; rerun with updated code: '+str(file))
        sources[label] = str(file)
        for period, period_peak in payload['period_peaks'].items():
            period_rows.append(peak_period_row(label, period, period_peak))
        for period, peak in payload['period_peaks'].items():
            if peak is None:
                print('{} {}: not observed'.format(label, period))
            else:
                print('{} {}: observed algorithm peak {:.3f} MiB, event {}, phase {}'.format(
                    label, period, peak['algorithm_bytes']/2**20, peak['sequence'], peak['phase']))
        with file.with_name('algorithm-memory-stages.csv').open() as f:
            rows.extend(dict(algorithm=label, **r) for r in csv.DictReader(f))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'comparison.csv').open('w', newline='') as f:
        if rows:
            fields = list(dict.fromkeys(key for row in rows for key in row))
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    with (args.output/'comparison-periods.csv').open('w', newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(period_rows[0]));w.writeheader();w.writerows(period_rows)
    (args.output/'sources.json').write_text(json.dumps(sources,indent=2)+'\n')
    comparison = peak_comparisons(period_rows)
    with (args.output/'comparison-peaks.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(comparison[0])); w.writeheader(); w.writerows(comparison)
    lines = ['# SF / BP 算法保留显存对比', '',
             '节省量 = BP 峰值 − SF 峰值；节省比例 = (BP − SF) / BP。负值表示 SF 更高。',
             '同一统计区间分别取各自全模型峰值，不要求峰值发生在相同层/阶段；不比较模型算子临时空间。', '',
             '|统计区间|SF 峰值 MiB|BP 峰值 MiB|SF 节省 MiB|SF 节省 %|SF 峰值位置|BP 峰值位置|',
             '|---|---:|---:|---:|---:|---|---|']
    for r in comparison:
        numbers = ['未观测' if r[k] is None else '{:.3f}'.format(r[k])
                   for k in ('sf_peak_mib','bp_peak_mib','saved_mib','saving_percent')]
        lines.append('| {} | {} | {} / {} | {} / {} |'.format(
            r['period'], ' | '.join(numbers), r['sf_layer'], r['sf_phase'], r['bp_layer'], r['bp_phase']))
    lines += ['',
             '## 峰值出现位置与构成', '',
             'epoch 从 0 开始；SF 是逐层 epoch，BP 是全模型 epoch，不代表等计算进度。',
             '事件序号仅用于各自运行内定位，不能跨运行相减得到时间差。相等峰值保留首次观测位置。',
             '执行层不等于存储归属层；峰值时各归属层占用在 CSV 的 peak_owners_json 中。', '',
             '|区间|SF 层 / 阶段 / epoch / 事件|BP 层 / 阶段 / epoch / 事件|执行层比较|阶段比较|SF 主导类别|BP 主导类别|',
             '|---|---|---|---|---|---|---|']
    for r in comparison:
        lines.append('| {} | {} | {} | {} | {} | {} | {} |'.format(
            r['period'], peak_location(r,'sf'), peak_location(r,'bp'),
            r['layer_comparison'],r['phase_comparison'],
            r['sf_dominant_category'] or '未知',r['bp_dominant_category'] or '未知'))
    lines += ['', '类别占用、归属层占用和前三大 storage 均来自各自同一个峰值快照，见 comparison-peaks.csv；不是类别独立峰值之和。', '',
             '|算法|统计区间|算法峰值 MiB|事件|', '|---|---|---:|---:|']
    for r in period_rows:
        lines.append('| {} | {} | {} | {} |'.format(r['algorithm'], r['period'],
            '{:.3f}'.format(r['algorithm_observed_peak_mib']) if r['observed'] else '未观测',
            r['sequence'] if r['observed'] else '—'))
    lines += ['',
             '每行使用该执行层/阶段/存储归属自己的观测峰值；不能将各行相加。',
             '原生临时空间未归因；autograd 保存量不是精确的无梯度反事实差值。', '',
             '|算法|统计区间|执行层|阶段|存储归属|算法峰值 MiB|', '|---|---|---|---|---|---:|']
    for row in rows:
        lines.append('| {} | {} | {} | {} | {} | {:.3f} |'.format(
            row['algorithm'], row['period'], row['active_layer'], row['phase'], row['owner'],
            float(row['algorithm_observed_peak_mib'])))
    (args.output/'comparison.md').write_text('\n'.join(lines)+'\n')
    print(args.output/'comparison.csv')
    print(args.output/'comparison.md')


if __name__ == '__main__':
    main()
