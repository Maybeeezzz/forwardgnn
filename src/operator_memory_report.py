"""Read Kineto Chrome traces and attribute allocation events without PyTorch."""
import argparse
import csv
import heapq
import json
from collections import defaultdict
from pathlib import Path


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_csv(path, rows, fields):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row.get(key), ensure_ascii=False)
                             if isinstance(row.get(key), (dict, list)) else row.get(key) for key in fields})


def analyze_trace(trace, device_index=0):
    events = trace.get('traceEvents', []) if isinstance(trace, dict) else trace
    operators = defaultdict(list)
    memories = []
    for index, event in enumerate(events):
        args = event.get('args', {})
        if event.get('ph') == 'X' and event.get('cat') in ('cpu_op', 'Operator'):
            operators[(event.get('pid'), event.get('tid'))].append(event)
        if event.get('name') == '[memory]':
            dtype = args.get('Device Type')
            if str(dtype).lower() not in ('1', 'cuda'):
                continue
            if number(args.get('Device Id')) != device_index or number(args.get('Bytes')) is None:
                continue
            memories.append((index, event))
    for values in operators.values():
        values.sort(key=lambda e: e['ts'])
    positions, heaps = defaultdict(int), defaultdict(list)
    allocations, timeline, live = [], [], {}
    peak_total, peak_time, peak_ids = None, None, []
    orphan_frees = 0
    for index, event in sorted(memories, key=lambda pair: (pair[1]['ts'], pair[0])):
        thread = (event.get('pid'), event.get('tid'))
        stamp = event['ts']
        candidates = operators[thread]
        while positions[thread] < len(candidates) and candidates[positions[thread]]['ts'] <= stamp:
            position = positions[thread]
            op = candidates[position]
            heapq.heappush(heaps[thread], (-op['ts'], op['dur'], position, op))
            positions[thread] += 1
        heap = heaps[thread]
        while heap and heap[0][3]['ts'] + heap[0][3]['dur'] < stamp:
            heapq.heappop(heap)
        owner = heap[0][3] if heap else None
        args = event['args']
        size = int(number(args['Bytes']))
        address = args.get('Addr')
        total = number(args.get('Total Allocated'))
        row = dict(event_index=index, timestamp_us=stamp, bytes=size, address=address,
                   total_allocated_bytes=total, total_reserved_bytes=number(args.get('Total Reserved')),
                   operator=owner['name'] if owner else '<unattributed>',
                   operator_start_us=owner['ts'] if owner else None,
                   input_shapes=owner.get('args', {}).get('Input Dims') if owner else None,
                   input_types=owner.get('args', {}).get('Input type') if owner else None,
                   allocation_id=None)
        # Addresses may be reused: each positive event receives a fresh lifetime ID.
        if size > 0:
            allocation = dict(row, allocation_id=len(allocations), freed_at_us=None)
            allocations.append(allocation)
            row['allocation_id'] = allocation['allocation_id']
            if address is not None:
                live[str(address)] = allocation['allocation_id']
        elif size < 0:
            identity = live.pop(str(address), None) if address is not None else None
            row['allocation_id'] = identity
            if identity is None:
                orphan_frees += 1
            else:
                allocations[identity]['freed_at_us'] = stamp
        timeline.append(row)
        if total is not None and (peak_total is None or total > peak_total):
            peak_total, peak_time = total, stamp
            peak_ids = list(live.values())
    by_operator = {}
    for allocation in allocations:
        name = allocation['operator']
        group = by_operator.setdefault(name, dict(operator=name, allocation_count=0,
                                                 gross_allocated_bytes=0, largest_allocation_bytes=0,
                                                 live_at_trace_peak_bytes=0))
        group['allocation_count'] += 1
        group['gross_allocated_bytes'] += allocation['bytes']
        group['largest_allocation_bytes'] = max(group['largest_allocation_bytes'], allocation['bytes'])
    for identity in peak_ids:
        allocation = allocations[identity]
        by_operator[allocation['operator']]['live_at_trace_peak_bytes'] += allocation['bytes']
    return dict(memory_events=timeline, allocations=allocations,
                operators=sorted(by_operator.values(), key=lambda r: r['gross_allocated_bytes'], reverse=True),
                trace_peak_allocated_bytes=peak_total, trace_peak_timestamp_us=peak_time,
                live_at_trace_peak=[allocations[i] for i in peak_ids],
                unmatched_frees=orphan_frees,
                allocation_owner_rule='Innermost enclosing CPU operator on the same pid/tid; not CUDA kernel ownership.',
                status='ok' if memories else 'no_cuda_memory_events')


def generate_report(directory):
    directory = Path(directory)
    metadata = json.loads((directory / 'operator-metadata.json').read_text())
    trace = json.loads((directory / 'operator-trace.json').read_text())
    result = analyze_trace(trace, metadata['device_index'])
    (directory / 'operator-allocation-summary.json').write_text(json.dumps(result, indent=2) + '\n')
    write_csv(directory / 'operator-memory-events.csv', result['memory_events'],
              ['event_index', 'timestamp_us', 'bytes', 'address', 'allocation_id', 'total_allocated_bytes',
               'total_reserved_bytes', 'operator', 'operator_start_us', 'input_shapes', 'input_types'])
    write_csv(directory / 'operator-allocations.csv', sorted(result['allocations'], key=lambda r: r['bytes'], reverse=True),
              ['allocation_id', 'operator', 'bytes', 'address', 'timestamp_us', 'freed_at_us', 'input_shapes', 'input_types'])
    write_csv(directory / 'operator-allocation-totals.csv', result['operators'],
              ['operator', 'allocation_count', 'gross_allocated_bytes', 'largest_allocation_bytes', 'live_at_trace_peak_bytes'])
    profiler_events_path = directory / 'operator-events.json'
    stack_by_name = {}
    if profiler_events_path.exists():
        for event in json.loads(profiler_events_path.read_text()):
            if event.get('stack') and event['name'] not in stack_by_name:
                stack_by_name[event['name']] = event['stack']
    lines = ['# 算子级显存分析', '', '状态：{}'.format(result['status']), '',
             '目标模块：{}；第 {} 次调用；record_shapes={}；tensor_ops={}'.format(
                 metadata.get('target_module') or '<整个 context>', metadata.get('target_call'),
                 metadata.get('record_shapes'), metadata.get('tensor_ops')), '',
             '| 窗口指标 | MiB |', '|---|---:|']
    for name in ('start_allocated_bytes', 'peak_allocated_bytes', 'end_allocated_bytes', 'peak_reserved_bytes'):
        value = metadata.get(name)
        lines.append('| {} | {} |'.format(name, '{:.3f}'.format(value / 2**20) if value is not None else '不可用'))
    lines += ['', '## 产生最大分配的 CPU 操作', '',
              '累计分配不等于同时存活显存；临时分配后释放仍计入累计值。不可把父/子操作的 inclusive 内存相加。', '',
              '| 操作 | 分配次数 | 累计分配 MiB | 最大单次 MiB | trace峰值时本窗口新分配且存活 MiB |',
              '|---|---:|---:|---:|---:|']
    for row in result['operators'][:30]:
        lines.append('| {} | {} | {:.3f} | {:.3f} | {:.3f} |'.format(
            row['operator'].replace('|', '\\|'), row['allocation_count'], row['gross_allocated_bytes']/2**20,
            row['largest_allocation_bytes']/2**20, row['live_at_trace_peak_bytes']/2**20))
    lines += ['', '## 最大单次分配及生命周期', '',
              '| ID | 操作 | MiB | 分配时间 us | 释放时间 us | 操作输入形状 |',
              '|---:|---|---:|---:|---|---|']
    for row in sorted(result['allocations'], key=lambda r: r['bytes'], reverse=True)[:20]:
        lines.append('| {} | {} | {:.3f} | {} | {} | {} |'.format(
            row['allocation_id'], row['operator'].replace('|', '\\|'), row['bytes']/2**20,
            row['timestamp_us'], row['freed_at_us'], str(row['input_shapes']).replace('|', '\\|')))
    lines += ['', '## 目标调用的输入/输出 Tensor', '',
              '这些是窗口边界 Tensor 元数据，不包括未暴露为 Python Tensor 的内部 workspace。', '',
              '| 变量 | shape | dtype | device | storage MiB |', '|---|---|---|---|---:|']
    for row in metadata.get('inputs', []) + metadata.get('outputs', []):
        size = row.get('storage_bytes')
        lines.append('| {} | {} | {} | {} | {} |'.format(
            row['path'], row['shape'], row['dtype'], row['device'],
            '{:.3f}'.format(size / 2**20) if size is not None else '不可用'))
    lines += ['', '## 大分配操作的调用栈示例', '']
    for row in result['operators'][:10]:
        stack = stack_by_name.get(row['operator'])
        lines += ['**{}**'.format(row['operator']), '',
                  '```text', '\n'.join(stack[:12]) if stack else '当前 profiler 未返回此操作的调用栈；请查看原始 trace。',
                  '```', '']
    lines += ['', '## 解释范围', '',
              '- allocator 峰值取自 CUDA API；trace 高水位只来自有 Total Allocated 的事件，两者需交叉检查。',
              '- 存活列表仅含本窗口记录到的新分配；窗口开始前已有的输入/参数不在该列表内。',
              '- 按相同 CPU 线程上的包围区间归属操作；没有匹配时保留 unattributed，不猜测 CUDA kernel。',
              '- Input Dims 是操作输入，不是分配块的形状；workspace 不一定对应 Python Tensor。',
              '- 启用 tensor_ops 后另有 operator-tensors.jsonl，记录可被 Python dispatcher 观察到的输入/输出 storage、形状、dtype；原生内部临时 Tensor 未必可见。',
              '- 地址会复用，比较 tensor storage 与分配记录时必须同时考虑生命周期；不能仅凭相同地址认定是同一 Tensor。',
              '- record_shapes 可能保留张量引用并改变峰值；需与不开启形状记录的独立运行对照。',
              '- operator-events.json 中 self/inclusive 内存为净变化，瞬时大分配可能净值为零；以分配事件和高水位为准。',
              '- trace、调用栈和算子形状能定位操作，但不能保证给底层每个临时缓冲区恢复变量名。']
    if result['status'] != 'ok':
        lines += ['', '**未发现 CUDA [memory] 事件，不能将缺失数据视作零分配。保留原始 trace 和 profiler 事件供检查。**']
    (directory / 'operator-memory-report.md').write_text('\n'.join(lines) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    generate_report(args.directory)
    print(args.directory / 'operator-memory-report.md')
