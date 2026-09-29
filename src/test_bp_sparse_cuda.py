"""CUDA BP matrix with official PyG operators and torch_sparse.SparseTensor.

Tasks x graph topologies x GNNs; paired initialization; isolated processes.
Reports CUDA allocator high-water marks, not sampled or whole-device memory.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from datetime import datetime
from time import perf_counter

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
import torch_geometric
from torch_geometric.nn import GCN, GraphSAGE, GIN, GAT


def inputs(args):
    g = torch.Generator().manual_seed(args.seed)
    x = torch.randn(args.nodes, args.features, generator=g)
    y = torch.randint(0, 6, (args.nodes,), generator=g)
    if args.topology == 'uniform':
        edge = torch.randint(args.nodes, (2, args.edges // 2), generator=g)
    else:
        probabilities = torch.arange(1, args.nodes + 1).float().rsqrt()
        edge = torch.multinomial(probabilities, args.edges, replacement=True, generator=g).reshape(2, -1)
    edge = torch.sort(edge, dim=0).values
    edge = torch.unique(edge[:, edge[0] != edge[1]], dim=1)
    if args.task == 'link-pred':
        # Hold out 20% of positive pairs from the message-passing graph.
        all_pairs = set(map(tuple, edge.t().tolist()))
        order = torch.randperm(edge.shape[1], generator=g)
        edge = edge[:, order[:int(edge.shape[1] * 0.8)]]
        available_negatives = args.nodes * (args.nodes - 1) // 2 - len(all_pairs)
        count = min(512, edge.shape[1], available_negatives)
        if count == 0:
            raise ValueError('Link prediction needs at least one training edge and one non-edge.')
        positive = edge[:, :count]
        negative = set()
        while len(negative) < count:
            candidates = torch.randint(args.nodes, (max(32, count * 2), 2), generator=g).tolist()
            for u, v in candidates:
                u, v = sorted((u, v))
                if u != v and (u, v) not in all_pairs:
                    negative.add((u, v))
                if len(negative) == count:
                    break
        negative = torch.tensor(sorted(negative)).t()
        y = {'pairs': torch.cat([positive, negative], dim=1),
             'labels': torch.cat([torch.ones(count), torch.zeros(count)])}
    edge = torch.cat([edge, edge.flip(0)], dim=1)
    return x, y, edge


def setup(args, backend):
    from torch_sparse import SparseTensor
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)
    x, y, edge = inputs(args)
    cls = {'GCN': GCN, 'SAGE': GraphSAGE, 'GIN': GIN, 'GAT': GAT}[args.architecture]
    kwargs = {'heads': 4} if args.architecture == 'GAT' else {}
    # Same BP architecture as the author's PyG baseline; GIN is an extension.
    model = cls(in_channels=args.features, hidden_channels=args.hidden, num_layers=2,
                out_channels=6 if args.task == 'node-class' else args.hidden,
                dropout=0.0, act='relu', **kwargs).to(args.device)
    edge_gpu = edge.to(args.device)
    if backend == 'sparse-tensor':
        graph = SparseTensor(row=edge_gpu[1], col=edge_gpu[0],
                             sparse_sizes=(args.nodes, args.nodes))
    else:
        graph = edge_gpu
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
    y = {k: v.to(args.device) for k, v in y.items()} if isinstance(y, dict) else y.to(args.device)
    return model, optimizer, x.to(args.device), y, graph, edge.shape[1]


def loss_for(model, x, y, graph):
    embeddings = model(x, graph)
    if isinstance(y, dict):
        pairs = y['pairs']
        logits = (embeddings[pairs[0]] * embeddings[pairs[1]]).sum(dim=1)
        return logits, torch.nn.functional.binary_cross_entropy_with_logits(logits, y['labels'])
    count = int(x.shape[0] * 0.6)
    return embeddings, torch.nn.functional.cross_entropy(embeddings[:count], y[:count])


def check(args):
    setups = [setup(args, name) for name in ['edge-index', 'sparse-tensor']]
    calls = [[0] * len(s[0].convs) for s in setups]
    handles = []
    for backend_i, s in enumerate(setups):
        for layer_i, conv in enumerate(s[0].convs):
            def count(module, inputs, output, b=backend_i, l=layer_i):
                calls[b][l] += 1
            handles.append(conv.register_message_and_aggregate_forward_hook(count))
    records = []
    all_pass = True
    for step in range(3):
        values = []
        for model, opt, x, y, graph, _ in setups:
            opt.zero_grad()
            logits, loss = loss_for(model, x, y, graph)
            loss.backward()
            gradients = [p.grad.detach().cpu().clone() for p in model.parameters()]
            opt.step()
            values.append([logits.detach().cpu(), loss.detach().cpu(), gradients,
                           [p.detach().cpu().clone() for p in model.parameters()]])
        a, b = values
        row = {'step': step + 1}
        for name, left, right in [('logits', [a[0]], [b[0]]), ('loss', [a[1]], [b[1]]),
                                  ('gradients', a[2], b[2]), ('parameters', a[3], b[3])]:
            row[name + '_max_abs_diff'] = max((u - v).abs().max().item() for u, v in zip(left, right))
            passed = all(torch.allclose(u, v, atol=1e-5, rtol=1e-4) for u, v in zip(left, right))
            row[name + '_allclose'] = passed
            all_pass = all_pass and passed
        records.append(row)
    for handle in handles:
        handle.remove()
    expected = [3 if conv.fuse else 0 for conv in setups[1][0].convs]
    path_valid = calls[0] == [0] * len(expected) and calls[1] == expected
    return {'allclose': all_pass, 'atol': 1e-5, 'rtol': 1e-4, 'steps': records,
            'fused_calls': {'edge-index': calls[0], 'sparse-tensor': calls[1]},
            'expected_sparse_fused_calls': expected, 'path_valid': path_valid}



def begin_window(args):
    torch.cuda.synchronize(args.device)
    baseline = {'allocated_bytes': torch.cuda.memory_allocated(args.device),
                'reserved_bytes': torch.cuda.memory_reserved(args.device)}
    torch.cuda.reset_peak_memory_stats(args.device)
    return baseline, perf_counter()


def end_window(args, token):
    torch.cuda.synchronize(args.device)
    baseline, started = token
    return {'baseline': baseline,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(args.device),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(args.device),
            'end_allocated_bytes': torch.cuda.memory_allocated(args.device),
            'end_reserved_bytes': torch.cuda.memory_reserved(args.device),
            'synchronized_seconds': perf_counter() - started}


def combine(windows):
    return {key: max(w[key] for w in windows)
            for key in ['peak_allocated_bytes', 'peak_reserved_bytes']}


def measure(args, backend):
    # First GPU graph/model allocations and sparse conversion are inside this window.
    token = begin_window(args)
    model, opt, x, y, graph, edges = setup(args, backend)
    initialization = end_window(args, token)

    def update(record=False):
        stages = []
        if record:
            token = begin_window(args)
        opt.zero_grad(set_to_none=True)
        logits, loss = loss_for(model, x, y, graph)
        if record:
            stages.append(dict(phase='forward_loss', **end_window(args, token)))
            token = begin_window(args)
        loss.backward()
        if record:
            stages.append(dict(phase='backward', **end_window(args, token)))
            token = begin_window(args)
        opt.step()
        if record:
            stages.append(dict(phase='adam_step', **end_window(args, token)))
        return stages

    cold = update(record=True)
    token = begin_window(args)
    for _ in range(args.warmup - 1):
        update()
    warmup = end_window(args, token)
    steady = []
    for step in range(args.steps):
        steady.extend(dict(step=step + 1, **w) for w in update(record=True))
    return {'backend': backend, 'actual_directed_edges': edges,
            'initialization': initialization, 'cold_stages': cold,
            'cold_training': combine(cold), 'additional_warmup': warmup,
            'steady_stages': steady, 'steady_training': combine(steady),
            'whole_run': combine([initialization, warmup] + cold + steady),
            'fused_layers': [bool(conv.fuse) for conv in model.convs],
            'warmup_updates_including_cold': args.warmup,
            'measured_updates': args.steps,
            'scope': 'Absolute CUDA allocator peaks. Initialization includes model/data '
                     'transfer and GPU SparseTensor construction. Cold step includes Adam '
                     'state creation; steady follows warmup. Whole-run is maximum across '
                     'all windows. Excludes CUDA context and external allocations not tracked '
                     'by PyTorch. No sampling or measurement hooks.'}


def command_for(args, destination, task, topology, architecture, worker):
    command = [sys.executable, str(Path(__file__).resolve()), '--worker', worker,
               '--output', str(destination.resolve()), '--task', task,
               '--topology', topology, '--architecture', architecture]
    for key in ['nodes', 'edges', 'features', 'hidden', 'steps', 'seed', 'gpu', 'warmup']:
        command.extend(['--' + key, str(getattr(args, key))])
    return command


def run_child(args, destination, task, topology, architecture, worker):
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / (worker + '.log')).open('w') as handle:
        completed = subprocess.run(command_for(args, destination, task, topology, architecture, worker),
                                   stdout=handle, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError('Worker failed; inspect {}'.format(destination / (worker + '.log')))
    return json.loads((destination / (worker + '.json')).read_text())


def run_matrix(args):
    dimensions = [('task', ['node-class', 'link-pred']),
                  ('topology', ['uniform', 'hub']),
                  ('architecture', ['GCN', 'SAGE', 'GIN', 'GAT'])]
    tasks, topologies, architectures = [[getattr(args, key)] if getattr(args, key) != 'all'
                                        else options for key, options in dimensions]
    rows = []
    for task in tasks:
        for topology in topologies:
            for architecture in architectures:
                name = '{}_{}_{}'.format(task, topology, architecture)
                destination = args.output / name
                row = dict(task=task, topology=topology, architecture=architecture, directory=name)
                print('Running', name, flush=True)
                try:
                    numerical = run_child(args, destination, task, topology, architecture, 'check')
                    row['check'] = numerical
                    if not numerical['allclose'] or not numerical['path_valid']:
                        row['status'] = 'numerical_or_path_mismatch'
                    else:
                        repeats = []
                        for repeat in range(args.repeats):
                            pair = {}
                            # Alternate execution order; every backend/repeat is a new process.
                            order = ['edge-index', 'sparse-tensor']
                            if repeat % 2:
                                order.reverse()
                            for backend in order:
                                pair[backend] = run_child(args, destination / ('repeat-{}'.format(repeat)),
                                                          task, topology, architecture, backend)
                            repeats.append(pair)
                        row.update(status='completed', repetitions=repeats)
                        row['summary'] = {}
                        for scope in ['initialization', 'cold_training', 'steady_training', 'whole_run']:
                            metrics = {}
                            for metric in ['peak_allocated_bytes', 'peak_reserved_bytes']:
                                values = {b: [p[b][scope][metric] / 2**20 for p in repeats]
                                          for b in ['edge-index', 'sparse-tensor']}
                                per_pair = [100 * (1 - p['sparse-tensor'][scope][metric] /
                                                       p['edge-index'][scope][metric]) for p in repeats]
                                metrics[metric] = {
                                    'mib': {b: {'values': v, 'mean': statistics.mean(v),
                                                'std': statistics.pstdev(v)} for b, v in values.items()},
                                    'paired_reduction_percent': per_pair,
                                    'mean_paired_reduction_percent': statistics.mean(per_pair)}
                            row['summary'][scope] = metrics
                except RuntimeError as exc:
                    row.update(status='execution_failed', error=str(exc))
                rows.append(row)
                (args.output / 'matrix.json').write_text(json.dumps(rows, indent=2) + '\n')
                print(name, row['status'], flush=True)
    lines = ['# CUDA BP SparseTensor matrix', '',
             'Synthetic graphs; CUDA allocator high-water marks. '
             'Not full-paper accuracy reproduction. See JSON for cold/initialization/whole-run peaks.', '',
             '| Task | Topology | GNN | Status | Steady allocated MiB (edge / sparse) | Reduction |',
             '|---|---|---|---|---|---|']
    for row in rows:
        peak, reduction = '—', '—'
        if row['status'] == 'completed':
            metric = row['summary']['steady_training']['peak_allocated_bytes']
            peak = ' / '.join('{:.2f} ± {:.2f}'.format(metric['mib'][b]['mean'], metric['mib'][b]['std'])
                              for b in ['edge-index', 'sparse-tensor'])
            reduction = '{:.2f}%'.format(metric['mean_paired_reduction_percent'])
        lines.append('| {} | {} | {} | {} | {} | {} |'.format(
            row['task'], row['topology'], row['architecture'], row['status'], peak, reduction))
    lines.extend(['', 'GAT sparse input does not imply fused SpMM; see check.json for actual fused calls.',
                  'Reserved includes allocator cache. Synchronization affects timings; no formal speedup claim.'])
    (args.output / 'report.md').write_text('\n'.join(lines) + '\n')
    print('Results:', args.output)
    return all(row['status'] == 'completed' for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in [('nodes', 2048), ('edges', 32768), ('features', 64), ('hidden', 64),
                          ('steps', 10), ('seed', 100), ('gpu', 0), ('warmup', 2), ('repeats', 3)]:
        parser.add_argument('--' + name, type=int, default=default)
    parser.add_argument('--architecture', choices=['all', 'GCN', 'SAGE', 'GIN', 'GAT'], default='all')
    parser.add_argument('--task', choices=['all', 'node-class', 'link-pred'], default='all')
    parser.add_argument('--topology', choices=['all', 'uniform', 'hub'], default='all')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--worker', choices=['check', 'edge-index', 'sparse-tensor'])
    args = parser.parse_args()
    if min(args.nodes, args.edges, args.features, args.hidden) < 2 or min(args.steps, args.warmup, args.repeats) < 1:
        parser.error('Graph/model sizes must be >=2 and step/warmup/repeat budgets >=1.')
    if args.architecture in ['all', 'GAT'] and args.hidden % 4:
        parser.error('GAT hidden size must be divisible by 4 heads.')
    if args.worker and 'all' in (args.architecture, args.task, args.topology):
        parser.error('Worker dimensions must be explicit.')
    if not torch.cuda.is_available() or not 0 <= args.gpu < torch.cuda.device_count():
        raise RuntimeError('A valid CUDA GPU is required; no CPU or other-device fallback.')
    try:
        torch_sparse = importlib.import_module('torch_sparse')
    except (ImportError, OSError) as exc:
        raise RuntimeError('Install torch_sparse compatible with your PyTorch/CUDA environment.') from exc
    args.device = torch.device('cuda:{}'.format(args.gpu))
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # gather-scatter CUDA may use nondeterministic atomics. Keep tolerance checks;
    # do not enforce a deterministic-only kernel set unsupported by the baseline.
    if args.output is None:
        args.output = Path(__file__).resolve().parents[1] / 'results' / (
            'bp-sparse-cuda-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        result = check(args) if args.worker == 'check' else measure(args, args.worker)
        (args.output / (args.worker + '.json')).write_text(json.dumps(result, indent=2) + '\n')
        return
    if any(args.output.iterdir()):
        raise FileExistsError('Use a new/empty --output directory to avoid mixing experiments.')
    configuration = {k: v for k, v in vars(args).items() if k not in ['output', 'device']}
    configuration.update(torch_version=torch.__version__, pyg_version=torch_geometric.__version__,
                         torch_sparse_version=torch_sparse.__version__, cuda_version=torch.version.cuda,
                         gpu_name=torch.cuda.get_device_name(args.device), tf32=False,
                         deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                         numerical_tolerance={'atol': 1e-5, 'rtol': 1e-4},
                         runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output / 'configuration.json').write_text(json.dumps(configuration, indent=2) + '\n')
    (args.output / 'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    if not run_matrix(args):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
