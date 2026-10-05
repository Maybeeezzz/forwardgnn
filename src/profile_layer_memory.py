"""Layer/stage CUDA diagnostics for the original 1..4-layer node-classification matrix."""
import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ['CitationFull-CiteSeer', 'CitationFull-Cora_ML',
            'CitationFull-PubMed', 'Amazon-Photo', 'GitHub']


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    p.add_argument('--architectures', nargs='+', choices=['GCN', 'SAGE', 'GAT'], default=['GCN', 'SAGE', 'GAT'])
    p.add_argument('--depths', nargs='+', type=int, default=[1, 2, 3, 4])
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--runs', type=int, default=5)
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--patience', type=int, default=100)
    p.add_argument('--val-every', type=int, default=2)
    p.add_argument('--seed', type=int, default=100)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--worker-config', type=Path, help=argparse.SUPPRESS)
    a = p.parse_args()
    if min(a.depths + [a.hidden, a.epochs, a.val_every]) < 1 or not 1 <= a.runs <= 5 or a.gpu < 0:
        p.error('Positive dimensions/epochs, 1..5 runs and CUDA gpu >= 0 required.')
    if 'GAT' in a.architectures and a.hidden % 4:
        p.error('GAT hidden size must be divisible by 4.')
    if len(set(a.depths)) != len(a.depths):
        p.error('Depths must be unique.')
    if any(d not in (1, 2, 3, 4) for d in a.depths):
        p.error('This diagnostic supports only the original 1..4 layers.')
    return a


def worker_impl(config, state):
    # Imports are deliberately lazy: planning and result aggregation need no ML runtime.
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    import torch_geometric
    import train_backprop as bp
    import train_forward as sf
    from datasets import load_node_classification_data
    from datasets.datasplit import DataSplit
    from utils.train_utils import ResultManager, SeedManager, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; CPU/MPS measurements are not substituted.')
    device = torch.device('cuda:{}'.format(config['gpu']))
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seeds = SeedManager(config['seed'])
    seeds.set_run_i(config['run'])
    set_seed(seeds.get_run_seed(), deterministic=config['architecture'] != 'GAT')
    for path in DataSplit(config['dataset'], num_splits=5).node_split_paths(config['run']).values():
        if not path.is_file():
            raise FileNotFoundError('Published dataset split required: {}'.format(path))
    args = argparse.Namespace(
        dataset=config['dataset'], model=('GNN_SingleForward-' if config['method'] == 'sf' else 'GNN-') + config['architecture'],
        num_layers=config['depth'], num_hidden=config['hidden'], task='node-class',
        lr=0.001, epochs=config['epochs'], patience=config['patience'],
        val_every=config['val_every'], val_from=0, device=device, cuda=True,
        append_label=None, aug_edge_direction='bidirection', temperature=1.0,
        results_dir=Path(config['directory']), seed=config['seed'])
    data = load_node_classification_data(args, split_i=config['run'])
    graph = {'nodes': data.num_nodes, 'edges': data.num_edges,
             'features': data.num_features, 'classes': int(data.num_classes)}
    model = (sf.build_node_classification_model(args.model, args.num_layers, args.num_hidden,
             'forwardforward_loss_fn', args.lr, data, args) if config['method'] == 'sf'
             else bp.build_model(data, args))
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    manager = ResultManager('layer-profile', args, seeds)
    from utils.layer_memory_trace import LayerMemoryTrace
    tracer = LayerMemoryTrace(torch, device, model, config['directory'])
    state['tracer'] = tracer
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    tracer.start()
    start = time.perf_counter()
    if config['method'] == 'sf':
        model = model.to(device)
        training = model.forward_train(data, manager, config['run'])
    else:
        trainer = bp.NodeClassificationTrainer(model, data, device, args.lr, args.epochs, args.patience, args)
        elapsed, epoch = trainer.train()
        training = {'train_time': elapsed, 'train_epochs': [epoch]}
    torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - start

    training_peak = tracer.end_window('training')
    # Separate final inference window, but include every window in whole-run maximum.
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        if config['method'] == 'sf':
            accuracy, _ = model.eval_model(data.test_mask)
        else:
            # Original BP reports a fraction; SF reports percent.
            accuracy = 100.0 * trainer.test()
    torch.cuda.synchronize(device)
    inference_peak = tracer.end_window('final_inference')
    return dict(config, status='ok', graph=graph, parameter_bytes=parameter_bytes,
                run_seed=seeds.get_run_seed(), accuracy_percent=float(accuracy),
                training=training, training_seconds=training_seconds,
                training_peak=training_peak, inference_peak=inference_peak,
                whole_peak={k: max(training_peak[k], inference_peak[k]) for k in training_peak},
                environment={'torch': torch.__version__, 'pyg': torch_geometric.__version__,
                             'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(device),
                             'total_memory_bytes': torch.cuda.get_device_properties(device).total_memory})


def worker(config):
    state = {}
    try:
        return worker_impl(config, state)
    finally:
        if 'tracer' in state:
            state['tracer'].close()


def summarize(rows, output):
    groups = {}
    for row in rows:
        key = (row['dataset'], row['architecture'], row['method'], row['depth'])
        groups.setdefault(key, []).append(row)
    table = []
    for key, values in sorted(groups.items()):
        good = [r for r in values if r['status'] == 'ok']
        peaks = [r['whole_peak']['allocated_bytes'] / 2**20 for r in good]
        table.append(dict(zip(['dataset', 'architecture', 'method', 'depth'], key),
                          completed=len(good), attempted=len(values),
                          mean_mib=statistics.mean(peaks) if peaks else None,
                          std_mib=statistics.pstdev(peaks) if peaks else None))
    with (output / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    # Only fit fully paired depth points; never turn missing/OOM runs into zero memory.
    trends = []
    for dataset, arch in sorted({(r['dataset'], r['architecture']) for r in rows}):
        subset = [r for r in table if r['dataset'] == dataset and r['architecture'] == arch]
        depths = sorted({r['depth'] for r in subset})
        common = [d for d in depths if all(any(r['depth'] == d and r['method'] == method
                  and r['completed'] == r['attempted'] for r in subset) for method in ['sf', 'bp'])]
        for method in ['sf', 'bp']:
            points = sorted((r['depth'], r['mean_mib']) for r in subset
                            if r['method'] == method and r['depth'] in common)
            if len(points) < 2:
                continue
            xs, ys = zip(*points)
            xm, ym = statistics.mean(xs), statistics.mean(ys)
            slope = sum((x-xm)*(y-ym) for x, y in points) / sum((x-xm)**2 for x in xs)
            trends.append({'dataset': dataset, 'architecture': arch, 'method': method,
                           'paired_depths': common, 'complete_depth_range': common == depths,
                           'slope_mib_per_layer': slope,
                           'endpoint_growth_percent': 100 * (ys[-1] / ys[0] - 1)})
    write_json(output / 'trends.json', trends)
    write_json(output / 'matrix.json', rows)
    lines = ['# Depth memory sweep', '', 'Whole-run CUDA peak allocated; mean ± population SD.', '',
             '| Dataset | Model | Method | Depth | Completed | MiB |',
             '|---|---|---|---:|---:|---:|']
    for r in table:
        value = '{:.2f} ± {:.2f}'.format(r['mean_mib'], r['std_mib']) if r['mean_mib'] is not None else '—'
        lines.append('| {dataset} | {architecture} | {method} | {depth} | {completed}/{attempted} | '.format(**r) + value + ' |')
    lines += ['', 'See trends.json for slopes on fully paired depths only. Missing/OOM points are excluded,',
              'not zero or GPU-capacity estimates. This report does not automatically assert the hypothesis.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n')


def main(a):
    if a.worker_config:
        config = json.loads(a.worker_config.read_text())
        try:
            row = worker(config)
        except Exception as exc:
            traceback.print_exc()
            row = dict(config, status='oom' if 'out of memory' in str(exc).lower() else 'error',
                       error='{}: {}'.format(type(exc).__name__, exc))
        write_json(a.output, row)
        return int(row['status'] != 'ok')
    output = a.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new or empty output directory.')
    output.mkdir(parents=True, exist_ok=True)
    configs = []
    for dataset, arch, depth, run in itertools.product(a.datasets, a.architectures, sorted(a.depths), range(a.runs)):
        for method in (['sf', 'bp'] if run % 2 == 0 else ['bp', 'sf']):
            directory = output / '{}-{}-L{}-run{}-{}'.format(dataset, arch, depth, run, method)
            configs.append(dict(dataset=dataset, architecture=arch, depth=depth, run=run,
                                method=method, directory=str(directory), hidden=a.hidden,
                                seed=a.seed, epochs=a.epochs, patience=a.patience,
                                val_every=a.val_every, gpu=a.gpu))
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((ROOT / 'src').rglob('*.py'))}
    write_json(output / 'plan.json', {'configurations': configs, 'source_sha256': hashes, 'dry_run': a.dry_run, 'instrumented': True})
    print('{} independent CUDA processes; output: {}'.format(len(configs), output), flush=True)
    if a.dry_run:
        return 0
    # Fail once before launching the matrix if the runtime is unavailable.
    import torch
    import torch_geometric  # noqa: F401
    import torch_sparse  # noqa: F401
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable.')
    from datasets.datasplit import DataSplit
    missing = sorted({str(path) for c in configs for path in
                      DataSplit(c['dataset'], num_splits=5).node_split_paths(c['run']).values()
                      if not path.is_file()})
    if missing:
        raise FileNotFoundError('Required published splits missing:\n' + '\n'.join(missing))
    rows = []
    for config in configs:
        directory = Path(config['directory'])
        directory.mkdir()
        write_json(directory / 'config.json', config)
        command = [sys.executable, str(Path(__file__).resolve()), '--worker-config',
                   str(directory / 'config.json'), '--output', str(directory / 'measurement.json')]
        with (directory / 'stdout.log').open('w') as log:
            completed = subprocess.run(command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
        result = directory / 'measurement.json'
        row = json.loads(result.read_text()) if result.exists() else dict(config, status='error', error='Worker exited without result')
        if completed.returncode and row['status'] == 'ok':
            row.update(status='error', error='Nonzero worker exit')
        rows.append(row)
        summarize(rows, output)
        print('{}/{} {}: {}'.format(len(rows), len(configs), directory.name, row['status']), flush=True)
    return int(any(row['status'] != 'ok' for row in rows))


if __name__ == '__main__':
    sys.exit(main(parse_args()))
