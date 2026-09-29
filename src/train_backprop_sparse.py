"""GCN node-classification BP with the official trainer and sparse adjacency.

Use --graph-backend edge-index for the paired reference. All optimization,
validation, checkpoint selection and testing are delegated to train_backprop.
"""
import argparse
import gc
import json
import sys
from timeit import default_timer as timer

import numpy as np
import torch

import train_backprop as official
from datasets.datasplit import DataSplit
from utils.train_utils import ResultManager, SeedManager, set_seed


def parse_args():
    extension = argparse.ArgumentParser(add_help=False)
    extension.add_argument('--graph-backend', choices=['sparse-tensor', 'edge-index'],
                           default='sparse-tensor')
    extra, remaining = extension.parse_known_args()
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]] + remaining
        args = official.parse_args()
    finally:
        sys.argv = original_argv
    args.graph_backend = extra.graph_backend
    if args.exp_setting == 'default':
        args.exp_setting = 'bp-sparse-results'
    if args.model != 'GNN-GCN' or args.task != 'node-class':
        raise ValueError('This experiment supports --model GNN-GCN --task node-class only.')
    if not 1 <= args.num_runs <= 5 or args.num_layers < 1 or args.num_hidden < 1:
        raise ValueError('Use 1..5 runs and positive layer/hidden sizes.')
    if args.epochs < 1 or args.val_every < 1:
        raise ValueError('epochs and val-every must be positive.')
    if args.gpu >= 0 and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Use --gpu -1 only for CPU correctness checks; '
                           'CPU runs do not provide GPU memory results.')
    if args.graph_backend == 'sparse-tensor':
        try:
            import torch_sparse  # noqa: F401
        except ImportError as exc:
            raise RuntimeError('Install torch_sparse matching PyTorch/CUDA in the official '
                               'ForwardLearningGNN environment before running this experiment.') from exc
    # Do not silently generate new partitions instead of the published splits.
    for run_i in range(args.num_runs):
        for path in DataSplit(args.dataset, num_splits=5).node_split_paths(run_i).values():
            if not path.is_file():
                raise FileNotFoundError('Published split missing: {}. Place the author-provided '
                                        'forwardgnn-datasplits files in datasplits/.'.format(path))
    return official.populate_args(args)


def graph_input(data, backend):
    if backend == 'sparse-tensor':
        from torch_sparse import SparseTensor
        # PyG's sparse message-passing interface expects adj_t (target, source).
        # Explicit shape preserves isolated nodes. Construct once per run on CPU;
        # the official trainer transfers/clones it just as it does edge_index.
        data.edge_index = SparseTensor(row=data.edge_index[1], col=data.edge_index[0],
                                      sparse_sizes=(data.num_nodes, data.num_nodes))
    return data


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def memory_peaks(device):
    if device.type != 'cuda':
        return None
    return {
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
        'peak_allocated_mib': torch.cuda.max_memory_allocated(device) / 2**20,
        'peak_reserved_mib': torch.cuda.max_memory_reserved(device) / 2**20,
    }


def main(args):
    seeds = SeedManager(args.seed)
    manager = ResultManager('bp-{}-results'.format(args.graph_backend), args, seeds)
    import torch_geometric
    metadata = {'torch': torch.__version__, 'pyg': torch_geometric.__version__,
                'cuda': torch.version.cuda, 'device': str(args.device),
                'gpu_name': torch.cuda.get_device_name(args.device) if args.cuda else None}
    if args.graph_backend == 'sparse-tensor':
        import torch_sparse
        metadata['torch_sparse'] = torch_sparse.__version__

    for run_i in range(args.num_runs):
        previous = manager.load_run_result(run_i)
        if previous is not None and not args.overwrite_result:
            keys = ('dataset', 'model', 'task', 'num_layers', 'num_hidden', 'seed',
                    'epochs', 'lr', 'val_every', 'patience', 'graph_backend')
            if any(previous.get(k) != getattr(args, k) for k in keys):
                raise ValueError('Existing result has different settings; use a new '
                                 '--exp-setting or explicitly --overwrite-result.')
            if previous.get('environment') != metadata:
                raise ValueError('Existing result has a different environment; use a new --exp-setting.')
            continue
        gc.collect()
        if args.cuda:
            torch.cuda.empty_cache()
        seeds.set_run_i(run_i)
        set_seed(seeds.get_run_seed(), deterministic=True)
        data = official.load_node_classification_data(args, split_i=run_i)
        model = official.build_model(data, args)
        started = timer()
        data = graph_input(data, args.graph_backend)
        preparation_seconds = timer() - started
        trainer = official.NodeClassificationTrainer(model, data, args.device,
                                                     args.lr, args.epochs, args.patience, args)
        synchronize(args.device)
        if args.cuda:
            torch.cuda.reset_peak_memory_stats(args.device)
        started = timer()
        train_time, train_epochs = trainer.train()
        synchronize(args.device)
        training_wall_seconds = timer() - started
        training_memory = memory_peaks(args.device)
        test_acc = trainer.test()
        synchronize(args.device)
        run_memory = memory_peaks(args.device)
        manager.save_run_result(run_i, {
            'perf': test_acc, 'train_time': train_time,
            'train_epochs': [train_epochs], 'best_val_epochs': [-1],
        })
        path = manager.run_result_path(run_i)
        result = json.loads(path.read_text())
        result.update({
            'environment': metadata,
            'graph_preparation_seconds': preparation_seconds,
            'training_wall_seconds': training_wall_seconds,
            'training_memory': training_memory,
            'training_and_test_memory': run_memory,
            'memory_scope': 'Device transfer, training, validation and checkpoint restore; '
                            'training_and_test_memory additionally includes official test(). '
                            'CPU graph preparation excluded. Absolute allocator peaks, no hooks.',
        })
        path.write_text(json.dumps(result, indent=2) + '\n')
        del trainer, model, data

    rows = [manager.load_run_result(i) for i in range(args.num_runs)]
    perf = [r['perf'] for r in rows]
    peaks = [r['training_memory']['peak_allocated_mib'] for r in rows
             if r['training_memory'] is not None]
    summary = {
        'dataset': args.dataset, 'model': args.model, 'num_layers': args.num_layers,
        'graph_backend': args.graph_backend, 'completed_runs': len(rows),
        'accuracy_mean_percent': float(np.mean(perf)),
        'accuracy_std_percent': float(np.std(perf, ddof=0)),
        'training_peak_allocated_mean_mib': float(np.mean(peaks)) if peaks else None,
        'training_peak_allocated_max_mib': max(peaks) if peaks else None,
        'training_wall_mean_seconds': float(np.mean([r['training_wall_seconds'] for r in rows])),
        'run_files': [manager.run_result_path(i).name for i in range(args.num_runs)],
    }
    destination = args.results_dir / 'summary-{}-{}-L{}.json'.format(
        args.graph_backend, args.model, args.num_layers)
    destination.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main(parse_args())
