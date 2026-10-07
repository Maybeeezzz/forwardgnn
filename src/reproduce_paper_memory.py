"""Run original node-classification entry points with author-script settings and observation only."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ['CitationFull-CiteSeer', 'CitationFull-Cora_ML', 'CitationFull-PubMed', 'Amazon-Photo', 'GitHub']


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def commands(args, output):
    configurations = []
    for method in args.methods:
        pairs = ((dataset, arch) for dataset in args.datasets for arch in args.architectures) if method == 'bp' else (
            (dataset, arch) for arch in args.architectures for dataset in args.datasets)
        for dataset, arch in pairs:
            for depth in ([1, 2, 3, 4] if method == 'bp' else [4]):
                name = '{}-{}-{}-L{}'.format(dataset, arch, method, depth)
                script = 'train_backprop.py' if method == 'bp' else 'train_forward.py'
                argv = ['--exp-setting', str(output / 'original-results'), '--dataset', dataset,
                        '--task', 'node-class', '--model', ('GNN-' if method == 'bp' else 'GNN_SingleForward-') + arch,
                        '--num-layers', str(depth), '--num-runs', '5', '--seed', '100',
                        '--epochs', '1000', '--val-every', '2', '--lr', '0.001',
                        '--patience', '100', '--num-hidden', '128', '--overwrite-result', '--gpu', str(args.gpu)]
                if method == 'sf':
                    argv += ['--append-label', 'none', '--aug-edge-direction', 'bidirection']
                configurations.append(dict(name=name, script=script, argv=argv, gpu=args.gpu,
                                           directory=str(output / name), monitor=args.monitor))
    return configurations


def worker(config):
    # Import the original module first, exactly as its entry point does, including
    # its CUBLAS_WORKSPACE_CONFIG assignment before torch initialization.
    module = __import__(Path(config['script']).stem)
    import torch
    import torch_geometric
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU fallback.')
    device = torch.device('cuda:{}'.format(config['gpu']))
    torch.cuda.set_device(device)
    directory = Path(config['directory'])
    save(directory / 'environment.json', dict(
        torch=torch.__version__, pyg=torch_geometric.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(device),
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        tf32_cudnn=torch.backends.cudnn.allow_tf32,
        monitor=config['monitor']))
    tracer = None
    if config['monitor'] == 'boundaries':
        from utils.layer_memory_trace import LayerMemoryTrace

        class EmptyModel:
            layers = []
            def modules(self):
                return []

        class OriginalTrace(LayerMemoryTrace):
            run = None

            def boundary(self, location):
                location = dict(location, run_i=self.run)
                super().boundary(location)

            def trace(self, frame, event, arg):
                name = frame.f_code.co_name
                filename = frame.f_code.co_filename
                original = filename.endswith(('/train_forward.py', '/train_backprop.py'))
                if original and name in ('build_model', 'build_node_classification_model'):
                    if event == 'return' and arg is not None:
                        layers = arg.layers if hasattr(arg, 'layers') else arg.convs
                        # Only integer ids; do not keep model/tensor references alive.
                        self.layers = {id(child): i + 1 for i, layer in enumerate(layers)
                                       for child in layer.modules()}
                    return self.trace
                if original and name == 'main':
                    import linecache
                    self.run = frame.f_locals.get('run_i', self.run)
                    if event in ('call', 'line', 'return', 'exception'):
                        self.boundary(dict(event=event, file=filename, function=name, line=frame.f_lineno,
                                           source=linecache.getline(filename, frame.f_lineno).strip()))
                    return self.trace
                return super().trace(frame, event, arg)

        tracer = OriginalTrace(torch, device, EmptyModel(), directory)
        torch.cuda.reset_peak_memory_stats(device)
        tracer.start()
    sys.argv = [str(ROOT / 'src' / config['script'])] + config['argv']
    try:
        # Use the original parser, argument population, main, five-run loop,
        # training, validation, checkpointing and testing without replacement.
        args = module.populate_args(module.parse_args())
        module.main(args)
        if tracer is not None:
            tracer.boundary({'event': 'original_main_finished'})
    finally:
        if tracer is not None:
            tracer.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    p.add_argument('--architectures', nargs='+', choices=['GCN', 'SAGE', 'GAT'], default=['GCN', 'SAGE', 'GAT'])
    p.add_argument('--methods', nargs='+', choices=['bp', 'sf'], default=['bp', 'sf'])
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--monitor', choices=['boundaries', 'none'], default='boundaries')
    p.add_argument('--output', type=Path)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--worker-config', type=Path, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker_config:
        worker(json.loads(args.worker_config.read_text()))
        return 0
    if args.output is None or args.gpu < 0:
        p.error('Provide --output and a nonnegative --gpu.')
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        p.error('Use a new or empty output directory.')
    output.mkdir(parents=True, exist_ok=True)
    configs = commands(args, output)
    save(output / 'plan.json', dict(configurations=configs, dry_run=args.dry_run,
        scope='Author node-classification SF/BP scripts; original five-run process lifecycle.',
        source_sha256={str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                       for folder in ['src', 'exp/nodeclass'] for f in sorted((ROOT / folder).rglob('*'))
                       if f.is_file() and f.suffix in ('.py', '.sh')}))
    print('{} processes, each retaining the original 5-run loop'.format(len(configs)), flush=True)
    if args.dry_run:
        return 0
    missing = [str(ROOT / 'datasplits' / dataset / 'node-5splits' / '{}-node-index-split{}.pt'.format(part, i))
               for dataset in args.datasets for i in range(5) for part in ['train', 'val', 'test']
               if not (ROOT / 'datasplits' / dataset / 'node-5splits' / '{}-node-index-split{}.pt'.format(part, i)).is_file()]
    if missing:
        raise FileNotFoundError('Place published splits before running:\n' + '\n'.join(missing))
    statuses = []
    for config in configs:
        directory = Path(config['directory'])
        directory.mkdir()
        save(directory / 'config.json', config)
        with (directory / 'stdout.log').open('w') as log:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker-config',
                                     str(directory / 'config.json')], cwd=str(ROOT / 'src'),
                                    stdout=log, stderr=subprocess.STDOUT)
        statuses.append(dict(name=config['name'], returncode=result.returncode))
        save(output / 'status.json', statuses)
        print('{}: {}'.format(config['name'], result.returncode), flush=True)
    return int(any(row['returncode'] for row in statuses))


if __name__ == '__main__':
    sys.exit(main())
