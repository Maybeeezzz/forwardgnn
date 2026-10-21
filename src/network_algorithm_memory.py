"""Isolated small network x learning-rule memory matrix. Controller is stdlib-only."""
import argparse
from contextlib import nullcontext
from datetime import datetime
import csv
import errno
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURES = ['mlp', 'cnn']
ALGORITHMS = ['bp', 'local', 'ff', 'dfa', 'spsa']
MODES = ['allocator', 'compact']


def atomic(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    try:
        temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
        os.replace(str(temp), str(path))
    finally:
        if temp.exists():
            temp.unlink()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--architectures', nargs='+', choices=ARCHITECTURES, default=ARCHITECTURES)
    p.add_argument('--algorithms', nargs='+', choices=ALGORITHMS, default=ALGORITHMS)
    p.add_argument('--modes', nargs='+', choices=MODES, default=MODES)
    p.add_argument('--output', type=Path, default=Path('results/network-algorithm-memory'))
    p.add_argument('--data-dir', type=Path, default=Path('data'))
    p.add_argument('--seed', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--train-samples', type=int, default=4096)
    p.add_argument('--test-samples', type=int, default=1024)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--perturbation', type=float, default=0.001)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--summarize-only', action='store_true')
    p.add_argument('--prepare-data', action='store_true', help='CPU-only dataset preparation; no training.')
    p.add_argument('--download', action='store_true', help='Allow Fashion-MNIST download only with --prepare-data.')
    p.add_argument('--fail-fast', action='store_true')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--signature', help=argparse.SUPPRESS)
    return p


def arguments(argv=None):
    p = parser()
    a = p.parse_args(argv)
    if min(a.batch_size, a.train_samples, a.test_samples, a.steps) < 1 or a.warmup < 0 or a.gpu < 0:
        p.error('Positive batch/sample/step sizes and nonnegative warmup/gpu required.')
    if a.train_samples < a.batch_size or a.train_samples > 60000 or a.test_samples > 10000:
        p.error('Require batch-size <= train-samples <= 60000 and test-samples <= 10000.')
    if not (math.isfinite(a.lr) and math.isfinite(a.perturbation) and a.lr > 0 and a.perturbation > 0):
        p.error('lr and perturbation must be finite and positive.')
    for key in ('architectures', 'algorithms', 'modes'):
        if len(set(getattr(a, key))) != len(getattr(a, key)):
            p.error('Duplicate '+key)
    if a.download and not a.prepare_data:
        p.error('--download requires --prepare-data; workers never download.')
    if a.worker and any(len(getattr(a, k)) != 1 for k in ('architectures', 'algorithms', 'modes')):
        p.error('Worker requires a single network/algorithm/mode.')
    a.output, a.data_dir = a.output.resolve(), a.data_dir.resolve()
    return a


def specification(a):
    keys = ('architectures', 'algorithms', 'modes', 'seed', 'batch_size', 'train_samples',
            'test_samples', 'warmup', 'steps', 'lr', 'perturbation')
    spec = {k: getattr(a, k) for k in keys}
    files = [Path(__file__), ROOT/'src/network_memory_models.py',
             ROOT/'src/memory_algorithm_summary.py', ROOT/'src/memory_stage_summary.py',
             *sorted((ROOT/'src/utils').glob('*memory*.py'))]
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    spec.update(dataset='FashionMNIST', data_dir=str(a.data_dir), version=1, source_sha256=digest.hexdigest())
    return spec


def plan(a):
    spec = specification(a)
    signature = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    cases = []
    for network in a.architectures:
        for algorithm in a.algorithms:
            for mode in a.modes:
                directory = a.output/network/algorithm/mode
                cmd = [sys.executable, str(Path(__file__).resolve()), '--worker',
                       '--architectures', network, '--algorithms', algorithm, '--modes', mode,
                       '--output', str(directory), '--data-dir', str(a.data_dir), '--signature', signature]
                for key in ('seed', 'batch_size', 'train_samples', 'test_samples', 'warmup', 'steps', 'lr', 'perturbation', 'gpu'):
                    cmd += ['--'+key.replace('_', '-'), str(getattr(a, key))]
                cases.append(dict(network=network, algorithm=algorithm, mode=mode,
                                  directory=str(directory), command=cmd))
    return dict(configuration=spec, signature=signature, cases=cases)


def completed(case, signature):
    for path in sorted(Path(case['directory']).glob('attempt-*/result.json'), reverse=True):
        try:
            data = json.loads(path.read_text())
            if data['complete'] and data['signature'] == signature:
                if any(data.get(k) != case[k] for k in ('algorithm', 'mode')) or data.get('network') != case['network']:
                    continue
                if not data.get('parameters_changed') or not data.get('losses'):
                    continue
                if not all(math.isfinite(float(data[p][k])) for p in ('training','evaluation')
                           for k in ('peak_allocated_mib','peak_reserved_mib','end_allocated_mib')):
                    continue
                if not math.isfinite(float(data['accuracy'])):
                    continue
                if case['mode'] == 'compact':
                    summary = json.loads((path.parent/'trace/algorithm-memory-summary.json').read_text())
                    if not summary['complete'] or not all(summary['period_peaks'][p] for p in ('training','evaluation')):
                        continue
                    if not (path.parent/'trace/algorithm-memory-stages.csv').is_file():
                        continue
                return path
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None


def dataset(a):
    # Lazy imports: --dry-run and --summarize-only require no ML dependencies.
    from torchvision.datasets import FashionMNIST
    train = FashionMNIST(str(a.data_dir), train=True, download=a.download)
    test = FashionMNIST(str(a.data_dir), train=False, download=a.download)
    return (train.data[:a.train_samples].unsqueeze(1).float()/255, train.targets[:a.train_samples],
            test.data[:a.test_samples].unsqueeze(1).float()/255, test.targets[:a.test_samples])


def memory_api():
    # Load the standalone utility without utils/__init__.py's GNN/PyG dependencies.
    package = types.ModuleType('_network_memory_utils')
    package.__path__ = [str(ROOT/'src/utils')]
    sys.modules.setdefault(package.__name__, package)
    return importlib.import_module(package.__name__+'.memory_analysis').MemoryAnalysis


def worker(a):
    import torch
    from network_memory_models import Network, update, predict
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; CPU does not substitute for GPU memory results.')
    torch.cuda.set_device(a.gpu)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(a.seed)
    data_x, data_y, test_x, test_y = dataset(a)
    network, algorithm, mode = a.architectures[0], a.algorithms[0], a.modes[0]
    device = torch.device('cuda', a.gpu)
    model = Network(network, algorithm, a.seed, a.lr).setup(device)
    generator = torch.Generator(device=device).manual_seed(a.seed+2000)
    a.output.mkdir(parents=True, exist_ok=True)
    atomic(a.output/'metadata.json', dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
           torch_version=torch.__version__, cuda_version=torch.version.cuda,
           device=torch.cuda.get_device_name(device)))
    rules = [{'pattern': '*.batch_x', 'category': 'model_baseline'},
             {'pattern': '*.batch_y', 'category': 'model_baseline'}]
    rule_path = a.output/'rules.json'
    atomic(rule_path, rules)

    def run_steps(start, count, memory=None):
        losses = []
        for step in range(start, start+count):
            offset = (step % (len(data_x)//a.batch_size))*a.batch_size
            batch_x = data_x[offset:offset+a.batch_size].to(device)
            batch_y = data_y[offset:offset+a.batch_size].to(device)
            value = update(model, batch_x, batch_y, step, memory, generator, a.perturbation)
            if not math.isfinite(value):
                raise ValueError('Non-finite loss; configuration is not a valid memory result.')
            losses.append(value)
            del batch_x, batch_y
        return losses

    def allocator():
        torch.cuda.synchronize(device)
        return dict(peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20,
                    peak_reserved_mib=torch.cuda.max_memory_reserved(device)/2**20,
                    end_allocated_mib=torch.cuda.memory_allocated(device)/2**20)

    # Warmup runs before installing saved-tensor hooks: no pending graphs cross this boundary.
    torch.cuda.reset_peak_memory_stats(device)
    warmup_losses = run_steps(0, a.warmup)
    warmup = allocator()
    torch.cuda.reset_peak_memory_stats(device)
    context = (memory_api()(True, model, device, a.output/'trace', algorithm=True,
                           rules=rule_path, output_mode='compact', trace_roots=[Path(__file__), ROOT/'src/network_memory_models.py'],
                           metadata=dict(network=network, algorithm=algorithm, signature=a.signature))
               if mode == 'compact' else nullcontext(None))
    # Only scalar CPU checks survive: no full backup contaminates measured peaks.
    before = [p.detach().double().sum().item() for p in model.parameters()]
    torch.cuda.reset_peak_memory_stats(device)
    with context as memory:
        losses = run_steps(a.warmup, a.steps, memory)
        training = allocator()
        if memory:
            window = memory.tracer.end_window('training_end')
            training['peak_allocated_mib'] = window['allocated_bytes']/2**20
            training['peak_reserved_mib'] = window['reserved_bytes']/2**20
        torch.cuda.reset_peak_memory_stats(device)
        correct = 0
        model.eval()
        with torch.no_grad(), (memory.stage('evaluation', period='evaluation') if memory else nullcontext()):
            for offset in range(0, len(test_x), a.batch_size):
                batch_x = test_x[offset:offset+a.batch_size].to(device)
                batch_y = test_y[offset:offset+a.batch_size].to(device)
                correct += (predict(model, batch_x) == batch_y).sum().item()
                del batch_x, batch_y
        evaluation = allocator()
        if memory:
            window = memory.tracer.end_window('evaluation_end')
            evaluation['peak_allocated_mib'] = window['allocated_bytes']/2**20
            evaluation['peak_reserved_mib'] = window['reserved_bytes']/2**20
    after = [p.detach().double().sum().item() for p in model.parameters()]
    if not all(math.isfinite(v) for v in after) or before == after:
        raise ValueError('Parameters non-finite or unchanged; not a valid training run.')
    result = dict(complete=True, signature=a.signature, network=network, algorithm=algorithm, mode=mode,
                  parameter_count=sum(p.numel() for p in model.parameters()),
                  algorithm_parameter_count=sum(p.numel() for p in model.parameters()
                                                if getattr(p, '_memory_algorithm_category', None)),
                  warmup=warmup, training=training, evaluation=evaluation,
                  warmup_losses=warmup_losses, losses=losses, accuracy=correct/len(test_x),
                  parameters_changed=True)
    atomic(a.output/'result.json', result)


def csv_file(path, rows):
    if not rows:
        path.write_text('')
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def summarize(a, p):
    rows, stages, missing = [], [], []
    for case in p['cases']:
        result = completed(case, p['signature'])
        if result is None:
            missing.append({k:case[k] for k in ('network','algorithm','mode')})
            continue
        data = json.loads(result.read_text())
        summary = (json.loads((result.parent/'trace/algorithm-memory-summary.json').read_text())
                   if case['mode'] == 'compact' else None)
        for period in ('training', 'evaluation'):
            peak = summary['period_peaks'][period] if summary else None
            rows.append(dict(network=case['network'], algorithm=case['algorithm'], mode=case['mode'],
                             period=period, parameter_count=data['parameter_count'],
                             algorithm_parameter_count=data['algorithm_parameter_count'],
                             **data[period], algorithm_peak_mib=peak['algorithm_bytes']/2**20 if peak else '',
                             peak_layer=peak['active_layer'] if peak else '',
                             peak_phase=peak['phase'] if peak else '',
                             last_loss=data['losses'][-1], accuracy=data['accuracy'], result=str(result)))
        if summary:
            with (result.parent/'trace/algorithm-memory-stages.csv').open() as stream:
                stages.extend(dict(network=case['network'], algorithm=case['algorithm'], **r)
                              for r in csv.DictReader(stream))
    csv_file(a.output/'matrix-periods.csv', rows)
    csv_file(a.output/'matrix-stages.csv', stages)
    atomic(a.output/'missing.json', missing)
    lines = ['# Network × algorithm memory', '',
             'MiB. Algorithm retained-storage observations are not exact causal extra memory.',
             'Allocator and compact are separate runs. Short-run accuracy is a sanity check, not a ranking.', '',
             '|Network|Algorithm|Mode|Period|Allocated peak|Algorithm peak|Layer / phase|',
             '|---|---|---|---|---:|---:|---|']
    for r in rows:
        lines.append('|{network}|{algorithm}|{mode}|{period}|{peak_allocated_mib:.3f}|{algorithm_peak_mib}|{peak_layer} / {peak_phase}|'.format(**r))
    lines += ['', 'Incomplete configurations: '+str(len(missing)), 'See missing.json and status.json.']
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')
    return not missing


def main(argv=None):
    a = arguments(argv)
    if a.worker:
        worker(a)
        return 0
    if a.prepare_data:
        if a.dry_run:
            print('Would prepare Fashion-MNIST at '+str(a.data_dir))
        else:
            dataset(a)
            print('Fashion-MNIST ready: '+str(a.data_dir))
        return 0
    p = plan(a)
    if a.resume or a.summarize_only:
        original = json.loads((a.output/'plan.json').read_text())
        if original['signature'] != p['signature']:
            raise ValueError('Configuration/source mismatch; use original settings/code or a new output directory.')
    elif a.output.exists() and any(a.output.iterdir()):
        raise ValueError('Use an empty output directory or --resume.')
    for case in p['cases']:
        case['action'] = 'skip' if completed(case, p['signature']) else 'run'
    if a.dry_run:
        print(json.dumps(p, indent=2))
        return 0
    if a.summarize_only:
        return 0 if summarize(a, p) else 1
    a.output.mkdir(parents=True, exist_ok=True)
    if not a.resume:
        atomic(a.output/'plan.json', p)
    statuses = []
    for case in p['cases']:
        status = dict(network=case['network'], algorithm=case['algorithm'], mode=case['mode'])
        if case['action'] == 'skip':
            status['status'] = 'skipped_complete'
            print('SKIP', status, flush=True)
        else:
            attempt = Path(case['directory'])/('attempt-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
            attempt.mkdir(parents=True)
            command = list(case['command'])
            command[command.index('--output')+1] = str(attempt)
            status.update(status='running', directory=str(attempt), command=command)
            atomic(a.output/'status.json', statuses+[status])
            print('RUN', case['network'], case['algorithm'], case['mode'], flush=True)
            with (attempt/'process.log').open('w') as log:
                result = subprocess.run(command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
            status.update(returncode=result.returncode,
                          status='complete' if result.returncode == 0 and completed(case, p['signature']) else 'failed')
            if status['status'] == 'failed':
                with (attempt/'process.log').open('rb') as log:
                    log.seek(max(0, (attempt/'process.log').stat().st_size-4096))
                    tail = log.read().decode(errors='replace')
                print(tail, file=sys.stderr)
                if 'No space left on device' in tail or 'Disk quota exceeded' in tail:
                    atomic(a.output/'status.json', statuses+[status])
                    raise OSError(errno.ENOSPC, 'Worker disk exhaustion; stopped.')
        statuses.append(status)
        atomic(a.output/'status.json', statuses)
        if a.fail_fast and status['status'] == 'failed':
            break
    return 0 if summarize(a, p) else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('Interrupted; completed attempts are preserved. Use --resume.', file=sys.stderr)
        sys.exit(130)
    except (OSError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
