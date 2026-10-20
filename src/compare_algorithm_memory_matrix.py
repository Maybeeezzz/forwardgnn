"""Plan/run isolated SF/BP retained-storage diagnostics (stdlib-only controller)."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ['CitationFull-CiteSeer', 'CitationFull-Cora_ML', 'CitationFull-PubMed', 'Amazon-Photo', 'GitHub']


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=['Amazon-Photo'])
    p.add_argument('--architectures', nargs='+', choices=['GCN', 'SAGE', 'GAT'], default=['GCN', 'SAGE', 'GAT'])
    p.add_argument('--depths', nargs='+', type=int, default=[2, 4, 8, 16])
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--runs', type=int, default=1)
    p.add_argument('--seed', type=int, default=100)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--patience', type=int, default=-1)
    p.add_argument('--val-every', type=int, default=1)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dry-run', action='store_true', help='Print plan; no imports of torch, files, or subprocesses.')
    p.add_argument('--fail-fast', action='store_true', help='Stop after the first failed training process.')
    a = p.parse_args(argv)
    if min(a.depths + [a.hidden, a.runs, a.epochs, a.val_every]) < 1 or a.lr <= 0 or a.gpu < 0:
        p.error('Depths/hidden/runs/epochs/val-every/lr must be positive and gpu nonnegative.')
    if 'GAT' in a.architectures and a.hidden % 4:
        p.error('GAT hidden width must be divisible by 4 heads.')
    for name in ('datasets', 'architectures', 'depths'):
        values = getattr(a, name)
        if len(values) != len(set(values)):
            p.error('Duplicate values in --'+name)
    a.output = a.output.resolve()
    return a


def build_plan(a):
    cases = []
    for dataset, arch, depth in itertools.product(a.datasets, a.architectures, a.depths):
        directory = a.output / dataset / arch / 'layers-{}'.format(depth)
        for method in ('sf', 'bp'):
            memory = directory / method
            model = ('GNN_SingleForward-' if method == 'sf' else 'GNN-') + arch
            cmd = [sys.executable, str(ROOT/'src'/('train_forward.py' if method == 'sf' else 'train_backprop.py')),
                   '--task', 'node-class', '--model', model, '--dataset', dataset,
                   '--num-layers', str(depth), '--num-hidden', str(a.hidden), '--num-runs', str(a.runs),
                   '--seed', str(a.seed), '--epochs', str(a.epochs), '--lr', str(a.lr),
                   '--patience', str(a.patience), '--val-every', str(a.val_every), '--gpu', str(a.gpu),
                   # Absolute exp-setting is resolved by pathlib under this new output root.
                   '--exp-setting', str(directory/'training-results'/method), '--overwrite-result',
                   '--memory-analysis', '--memory-analysis-algorithm',
                   '--memory-analysis-rules', str(ROOT/'configs/memory'/ (method+'.json')),
                   '--memory-analysis-dir', str(memory)]
            if method == 'sf':
                cmd += ['--append-label', 'none', '--aug-edge-direction', 'bidirection', '--temperature', '1.0']
            cases.append(dict(dataset=dataset, architecture=arch, depth=depth, method=method,
                              runs=a.runs, memory_directory=str(memory), command=cmd,
                              log=str(directory/(method+'-process.log'))))
    return dict(schema_version=1, scope='Algorithm retained-storage diagnostic; not full accuracy reproduction',
                configuration={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                process_count=len(cases), measured_run_count=len(cases)*a.runs, cases=cases)


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def read_run(case, run_i):
    """Match every seed/split by explicit run metadata; never take just latest."""
    found = []
    for path in sorted(Path(case['memory_directory']).glob('run*/algorithm-memory-summary.json')):
        metadata = json.loads(path.with_name('memory-run.json').read_text())
        if metadata['arguments']['run_i'] != run_i:
            continue
        summary = json.loads(path.read_text())
        if summary.get('schema_version', 0) < 2 or not summary.get('complete', False):
            raise ValueError('Incomplete or legacy summary: '+str(path))
        found.append(path.parent)
    if len(found) != 1:
        raise ValueError('Expected one complete run {} in {}, found {}'.format(run_i, case['memory_directory'],len(found)))
    return found[0]


def aggregate(plan, statuses, output, execute=subprocess.run):
    periods, stages, pairs = [], [], []
    cases = plan['cases']
    for first in range(0, len(cases), 2):
        sf, bp = cases[first:first+2]
        for run_i in range(sf['runs']):
            identity = dict(dataset=sf['dataset'], architecture=sf['architecture'], depth=sf['depth'], run_i=run_i)
            pair = dict(identity, status='unavailable', error='')
            try:
                if any(statuses[i]['status'] != 'completed' for i in (first,first+1)):
                    raise ValueError('Training process did not complete; see process-status.json and process logs.')
                sf_dir, bp_dir = read_run(sf,run_i), read_run(bp,run_i)
                target = Path(sf['memory_directory']).parent/'comparison'/('run{}'.format(run_i))
                command = [sys.executable,str(ROOT/'src/memory_algorithm_summary.py'),
                           '--sf',str(sf_dir),'--bp',str(bp_dir),'--output',str(target)]
                target.mkdir(parents=True,exist_ok=True)
                with (target/'compare.log').open('w') as log:
                    result=execute(command,cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT)
                if result.returncode:
                    raise ValueError('Comparison failed; see '+str(target/'compare.log'))
                pair_rows = []
                for filename in ('comparison-periods.csv','comparison.csv'):
                    with (target/filename).open() as f:
                        pair_rows.append([dict(identity,**row) for row in csv.DictReader(f)])
                periods.extend(pair_rows[0]);stages.extend(pair_rows[1])
                pair.update(status='completed', directory=str(target))
            except (OSError, ValueError, KeyError) as exc:
                pair['error']=str(exc)
            pairs.append(pair)
    save(output/'pair-status.json',pairs)
    write_csv(output/'matrix-periods.csv',periods)
    write_csv(output/'matrix-stages.csv',stages)
    lines=['# SF/BP 算法保留存储矩阵', '',
           '单位 MiB；按 run 保留，不将不同层或分类的独立峰值相加。失败配置没有数值，不当成零。',
           '评估值含仍常驻的训练状态，不等于评估新增量。', '',
           '|数据集|骨干|层数|run|方法|区间|峰值 MiB|', '|---|---|---:|---:|---|---|---:|']
    for row in periods:
        lines.append('| {} | {} | {} | {} | {} | {} | {} |'.format(
            row['dataset'],row['architecture'],row['depth'],row['run_i'],row['algorithm'],row['period'],
            '{:.3f}'.format(float(row['algorithm_observed_peak_mib'])) if row['observed']=='True' else '未观测'))
    lines += ['', '## 未完成配置', '']
    lines += ['- {dataset}/{architecture}/L{depth}/run{run_i}: {error}'.format(**pair)
              for pair in pairs if pair['status']!='completed'] or ['无。']
    (output/'matrix-report.md').write_text('\n'.join(lines)+'\n')
    return all(pair['status']=='completed' for pair in pairs)


def main(argv=None):
    a=parse_args(argv);plan=build_plan(a)
    if a.dry_run:
        print(json.dumps(plan,indent=2))
        return 0
    if a.output.exists() and any(a.output.iterdir()):
        raise SystemExit('Use a new or empty --output directory; existing results will not be overwritten.')
    a.output.mkdir(parents=True,exist_ok=True)
    save(a.output/'plan.json',plan)
    statuses=[dict(case,status='pending',returncode=None) for case in plan['cases']]
    save(a.output/'process-status.json',statuses)
    for index, case in enumerate(plan['cases']):
        log=Path(case['log']);log.parent.mkdir(parents=True,exist_ok=True)
        statuses[index]['status']='running';save(a.output/'process-status.json',statuses)
        print('[{}/{}] {}'.format(index+1,len(statuses),shlex.join(case['command'])),flush=True)
        try:
            with log.open('w') as stream:
                result=subprocess.run(case['command'],cwd=str(ROOT),stdout=stream,stderr=subprocess.STDOUT)
            statuses[index].update(status='completed' if result.returncode==0 else 'failed',returncode=result.returncode)
        except KeyboardInterrupt:
            statuses[index]['status']='interrupted'
            save(a.output/'process-status.json',statuses)
            return 130
        except OSError as exc:
            statuses[index].update(status='failed',error=str(exc))
        save(a.output/'process-status.json',statuses)
        if a.fail_fast and statuses[index]['status']!='completed':break
    success=aggregate(plan,statuses,a.output)
    print('Report:',a.output/'matrix-report.md')
    return 0 if success else 1


if __name__=='__main__':
    sys.exit(main())
