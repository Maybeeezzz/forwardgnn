"""Offline planning and aggregation tests: never launch training."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import compare_algorithm_memory_matrix as matrix


class MatrixTests(unittest.TestCase):
    def complete_run(self, case, name='run0-fixture'):
        folder = Path(case['memory_directory'])/name
        folder.mkdir(parents=True)
        command = case['command']
        args = {'run_i': 0}
        for flag in ('dataset', 'model', 'task', 'num-layers', 'num-hidden', 'num-runs',
                     'seed', 'epochs', 'lr', 'patience', 'val-every', 'append-label',
                     'aug-edge-direction', 'temperature'):
            if '--'+flag in command:
                value = command[command.index('--'+flag)+1]
                try:
                    value = json.loads(value)
                except ValueError:
                    pass
                args[flag.replace('-', '_')] = None if value == 'none' else value
        matrix.save(folder/'memory-run.json', {'arguments': args})
        rules = json.loads(Path(command[command.index('--memory-analysis-rules')+1]).read_text())
        matrix.save(folder/'algorithm-memory-summary.json',
                    dict(schema_version=2, complete=True, rules=rules,
                         period_peaks={'training': {}}, global_peak={}))
        (folder/'algorithm-memory-stages.csv').write_text('period,stage\ntraining,forward\n')
        return folder

    def test_resume_validates_results_and_dry_run_is_read_only(self):
        with tempfile.TemporaryDirectory() as d:
            argv = ['--output', d, '--architectures', 'SAGE', '--depths', '16']
            plan = matrix.build_plan(matrix.parse_args(argv))
            matrix.save(Path(d)/'plan.json', plan)
            complete = self.complete_run(plan['cases'][0])
            partial = self.complete_run(plan['cases'][0], 'run0-zpartial')
            (partial/'algorithm-memory-summary.json').write_text('{"complete":false}')
            before = (Path(d)/'plan.json').read_bytes()
            with patch.object(matrix.subprocess, 'run', side_effect=AssertionError('No training')), contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(matrix.main(argv+['--resume', '--dry-run']), 0)
            cases = json.loads(out.getvalue())['cases']
            self.assertEqual([c['action'] for c in cases], ['skip', 'run'])
            self.assertEqual(cases[0]['selected_runs']['0'], str(complete))
            self.assertEqual((Path(d)/'plan.json').read_bytes(), before)
            self.assertFalse((Path(d)/'process-status.json').exists())
            metadata = json.loads((complete/'memory-run.json').read_text())
            metadata['arguments']['epochs'] = 999
            matrix.save(complete/'memory-run.json', metadata)
            with self.assertRaises(ValueError):
                matrix.read_run(plan['cases'][0], 0, validate=True)
            with self.assertRaises(ValueError):
                a = matrix.parse_args(argv+['--resume', '--epochs', '4'])
                matrix.resume_plan(a, matrix.build_plan(a))

    def test_atomic_save_and_attempt_isolation(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'status.json'
            matrix.save(path, {'status': 'old'})
            with patch.object(matrix.os, 'replace', side_effect=OSError(28, 'No space left on device')):
                with self.assertRaises(OSError):
                    matrix.save(path, {'status': 'new'})
            self.assertEqual(json.loads(path.read_text()), {'status': 'old'})
            self.assertFalse(path.with_name('status.json.tmp').exists())
            case = matrix.build_plan(matrix.parse_args(['--output', d]))['cases'][0]
            original = case['memory_directory']
            matrix.isolate_attempt(case, 'test')
            self.assertEqual(case['memory_directory'], original)
            self.assertIn(original+'/attempt-test', case['command'])
            self.assertTrue(case['log'].endswith('sf-process-test.log'))

    def test_worker_disk_exhaustion_stops_matrix(self):
        with tempfile.TemporaryDirectory() as d:
            def failed_worker(command, **kwargs):
                kwargs['stdout'].write('OSError: [Errno 28] No space left on device\n')
                return SimpleNamespace(returncode=120)
            with patch.object(matrix.subprocess, 'run', side_effect=failed_worker) as worker, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(matrix.main(['--output', d]), 2)
            self.assertEqual(worker.call_count, 1)
            statuses = json.loads((Path(d)/'process-status.json').read_text())
            self.assertEqual(statuses[0]['status'], 'failed')
            self.assertTrue(all(s['status'] == 'pending' for s in statuses[1:]))

    def test_default_commands_isolated_and_paired(self):
        a=matrix.parse_args(['--output','/tmp/memory-matrix-fixture'])
        p=matrix.build_plan(a)
        self.assertEqual(p['process_count'],24)
        self.assertEqual(len({c['memory_directory'] for c in p['cases']}),24)
        for i in range(0,24,2):
            sf,bp=p['cases'][i:i+2]
            self.assertEqual(sf['depth'],bp['depth'])
            self.assertEqual(sf['architecture'],bp['architecture'])
            for case in (sf,bp):
                cmd=case['command']
                self.assertIn('--memory-analysis-algorithm',cmd)
                self.assertTrue(Path(cmd[cmd.index('--exp-setting')+1]).is_absolute())
                self.assertTrue(Path(cmd[1]).is_file())
                self.assertTrue(Path(cmd[cmd.index('--memory-analysis-rules')+1]).is_file())

    def test_summarize_only_never_launches_training(self):
        with tempfile.TemporaryDirectory() as d:
            argv = ['--output', d, '--architectures','SAGE','--depths','2']
            p = matrix.build_plan(matrix.parse_args(argv))
            matrix.save(Path(d)/'plan.json',p)
            with patch.object(matrix.subprocess,'run',side_effect=AssertionError('No training')), patch.object(matrix,'aggregate',return_value=False) as aggregate:
                self.assertEqual(matrix.main(argv+['--summarize-only']),1)
            self.assertEqual(aggregate.call_count,1)
            self.assertFalse((Path(d)/'process-status.json').exists())

    def test_dry_run_has_no_side_effects(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'not-created'
            with patch.object(matrix.subprocess,'run',side_effect=AssertionError('No subprocess allowed')), contextlib.redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(matrix.main(['--output',str(out),'--dry-run']),0)
            self.assertFalse(out.exists())
            self.assertEqual(json.loads(stdout.getvalue())['measured_run_count'],24)

    def test_invalid_and_duplicate_parameters(self):
        for args in [['--depths','0'],['--hidden','127'],['--runs','0'],['--architectures','GCN','GCN']]:
            with self.subTest(args=args),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                matrix.parse_args(['--output','/tmp/unused']+args)

    def test_all_run_ids_aggregated_and_failed_not_zero(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            plan=matrix.build_plan(matrix.parse_args(['--output',d,'--architectures','SAGE','--depths','2','--runs','2']))
            statuses=[dict(status='completed'),dict(status='completed')]
            for case in plan['cases']:
                for run_i in (0,1):
                    folder=Path(case['memory_directory'])/('run{}-fixture'.format(run_i));folder.mkdir(parents=True)
                    matrix.save(folder/'memory-run.json',dict(arguments=dict(run_i=run_i)))
                    matrix.save(folder/'algorithm-memory-summary.json',dict(schema_version=2,complete=True))
            calls=[]
            def fake_compare(command,**kwargs):
                target=Path(command[command.index('--output')+1]);calls.append(command)
                (target/'comparison-periods.csv').write_text('algorithm,period,observed,algorithm_observed_peak_mib\nSF,training,True,1.0\nBP,training,True,2.0\n')
                (target/'comparison.csv').write_text('algorithm,period,owner\nSF,training,all\nBP,training,all\n')
                return SimpleNamespace(returncode=0)
            self.assertTrue(matrix.aggregate(plan,statuses,out,execute=fake_compare))
            self.assertEqual(len(calls),2)
            self.assertIn('run0-fixture',calls[0][calls[0].index('--sf')+1])
            self.assertIn('run1-fixture',calls[1][calls[1].index('--sf')+1])
            statuses[1]['status']='failed'
            self.assertFalse(matrix.aggregate(plan,statuses,out,execute=fake_compare))
            self.assertEqual((out/'matrix-periods.csv').read_text(),'')
            self.assertTrue(all(r['status']=='unavailable' for r in json.loads((out/'pair-status.json').read_text())))


if __name__=='__main__':unittest.main()
