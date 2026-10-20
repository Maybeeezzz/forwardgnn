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
