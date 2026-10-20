"""Offline regression tests only; no dataset, training or CUDA experiment."""
import ast
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from memory_algorithm_summary import AlgorithmMemorySummary
from memory_stage_summary import StageResolver

ROOT = Path(__file__).resolve().parents[1]


class PeriodTests(unittest.TestCase):
    def test_training_transition_validation_and_native_backward(self):
        source = '''def train(self):
    epoch = -1
    for epoch in range(3):
        out = model(x)
        val_out = model(x)
    node_feats = layer.forward(x)
'''
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'trainer.py';path.write_text(source)
            resolver=StageResolver()
            for line, period in [(2,'other'),(4,'training'),(5,'evaluation'),(6,'other')]:
                frame=SimpleNamespace(f_code=SimpleNamespace(co_filename=str(path),co_name='train'),
                                      f_lineno=line,f_locals={'epoch':2},f_back=None)
                self.assertEqual(resolver.resolve(frame,{}, {})['period'],period)
            self.assertEqual(resolver.resolve(None,{'phase':'backward'}, {})['period'],'training')

    def test_period_peaks_are_independent_and_missing_is_null(self):
        summary=AlgorithmMemorySummary([])
        for sequence,(period,size) in enumerate([('training',10),('evaluation',90),('training',20),('other',200)]):
            summary.observe(dict(sequence=sequence,allocated_bytes=1000,interval_peak_allocated_bytes=9999),
                dict(variables=[dict(path='incoming_gradient',storage_id='g')],
                     storages=[dict(storage_id='g',bytes=size,aliases=['incoming_gradient'])]),
                dict(layer='layers.0',phase='forward',period=period))
        self.assertEqual(summary.global_peak['algorithm_bytes'],200)
        self.assertEqual(summary.period_peaks['training']['algorithm_bytes'],20)
        self.assertEqual(summary.period_peaks['evaluation']['algorithm_bytes'],90)
        self.assertIsNone(summary.period_peaks['unclassified'])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for method in ('sf','bp'):
                target=root/method;target.mkdir();summary.write(target)
                p=target/'algorithm-memory-summary.json';data=json.loads(p.read_text())
                data['complete']=True;p.write_text(json.dumps(data))
            cmd=[sys.executable,str(ROOT/'src/memory_algorithm_summary.py'),'--sf',str(root/'sf'),
                 '--bp',str(root/'bp'),'--output',str(root/'comparison')]
            subprocess.run(cmd,check=True,capture_output=True)
            with (root/'comparison/comparison-periods.csv').open() as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),8)
            self.assertEqual(next(r for r in rows if r['period']=='unclassified')['algorithm_observed_peak_mib'],'')
            self.assertIn('training',(root/'comparison/comparison.md').read_text())
            data['schema_version']=1;p.write_text(json.dumps(data))
            result=subprocess.run(cmd,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Legacy trace',result.stderr)

    def test_both_bp_test_methods_disable_grad_and_restore_state(self):
        # Execute the actual test methods with small doubles, avoiding imports of
        # PyG/datasets and avoiding any training experiment.
        state={'grad':True,'eval':False,'calls':0}
        def no_grad():
            def decorate(fn):
                def wrapped(*args,**kwargs):
                    old=state['grad'];state['grad']=False
                    try:return fn(*args,**kwargs)
                    finally:state['grad']=old
                return wrapped
            return decorate
        class Output:
            def argmax(self,dim):return self
            def sigmoid(self):return self
        class Model:
            def eval(self):state['eval']=True
            def __call__(self,*args):
                self_test.assertFalse(state['grad']);state['calls']+=1
                return Output()
        class Data:
            x=edge_index=y=test_mask=edge_label_index=edge_label=None
            def to(self,device):return self
        self_test=self
        tree=ast.parse((ROOT/'src/train_backprop.py').read_text())
        for cls in [n for n in tree.body if isinstance(n,ast.ClassDef)]:
            method=next((n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='test'),None)
            if method is None:continue
            scope={'torch':SimpleNamespace(no_grad=no_grad),
                   'eval_node_classification':lambda *a,**k:{'acc':50.},
                   'eval_link_prediction':lambda *a,**k:{'rocauc':60.}}
            exec(compile(ast.Module(body=[method],type_ignores=[]),'<test-method>','exec'),scope)
            owner=SimpleNamespace(model=Model(),data=Data(),test_data=Data(),device='cpu',
                                  link_predict=lambda *a:Output())
            scope['test'](owner)
            self.assertTrue(state['grad']);self.assertTrue(state['eval'])
        self.assertEqual(state['calls'],2)


if __name__=='__main__':unittest.main()
