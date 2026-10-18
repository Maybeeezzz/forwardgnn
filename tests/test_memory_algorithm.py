"""Attribution, coherent peaks, weak saved lifetimes and optional real autograd."""
import csv
import gc
import json
from pathlib import Path
import sys
import tempfile
import unittest
import weakref
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from memory_algorithm_summary import classify, AlgorithmMemorySummary, layer_name
import importlib.util

def load_module(name, filename):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).resolve().parents[1]/'src/utils'/filename)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

SavedMemory=load_module('saved_memory','saved_memory.py').SavedMemory
api=load_module('utils.memory_analysis','memory_analysis.py')
memory_analysis=api.memory_analysis
from types import SimpleNamespace


def details(*rows):
    variables, storages = [], {}
    for key, size, path, attributes in rows:
        variables.append(dict(storage_id=key, path=path, **attributes))
        storages.setdefault(key, dict(storage_id=key, bytes=size, aliases=[]))['aliases'].append(path)
    return dict(variables=variables, storages=list(storages.values()))


class AttributionTests(unittest.TestCase):
    def test_aliases_baseline_gradient_and_layer_owners(self):
        d = details(('p',64,"model.layers.0._parameters['weight']",{'is_parameter':True}),
                    ('p',64,'autograd_saved.layer2.0',{'saved_for_backward':True}),
                    ('g',64,"model.layers.0._parameters['weight'].grad",{}),
                    ('x',128,'baseline_input.1',{'baseline_storage':True}),
                    ('x',128,'autograd_saved.layer1.1',{'saved_for_backward':True}),
                    ('a',128,'autograd_saved.layer1.2',{'saved_for_backward':True,'saved_owner':'layer1'}))
        by_id={x['storage_id']:x for x in classify(d,[])}
        self.assertEqual(by_id['p']['category'],'model_baseline')
        self.assertEqual(by_id['x']['category'],'model_baseline')
        self.assertEqual(by_id['g']['category'],'gradients')
        self.assertEqual(by_id['g']['owner'],'layer1')
        self.assertEqual(by_id['a']['category'],'saved_for_backward')
        self.assertEqual(sum(x['bytes'] for x in by_id.values()),384)
        self.assertEqual(layer_name("model._modules['convs']._modules['1']._parameters['w']"),'layer2')

    def test_explicit_rules_and_unknown_not_charged(self):
        d=details(('x',100,'model.aug_graph.x',{}),('y',12,'frame0.forward.x',{}))
        d['variables'].append(dict(storage_id='x',path='frame0.forward_train.node_feats'))
        rules=[{'pattern':'*.aug_graph.*','category':'algorithm_data'},
               {'pattern':'*.forward_train.node_feats','category':'algorithm_cache'}]
        values=classify(d,rules)
        self.assertEqual(values[0]['category'],'algorithm_data')
        self.assertEqual(values[1]['category'],'unattributed_visible')

    def test_algorithm_peak_independent_of_allocator_peak(self):
        summary=AlgorithmMemorySummary([])
        for seq,(gradient,state,allocator) in enumerate([(64,0,100000),(0,128,200)]):
            d=details(('g',gradient,'weight.grad',{'memory_owner':'layer1'}),
                      ('s',state,"optimizer.state[key#0]['exp_avg']",{'memory_owner':'layer1'}))
            summary.observe(dict(sequence=seq, allocated_bytes=allocator, interval_peak_allocated_bytes=allocator),d,
                            dict(layer='convs.0',phase='backward',epoch_zero_based=seq))
        self.assertEqual(summary.global_peak['algorithm_bytes'],128)
        self.assertEqual(summary.global_peak['sequence'],1)
        with tempfile.TemporaryDirectory() as directory:
            summary.write(directory)
            with open(Path(directory)/'algorithm-memory-stages.csv') as f:
                row=next(r for r in csv.DictReader(f) if r['owner']=='layer1')
            self.assertEqual(float(row['gradients_mib']),0)
            self.assertEqual(float(row['optimizer_state_mib'])*2**20,128)

    def test_saved_registry_weak_and_unpack_preserves_tensor(self):
        class Tensor:
            def detach(self): return self
        saved=SavedMemory(None,lambda:'layer1')
        tensor=Tensor();packed=saved.pack(tensor)
        self.assertIs(saved.unpack(packed),tensor)
        self.assertEqual(len(saved.roots()),1)
        ref=weakref.ref(packed);del packed;gc.collect()
        self.assertIsNone(ref());self.assertEqual(saved.roots(),[])

    def test_invalid_flag_combinations_fail_before_torch_import(self):
        for options in [dict(memory_analysis=False),dict(memory_analysis=True,memory_analysis_operators=True),
                        dict(memory_analysis=True,memory_analysis_no_variables=True)]:
            with self.assertRaises(ValueError):
                memory_analysis(SimpleNamespace(memory_analysis_algorithm=True,**options),None)


try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None, 'PyTorch unavailable')
class RealAutogradTests(unittest.TestCase):
    def test_gradients_and_saved_handle_release(self):
        torch.manual_seed(7)
        model=torch.nn.Linear(3,2);x=torch.randn(4,3)
        model(x).square().sum().backward()
        expected=model.weight.grad.clone();model.zero_grad()
        saved=SavedMemory(torch,lambda:'layer1');saved.start()
        try:
            loss=model(x).square().sum()
            self.assertTrue(saved.roots())
            loss.backward()
            self.assertTrue(torch.allclose(model.weight.grad,expected))
            self.assertFalse(saved.roots())
        finally: saved.close()

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_end_to_end(self):
        from utils.memory_analysis import MemoryAnalysis
        model=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.ReLU(),torch.nn.Linear(4,2)).cuda()
        optimizer=torch.optim.Adam(model.parameters())
        x=torch.randn(5,3,device='cuda')
        with tempfile.TemporaryDirectory() as directory:
            with MemoryAnalysis(True,model,'cuda:0',directory,algorithm=True) as memory:
                with memory.stage('forward',layer='layers.0'):
                    loss=model(x).square().mean()
                with memory.stage('backward',layer='layers.0'):
                    loss.backward()
                with memory.stage('optimizer_step'):
                    optimizer.step()
            payload=json.loads((Path(directory)/'algorithm-memory-summary.json').read_text())
            self.assertTrue(payload['complete'])
            self.assertGreater(payload['global_peak']['algorithm_bytes'],0)


if __name__=='__main__': unittest.main()
