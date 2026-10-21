"""Offline contracts and CPU tensor checks; no dataset downloads or CUDA training."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import network_algorithm_memory as matrix
from memory_algorithm_summary import AlgorithmMemorySummary, classify

try:
    import torch
    from network_memory_models import Network, update, predict, inject
except ImportError:
    torch = None


class PlannerTests(unittest.TestCase):
    def test_summary_separates_modes_and_missing_is_not_zero(self):
        import csv
        with tempfile.TemporaryDirectory() as d:
            a = matrix.arguments(['--output',d,'--architectures','mlp','--algorithms','bp'])
            p = matrix.plan(a)
            matrix.atomic(Path(d)/'plan.json',p)
            case = p['cases'][0]
            folder=Path(case['directory'])/'attempt-1';folder.mkdir(parents=True)
            stats=dict(peak_allocated_mib=10,peak_reserved_mib=12,end_allocated_mib=5)
            matrix.atomic(folder/'result.json',dict(complete=True,signature=p['signature'],
                network='mlp',algorithm='bp',mode='allocator',parameters_changed=True,
                parameter_count=100,algorithm_parameter_count=0,losses=[1.0],accuracy=0.2,
                training=stats,evaluation=stats))
            with patch.object(matrix.subprocess,'run',side_effect=AssertionError('No training')):
                self.assertFalse(matrix.summarize(a,p))
            with (Path(d)/'matrix-periods.csv').open() as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),2)
            self.assertEqual(rows[0]['algorithm_peak_mib'],'')
            self.assertEqual(rows[0]['peak_allocated_mib'],'10')
            self.assertEqual(json.loads((Path(d)/'missing.json').read_text())[0]['mode'],'compact')

    def test_default_dry_run_without_workers_or_files(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/'absent'
            with patch.object(matrix.subprocess, 'run', side_effect=AssertionError('No workers')), contextlib.redirect_stdout(io.StringIO()) as stream:
                self.assertEqual(matrix.main(['--output', str(out), '--dry-run']), 0)
            p = json.loads(stream.getvalue())
            self.assertEqual(len(p['cases']), 20)
            self.assertEqual(len({c['directory'] for c in p['cases']}), 20)
            self.assertFalse(out.exists())

    def test_resume_validation_and_incomplete_attempt(self):
        with tempfile.TemporaryDirectory() as d:
            argv = ['--output', d, '--architectures', 'mlp', '--algorithms', 'bp', '--modes', 'allocator']
            a = matrix.arguments(argv)
            p = matrix.plan(a)
            matrix.atomic(Path(d)/'plan.json', p)
            case = p['cases'][0]
            folder = Path(case['directory'])/'attempt-1'
            folder.mkdir(parents=True)
            matrix.atomic(folder/'result.json', {'complete':False, 'signature':p['signature']})
            self.assertIsNone(matrix.completed(case, p['signature']))
            matrix.atomic(folder/'result.json', dict(complete=True, signature=p['signature'],
                          network='mlp', algorithm='bp', mode='allocator', parameters_changed=True,
                          losses=[1.0], accuracy=0.1,
                          training=dict(peak_allocated_mib=1,peak_reserved_mib=2,end_allocated_mib=1),
                          evaluation=dict(peak_allocated_mib=1,peak_reserved_mib=2,end_allocated_mib=1)))
            self.assertEqual(matrix.completed(case, p['signature']), folder/'result.json')
            with patch.object(matrix.subprocess, 'run', side_effect=AssertionError('No worker')), contextlib.redirect_stdout(io.StringIO()) as stream:
                matrix.main(argv+['--resume','--dry-run'])
            self.assertEqual(json.loads(stream.getvalue())['cases'][0]['action'], 'skip')
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                matrix.main(argv+['--resume', '--steps', '21', '--dry-run'])

    def test_explicit_parameter_and_buffer_override_deduplicates(self):
        for category in ('local_heads', 'feedback_state', 'parameter_backup', 'perturbation_state'):
            d = dict(variables=[dict(storage_id='s', path='model._buffers[x]', algorithm_category=category),
                                dict(storage_id='s', path='autograd_saved.layer1.0', saved_for_backward=True)],
                     storages=[dict(storage_id='s', bytes=64, aliases=['model._buffers[x]', 'autograd_saved.layer1.0'])])
            rows = classify(d, [])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['category'], category)
            self.assertEqual(rows[0]['bytes'], 64)
            del d['variables'][0]['algorithm_category']
            self.assertEqual(classify(d, [])[0]['category'], 'model_baseline')
            self.assertEqual(classify(d, [{'pattern':'model.*','category':category,'override_baseline':True}])[0]['category'], category)

    def test_warmup_does_not_replace_training_period(self):
        summary = AlgorithmMemorySummary([])
        for period, size in [('warmup',1000), ('training',10)]:
            summary.observe(dict(sequence=0, allocated_bytes=size, interval_peak_allocated_bytes=size),
                            dict(variables=[dict(storage_id='s', path='incoming_gradient')],
                                 storages=[dict(storage_id='s',bytes=size,aliases=['incoming_gradient'])]),
                            dict(period=period,phase='backward'))
        self.assertEqual(summary.period_peaks['training']['algorithm_bytes'], 10)
        self.assertEqual(summary.period_peaks['warmup']['algorithm_bytes'], 1000)


@unittest.skipIf(torch is None, 'PyTorch not installed')
class LearningRuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def sample(self):
        g = torch.Generator().manual_seed(17)
        return torch.rand(2,1,28,28,generator=g), torch.tensor([2,7])

    def test_common_initialization_all_algorithms(self):
        for arch in matrix.ARCHITECTURES:
            baseline = Network(arch, 'bp')
            for algorithm in matrix.ALGORITHMS:
                net = Network(arch, algorithm)
                for a,b in zip(net.layers.parameters(), baseline.layers.parameters()):
                    self.assertTrue(torch.equal(a,b))
                if net.head is not None:
                    self.assertTrue(torch.equal(net.head.weight, baseline.head.weight))

    def test_all_rules_update_finitely_and_predict(self):
        x,y = self.sample()
        for arch in matrix.ARCHITECTURES:
            for algorithm in matrix.ALGORITHMS:
                with self.subTest(arch=arch, algorithm=algorithm):
                    net = Network(arch, algorithm).setup('cpu')
                    before = [p.detach().clone() for p in net.parameters()]
                    value = update(net,x,y,generator=torch.Generator().manual_seed(9))
                    self.assertTrue(torch.isfinite(torch.tensor(value)))
                    self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,net.parameters())))
                    self.assertTrue(all(torch.isfinite(p).all() for p in net.parameters()))
                    prediction = predict(net,x)
                    self.assertEqual(prediction.shape,y.shape)
                    self.assertTrue(((prediction>=0)&(prediction<10)).all())

    def test_local_ff_dfa_inputs_are_detached(self):
        x,y = self.sample()
        for algorithm in ('local','ff','dfa'):
            net = Network('mlp',algorithm).setup('cpu')
            seen=[]
            handles=[layer.register_forward_pre_hook(lambda m,args: seen.append(args[0].requires_grad))
                     for layer in net.layers]
            update(net,x,y,generator=torch.Generator().manual_seed(9))
            for h in handles: h.remove()
            self.assertTrue(seen)
            self.assertFalse(any(seen))

    def test_dfa_feedback_is_fixed_and_weight_gradient_matches_vjp(self):
        x,y = self.sample()
        net = Network('mlp','dfa').setup('cpu')
        original = net.feedback_0.clone()
        h0 = net.layers[0](x.flatten(1))
        h = net.layers[2](net.layers[1](h0.detach()).detach())
        logits = net.head(h.detach())
        error = (logits.softmax(-1)-torch.nn.functional.one_hot(y,10))/len(y)
        expected = torch.autograd.grad(h0, net.layers[0][0].weight, error.detach() @ original)[0]
        update(net,x,y)
        self.assertTrue(torch.equal(original, net.feedback_0))
        self.assertTrue(torch.allclose(expected, net.layers[0][0].weight.grad, atol=1e-6))
        self.assertEqual(net.feedback_0._memory_algorithm_category,'feedback_state')

    def test_spsa_restores_on_failure_and_never_builds_graph(self):
        x,y = self.sample()
        net = Network('mlp','spsa').setup('cpu')
        original=[p.clone() for p in net.parameters()]
        real_forward=net.forward
        calls=[]
        def fail_second(*args,**kwargs):
            calls.append(torch.is_grad_enabled())
            if len(calls)==2: raise RuntimeError('injected failure')
            return real_forward(*args,**kwargs)
        with patch.object(net,'forward',side_effect=fail_second), self.assertRaisesRegex(RuntimeError,'injected'):
            update(net,x,y,generator=torch.Generator().manual_seed(9))
        self.assertEqual(calls,[False,False])
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(original,net.parameters())))

    def test_injection_changes_only_reserved_pixels(self):
        x,y = self.sample()
        result=inject(x,y)
        self.assertTrue(torch.equal(result.flatten(1)[:,10:],x.flatten(1)[:,10:]))
        self.assertTrue(torch.equal(result.flatten(1)[:,:10],torch.nn.functional.one_hot(y,10)))
        self.assertNotEqual(result.data_ptr(),x.data_ptr())

    def test_spsa_restores_before_update_and_estimates_two_sided_direction(self):
        x,y = self.sample()
        net = Network('mlp','spsa').setup('cpu')
        original = [p.clone() for p in net.parameters()]
        generator = torch.Generator().manual_seed(9)
        directions = [torch.empty_like(p).bernoulli_(0.5,generator=generator).mul_(2).sub_(1)
                      for p in net.parameters()]
        losses = []
        real_forward = net.forward
        def forward(*args,**kwargs):
            value = real_forward(*args,**kwargs)
            losses.append(torch.nn.functional.cross_entropy(value,y).item())
            return value
        real_step = net.optimizer.step
        def step():
            self.assertEqual(len(losses),2)
            for p,old,direction in zip(net.parameters(),original,directions):
                self.assertTrue(torch.equal(p,old))
                self.assertTrue(torch.allclose(p.grad,direction*((losses[0]-losses[1])/0.002)))
            real_step()
        with patch.object(net,'forward',side_effect=forward), patch.object(net.optimizer,'step',side_effect=step):
            update(net,x,y,generator=torch.Generator().manual_seed(9))


if __name__ == '__main__':
    unittest.main()
