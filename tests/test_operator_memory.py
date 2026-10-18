"""Trace parsing, allocation lifetimes and targeted-profiler contracts without CUDA."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from operator_memory_report import analyze_trace, generate_report
spec = importlib.util.spec_from_file_location('operator_trace', ROOT / 'src/utils/operator_memory_trace.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def op(name, start, duration, tid=1):
    return dict(name=name, ph='X', cat='cpu_op', ts=start, dur=duration, pid=1, tid=tid,
                args={'Input Dims': [[4, 8]]})


def mem(stamp, address, size, total, tid=1, device=0):
    return dict(name='[memory]', ph='i', ts=stamp, pid=1, tid=tid,
                args={'Device Type': 1, 'Device Id': device, 'Addr': address, 'Bytes': size,
                      'Total Allocated': total, 'Total Reserved': 4096})


class ParserTests(unittest.TestCase):
    def test_transient_allocation_owner_and_address_reuse(self):
        trace = dict(traceEvents=[op('aten::scatter', 0, 100), op('aten::empty', 5, 10),
                                 mem(6, 100, 1024, 2024), mem(20, 100, -1024, 1000),
                                 mem(30, 100, 128, 1128), mem(40, 100, -128, 1000),
                                 mem(50, 200, -10, 990), mem(60, 300, 9999, 9999, device=1)])
        result = analyze_trace(trace)
        self.assertEqual(len(result['allocations']), 2)
        a, b = result['allocations']
        self.assertEqual(a['operator'], 'aten::empty')
        self.assertEqual(b['operator'], 'aten::scatter')
        self.assertNotEqual(a['allocation_id'], b['allocation_id'])
        self.assertEqual(a['freed_at_us'], 20)
        self.assertEqual(result['trace_peak_allocated_bytes'], 2024)
        self.assertEqual(sum(x['bytes'] for x in result['live_at_trace_peak']), 1024)
        self.assertEqual(result['unmatched_frees'], 1)
        self.assertEqual(sum(x['gross_allocated_bytes'] for x in result['operators']), 1152)

    def test_thread_scope_and_missing_events(self):
        result = analyze_trace([op('wrong_thread', 0, 20), mem(5, 1, 64, 64, tid=2)])
        self.assertEqual(result['allocations'][0]['operator'], '<unattributed>')
        missing = analyze_trace([op('only_op', 0, 20)])
        self.assertEqual(missing['status'], 'no_cuda_memory_events')
        self.assertIsNone(missing['trace_peak_allocated_bytes'])

    def test_report_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / 'operator-metadata.json').write_text(json.dumps({'device_index': 0}))
            (directory / 'operator-trace.json').write_text(json.dumps([op('aten::empty', 0, 10), mem(1, 2, 1024, 2048)]))
            generate_report(directory)
            self.assertIn('aten::empty', (directory / 'operator-memory-report.md').read_text())
            self.assertTrue((directory / 'operator-allocations.csv').exists())


class Handle:
    def __init__(self):
        self.removed = False
    def remove(self):
        self.removed = True


class Target:
    training = True
    def register_forward_pre_hook(self, hook):
        self.pre = hook
        self.pre_handle = Handle()
        return self.pre_handle
    def register_forward_hook(self, hook):
        self.post = hook
        self.post_handle = Handle()
        return self.post_handle


class TargetTests(unittest.TestCase):
    def fake(self):
        return SimpleNamespace(__version__='test', version=SimpleNamespace(cuda='test'),
                               cuda=SimpleNamespace(get_device_name=lambda d: 'fake'),
                               are_deterministic_algorithms_enabled=lambda: True,
                               is_grad_enabled=lambda: True, is_tensor=lambda x: False)

    def test_target_call_and_cleanup(self):
        target = Target()
        model = SimpleNamespace(named_modules=lambda: [('', target), ('encoder.aggregate', target)])
        with tempfile.TemporaryDirectory() as directory:
            tracer = module.OperatorMemoryTrace(self.fake(), SimpleNamespace(index=0), model, directory,
                                                 target_module='encoder.aggregate', target_call=2)
            begins = []
            def begin():
                tracer.active = True
                begins.append(tracer.calls)
            def finish(error=None):
                tracer.active = False
                tracer.finished = True
            tracer.begin, tracer.finish = begin, finish
            tracer.start()
            target.pre(target, ())
            target.post(target, (), None)
            self.assertEqual(begins, [])
            target.pre(target, ())
            target.post(target, (), None)
            target.pre(target, ())
            target.post(target, (), None)
            self.assertEqual(begins, [2])
            tracer.close()
            self.assertTrue(target.pre_handle.removed and target.post_handle.removed)

    def test_finish_failure_still_stops_profiler_and_closes_stream(self):
        target = Target()
        model = SimpleNamespace(named_modules=lambda: [('encoder', target)])
        with tempfile.TemporaryDirectory() as directory:
            torch = self.fake()
            def fail(device):
                raise RuntimeError('device sync failed')
            torch.cuda.synchronize = fail
            tracer = module.OperatorMemoryTrace(torch, SimpleNamespace(index=0), model, directory, target_module='encoder')
            tracer.start()
            stops = []
            tracer.profiler = SimpleNamespace(stop=lambda: stops.append('stopped'))
            stream = (Path(directory) / 'tensor-log').open('w')
            tracer.tensor_stream = stream
            tracer.active = True
            tracer.export = lambda: None
            with self.assertRaisesRegex(RuntimeError, 'device sync failed'):
                tracer.close()
            self.assertEqual(stops, ['stopped'])
            self.assertTrue(stream.closed)
            self.assertTrue(target.pre_handle.removed and target.post_handle.removed)
            self.assertEqual(tracer.metadata['status'], 'partial')

    def test_uncaptured_call_reports_error_and_removes_hooks(self):
        target = Target()
        model = SimpleNamespace(named_modules=lambda: [('encoder', target)])
        with tempfile.TemporaryDirectory() as directory:
            tracer = module.OperatorMemoryTrace(self.fake(), SimpleNamespace(index=0), model, directory, target_module='encoder')
            tracer.start()
            with self.assertRaisesRegex(RuntimeError, 'not reached'):
                tracer.close()
            self.assertTrue(target.pre_handle.removed and target.post_handle.removed)
            self.assertEqual(json.loads((Path(directory)/'operator-metadata.json').read_text())['status'], 'not_captured')


class NativeCudaTests(unittest.TestCase):
    def test_real_profiler_export_and_metadata(self):
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch unavailable')
        if not torch.cuda.is_available():
            self.skipTest('CUDA unavailable')
        class Buffer(torch.nn.Module):
            def forward(self, x):
                temp = x.repeat(32, 1)
                return temp.sum(dim=0)
        model = torch.nn.Sequential(Buffer()).cuda()
        device = torch.device('cuda', torch.cuda.current_device())
        for detailed in (False, True):
            with self.subTest(detailed=detailed), tempfile.TemporaryDirectory() as directory:
                tracer = module.OperatorMemoryTrace(torch, device, model, directory, target_module='0',
                                                     record_shapes=detailed, tensor_ops=detailed)
                tracer.start()
                try:
                    x = torch.ones(16, 16, device=device)
                    result = model(x)
                    torch.testing.assert_close(result, torch.full((16,), 512., device=device))
                finally:
                    tracer.close()
                metadata = json.loads((Path(directory)/'operator-metadata.json').read_text())
                self.assertEqual(metadata['status'], 'complete')
                summary = json.loads((Path(directory)/'operator-allocation-summary.json').read_text())
                self.assertEqual(summary['status'], 'ok', 'Profiler build lacks CUDA memory events')
                self.assertTrue(summary['allocations'])
                if detailed:
                    tensors = [json.loads(line) for line in (Path(directory)/'operator-tensors.jsonl').read_text().splitlines()]
                    self.assertTrue(any(row['outputs'] for row in tensors))


if __name__ == '__main__':
    unittest.main()
