"""Dependency-free contracts for alias accounting and non-retention."""
import gc
import importlib.util
import json
from pathlib import Path
import tempfile
import subprocess
import sys
from types import SimpleNamespace
import unittest
import weakref
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

spec = importlib.util.spec_from_file_location('memory_trace', Path(__file__).resolve().parents[1] / 'src/utils/layer_memory_trace.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Device:
    type, index = 'cuda', 0

    def __str__(self):
        return 'cuda:0'


class Storage:
    def __init__(self, identity, size):
        self._cdata, self.size = identity, size

    def nbytes(self):
        return self.size

    def data_ptr(self):
        return self._cdata * 100


class Tensor:
    device, layout, dtype = Device(), 'strided', 'float32'
    requires_grad, grad_fn = False, None

    def __init__(self, storage, count=4):
        self.backing, self.shape = storage, (count,)

    def untyped_storage(self):
        return self.backing

    def numel(self):
        return self.shape[0]

    def element_size(self):
        return 4

    def storage_offset(self):
        return 0

    def stride(self):
        return (1,)


class Parameter(Tensor):
    grad = None


class Module:
    def modules(self):
        return [self]

    def named_modules(self):
        return [('', self)]

    def named_parameters(self):
        return []


class Optimizer:
    pass


class Cuda:
    allocated = 128
    def synchronize(self, device):
        pass
    def memory_allocated(self, device):
        return self.allocated
    def memory_reserved(self, device):
        return 256
    def max_memory_allocated(self, device):
        return 192
    def max_memory_reserved(self, device):
        return 256
    def reset_peak_memory_stats(self, device):
        pass


def fake_torch():
    return SimpleNamespace(is_tensor=lambda x: isinstance(x, Tensor), strided='strided',
                           nn=SimpleNamespace(Module=Module, Parameter=Parameter),
                           optim=SimpleNamespace(Optimizer=Optimizer), cuda=Cuda())


class InventoryTests(unittest.TestCase):
    def test_aliases_resize_disappearance_and_no_retention(self):
        inventory = module.TensorInventory(fake_torch(), Device())
        backing = Storage(1, 64)
        tensor, view = Tensor(backing, 16), Tensor(backing, 2)
        reference = weakref.ref(tensor)
        first = inventory.capture([('x', tensor), ('view', view)])
        self.assertEqual(first['unique_storage_bytes'], 64)
        self.assertEqual(first['logical_bytes_with_aliases'], 72)
        self.assertEqual(len(first['storages']), 1)
        second = inventory.capture([('view', view)])
        self.assertFalse(second['disappeared_storages'])
        self.assertEqual(second['alias_changes'][0]['removed'], ['x'])
        backing.size = 128
        self.assertEqual(inventory.capture([('view', view)])['resized_storages'][0]['after_bytes'], 128)
        del tensor
        gc.collect()
        self.assertIsNone(reference())
        last = inventory.capture([])
        self.assertEqual(last['disappeared_storages'][0]['bytes'], 128)
        json.dumps(last)

    def test_parameter_gradient_optimizer_and_cycles(self):
        model, optimizer = Module(), Optimizer()
        model.weight = Parameter(Storage(1, 16))
        model.weight.grad = Tensor(Storage(2, 16))
        model.optimizer = optimizer
        optimizer.state = {model.weight: {'exp_avg': Tensor(Storage(3, 16)),
                                        'exp_avg_sq': Tensor(Storage(4, 16))}}
        optimizer.param_groups = [{'params': [model.weight]}]
        model.cycle = model
        inventory = module.TensorInventory(fake_torch(), Device())
        result = inventory.capture([('model', model)])
        self.assertEqual(result['unique_storage_bytes'], 64)
        self.assertEqual(len(result['storages']), 4)
        self.assertTrue(any('exp_avg' in row['path'] for row in result['variables']))

    def test_device_filter_and_non_strided_diagnostic(self):
        tensor = Tensor(Storage(1, 16))
        tensor.device = SimpleNamespace(type='cuda', index=1)
        sparse = Tensor(Storage(2, 16))
        sparse.layout = 'sparse'
        result = module.TensorInventory(fake_torch(), Device()).capture([('other', tensor), ('sparse', sparse)])
        self.assertEqual(result['unique_storage_bytes'], 0)
        self.assertEqual(len(result['inspection_errors']), 1)

    def test_native_gradient_hook_without_python_parent(self):
        torch, model = fake_torch(), Module()
        gradient = Tensor(Storage(7, 16))
        reference = weakref.ref(gradient)
        with tempfile.TemporaryDirectory() as directory:
            tracer = module.LayerMemoryTrace(torch, Device(), model, directory)
            tracer.active = True
            # Reproduce a C++ -> Python callback with only the hook frame.
            with patch.object(module.sys, '_getframe', return_value=SimpleNamespace(f_back=None)) as getframe:
                self.assertIsNone(tracer.gradient_hook('encoder', 'weight')(gradient))
                getframe.assert_called_once_with()
            tracer.active = False  # This test did not install sys.settrace.
            tracer.close()
            event = json.loads((Path(directory) / 'memory-events.jsonl').read_text())
            details = json.loads((Path(directory) / 'memory-variables.jsonl').read_text())
            self.assertFalse(event['end']['python_parent_frame_available'])
            self.assertEqual(event['stage_context']['phase'], 'backward')
            self.assertEqual(details['unique_storage_bytes'], 16)
            self.assertEqual(details['variables'][0]['path'], 'incoming_gradient')
            del gradient
            gc.collect()
            self.assertIsNone(reference())

    def test_boundary_output_and_disabled_inventory(self):
        torch, model = fake_torch(), Module()
        model.weight = Parameter(Storage(1, 16))
        with tempfile.TemporaryDirectory() as directory:
            tracer = module.LayerMemoryTrace(torch, Device(), model, directory)
            tracer.boundary({'event': 'before'})
            model.weight = None
            torch.cuda.allocated = 112
            tracer.boundary({'event': 'after'})
            tracer.close()
            rows = [json.loads(line) for line in (Path(directory) / 'memory-events.jsonl').read_text().splitlines()]
            details = [json.loads(line) for line in (Path(directory) / 'memory-variables.jsonl').read_text().splitlines()]
            self.assertEqual(rows[1]['allocated_delta_bytes'], -16)
            self.assertEqual(rows[1]['disappeared_visible_storage_bytes'], 16)
            self.assertEqual(details[1]['sequence'], rows[1]['sequence'])
            self.assertIn('Visible storages', (Path(directory) / 'memory-trace-report.md').read_text())
            command = [sys.executable, str(Path(__file__).resolve().parents[1] / 'src/inspect_memory_event.py'),
                       directory, '--sequence', '1']
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            self.assertIn('disappeared_storages', result.stdout)
            self.assertIn('model.weight', result.stdout)
            stage_file = Path(directory) / 'memory-stage-summary.json'
            self.assertTrue(stage_file.exists())
            command = [sys.executable, str(Path(__file__).resolve().parents[1] / 'src/memory_stage_summary.py'), directory]
            subprocess.run(command, capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(stage_file.read_text())[0]['snapshot']['sequence'], 0)
        with tempfile.TemporaryDirectory() as directory:
            tracer = module.LayerMemoryTrace(torch, Device(), model, directory, variable_details=False)
            tracer.boundary({'event': 'before'})
            tracer.close()
            self.assertFalse((Path(directory) / 'memory-variables.jsonl').exists())


class NativeCudaTests(unittest.TestCase):
    def test_real_storage_views_adam_and_non_retention(self):
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch unavailable')
        if not torch.cuda.is_available():
            self.skipTest('CUDA unavailable')
        device = torch.device('cuda', torch.cuda.current_device())
        model = torch.nn.Linear(4, 2).to(device)
        model.optimizer = torch.optim.Adam(model.parameters())
        x = torch.ones(3, 4, device=device)
        model(x).sum().backward()
        model.optimizer.step()
        inventory = module.TensorInventory(torch, device)
        result = inventory.capture([('model', model), ('x', x), ('alias', x.detach()), ('view', x[:, :1])])
        rows = [row for row in result['variables'] if row['path'] in ('x', 'alias', 'view')]
        self.assertEqual(len({row['storage_id'] for row in rows}), 1)
        self.assertEqual(rows[0]['storage_bytes'], x.numel() * x.element_size())
        self.assertTrue(any('exp_avg_sq' in row['path'] for row in result['variables']))
        reference = weakref.ref(x)
        del x
        gc.collect()
        self.assertIsNone(reference())
        json.dumps(result)


if __name__ == '__main__':
    unittest.main()
