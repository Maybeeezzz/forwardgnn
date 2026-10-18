"""Opt-in API: no-op behavior, generic model support and scope cleanup."""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

from test_memory_inventory import fake_torch, Device, Module, module as trace_module

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('utils.memory_analysis', ROOT / 'src/utils/memory_analysis.py')
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)


class AnalysisTests(unittest.TestCase):
    def test_disabled_is_noop_even_with_invalid_device_and_model(self):
        parser = api.add_memory_arguments(argparse.ArgumentParser())
        args = parser.parse_args([])
        self.assertFalse(args.memory_analysis)
        original_trace = sys.gettrace()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'absent'
            # Any attempted torch import would fail.
            with patch.dict(sys.modules, {'torch': None}):
                with api.MemoryAnalysis(model=object(), device='bad', directory=output) as memory:
                    with memory.stage('anything', layer='arbitrary'):
                        pass
                    self.assertIsNone(memory.tracer)
                with api.memory_analysis(args, object()):
                    pass
            self.assertFalse(output.exists())
        self.assertIs(sys.gettrace(), original_trace)

    def test_entry_parsers_expose_default_off_flag(self):
        for filename in ('train_forward.py', 'train_backprop.py'):
            tree = ast.parse((ROOT / 'src' / filename).read_text())
            function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'parse_args')
            scope = {'argparse': argparse, 'add_memory_arguments': api.add_memory_arguments}
            exec(compile(ast.Module(body=[function], type_ignores=[]), filename, 'exec'), scope)
            with patch.object(sys, 'argv', [filename]):
                self.assertFalse(scope['parse_args']().memory_analysis)
            with patch.object(sys, 'argv', [filename, '--memory-analysis', '--memory-analysis-dir', '/tmp/example']):
                args = scope['parse_args']()
                self.assertTrue(args.memory_analysis)
                self.assertEqual(args.memory_analysis_dir, '/tmp/example')
            with patch.object(sys, 'argv', [filename, '--memory-analysis', '--memory-analysis-operators',
                                           '--memory-analysis-module', 'encoder.aggregate',
                                           '--memory-analysis-call', '2', '--memory-analysis-record-shapes',
                                           '--memory-analysis-tensor-ops']):
                args = scope['parse_args']()
                self.assertTrue(args.memory_analysis_operators)
                self.assertEqual(args.memory_analysis_module, 'encoder.aggregate')
                self.assertEqual(args.memory_analysis_call, 2)
                self.assertTrue(args.memory_analysis_record_shapes and args.memory_analysis_tensor_ops)


    def test_enabled_scope_and_exception_restore_trace(self):
        torch = fake_torch()
        torch.device = lambda value, index=None: Device()
        torch.cuda.is_available = lambda: True
        package = ModuleType('utils')
        package.__path__ = []
        original_trace = sys.gettrace()
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(sys.modules, {'torch': torch, 'utils': package,
                                          'utils.layer_memory_trace': trace_module}):
                model = Module()  # No layers/convs properties.
                with self.assertRaisesRegex(ValueError, 'training failed'):
                    with api.MemoryAnalysis(True, model, Device(), directory, trace_roots=[directory]) as memory:
                        with memory.stage('custom_forward', layer='encoder', epoch=2):
                            memory.tracer.boundary({'event': 'test'})
                        raise ValueError('training failed')
                self.assertTrue(memory.tracer.closed)
            self.assertIs(sys.gettrace(), original_trace)
            rows = json.loads((Path(directory) / 'memory-stage-summary.json').read_text())
            self.assertTrue(any(row['snapshot']['phase'] == 'custom_forward'
                                and row['snapshot']['layer'] == 'encoder' for row in rows))
            # Repeated close is harmless and does not rewrite/erase diagnostics.
            memory.tracer.close()


if __name__ == '__main__':
    unittest.main()
