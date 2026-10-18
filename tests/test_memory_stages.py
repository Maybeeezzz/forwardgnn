"""Stage semantics and coherent, storage-deduplicated table snapshots."""
import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from memory_stage_summary import StageMemorySummary, StageResolver, categorize


def snapshot(parameter, activation):
    return dict(variables=[
        dict(path="model._parameters['weight']", storage_id='p', is_parameter=True),
        dict(path='frame0.forward.x', storage_id='p'),
        dict(path='frame0.forward.node_emb', storage_id='a'),
        dict(path='frame1.forward_train.node_emb', storage_id='a'),
    ], storages=[dict(storage_id='p', bytes=parameter, aliases=['weight', 'x']),
                 dict(storage_id='a', bytes=activation, aliases=['node_emb'])], inspection_errors=[])


class StageTests(unittest.TestCase):
    def test_alias_categories_are_exclusive(self):
        totals, entries = categorize(snapshot(64, 128))
        self.assertEqual(sum(totals.values()), 192)
        self.assertEqual(totals['parameters'], 64)
        self.assertEqual(totals['activations'], 128)
        self.assertEqual(len(entries), 2)
        paths = [('.optimizer.state[key#0][\'exp_avg\']', 'optimizer_state'),
                 ('.weight.grad', 'gradients'), ('.best_model_state_dict[\'w\']', 'checkpoint'),
                 ('._cached_propagate_out', 'cache'), ('.aug_graph.x', 'graph_data'),
                 ('.forward_train.logits', 'logits_loss')]
        for index, (path, category) in enumerate(paths):
            details = dict(variables=[dict(path='model' + path, storage_id=str(index))],
                           storages=[dict(storage_id=str(index), bytes=16, aliases=[path])])
            self.assertEqual(categorize(details)[0][category], 16)

    def test_same_snapshot_not_sum_of_independent_category_peaks(self):
        summary = StageMemorySummary()
        for sequence, (allocated, parameter, activation) in enumerate([(300, 64, 128), (280, 128, 64)]):
            summary.observe(dict(sequence=sequence, allocated_bytes=allocated, reserved_bytes=512,
                                 interval_peak_allocated_bytes=400), snapshot(parameter, activation),
                            dict(layer=1, phase='forward', epoch_zero_based=sequence))
        summary.observe(dict(sequence=2, allocated_bytes=350, reserved_bytes=512,
                             interval_peak_allocated_bytes=450), snapshot(64, 128),
                        dict(layer=1, phase='backward', epoch_zero_based=0))
        with tempfile.TemporaryDirectory() as directory:
            summary.write(directory)
            with (Path(directory) / 'memory-stage-summary.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            row = next(r for r in rows if r['phase'] == 'forward')
            self.assertEqual(int(row['sequence']), 0)
            self.assertEqual(float(row['parameters_mib']) * 2**20, 64)
            self.assertEqual(float(row['activations_mib']) * 2**20, 128)
            self.assertEqual(float(row['unaccounted_mib']) * 2**20, 108)
            self.assertEqual(float(row['interval_peak_allocated_mib']) * 2**20, 400)
            groups = json.loads((Path(directory) / 'memory-stage-summary.json').read_text())
            self.assertEqual(len(groups), 2)
            self.assertTrue(all('storages' in group['snapshot'] for group in groups))

    def test_statement_stage_multiline_and_evaluation_override(self):
        source = '''class ArbitraryBlock:
    def forward_train(self, x):
        node_emb = self.forward(
            x.detach()
        )
        logits = make_logits(node_emb)
        self.optimizer.zero_grad()
        loss = self.criterion(logits)
        loss.backward()
        self.optimizer.step()
        return loss.item()
'''
        owner = type('ArbitraryBlock', (), {})()
        optimizer_base = type('Optimizer', (), {'__module__': 'torch.optim.optimizer'})
        owner.optimizer = optimizer_base()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'custom_model.py'
            path.write_text(source)
            resolver = StageResolver()
            for line, expected in [(3, 'forward'), (4, 'forward'), (6, 'other'),
                                   (7, 'zero_grad'), (8, 'other'), (9, 'backward'),
                                   (10, 'optimizer_step'), (11, 'other')]:
                frame = SimpleNamespace(f_code=SimpleNamespace(co_filename=str(path), co_name='forward_train'),
                                        f_lineno=line, f_locals={'self': owner, 'epoch': 0}, f_back=None)
                context = resolver.resolve(frame, {}, {id(owner): 1})
                self.assertEqual(context['phase'], expected)
                self.assertEqual(context['layer'], 1)
            evaluator = SimpleNamespace(f_code=SimpleNamespace(co_filename=str(path), co_name='eval_model'),
                                        f_lineno=1, f_locals={}, f_back=None)
            frame.f_back = evaluator
            self.assertEqual(resolver.resolve(frame, {}, {})['phase'], 'evaluation')


if __name__ == '__main__':
    unittest.main()
