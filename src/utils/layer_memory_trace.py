"""Diagnostic CUDA boundary tracing; no tensor references or forced collection."""
import json
import linecache
from pathlib import Path
import sys
import time


class LayerMemoryTrace:
    def __init__(self, torch, device, model, directory):
        self.torch, self.device = torch, device
        self.path = Path(directory)
        self.stream = (self.path / 'memory-events.jsonl').open('w')
        self.layers = {id(layer): i + 1 for i, layer in enumerate(
            model.layers if hasattr(model, 'layers') else model.convs)}
        for module in model.modules():
            for layer_id, index in list(self.layers.items()):
                if id(module) == layer_id:
                    for child in module.modules():
                        self.layers[id(child)] = index
                    break
        self.handles = []
        self.previous = None
        self.count = 0
        self.started = time.perf_counter()
        self.maximum = {'allocated_bytes': 0, 'reserved_bytes': 0}
        self.best = None
        self.releases = []
        self.top = []
        self.window = dict(self.maximum)
        self.by_layer = {}
        self.active = False
        self.old_trace = None
        layers = model.layers if hasattr(model, 'layers') else model.convs
        for i, layer in enumerate(layers):
            for name, param in layer.named_parameters():
                if param.requires_grad:
                    self.handles.append(param.register_hook(self.gradient_hook(i + 1, name)))

    def gradient_hook(self, layer, name):
        def hook(gradient):
            # Returning None preserves the incoming gradient. Never retain it.
            if self.active:
                self.boundary({'event': 'parameter_gradient_ready', 'layer': layer,
                               'parameter': name, 'phase': 'backward'})
        return hook

    def boundary(self, location):
        t, d = self.torch, self.device
        t.cuda.synchronize(d)
        allocated = t.cuda.memory_allocated(d)
        reserved = t.cuda.memory_reserved(d)
        peak_a = t.cuda.max_memory_allocated(d)
        peak_r = t.cuda.max_memory_reserved(d)
        row = dict(sequence=self.count, seconds=time.perf_counter() - self.started,
                   end=location, interval_after=self.previous,
                   allocated_bytes=allocated, reserved_bytes=reserved,
                   interval_peak_allocated_bytes=peak_a, interval_peak_reserved_bytes=peak_r)
        label = self.previous or location
        layer = str(label.get('layer', 'unassigned'))
        self.by_layer[layer] = max(self.by_layer.get(layer, 0), peak_a)
        if self.previous is not None:
            row['allocated_delta_bytes'] = allocated - self.last_allocated
            row['reserved_delta_bytes'] = reserved - self.last_reserved
            row['net_released_bytes'] = max(0, self.last_allocated - allocated)
            row['peak_to_end_drop_bytes'] = max(0, peak_a - allocated)
            if row['net_released_bytes'] or row['peak_to_end_drop_bytes']:
                self.releases.append(row)
                self.releases = sorted(self.releases, key=lambda r: r['peak_to_end_drop_bytes'], reverse=True)[:20]
        self.maximum = {k: max(self.maximum[k], v) for k, v in
                        [('allocated_bytes', peak_a), ('reserved_bytes', peak_r)]}
        self.window = {k: max(self.window[k], v) for k, v in
                       [('allocated_bytes', peak_a), ('reserved_bytes', peak_r)]}
        if self.best is None or peak_a > self.best['interval_peak_allocated_bytes']:
            self.best = row
        self.top.append(row)
        self.top = sorted(self.top, key=lambda r: r['interval_peak_allocated_bytes'], reverse=True)[:20]
        self.stream.write(json.dumps(row) + '\n')
        self.stream.flush()
        self.count += 1
        self.previous, self.last_allocated, self.last_reserved = location, allocated, reserved
        t.cuda.reset_peak_memory_stats(d)

    def trace(self, frame, event, arg):
        filename = frame.f_code.co_filename.replace('\\', '/')
        function = frame.f_code.co_name
        selected = (
            filename.endswith('/train_backprop.py') and function in ('train', 'test') or
            filename.endswith('/gnn_sf.py') and function in ('forward_train', 'eval_model', 'forward', '__init__', 'augment') or
            filename.endswith('/train_utils.py') and function in ('step', 'save_checkpoint', 'load_checkpoint') or
            '/torch_geometric/nn/conv/' in filename and function == 'forward' or
            filename.endswith('/gnn/gnn_conv.py') and function == 'forward'
        )
        if not selected:
            return None
        if event in ('call', 'line', 'return', 'exception'):
            location = dict(event=event, file=filename, function=function, line=frame.f_lineno)
            location['source'] = linecache.getline(filename, frame.f_lineno).strip()
            owner = frame.f_locals.get('self')
            location['class'] = type(owner).__name__
            if id(owner) in self.layers:
                location['layer'] = self.layers[id(owner)]
            ancestor = frame
            while ancestor is not None:
                if ancestor.f_code.co_filename.endswith(('gnn_sf.py', 'train_backprop.py')):
                    if 'epoch' in ancestor.f_locals and isinstance(ancestor.f_locals['epoch'], int):
                        location.setdefault('epoch_zero_based', ancestor.f_locals['epoch'])
                    if 'i' in ancestor.f_locals and isinstance(ancestor.f_locals['i'], int):
                        location.setdefault('layer', ancestor.f_locals['i'] + 1)
                ancestor = ancestor.f_back
            self.boundary(location)
        return self.trace

    def start(self):
        self.old_trace = sys.gettrace()
        if self.old_trace is not None:
            raise RuntimeError('Run memory tracing without an existing Python debugger/trace.')
        self.active = True
        self.boundary({'event': 'start', 'phase': 'before_gpu_transfer'})
        sys.settrace(self.trace)

    def end_window(self, name):
        self.boundary({'event': 'window_end', 'phase': name})
        result = dict(self.window)
        self.window = dict.fromkeys(self.window, 0)
        return result

    def close(self):
        if self.active:
            sys.settrace(self.old_trace)
            self.active = False
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.stream.close()
        summary = dict(events=self.count, whole_peak=self.maximum,
                       interval_peaks_by_layer_bytes=self.by_layer,
                       largest_peak_interval=self.best, top_peak_intervals=self.top,
                       largest_release_intervals=self.releases,
                       scope='Synchronized Python boundaries and parameter-gradient-ready hooks. '
                       'Drops are allocator observations, not exact individual cudaFree timestamps.')
        (self.path / 'memory-trace-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        def describe(row):
            loc = row.get('interval_after') or row['end']
            return '{}:{} {} layer={} epoch={} {}'.format(
                Path(loc.get('file', '')).name, loc.get('line', ''),
                loc.get('function', loc.get('event')), loc.get('layer', '—'),
                loc.get('epoch_zero_based', '—'), loc.get('source', ''))
        lines = ['# Layer memory diagnostic', '',
                 'Absolute allocator peaks; instrumented timings are not benchmark timings.', '',
                 '## Highest peak intervals', '',
                 '| MiB | Interval begins at |', '|---:|---|']
        for row in self.top:
            lines.append('| {:.2f} | {} |'.format(
                row['interval_peak_allocated_bytes'] / 2**20, describe(row).replace('|', '\\|')))
        lines += ['', '## Largest observed peak-to-end drops', '',
                  '| Peak-to-end MiB | Net released MiB | Reserved delta MiB | Interval begins at |',
                  '|---:|---:|---:|---|']
        for row in self.releases:
            lines.append('| {:.2f} | {:.2f} | {:.2f} | {} |'.format(
                row['peak_to_end_drop_bytes'] / 2**20, row['net_released_bytes'] / 2**20,
                row['reserved_delta_bytes'] / 2**20, describe(row).replace('|', '\\|')))
        lines += ['', 'See memory-events.jsonl for chronological boundaries and gradient-ready events.',
                  'A drop in allocated with unchanged reserved means allocator reuse, not release to the driver.',
                  'Layer-associated values include all live tensors; they are not exclusive per-layer sizes.']
        (self.path / 'memory-trace-report.md').write_text('\n'.join(lines) + '\n')
