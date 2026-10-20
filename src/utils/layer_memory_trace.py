"""Diagnostic CUDA boundary tracing; no tensor references or forced collection."""
import json
import linecache
from pathlib import Path
import sys
import time
import weakref
import inspect
import types
from contextlib import contextmanager

from memory_stage_summary import StageResolver, StageMemorySummary


class TensorInventory:
    """Inspect reachable tensors without retaining them or evaluating properties.

    Storage identities describe observations, not allocator allocation lifetimes.
    Missing storage may still be retained by autograd or native code.
    """
    def __init__(self, torch, device):
        self.torch = torch
        self.device = device
        self.previous = {}
        self.on_tensor = None
        self.tensor_owners = {}

    def capture(self, roots):
        variables, storages, errors = [], {}, []

        def visit(value, path, ancestors):
            if self.torch.is_tensor(value):
                if value.device.type != "cuda" or value.device.index != self.device.index:
                    return
                try:
                    if value.layout != self.torch.strided:
                        errors.append({"path": path, "reason": "non-strided tensor; storage not enumerated"})
                        return
                    storage = (value.untyped_storage() if hasattr(value, "untyped_storage")
                               else value.storage())
                    size = storage.nbytes()
                    pointer = storage.data_ptr()
                    # _cdata distinguishes simultaneously live zero-sized storages.
                    identity = getattr(storage, "_cdata", pointer)
                    key = "{}:{}:{}".format(value.device, identity, pointer)
                    row = dict(path=path, shape=list(value.shape), dtype=str(value.dtype),
                               device=str(value.device), logical_bytes=value.numel() * value.element_size(),
                               storage_id=key, storage_bytes=size, storage_offset=value.storage_offset(),
                               stride=list(value.stride()), requires_grad=value.requires_grad,
                               grad_fn=type(value.grad_fn).__name__ if value.grad_fn is not None else None,
                               is_parameter=isinstance(value, self.torch.nn.Parameter))
                    if path.startswith('autograd_saved.'):
                        row['saved_for_backward'] = True
                        row['saved_owner'] = path.split('.')[1]
                    if id(value) in self.tensor_owners:
                        row['memory_owner'] = self.tensor_owners[id(value)]
                    if self.on_tensor:
                        self.on_tensor(value, row)
                    variables.append(row)
                    entry = storages.setdefault(key, dict(storage_id=key, bytes=size, aliases=[]))
                    entry["aliases"].append(path)
                    if isinstance(value, self.torch.nn.Parameter) and value.grad is not None:
                        visit(value.grad, path + ".grad", ancestors)
                except (RuntimeError, TypeError, AttributeError) as exc:
                    errors.append({"path": path, "reason": str(exc)})
                return
            if value is None or isinstance(value, (str, bytes, int, float, bool)):
                return
            if id(value) in ancestors:
                return
            if len(ancestors) >= 32:
                errors.append({"path": path, "reason": "container depth limit"})
                return
            ancestors = ancestors | {id(value)}
            if isinstance(value, dict):
                for index, (key, child) in enumerate(value.items()):
                    # Never stringify tensor keys (optimizer.state uses Parameters).
                    name = repr(key) if isinstance(key, (str, int, float, bool)) else "key#{}".format(index)
                    visit(child, path + "[" + name + "]", ancestors)
            elif isinstance(value, (list, tuple)):
                for index, child in enumerate(value):
                    visit(child, path + "[{}]".format(index), ancestors)
            else:
                # Traverse existing attributes only; never invoke getters or lazy tensor operations.
                if type(value).__module__.split('.')[-1] in ('memory_analysis', 'layer_memory_trace', 'memory_stage_summary'):
                    return
                excluded = (type, types.ModuleType, types.FunctionType, types.MethodType)
                if not isinstance(value, excluded) and hasattr(value, '__dict__'):
                    for name, child in vars(value).items():
                        if "hook" not in name:
                            visit(child, path + "." + name, ancestors)


        for name, value in roots:
            visit(value, name, set())
        appeared = [entry for key, entry in storages.items() if key not in self.previous]
        disappeared = [entry for key, entry in self.previous.items() if key not in storages]
        resized = [dict(storage_id=key, before_bytes=self.previous[key]["bytes"], after_bytes=entry["bytes"])
                   for key, entry in storages.items()
                   if key in self.previous and self.previous[key]["bytes"] != entry["bytes"]]
        alias_changes = [dict(storage_id=key,
                              added=sorted(set(entry['aliases']) - set(self.previous[key]['aliases'])),
                              removed=sorted(set(self.previous[key]['aliases']) - set(entry['aliases'])))
                         for key, entry in storages.items() if key in self.previous
                         and set(entry['aliases']) != set(self.previous[key]['aliases'])]
        self.previous = storages  # Metadata only: no tensor/storage/frame references.
        return dict(variables=variables, storages=list(storages.values()),
                    unique_storage_bytes=sum(entry["bytes"] for entry in storages.values()),
                    logical_bytes_with_aliases=sum(row["logical_bytes"] for row in variables),
                    appeared_storages=appeared, disappeared_storages=disappeared,
                    resized_storages=resized, alias_changes=alias_changes, inspection_errors=errors)


class LayerMemoryTrace:
    def __init__(self, torch, device, model, directory, variable_details=True, trace_roots=None,
                 algorithm=False, rules=None):
        self.torch, self.device = torch, device
        self.path = Path(directory)
        self.stream = (self.path / 'memory-events.jsonl').open('w')
        self.model_ref = weakref.ref(model)
        self.inventory = TensorInventory(torch, device) if variable_details else None
        self.variable_stream = ((self.path / 'memory-variables.jsonl').open('w')
                                if variable_details else None)
        self.variable_peak = None
        self.stage_resolver = StageResolver()
        self.stage_summary = StageMemorySummary()
        self.algorithm_summary = None
        self.saved_memory = None
        self.optimizer_refs = {}
        self.baseline_refs = {}
        if algorithm:
            if not variable_details:
                raise ValueError('Algorithm attribution requires variable inventory.')
            from memory_algorithm_summary import AlgorithmMemorySummary, load_rules
            from .saved_memory import SavedMemory
            self.algorithm_summary = AlgorithmMemorySummary(load_rules(rules))
            self.saved_memory = SavedMemory(torch, self.saved_owner)
            self.inventory.on_tensor = self.observe_baseline
        self.scopes = []
        self.closed = False
        self.trace_roots = tuple(Path(p).resolve() for p in (trace_roots or [Path(__file__).resolve().parents[1]]))
        self.excluded_files = {str(Path(__file__).resolve()),
                               str(Path(__file__).with_name('memory_analysis.py').resolve()),
                               str(Path(__file__).resolve().parents[1] / 'memory_stage_summary.py'),
                               str(Path(__file__).resolve().parents[1] / 'memory_algorithm_summary.py'),
                               str(Path(__file__).with_name('saved_memory.py').resolve())}
        self.model_files = set()
        self.layers = {}
        for name, module in model.named_modules():
            self.layers[id(module)] = name or 'model'
            if self.saved_memory:
                for value in vars(module).values():
                    if isinstance(value, torch.optim.Optimizer):
                        self.optimizer_refs[id(value)] = weakref.ref(value)
            try:
                self.model_files.add(str(Path(inspect.getfile(type(module))).resolve()))
            except (TypeError, OSError):
                pass
        self.stage_resolver.select_file = self.select_file
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
        for name, param in model.named_parameters():
            if param.requires_grad:
                owner = name.rsplit('.', 1)[0] if '.' in name else 'model'
                self.handles.append(param.register_hook(self.gradient_hook(owner, name)))

    def observe_baseline(self, tensor, row):
        import fnmatch
        path = row['path']
        baseline = path.startswith('baseline_input.') or any(
            r['category'] == 'model_baseline' and fnmatch.fnmatchcase(path, r['pattern'])
            for r in self.algorithm_summary.rules)
        if baseline:
            row['baseline_storage'] = True
            self.baseline_refs[id(tensor)] = weakref.ref(tensor)

    def saved_owner(self):
        from memory_algorithm_summary import layer_name
        frame = sys._getframe().f_back
        try:
            while frame is not None:
                name = self.layers.get(id(frame.f_locals.get('self')))
                if name and layer_name(name) != 'shared':
                    return layer_name(name)
                frame = frame.f_back
            return 'shared'
        finally:
            del frame

    def select_file(self, filename):
        path = Path(filename).resolve()
        if str(path) in self.excluded_files:
            return False
        return str(path) in self.model_files or any(path == root or root in path.parents for root in self.trace_roots)

    @contextmanager
    def stage(self, name, layer=None, epoch=None, period=None):
        """Optional explicit labels for any algorithm; no tensor lifetime changes."""
        if period is None:
            period = ('evaluation' if name == 'evaluation' else
                      'training' if name in ('forward', 'loss_build', 'backward', 'optimizer_step', 'zero_grad') else 'other')
        if period not in ('training', 'evaluation', 'other', 'unclassified'):
            raise ValueError('Unknown memory period: '+str(period))
        context = dict(phase=name, layer=layer, epoch_zero_based=epoch, period=period)
        self.scopes.append(context)
        self.boundary({'event': 'stage_enter', 'phase': name}, frame=sys._getframe(2))
        try:
            yield
        finally:
            self.boundary({'event': 'stage_exit', 'phase': name}, frame=sys._getframe(2))
            self.scopes.pop()
            outer = (self.scopes[-1] if self.scopes else
                     dict(phase='other', layer=None, epoch_zero_based=None, period='other'))
            self.previous = dict(self.previous, stage_context=dict(outer))

    def gradient_hook(self, layer, name):
        def hook(gradient):
            # Returning None preserves the incoming gradient. Never retain it.
            if self.active:
                # Native autograd workers may enter Python with no caller frame.
                # The current hook frame always exists; its f_back may be None.
                parent_frame = sys._getframe().f_back
                self.boundary({'event': 'parameter_gradient_ready', 'layer': layer,
                               'parameter': name, 'phase': 'backward',
                               'python_parent_frame_available': parent_frame is not None},
                              frame=parent_frame,
                              extra_roots=[('incoming_gradient', gradient)])
        return hook

    def boundary(self, location, frame=None, extra_roots=None):
        location = dict(location)
        location['stage_context'] = (dict(self.scopes[-1]) if self.scopes else
                                     self.stage_resolver.resolve(frame, location, self.layers))
        stage_context = (location if location.get('event') == 'parameter_gradient_ready'
                         else self.previous or location)['stage_context']
        t, d = self.torch, self.device
        t.cuda.synchronize(d)
        allocated = t.cuda.memory_allocated(d)
        reserved = t.cuda.memory_reserved(d)
        peak_a = t.cuda.max_memory_allocated(d)
        peak_r = t.cuda.max_memory_reserved(d)
        row = dict(sequence=self.count, seconds=time.perf_counter() - self.started,
                   end=location, interval_after=self.previous, stage_context=stage_context,
                   allocated_bytes=allocated, reserved_bytes=reserved,
                   interval_peak_allocated_bytes=peak_a, interval_peak_reserved_bytes=peak_r)
        if self.inventory is not None:
            roots = [('model', self.model_ref())]
            if self.saved_memory:
                from memory_algorithm_summary import layer_name
                self.inventory.tensor_owners = {}
                for name, parameter in self.model_ref().named_parameters():
                    owner = layer_name(name)
                    self.inventory.tensor_owners[id(parameter)] = owner
                    if parameter.grad is not None:
                        self.inventory.tensor_owners[id(parameter.grad)] = owner
                # BP optimizer lives outside model; register it from frame locals
                # below and keep only weak references between boundaries.
                for key, ref in list(self.baseline_refs.items()):
                    tensor = ref()
                    if tensor is None:
                        del self.baseline_refs[key]
                    else:
                        roots.append(('baseline_input.{}'.format(key), tensor))
                tensor = None
            ancestor, depth = frame, 0
            while ancestor is not None:
                filename = ancestor.f_code.co_filename.replace('\\', '/')
                if self.select_file(filename):
                    for name, value in ancestor.f_locals.items():
                        roots.append(('frame{}.{}.{}'.format(depth, ancestor.f_code.co_name, name), value))
                        if self.saved_memory and isinstance(value, self.torch.optim.Optimizer):
                            self.optimizer_refs[id(value)] = weakref.ref(value)
                ancestor, depth = ancestor.f_back, depth + 1
            if self.saved_memory:
                roots.extend(self.saved_memory.roots())
                for key, ref in list(self.optimizer_refs.items()):
                    optimizer = ref()
                    if optimizer is not None:
                        roots.append(('tracked_optimizer{}'.format(key), optimizer))
                        for parameter, state in optimizer.state.items():
                            owner = self.inventory.tensor_owners.get(id(parameter), 'shared')
                            def assign(value):
                                if self.torch.is_tensor(value):
                                    self.inventory.tensor_owners[id(value)] = owner
                                elif isinstance(value, dict):
                                    for child in value.values(): assign(child)
                                elif isinstance(value, (list, tuple)):
                                    for child in value: assign(child)
                            assign(state)
                    else:
                        del self.optimizer_refs[key]
                optimizer = None  # only weak references survive the snapshot
            roots.extend(extra_roots or [])
            if self.saved_memory and location.get('event') == 'parameter_gradient_ready':
                for name, tensor in extra_roots or []:
                    self.inventory.tensor_owners[id(tensor)] = layer_name(str(location.get('layer', '')))
            details = self.inventory.capture(roots)
            del roots, ancestor
            details.update(sequence=self.count, end=location, stage_context=stage_context)
            self.stage_summary.observe(row, details, stage_context)
            if self.algorithm_summary:
                snapshot = self.algorithm_summary.observe(row, details, stage_context)
                row['algorithm_observed_bytes'] = snapshot['algorithm_bytes']
            details['allocator_minus_visible_storage_bytes'] = allocated - details['unique_storage_bytes']
            self.variable_stream.write(json.dumps(details) + '\n')
            self.variable_stream.flush()
            row['visible_unique_storage_bytes'] = details['unique_storage_bytes']
            row['allocator_minus_visible_storage_bytes'] = details['allocator_minus_visible_storage_bytes']
            row['appeared_visible_storage_bytes'] = sum(x['bytes'] for x in details['appeared_storages'])
            row['disappeared_visible_storage_bytes'] = sum(x['bytes'] for x in details['disappeared_storages'])
            if self.variable_peak is None or allocated > self.variable_peak['allocated_bytes']:
                self.variable_peak = dict(sequence=self.count, allocated_bytes=allocated,
                                          storages=sorted(details['storages'], key=lambda x: x['bytes'], reverse=True)[:20])
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
        selected = self.select_file(filename)
        if not selected:
            return None
        if event in ('call', 'line', 'return', 'exception'):
            location = dict(event=event, file=filename, function=function, line=frame.f_lineno)
            location['source'] = linecache.getline(filename, frame.f_lineno).strip()
            owner = frame.f_locals.get('self')
            location['class'] = type(owner).__name__
            if id(owner) in self.layers:
                location['layer'] = self.layers[id(owner)]
            self.boundary(location, frame=frame,
                          extra_roots=[('return_value', arg)] if event == 'return' else None)
        return self.trace

    def start(self):
        self.old_trace = sys.gettrace()
        if self.old_trace is not None:
            raise RuntimeError('Run memory tracing without an existing Python debugger/trace.')
        self.active = True
        self.boundary({'event': 'start', 'phase': 'before_gpu_transfer'}, frame=sys._getframe(1))
        if self.saved_memory:
            self.saved_memory.start()
        sys.settrace(self.trace)

    def end_window(self, name):
        self.boundary({'event': 'window_end', 'phase': name}, frame=sys._getframe(1))
        result = dict(self.window)
        self.window = dict.fromkeys(self.window, 0)
        return result

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.active:
            sys.settrace(self.old_trace)
            self.active = False
        if self.saved_memory:
            self.saved_memory.close()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.stream.close()
        if self.variable_stream is not None:
            self.variable_stream.close()
            self.stage_summary.write(self.path)
            if self.algorithm_summary:
                self.algorithm_summary.write(self.path)
                target = self.path / 'algorithm-memory-summary.json'
                data = json.loads(target.read_text())
                data['complete'] = bool(self.previous and self.previous.get('phase') == 'analysis_end')
                target.write_text(json.dumps(data, indent=2) + '\n')
        summary = dict(events=self.count, whole_peak=self.maximum,
                       largest_observed_boundary_storages=self.variable_peak,
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
        if self.variable_peak is not None:
            lines += ['', '## Visible storages at highest observed boundary allocation', '',
                      'Event sequence: {}'.format(self.variable_peak['sequence']), '',
                      '| MiB | Variable aliases (first 3) |', '|---:|---|']
            for entry in self.variable_peak['storages']:
                aliases = '; '.join(entry['aliases'][:3]).replace('|', '\\|')
                lines.append('| {:.3f} | {} |'.format(entry['bytes'] / 2**20, aliases))
            lines += ['', 'Full variable metadata and storage transitions: memory-variables.jsonl.',
                      'Disappeared means no longer visible to this inventory, NOT proven allocator free.',
                      'Transient interval peaks can exceed boundary observations; native/autograd tensors may be invisible.',
                      'Storage identities may be reused between observations; transitions are not allocation histories.']
        (self.path / 'memory-trace-report.md').write_text('\n'.join(lines) + '\n')
