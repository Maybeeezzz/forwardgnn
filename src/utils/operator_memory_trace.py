"""Opt-in operator profiling for one named module invocation or a whole context.

Uses the PyTorch 1.13 profiler API. No algorithm-specific imports or mutations.
"""
import json
import time
from pathlib import Path


def tensor_metadata(torch, value, path='input'):
    """Metadata only: never retain Tensor/storage references in the result."""
    rows = []
    if torch.is_tensor(value):
        row = dict(path=path, shape=list(value.shape), dtype=str(value.dtype), device=str(value.device),
                   logical_bytes=value.numel() * value.element_size())
        if value.layout == torch.strided:
            storage = value.untyped_storage() if hasattr(value, 'untyped_storage') else value.storage()
            row.update(storage_address=storage.data_ptr(), storage_bytes=storage.nbytes(),
                       storage_offset=value.storage_offset(), stride=list(value.stride()))
        rows.append(row)
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            rows.extend(tensor_metadata(torch, child, '{}[{}]'.format(path, index)))
    elif isinstance(value, dict):
        for index, (key, child) in enumerate(value.items()):
            label = key if isinstance(key, (str, int)) else 'key#{}'.format(index)
            rows.extend(tensor_metadata(torch, child, '{}[{}]'.format(path, label)))
    return rows


def tensor_mode(torch, stream):
    # This optional API is imported only when explicitly requested.
    from torch.utils._python_dispatch import TorchDispatchMode

    class MetadataMode(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.sequence = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            inputs = tensor_metadata(torch, args) + tensor_metadata(torch, kwargs or {}, 'kwargs')
            start = time.perf_counter_ns()
            output = func(*args, **(kwargs or {}))
            row = dict(sequence=self.sequence, operator=str(func), monotonic_start_ns=start,
                       monotonic_end_ns=time.perf_counter_ns(), inputs=inputs,
                       outputs=tensor_metadata(torch, output, 'output'))
            stream.write(json.dumps(row) + '\n')
            stream.flush()
            self.sequence += 1
            return output
    return MetadataMode()


class OperatorMemoryTrace:
    def __init__(self, torch, device, model, directory, target_module=None, target_call=1,
                 record_shapes=False, tensor_ops=False):
        if target_call < 1:
            raise ValueError('Operator target call must be >= 1.')
        self.torch, self.device = torch, device
        self.directory = Path(directory)
        self.target_module, self.target_call = target_module, target_call
        self.record_shapes, self.tensor_ops = record_shapes, tensor_ops
        self.handles, self.profiler, self.scope, self.mode, self.tensor_stream = [], None, None, None, None
        self.calls, self.depth = 0, 0
        self.active, self.finished, self.closed = False, False, False
        self.metadata = dict(target_module=target_module, target_call=target_call,
                             record_shapes=record_shapes, tensor_ops=tensor_ops,
                             torch_version=torch.__version__, cuda_version=torch.version.cuda,
                             device_index=device.index, gpu=torch.cuda.get_device_name(device),
                             deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                             status='waiting')
        self.target = None
        if target_module is not None:
            modules = dict(model.named_modules())
            if target_module not in modules:
                raise ValueError('Unknown memory-analysis module {!r}. Available: {}'.format(
                    target_module, ', '.join(modules)))
            self.target = modules[target_module]

    def save_metadata(self):
        (self.directory / 'operator-metadata.json').write_text(json.dumps(self.metadata, indent=2) + '\n')

    def start(self):
        self.save_metadata()
        if self.target is None:
            self.begin()
        else:
            self.handles.append(self.target.register_forward_pre_hook(self.before))
            self.handles.append(self.target.register_forward_hook(self.after))

    def before(self, module, inputs):
        if self.finished:
            return
        if self.active:
            self.depth += 1
            return
        self.calls += 1
        if self.calls == self.target_call:
            self.depth = 1
            self.metadata.update(inputs=tensor_metadata(self.torch, inputs),
                                 module_training=module.training,
                                 grad_enabled=self.torch.is_grad_enabled(), observed_call=self.calls)
            self.begin()

    def after(self, module, inputs, output):
        if not self.active:
            return
        self.depth -= 1
        if self.depth == 0:
            self.metadata['outputs'] = tensor_metadata(self.torch, output, 'output')
            self.finish()

    def begin(self):
        torch, device = self.torch, self.device
        torch.cuda.synchronize(device)
        self.metadata['start_allocated_bytes'] = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        self.profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            profile_memory=True, record_shapes=self.record_shapes, with_stack=True)
        self.profiler.start()
        self.active = True
        self.metadata['status'] = 'recording'
        self.scope = torch.profiler.record_function('memory_target::' + (self.target_module or 'context'))
        self.scope.__enter__()
        if self.tensor_ops:
            self.tensor_stream = (self.directory / 'operator-tensors.jsonl').open('w')
            self.mode = tensor_mode(torch, self.tensor_stream)
            self.mode.__enter__()
        self.save_metadata()

    def finish(self, error=None):
        if not self.active:
            return
        self.active = False
        failure = None
        try:
            self.torch.cuda.synchronize(self.device)
            self.metadata.update(end_allocated_bytes=self.torch.cuda.memory_allocated(self.device),
                                 peak_allocated_bytes=self.torch.cuda.max_memory_allocated(self.device),
                                 peak_reserved_bytes=self.torch.cuda.max_memory_reserved(self.device))
        except Exception as exc:
            failure = exc
        # Attempt every cleanup even if device synchronization or a context exit fails.
        for attribute in ('mode', 'scope'):
            context = getattr(self, attribute)
            if context is not None:
                try:
                    context.__exit__(None, None, None)
                except Exception as exc:
                    failure = failure or exc
                finally:
                    setattr(self, attribute, None)
        try:
            self.profiler.stop()
        except Exception as exc:
            failure = failure or exc
        finally:
            if self.tensor_stream is not None:
                self.tensor_stream.close()
                self.tensor_stream = None
        reason = error or failure
        self.metadata.update(status='partial' if reason else 'complete', error=str(reason) if reason else None)
        self.finished = True
        self.save_metadata()
        try:
            self.export()
        except Exception as exc:
            failure = failure or exc
            self.metadata.update(status='partial', export_error=str(exc))
            self.save_metadata()
        if failure is not None:
            raise failure

    def export(self):
        self.profiler.export_chrome_trace(str(self.directory / 'operator-trace.json'))
        rows = []
        for event in self.profiler.events():
            rows.append(dict(name=event.name, thread=event.thread,
                             start_us=event.time_range.start, end_us=event.time_range.end,
                             input_shapes=event.input_shapes, stack=event.stack,
                             self_cuda_memory_bytes=getattr(event, 'self_cuda_memory_usage',
                                                            getattr(event, 'self_device_memory_usage', None)),
                             inclusive_cuda_memory_bytes=getattr(event, 'cuda_memory_usage',
                                                                 getattr(event, 'device_memory_usage', None))))
        (self.directory / 'operator-events.json').write_text(json.dumps(rows, indent=2) + '\n')
        from operator_memory_report import generate_report
        generate_report(self.directory)

    def close(self, error=None):
        if self.closed:
            return
        self.closed = True
        try:
            if self.active:
                self.finish(error=error)
            elif not self.finished:
                self.metadata.update(status='not_captured', observed_calls=self.calls,
                                     error=str(error) if error else 'Requested module invocation was not reached.')
                self.save_metadata()
                if error is None:
                    raise RuntimeError(self.metadata['error'])
        finally:
            for handle in self.handles:
                handle.remove()
            self.handles.clear()
