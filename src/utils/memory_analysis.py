"""Opt-in CUDA memory diagnostics, independent of a model or learning algorithm.

with MemoryAnalysis(enabled=True, model=model, device=device, directory=path) as memory:
    with memory.stage('forward', layer='encoder', epoch=0):
        output = model(inputs)

Disabled analysis imports no torch, installs no hooks, and creates no files.
"""
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


def add_memory_arguments(parser):
    group = parser.add_argument_group('optional memory analysis')
    group.add_argument('--memory-analysis', action='store_true',
                       help='Enable CUDA boundary, variable/storage and stage diagnostics (default: off).')
    group.add_argument('--memory-analysis-dir', type=str, default=None,
                       help='Analysis output root; each run gets a new subdirectory.')
    group.add_argument('--memory-analysis-no-variables', action='store_true',
                       help='Record allocator boundaries only; omit variable inventory and stage tables.')
    group.add_argument('--memory-analysis-operators', action='store_true',
                       help='Use operator profiling instead of Python boundary tracing; requires --memory-analysis.')
    group.add_argument('--memory-analysis-module', default=None,
                       help='Exact named_modules() path; capture only this module forward invocation.')
    group.add_argument('--memory-analysis-call', type=int, default=1,
                       help='1-based invocation of the target module to capture (default: 1).')
    group.add_argument('--memory-analysis-record-shapes', action='store_true',
                       help='Profiler input shapes; can retain tensors and alter memory, use a separate diagnostic run.')
    group.add_argument('--memory-analysis-tensor-ops', action='store_true',
                       help='Optional dispatcher input/output Tensor metadata; use a separate diagnostic run.')
    return parser


class MemoryAnalysis:
    def __init__(self, enabled=False, model=None, device=None, directory=None,
                 variable_details=True, trace_roots=None, operators=False,
                 target_module=None, target_call=1, record_shapes=False, tensor_ops=False):
        self.enabled = enabled
        self.model, self.device = model, device
        self.directory = Path(directory) if directory is not None else None
        self.variable_details, self.trace_roots = variable_details, trace_roots
        self.tracer = None
        self.operator_tracer = None
        self.operators = operators
        self.operator_options = dict(target_module=target_module, target_call=target_call,
                                     record_shapes=record_shapes, tensor_ops=tensor_ops)

    def __enter__(self):
        if not self.enabled:
            return self
        import torch
        device = torch.device(self.device)
        if device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('--memory-analysis requires CUDA; no CPU/MPS memory substitution.')
        if device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        if self.model is None or self.directory is None:
            raise ValueError('Enabled memory analysis requires model and directory.')
        if self.directory.exists() and any(self.directory.iterdir()):
            raise ValueError('Memory analysis output directory must be new or empty.')
        self.directory.mkdir(parents=True, exist_ok=True)
        try:
            if self.operators:
                from .operator_memory_trace import OperatorMemoryTrace
                self.operator_tracer = OperatorMemoryTrace(torch, device, self.model, self.directory,
                                                           **self.operator_options)
                self.operator_tracer.start()
            else:
                from .layer_memory_trace import LayerMemoryTrace
                self.tracer = LayerMemoryTrace(torch, device, self.model, self.directory,
                                               variable_details=self.variable_details, trace_roots=self.trace_roots)
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                self.tracer.start()
        except BaseException as error:
            if self.tracer is not None:
                self.tracer.close()
            if self.operator_tracer is not None:
                self.operator_tracer.close(error=error)
            raise
        print('Memory analysis: {}'.format(self.directory))
        return self

    @contextmanager
    def stage(self, name, layer=None, epoch=None):
        if self.tracer is None:
            yield
        else:
            with self.tracer.stage(name, layer=layer, epoch=epoch):
                yield

    def __exit__(self, exc_type, exc, traceback):
        if self.operator_tracer is not None:
            try:
                self.operator_tracer.close(error=exc)
            except Exception:
                if exc_type is None:
                    raise
                import warnings
                warnings.warn('Operator profile export failed while handling a training exception; inspect partial files.')
        if self.tracer is not None:
            try:
                if exc_type is None:
                    self.tracer.end_window('analysis_end')
            finally:
                self.tracer.close()
        return False


def memory_analysis(args, model, run_i=0):
    """CLI adapter. Constructed around a run, not inside model algorithms."""
    operators = getattr(args, 'memory_analysis_operators', False)
    if not getattr(args, 'memory_analysis', False):
        if operators:
            raise ValueError('--memory-analysis-operators requires --memory-analysis.')
        return MemoryAnalysis()
    if not operators and (getattr(args, 'memory_analysis_module', None) is not None
                          or getattr(args, 'memory_analysis_record_shapes', False)
                          or getattr(args, 'memory_analysis_tensor_ops', False)):
        raise ValueError('Operator-specific options require --memory-analysis-operators.')
    root = getattr(args, 'memory_analysis_dir', None)
    root = Path(root) if root else Path(args.results_dir) / 'memory-analysis'
    directory = root / 'run{}-{}'.format(run_i, datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    return MemoryAnalysis(enabled=True, model=model, device=args.device, directory=directory,
                          variable_details=not getattr(args, 'memory_analysis_no_variables', False),
                          operators=operators, target_module=getattr(args, 'memory_analysis_module', None),
                          target_call=getattr(args, 'memory_analysis_call', 1),
                          record_shapes=getattr(args, 'memory_analysis_record_shapes', False),
                          tensor_ops=getattr(args, 'memory_analysis_tensor_ops', False))
