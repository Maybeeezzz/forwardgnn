# 可选的通用显存分析 utility

显存分析由 `src/utils/memory_analysis.py` 提供，**普通训练默认关闭**，使用
`--memory-analysis` 启用。不需要修改模型算法，不限定 SF、BP、节点分类或链接预测，
不要求模型具有 `.layers` 或 `.convs` 属性。模型通过标准 `named_modules()` 和
`named_parameters()` 接口识别。低层采集仍在 `LayerMemoryTrace`，阶段汇总在
`src/memory_stage_summary.py`。

## 命令行开关

`train_forward.py`、`train_backprop.py`、`train_backprop_sparse.py` 支持相同参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--memory-analysis` | 关闭 | 开启 CUDA 逐边界、变量/storage 和阶段分析 |
| `--memory-analysis-dir PATH` | 当前实验结果目录下 `memory-analysis/` | 分析输出根目录，不单独启用分析 |
| `--memory-analysis-no-variables` | 关闭 | 仅保留 allocator 边界测量，不输出变量清单和阶段分类表 |

不开启时 context 是 no-op，不实例化采集器、不注册 hooks、不修改 Python trace、不创建
分析文件、不调用 CUDA 显存统计。开启后需要 CUDA，CPU/MPS 不提供替代显存结果。
每次实际执行的 run 建立 `runN-时间戳/` 独立目录，终端打印路径。
原训练入口的“已有结果则跳过”规则不变；分析实验建议使用新的 `--exp-setting`，不要
因为启用了分析就默默重跑已有实验。测量覆盖 context 内的设备传输、训练和最终测试；
CPU 数据加载与模型构造不包含在窗口内。

从仓库根目录运行 SF 示例（SF 只是调用者，不是 utility 的实现前提）：

```bash
python src/train_forward.py \
  --task node-class --model GNN_SingleForward-SAGE \
  --dataset Amazon-Photo --num-layers 2 --num-hidden 128 \
  --num-runs 1 --epochs 3 --patience -1 --gpu 0 \
  --exp-setting memory-debug-sf \
  --memory-analysis --memory-analysis-dir results/memory-debug-sf
```

BP 使用同一个开关：

```bash
python src/train_backprop.py \
  --task node-class --model GNN-SAGE \
  --dataset Amazon-Photo --num-layers 2 --num-hidden 128 \
  --num-runs 1 --epochs 3 --patience -1 --gpu 0 \
  --exp-setting memory-debug-bp \
  --memory-analysis --memory-analysis-dir results/memory-debug-bp
```

去掉 `--memory-analysis` 即恢复普通训练。原训练器的超参数、数据划分和训练行为不变。
Sparse BP 开启分析时，其峰值字段使用采集器累计高水位，避免逐边界 reset 使原结果变成
最后一个区间的峰值；结果中明确记录 instrumentation，不能和无插桩耗时直接比较。

## 作为 Python utility 使用

在 `src` 可导入的环境中：

```python
from utils.memory_analysis import MemoryAnalysis

with MemoryAnalysis(
    enabled=args.memory_analysis,
    model=model,
    device=device,
    directory="results/custom-memory",
    trace_roots=["/absolute/path/to/my/training/code"],
) as memory:
    model = model.to(device)
    for epoch in range(epochs):
        with memory.stage("forward", layer="encoder", epoch=epoch):
            output = model(inputs)
        with memory.stage("loss_build", layer="encoder", epoch=epoch):
            loss = criterion(output, targets)
        with memory.stage("zero_grad", epoch=epoch):
            optimizer.zero_grad()
        with memory.stage("backward", epoch=epoch):
            loss.backward()
        with memory.stage("optimizer_step", epoch=epoch):
            optimizer.step()
```

`stage()` 可选，支持任意阶段名称及嵌套；关闭分析时同样是 no-op。显式 scope 优先于
自动识别，不必为新算法改 utility。手动 API 输出目录必须新建或为空，避免覆盖。
退出 context 时移除 hooks、恢复 trace 并写表；训练异常也会关闭采集器、保存已有观测，
不会吞掉训练异常。不支持与已启用的 Python debugger/trace 同时运行。

默认追踪仓库 `src` 下代码与模型模块的源码文件；外部训练脚本使用 `trace_roots` 指定
自己的目录。自动阶段识别根据通用的 forward、损失模块、backward、optimizer.step、
zero_grad 调用和 evaluate/test 等函数约定推断，不引用 SF 类名、算法文件名或固定行号。
跨行调用使用 AST 语句范围。无法可靠推断的语句归为 `other`；自定义层间传递、图准备等
业务语义请使用 `stage()` 明确标记。模块列使用标准模块路径（例如 `layers.0.gnn_layer`、
`convs.0.lin`），没有模块归属的全局步骤显示 `—`，不是虚构的数字层号。

## 输出与阶段显存表

| 文件 | 内容 |
|---|---|
| `memory-events.jsonl` | 位置、模块、epoch、当前 allocated/reserved、区间峰值、净变化和回落 |
| `memory-variables.jsonl` | 同一 sequence 的变量清单、storage 去重、别名变化和存储可见性变化 |
| `memory-stage-summary.md` | 模块/层 × 阶段 × 类别的 MiB 表 |
| `memory-stage-summary.csv` | 同样的宽表，用于 Excel 分析 |
| `memory-stage-summary.json` | 阶段选中快照的分类 storage 及全部变量别名 |
| `memory-trace-report.md` | 最高峰区间、最大回落及大 storage 摘要 |
| `memory-trace-summary.json` | 完整摘要与事件索引 |

阶段表列包括模块/层、阶段、epoch、sequence、allocated，以及参数、梯度、优化器状态、
早停副本、缓存、图数据、logits/loss、激活、其他可见、未归因差额与区间峰值。

每行取该模块、阶段跨 epoch 的 **allocated 最高已观测边界**，所有类别来自同一快照，
按 storage 去重。同 allocated 时取最早快照。分类总和加未归因差额等于该行 allocated。
这是该时刻整个进程的显存构成，包含其他模块常驻状态；不是当前模块独占显存，也不是
该阶段新增量。区间内部瞬时峰值可能更高，不能用边界清单完整解释它。

变量记录路径、shape、dtype、device、逻辑字节数、storage 字节数/身份/偏移、stride、
requires_grad、grad_fn、是否 Parameter。遍历模型、选定调用栈局部变量、容器及对象已有
属性，不调用属性 getter 或进行 tensor 计算，不把张量、storage、frame 保留到历史日志。
扫描范围包含参数梯度、优化器状态、数据、激活、缓存、checkpoint，以及 hook 的
`incoming_gradient` 和返回值。不会扫描全局 GC 或遍历 autograd 图。

同一 storage 的别名可能具有不同语义，唯一分类优先级为：梯度 → 参数 → 优化器状态 →
早停副本 → 缓存 → 图数据 → logits/loss → 激活 → 其他可见。参数使用 Parameter 标记，
其他类别基于属性路径和常见变量名推断；分类不是 allocator 的原生标签。无法辨认的对象
保留“其他可见”。输入与图数据共享存储时归图数据，参数的局部别名仍归参数。

- `unique_storage_bytes`：可见 storage 去重总量。
- `logical_bytes_with_aliases`：含别名的逻辑大小之和，不能当成真实显存。
- `appeared_storages` / `disappeared_storages`：相邻边界可见 storage 集合变化。
- `alias_changes` / `resized_storages`：别名和存储容量变化。
- `allocator_minus_visible_storage_bytes`：allocated 与可见存储的有符号差额，不自动归因为激活或泄漏。
- `inspection_errors`：非 strided 张量不支持、遍历深度限制或读取失败。

**不可见不等于已释放。** autograd 或原生算子可能仍持有 storage；需要结合 allocated
净下降和峰值回落判断。它们也不是累计 free 字节数。storage 身份/地址可能复用，因此
快照差分不能重建完整 allocation/free 历史。allocated 下降而 reserved 不变通常说明
分配器保留空间供复用。临时张量和原生对象需要结合算子 profiler 进一步归因。

Python line 事件在执行前发生，变化归属 `interval_after` 所指的执行区间；return 事件
也早于局部引用销毁。日志不执行 gc.collect/empty_cache。同步和逐边界清单会显著改变
耗时及异步时序，建议先跑 1 run、2 层、3 epoch，并用无插桩运行交叉验证峰值。

## 查看或重建表（无需 PyTorch）

用终端输出的实际 run 子目录替换下方路径：

```bash
python src/inspect_memory_event.py results/memory-debug-sf/run0-TIMESTAMP --top 20
python src/inspect_memory_event.py results/memory-debug-sf/run0-TIMESTAMP --sequence 100
python src/memory_stage_summary.py results/memory-debug-sf/run0-TIMESTAMP
```

阶段字段出现之前的旧日志需要重跑，不能可靠补全。

## 保留的批量实验入口

`profile_layer_memory.py` 是显式调用分析器的历史 SF/BP 配对实验驱动，不是 utility 的
使用前提。它仍支持原论文 1–4 层矩阵、`--methods sf`、`--no-variable-details`、
`--dry-run` 和作者划分检查；层字段现在采用通用模块路径。普通训练使用上面的可选开关。

## 验证

```bash
python -m unittest discover -s tests -p 'test_memory*.py' -v
```

无依赖测试覆盖默认关闭、普通入口参数、没有 layers/convs 的模型、异常清理、自定义
scope、storage 别名去重、优化器/梯度、同快照阶段表和离线查看器。原生 CUDA 测试在没有
PyTorch/CUDA 时明确 skip。本机未执行真实 GPU 训练跟踪。

## 算子级分析：定位聚合内部的临时分配

新增独立的 `--memory-analysis-operators` 模式。在 `--memory-analysis` 开启的前提下，
它以 PyTorch profiler **替代**逐行 Python trace，不同时运行变量清单采集器。训练算法
保持不变，只在指定模块的指定一次 forward 前后开启/关闭 profiler。模型路径由调用者
传入，utility 不绑定 SAGE 或 SF。

| 参数 | 默认 | 作用 |
|---|---|---|
| `--memory-analysis-operators` | 关闭 | 算子/分配事件模式 |
| `--memory-analysis-module PATH` | 不指定 | 精确匹配 named_modules 路径；不指定时记录整个 context |
| `--memory-analysis-call N` | 1 | 目标模块的第 N 次 forward，含训练、评估、层间重算 |
| `--memory-analysis-record-shapes` | 关闭 | 记录 profiler 操作输入形状，可能影响张量生命周期 |
| `--memory-analysis-tensor-ops` | 关闭 | 额外记录 Python dispatcher 可见的输入/输出 Tensor 元数据 |

支持 `train_forward.py` 和 `train_backprop.py`，也支持直接 `MemoryAnalysis(operators=True,
...)`。专门的 `train_backprop_sparse.py` 拥有独立峰值窗口，当前拒绝算子模式以避免相互
reset 污染结果；稀疏模型可通过 Python utility 自行包裹。普通逐行模式仍可用于该入口。

采用 PyTorch 1.13 已有的 profile_memory、with_stack、export_chrome_trace API，
不依赖新版本的 export_memory_timeline 或 allocator history 私有接口。可选 Tensor
元数据使用 `torch.utils._python_dispatch.TorchDispatchMode`，与基础 profiler 分开开关。

### CUDA 服务器命令（无需环境变量）

在更新后的仓库根目录，激活 `ForwardLearningGNN` 环境。先记录分配事件，不记录形状：

```bash
python src/train_forward.py \
  --task node-class --model GNN_SingleForward-SAGE --dataset Amazon-Photo \
  --num-layers 2 --num-hidden 128 --num-runs 1 --seed 100 \
  --epochs 3 --lr 0.001 --patience -1 --val-every 1 \
  --append-label none --aug-edge-direction bidirection --temperature 1.0 --gpu 0 \
  --exp-setting memory-op-sf-alloc --overwrite-result \
  --memory-analysis --memory-analysis-operators \
  --memory-analysis-module layers.0.gnn_layer.gnn.aggr_module \
  --memory-analysis-call 6 \
  --memory-analysis-dir results/memory-op-sf-alloc
```

对于此前“两层、每层三 epoch、关闭早停”的 SF 流程，第一层聚合依次执行：3 次训练、
第一层结束测试、第一层层间重算、第二层结束测试、最终测试。第 6 次调用对应原事件
1143 的评估聚合位置。改变训练预算/验证规则后次数可能不同；要定位第一次训练的聚合，
用 `--memory-analysis-call 1`。不存在模块或目标调用未发生会明确报错，不生成零值结论。

再独立运行形状与 Tensor 元数据诊断：

```bash
python src/train_forward.py \
  --task node-class --model GNN_SingleForward-SAGE --dataset Amazon-Photo \
  --num-layers 2 --num-hidden 128 --num-runs 1 --seed 100 \
  --epochs 3 --lr 0.001 --patience -1 --val-every 1 \
  --append-label none --aug-edge-direction bidirection --temperature 1.0 --gpu 0 \
  --exp-setting memory-op-sf-tensors --overwrite-result \
  --memory-analysis --memory-analysis-operators \
  --memory-analysis-module layers.0.gnn_layer.gnn.aggr_module \
  --memory-analysis-call 6 \
  --memory-analysis-record-shapes --memory-analysis-tensor-ops \
  --memory-analysis-dir results/memory-op-sf-tensors
```

第二层目标路径为 `layers.1.gnn_layer.gnn.aggr_module`。BP 的 SAGE 聚合通常为
`convs.0.aggr_module`，可在普通 BP 命令中使用同样的参数。参数错误会列出可用模块。

### 输出文件与读法

每次 run 的时间戳子目录内保存：

| 文件 | 内容 |
|---|---|
| `operator-metadata.json` | 目标调用、软件/GPU、确定性设置、是否训练/grad_enabled、输入/输出元数据、窗口峰值 |
| `operator-trace.json` | 原始 Chrome/Kineto trace，可在 Perfetto / Chrome trace viewer 中打开 |
| `operator-events.json` | 操作名称、输入形状、调用栈、self/inclusive 净内存变化 |
| `operator-memory-events.csv` | CUDA 分配/释放时间线、地址、字节数、分配器总量、所属 CPU 操作 |
| `operator-allocations.csv` | 按大小排列的单次分配、独立 allocation ID 和匹配的释放时间 |
| `operator-allocation-totals.csv` | 每操作累计分配、最大单次分配、trace 峰值时本窗口新分配且存活的字节 |
| `operator-allocation-summary.json` | 上述结果及 trace 峰值时的活跃分配列表 |
| `operator-memory-report.md` | 汇总报告，包括大分配操作、生命周期、调用栈示例和边界 Tensor |
| `operator-tensors.jsonl` | 仅 tensor-ops 模式：操作输入/输出 shape、dtype、storage 地址/大小、偏移 |

先比较 metadata 的 start/peak/end allocated，确认新窗口是否复现约 12.44 GiB 的临时
回落；再看大分配表的操作名称、单次字节、分配/释放时间。相同地址可能复用，所以按
allocation ID 区分生命周期。trace 峰值的活跃列表只含窗口内新分配，**不含窗口前已有
的 704.67 MiB 输入消息或模型参数**。

累计分配不能当成峰值（反复分配后释放也累加）。profiler 的 self/inclusive memory 是
净变化；某操作内部先分配再释放数 GiB，净值仍可能是 0。所有分配事件按同一 CPU 线程
上最内层包围操作归因，不能把它说成直接测到了某个 CUDA kernel 的私有变量。

Input Dims 是**操作输入形状**，不能拿来冒充新分配块的形状。tensor-ops 输出可识别一部分
实际 Tensor 的 shape/dtype/storage；C++ 内部中间 Tensor 或 workspace 未必会经过 Python
可见 dispatcher。两种日志的时间基准不同，地址比较必须结合生命周期，工具不自动把
相同地址认定为同一张量。无法恢复变量名的临时分配保留为分配块，不猜测。

[PyTorch 1.13 profiler 源码说明](https://github.com/pytorch/pytorch/blob/v1.13.1/torch/profiler/profiler.py)
明确指出 record_shapes 可能保留 Tensor 引用或引入额外复制。因此以第一轮定位分配高水位，
第二轮帮助解释形状；两者峰值不一致时不能混合为一份测量。原阶段诊断也是独立的插桩结果。

若 trace 没有 CUDA `[memory]` 事件，报告标为 `no_cuda_memory_events`，不将缺失当作零；
需要检查该服务器 profiler/CUPTI 构建和原始 trace。调用栈缺失同样显式显示为不可用。
异常时保留 partial 记录并移除 hooks，不能把失败窗口当成完整结果。

无需 PyTorch 的报告重建：

```bash
python src/operator_memory_report.py results/memory-op-sf-alloc/run0-TIMESTAMP
```

自动查看两次最新报告（无需替换时间戳）：

```bash
python - <<'PY'
from pathlib import Path
for name in ('memory-op-sf-alloc', 'memory-op-sf-tensors'):
    root = Path('results') / name
    runs = sorted(p for p in root.glob('run0-*') if p.is_dir())
    if not runs:
        raise SystemExit('没有运行目录: ' + str(root))
    report = runs[-1] / 'operator-memory-report.md'
    if not report.exists():
        raise SystemExit('尚未生成报告，请检查 operator-metadata.json: ' + str(runs[-1]))
    print('\n=== ' + str(report) + ' ===\n' + report.read_text())
PY
```

运行 `python -m unittest discover -s tests -v` 包括算子分析测试。无 CUDA 时仅做解析器、
调用次数选择、地址复用、临时分配归因、异常清理等测试；真实 profiler/dispatcher smoke
测试明确 skip。实际 12.44 GiB 来源必须依据服务器新生成的日志，不能从旧边界日志反推
具体底层操作或张量名。
