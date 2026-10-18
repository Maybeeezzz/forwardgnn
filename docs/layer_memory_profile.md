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
