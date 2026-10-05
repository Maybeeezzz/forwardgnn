# 原论文范围内的逐层显存诊断

本功能全部为新增文件，不修改原 SF/BP 训练代码，也不修改已有深度扫描入口。
入口为 `src/profile_layer_memory.py`，脚本为 `exp/nodeclass/nodeclass-layer-memory.sh`。
监控实现为 `src/utils/layer_memory_trace.py`。

## 实验范围

延续当前节点分类 SF/BP 对照，默认采用原论文的五个数据集：CitationFull-CiteSeer、
CitationFull-Cora_ML、CitationFull-PubMed、Amazon-Photo、GitHub；GCN/SAGE/GAT；
1、2、3、4 层；宽度 128；五个作者划分。明确拒绝超过 4 层的请求。
不包含合成图、更多深度或链接预测扩展。

默认每层/整体最多 1000 epoch（SF/BP），patience=100，val_every=2，Adam lr=0.001、
weight_decay=0.0005。模型和训练算法直接调用仓库实现。沿用此前实验的 FP32、关闭 TF32、
GCN/SAGE 确定性设置；这并非已证实与原论文显存测量口径一致的复现。
完整矩阵包含 600 个独立子进程，逐行同步插桩会显著减慢运行，建议先单配置验证。

## 运行

以下命令从仓库根目录执行，结果目录必须新建或为空：

```bash
# 无需 torch 的计划检查
python3 src/profile_layer_memory.py --output results/layer-plan --dry-run

# 小预算验证监控是否工作；不能用来代表完整训练结果
bash exp/nodeclass/nodeclass-layer-memory.sh --gpu 0 \
  --datasets Amazon-Photo --architectures GCN --depths 4 \
  --runs 1 --epochs 3 --patience -1 --output results/layer-smoke

# 定位已观察到的 SAGE 高峰值
bash exp/nodeclass/nodeclass-layer-memory.sh --gpu 0 \
  --datasets Amazon-Photo --architectures SAGE --depths 1 2 3 4 \
  --runs 1 --epochs 3 --patience -1 --output results/layer-sage-diagnostic

# 完整原论文范围矩阵
bash exp/nodeclass/nodeclass-layer-memory.sh --gpu 0 \
  --output results/layer-paper
```

CUDA、torch_geometric、torch_sparse 必须可用。父进程在启动训练前检查全部所需的
作者划分，缺失时一次列出，避免 GitHub 缺划分导致大量重复失败。
脚本不自动生成新划分，不使用 CPU/MPS 替代显存结果。

## 记录什么

监控直接跟踪原 SF/BP 函数及 PyG 卷积 forward 的 Python 执行边界，保留文件、源码行、
函数、层号和可获取的 epoch；每个边界执行 CUDA synchronize，读取当前 allocated/reserved
和自上次边界以来的峰值，然后重置峰值统计。全部区间峰值取最大值恢复监控窗口的高水位。

包括：

- GPU 传输、SF 图增强和训练准备。
- 每层前向及卷积 forward 内部 Python 步骤，如聚合、线性变换。
- loss 构造、zero_grad、backward 和 optimizer.step 对应语句。
- 各层参数梯度就绪回调，帮助观察 BP 反向阶段的层顺序。
- 验证、测试、early-stopping checkpoint 保存/恢复。
- SF 层间前向、detach、局部函数返回和下一次边界之间的释放。

梯度 hook 返回 None，不修改梯度；日志仅保存标量和文本，不保留张量引用。
不调用 gc.collect 或 empty_cache，不修改张量生命周期策略。

每个配置目录新增：

| 文件 | 用途 |
|---|---|
| memory-events.jsonl | 完整时间顺序记录；每条立即刷新，失败时也保留已写事件 |
| memory-trace-summary.json | 最大峰值区间、峰值最高的 20 个区间、回落最大的 20 个区间、按层关联的绝对峰值 |
| memory-trace-report.md | 可直接阅读的峰值及释放位置表 |
| measurement.json | 插桩下的训练/最终测试/全程峰值及模型结果 |
| stdout.log | 原训练日志及错误 |

根目录另提供原样式 matrix.json、summary.csv、report.md、trends.json 与 plan.json。
这些结果属于插桩诊断，应与之前无插桩的深度扫描结果分开解释。

## 如何理解峰值和回收

`interval_after` 表示区间起始位置，`end` 表示结束边界。Python `line` 事件发生在语句执行前，
所以区间的内存变化应归到前一个事件之后的执行过程，不能直接归给结束行。
函数 return 事件发生在局部变量真正销毁之前，释放可能在下一个边界才被观测。
层号从 1 开始；epoch_zero_based 从 0 开始。

- `allocated_delta_bytes < 0`：两个边界间，存活张量占用净减少。
- `peak_to_end_drop_bytes > 0`：区间内达到峰值后已回落；可发现同一条原生调用内创建又释放的临时内存。
- allocated 下降、reserved 不变：内存已可供 PyTorch 分配器复用，但缓存仍被保留。
- reserved 下降：分配器保留的内存减少，可能已归还 CUDA 驱动。

“净减少”和“峰值回落”都不是累计释放字节数。区间内若发生多次分配和释放，不能从两个
边界完整重建每次操作。时间戳是同步后的 CPU 相对时间，而非每次 cudaFree 的精确时间。
本工具定位的是释放发生的代码区间，而不是每个 allocator free 事件。

反向阶段用参数梯度就绪回调提供层关联观测，不声称它精确对应整个层的 backward 开始/结束。
底层 C++/CUDA 算子内部归因需要进一步使用专门的算子 profiler。按层汇总的数值是该层关联
区间中的**全进程绝对显存**，包含其他层的存活状态，不能相加或当作该层独占显存。

逐行同步和梯度 hook 会改变异步执行时序，因此诊断耗时不能作为性能速度结论，峰值也应
用无插桩实验交叉验证。CUDA OOM 时总结仅包含已完成边界；尚未成功记录的区间不可视作零。
监控范围在最终推理结束，进程退出后的统一显存释放不在其中。

## 本机验证状态

已进行 Python 3.8 语法检查、CLI/原论文矩阵范围检查，以及模拟 allocator 的峰值、窗口重置、
释放和日志输出验证。本机默认 Python 无 PyTorch/CUDA，尚未验证真实 GPU hooks、峰值和梯度
一致性。应在原 ForwardLearningGNN CUDA 环境先执行上面的小预算验证，再运行完整矩阵。
