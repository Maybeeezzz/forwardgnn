# SF 训练算法显存峰值分析

整理日期：2026-10-21。

本记录对应本目录的 SF/BP 算法显存矩阵实验，依据用户提供的 CUDA 服务器峰值查询输出整理。本文只分析 SF；没有重新执行实验，也没有重新读取完整事件日志。实验配置及运行方式见 [README](README.md)。

## 数据来源与统计口径

- 数据集：Amazon-Photo；骨干：GCN、SAGE、GAT；深度：2、4、8、16 层。
- 对应实验计划：hidden=128，runs=1，seed=100，epochs=3，lr=0.001，patience=-1，val-every=1。峰值摘录本身未打印全部运行参数，严格复核时应以各 run 的 `memory-run.json` 为准。
- 服务器结果根目录：`results/algorithm-memory-matrix/Amazon-Photo/`。本次输出中，GCN run 时间为 20261021-001704 至 002355，SAGE 为 004223 至 004856，GAT 为 010636 至 011634。
- 查询从各 SF run 的 `algorithm-memory-summary.json` 读取 `period_peaks.training`，仅纳入 `complete=true` 的结果，并打印峰值时最大的 10 个算法 storage。
- 指标为**按底层 storage 去重的训练保留存储观测峰值**，单位 MiB（2²⁰ 字节）。它不是整个进程的 CUDA 峰值，也不是精确的因果额外显存。
- 模型基线、普通 forward 可见存储和未归因存储被该查询过滤；此前约 12.44 GiB 的 SAGE 模型算子临时空间不计入本表。
- 本文依据打印到三位小数的结果计算比例与余量，因此存在舍入误差。

## 峰值出现的位置

**所有 12 个 SF 配置记录的训练算法峰值都位于第一层、第 2 个 epoch 的 `loss_build` 阶段。**

| 骨干 | 总层数 | 训练算法峰值 MiB | 执行层 | 阶段 | epoch（从 0 开始） | 事件序号 |
|---|---|---:|---|---|---:|---:|
| GCN | 2、4、8、16 | 61.744 | layer1 | loss_build | 1 | 380 |
| SAGE | 2、4、8、16 | 79.724 | layer1 | loss_build | 1 | 331 |
| GAT | 2、4、8、16 | 205.157 | layer1 | loss_build | 1 | 380 |

`layer1` 使用从 1 开始的展示编号，对应代码 `layers[0]`。`epoch=1` 表示第 2 个 epoch。工具在严格超过已有峰值时更新快照，因此相等峰值保留首次记录；不能据此认定后续 epoch 没有再次达到同样峰值。不同进程的事件序号也不是统一时间轴。

当前已测深度从 2 增加到 16 时，SF 的训练算法观测峰值没有上升。这是本数据集、预算和配置下的结果，不代表任意数据集、隐藏宽度或训练预算下都如此。

## 为什么峰值标记为 loss_build

`loss_build` 是峰值快照的阶段归属，不代表主要存储都由损失运算分配。

此时第一层 forward 已经完成，而 backward 尚未消费保存的张量。增强图及特征、forward 为 backward 保存的张量、节点表示和局部目标同时存活；第 2 个 epoch 时还已有 Adam 状态，快照中也观察到了参数梯度。

这些对象的重叠形成峰值。阶段标签不足以确定峰值对应的精确源码语句；若需要这一细节，应继续核对事件位置或补充追踪信息。

## SAGE：主要占用是三块 21.764 MiB 的存储

| 峰值时存活的 storage / 变量 | 大小 MiB | 类别 | 解释 |
|---|---:|---|---|
| `graph_augmenter.aug_x` | 21.764 | algorithm_data | 增强输入特征；与 `aug_graph.x`、首层 `node_feats` 等共享 storage |
| `autograd_saved.layer1.12` | 21.764 | saved_for_backward | 为 backward 保存的独立 storage，具体语义尚待定位 |
| `autograd_saved.layer1.13` | 21.764 | saved_for_backward | 另一块独立 storage，具体语义尚待定位 |
| `aug_graph.edge_index` | 3.783 | algorithm_data | 增强图连接关系 |
| `node_emb` | 3.739 | saved_for_backward | 第一层输出，也由 autograd 保存引用 |
| `train_node_emb` | 2.391 | objective | 用于局部目标的训练节点表示，也存在 autograd 别名 |
| `aug_graph.edge_type` | 1.892 | algorithm_data | 增强图边类型 |
| 其余存储 | 约 2.627 | 多种类别 | 总峰值减上述项目；包含梯度、Adam 状态及其他存储，并非完整类别分解 |

前三块合计 **65.292 MiB，约占 81.9%**。它们是降低当前峰值最值得研究的对象。两个 saved storage 大小相同，并不证明内容相同或属于可删除的冗余副本。

输出还明确列出第一层 `lin_l.weight.grad` 和 `lin_r.weight.grad` 各约 0.364 MiB，以及一个 Adam `exp_avg` storage 约 0.364 MiB。查询仅展示前 10 项，不能将这些项目当作全部梯度或全部优化器状态。

## GAT：单块 autograd 保存存储占约 60.8%

| 主要 storage | 大小 MiB | 占总峰值比例 |
|---|---:|---:|
| `autograd_saved.layer1.48` | 124.811 | 约 60.8% |
| `graph_augmenter.aug_x` | 21.764 | 约 10.6% |
| `autograd_saved.layer1.30` | 21.764 | 约 10.6% |

前三块合计 **168.339 MiB，约占 82.1%**。此外，前 10 项中还包含五块各约 3.900 MiB 的 saved storage、3.783 MiB 的增强边索引，以及 3.739 MiB 的节点表示。

124.811 MiB 的 storage 是首要定位对象。但 `.48` 是追踪生成的编号，不是算子名称。当前摘录没有 shape、dtype、保存调用位置或消费它的 backward 节点，不能直接认定它是某一种注意力张量或边消息张量。

## GCN：增强输入和一块大型 saved storage 占主要部分

GCN 的峰值为 61.744 MiB。其中增强输入 `aug_x` 与 `autograd_saved.layer1.14` 各为 21.764 MiB，合计 **43.528 MiB，约占 70.5%**。

其余主要项目包括 3.900 MiB 的 saved storage、3.783 MiB 的增强边索引、3.739 MiB 的节点表示、2.391 MiB 的训练节点表示、1.892 MiB 的边类型，以及 0.975 MiB 的 saved storage。前 10 项也出现第一层参数梯度和 Adam 状态。

## 针对降低峰值的优化优先级

### 1. 优先定位并减少第一层大型 saved tensors 的 GPU 保留量

补充这些 storage 的 shape、dtype、生成/保存位置和 backward 消费位置，再评估重计算或卸载是否适用。SAGE 重点是两块 21.764 MiB 的 saved storage；GAT 重点是 124.811 MiB 的 saved storage。

这些是反向计算所需的数据，不能直接删除。重计算或卸载也可能把峰值移到 backward，必须重新测量整个训练区间，而非只观察原峰值事件是否下降。以上属于待验证方案，本记录没有实施或测得收益。

### 2. 检查增强数据中可避免的分配

当前普通 GCN/SAGE/GAT forward 不使用关系型 `edge_type`，但峰值快照中它仍占 1.892 MiB。确认其他训练、评估路径也不需要后，可考虑按骨干类型避免创建该 CUDA 张量。是否真正降低全程峰值仍需实测。

`aug_x` 有多个别名，只删除一个别名不会释放其 storage；它仍是首层输入，不能简单删除。应检查构造方式或表示方式，而不是将多个引用误判为多个副本。

### 3. 提前清理旧梯度属于小幅优化

可评估在每次 forward 前使用 `zero_grad(set_to_none=True)`，减少上一轮梯度与本轮 forward/目标张量重叠。当前快照中确有梯度，但从大小看不是主要来源；收益及数值等价性需要验证。

### 4. 清理旧层状态不是本次峰值优化的首要方向

清理已完成层的 optimizer 状态、为层间表示生成使用 `no_grad()`，可以改善后续阶段的常驻或临时显存。但本次记录的最大峰值在第一层训练期间发生，尚未进入后续层，所以这些改动不能直接消除该峰值。

## 结论的边界与后续验证

后续对照必须同时检查 BP 的峰值位置，而非只比较 SF 与 BP 的 MiB。当前提供给本文的摘录只有 SF，无法据此认定 BP 峰值一定出现在最后一层或 backward。已扩展汇总工具：通过 README 中的 `--summarize-only` 命令，从双方已有快照输出执行层、阶段、epoch、主导类别、存储归属层及变量别名。实际 BP 位置应以生成的 `matrix-peak-comparison.csv` 为准，不用机制推测代替测量。

本次结果支持将优化重点从“已训练层状态累积”调整为“第一层 forward 保留张量与局部目标的重叠”。大型 saved tensors 由骨干计算产生、因反向传播而保留，按当前规则计入算法保留显存，但不属于 SF 独有开销；BP 对照必须使用同一规则。

下一轮验证应同时报告训练算法观测峰值、峰值层/阶段、峰值分类构成及整个进程 CUDA allocated 峰值，并检查损失和参数更新是否保持预期。算法保留峰值下降不保证模型算子主导的整个进程峰值同步下降。

当前摘录只包含训练峰值及其前 10 个 storage，不能据此给出完整生命周期、完整类别总量、评估峰值或精确底层分配/释放时刻。
