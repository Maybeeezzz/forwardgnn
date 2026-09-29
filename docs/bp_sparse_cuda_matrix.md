# CUDA BP SparseTensor 轻量矩阵

入口 `src/test_bp_sparse_cuda.py`，启动脚本 `exp/bp-sparse-cuda-matrix.sh`。
本实验直接使用官方 PyG 模型和 `torch_sparse.SparseTensor`，不替换算子的聚合函数。
不修改作者已有训练入口，也不依赖个人 GNN 仓库。它是内存机制实验，不能当作完整论文精度复现。

## 矩阵与训练预算

- 任务：节点分类 / 链接预测。
- 网络架构：GCN / GraphSAGE(mean) / GIN(sum) / GAT(4 heads)。GIN 是额外扩展。
- 图拓扑：均匀随机 / 枢纽偏置合成图。
- 输入：edge_index / torch_sparse.SparseTensor，后者使用 row=target、col=source。
- 默认 2048 节点、约 32768 条有向边、输入/隐藏维度 64、两层、节点类别数 6。
- Adam lr=0.001、weight_decay=0.0005；每配置同一随机种子 100 和相同初始化。
- 首次更新计为第 1 次预热；默认共预热 2 次，再测量 10 次更新。
- 每个配置默认重复测量三对独立进程，交替后端运行顺序。重复测量不是多 seed 精度实验。
- 链接预测移除 20% 正边；最多使用 512 条训练正边和等量负边做点积解码及 BCE。
  负采样排除完整图的正边；不报告留出集精度。节点分类使用前 60% 节点的交叉熵。

## 环境与运行

需要 CUDA GPU、PyTorch、PyG 和与 PyTorch/CUDA 匹配的 torch_sparse。
可使用作者的 ForwardLearningGNN 环境（其安装脚本指定 Python 3.8、PyTorch 1.13.1、
CUDA 11.7、PyG 2.2.0），并确认 torch_sparse 可以导入。

完整 16 配置矩阵，共 16 个检查进程和 96 个测量进程：

```bash
cd /path/to/forwardgnn
conda activate ForwardLearningGNN
bash exp/bp-sparse-cuda-matrix.sh --gpu 0
```

先运行单配置小预算：

```bash
python src/test_bp_sparse_cuda.py --gpu 0 \
  --task node-class --architecture GCN --topology uniform \
  --steps 3 --repeats 1
```

可用 `--nodes --edges --features --hidden --steps --warmup --repeats --seed` 调整预算。
默认至少需要 1 次预热。`--output` 必须指向新的或空目录，以免混用结果。
缺少 CUDA/torch_sparse 时明确报错；无 CPU 或其他设备回退。

## 数值与调用路径验证

独立检查进程比较相同初始化模型的三次更新：logits、loss、参数梯度及 Adam 更新后的参数。
容差预先固定为 atol=1e-5、rtol=1e-4。关闭 TF32；gather-scatter CUDA 可能使用原子操作，
不要求逐位确定性，也不因数值检查失败而自动放宽容差。

仅在检查进程注册 message_and_aggregate hooks，记录每层调用次数。
edge_index 应为 0；SparseTensor 对 fuse=True 的层应为 3。
GAT 通常 fuse=False，支持稀疏输入不代表其注意力消息已被融合成 SpMM。
检查失败的配置保留日志并标记，不输出有效降幅；整个矩阵继续其他配置，最终以非零状态退出。

## 显存口径

每个后端每次测量使用新进程，避免上一后端分配器缓存污染。
使用 `torch.cuda.reset_peak_memory_stats()` 和 `max_memory_allocated/max_memory_reserved()`：
这是 CUDA 分配器的高水位，不是定时采样下界。

结果分别包含：

1. initialization：模型/数据传 GPU、GPU SparseTensor 构造，包括临时 edge_index。
2. cold_training：首次 forward+loss、backward、Adam step，包含 Adam 状态初始化。
3. steady_training：预热后的完整训练步。
4. whole_run：初始化、首次更新、其余预热、测量期所有窗口峰值的最大值。

每阶段边界同步并重置峰值；窗口取最大值可恢复指定范围内的绝对 allocated/reserved 高水位。
同时保存窗口开始和结束的当前分配，区分常驻内存与临时增量。
测量进程不注册性能 hooks、不清理训练循环中的缓存。
SparseTensor 只构造一次，GCN 自身的归一化遵循 PyG 默认行为。

`allocated` 用于比较活跃分配峰值，`reserved` 包含分配器缓存。
PyTorch 未追踪的 CUDA context/外部库分配不包含在这些数值里。
CPU 数据准备的内存不属于 GPU 内存；初始化计时可能包含 CPU 准备时间。
逐阶段同步影响耗时，当前同步计时不能作为正式加速比。

## 输出

`results/bp-sparse-cuda-<timestamp>/` 保存：

- configuration.json、runner_snapshot.py：参数、设备、依赖版本、源码快照及 SHA256。
- 每配置 check.json 与日志：数值差异、融合调用次数。
- repeat-N 下各后端 JSON：初始化/首次/稳定/全程峰值，及逐阶段详细读数。
- matrix.json、report.md：成功/失败状态、各重复值、均值、总体标准差和配对降幅。

降幅按每对独立测量计算 `(baseline - sparse) / baseline * 100%`，再汇总；负值表示增加。
默认报告表以稳定训练 peak allocated 为主，JSON 同时提供全部范围与 reserved 结果。

## 当前验证状态

开发机没有 CUDA 和 torch_sparse；已做 Python 3.8 语法、CLI、图构造与任务损失的 CPU
检查，尚未执行 CUDA 数值一致性、融合调用或显存测量。部署到 CUDA 环境后应先执行上述
单配置命令，再运行完整矩阵。未生成或宣称 GPU 内存降幅。
