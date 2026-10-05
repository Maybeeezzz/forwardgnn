# SF 深层网络显存实验

## 研究问题与可检验假设

目标是检验：在固定数据、算子和隐藏宽度时，SF 的 CUDA 峰值显存随深度的增幅是否显著小于 BP。
不预设结果，也不将“逐层训练”解释成总显存严格 O(1)。本实验只讨论 GPU 分配器显存，
不代表 CPU RAM 或整机内存。

仓库原 SF 脚本仅设置 4 层。`GNNSingleForwardLayer.forward_train()` 对输入 detach，
反向传播局限于当前层；但所有层的参数、已训练层的梯度与 Adam 状态仍保留在 GPU。
此外，`eval_model()` 的 `accumulated_probs` 保留各层概率，约有 O(L × N_eval × C) 开销。
粗略机制预期是 SF 减少跨层激活保存，而参数/优化器内存仍随 L 增长；
BP 的跨层反向传播还需保存多层激活。第一层输入宽度很大时也可能主导峰值、掩盖增长。
因此必须报告绝对 MiB、增长比例以及 MiB/层斜率，不能只用归一化曲线判定。

## 主实验：原实现端到端深度扫描

入口：`src/benchmark_depth_memory.py`；原 SF/BP 模型和训练实现均不修改。

| 控制量 | 默认设置 |
|---|---|
| 数据 | Amazon-Photo、GitHub；扩展可用五个原始数据集 |
| 方法 | SF、BP |
| 算子 | GCN、GraphSAGE；GAT 可单独扩展 |
| 深度 L | 2、4、8、16、32、64；有余量再加 128 |
| 隐藏宽度 | 128；独立扫描 64、256 作为稳健性实验 |
| 划分/种子 | 前 3 个作者划分，seed=100 → 10100、10103、10106；正式结果用 5 个 |
| 优化 | FP32，Adam lr=0.001，weight_decay=0.0005，关闭 TF32 |
| 内存扫描预算 | BP 20 epoch；SF 每层 20 epoch，禁用早停 |
| SF 配置 | append_label=None，双向虚拟节点边，temperature=1 |

默认共 144 个独立 CUDA 子进程。每对方法使用相同 seed/划分，奇偶 run 交替方法顺序。
这不是相同参数初始化或相同模型：SF 包含虚拟节点、局部损失、归一化、逐层概率聚合，
BP 使用原始图和类别输出层。报告原始图规模、参数字节数，并将结果解释为原实现比较，
不单凭这个实验将所有差异归因于 detach。固定宽度，不固定总参数量。

每层 20 epoch 的 SF 与整体 20 epoch 的 BP 也不是等计算预算；短训练只用于触发
完整反向传播和 Adam 状态分配。不可据此比较最终精度、收敛速度或计算效率。
SF 原训练器仍会在每层结束时测试，且第一层、层间过渡也进入测量。

## 显存测量边界

- 每配置、每划分独立进程；CUDA 可用性强制检查，不回退 CPU/MPS。
- CPU 数据加载与 CPU 模型初始化后，GPU 模型/数据传输前同步并重置峰值。
- `training_peak` 包括 GPU 初始化、首次更新、所有训练、原训练器内部验证/测试、
  层间表示传递和 checkpoint（启用早停时）。它不是纯训练步峰值。
- 另一个窗口记录最终 `no_grad` 测试的 `inference_peak`；BP、SF 均使用 no_grad。
- 主指标 `whole_peak` 是上述窗口最大值。记录 allocated 和 reserved 字节数；
  allocated 用于主结论，reserved 用于检查缓存影响。窗口切换时不释放 GPU 常驻数据。
- 不在训练循环清缓存，不剔除第一层/冷启动，不做额外 warm-up。
  该协议测实际流程高水位，而不是稳态单步内存。耗时包含冷启动，不用于宣称加速比。
- CUDA context、外部库未经过 PyTorch allocator 的分配不在统计内。
- OOM 和其他错误保留日志、状态；矩阵继续，最终退出码非零。禁止把 OOM 记为 0 或显卡容量。

## 执行

使用仓库 `install/install_packages.sh` 对应环境，并提前放置作者发布的 datasplits。
缺少划分时拒绝自动生成。数据本体仍由原 loader 下载/读取。
每次使用新的 output 目录，以免混合不同预算、环境或源码。

先检查计划（无需 PyTorch，不执行训练）：

```bash
python3 src/benchmark_depth_memory.py --output results/depth-plan --dry-run
```

CUDA 冒烟验证（检查日志、JSON、精度字段及两种方法均成功）：

```bash
conda activate ForwardLearningGNN
bash exp/nodeclass/nodeclass-depth-memory.sh --gpu 0 \
  --datasets Amazon-Photo --architectures GCN --depths 2 8 \
  --runs 1 --epochs 3 --output results/depth-smoke
```

主实验（五个划分，共 240 个进程）：

```bash
bash exp/nodeclass/nodeclass-depth-memory.sh --gpu 0 \
  --runs 5 --output results/depth-main
```

宽度与更多图的稳健性实验：

```bash
bash exp/nodeclass/nodeclass-depth-memory.sh --gpu 0 --runs 5 --hidden 256 \
  --datasets CitationFull-CiteSeer CitationFull-Cora_ML CitationFull-PubMed Amazon-Photo GitHub \
  --output results/depth-width256
```

质量复核：主实验之外，至少在 4、16、32、64 层运行原论文预算，检查深层 SF 是否仍有实用精度。
避免用浅层精度代表深层；按验证集选 checkpoint，不按测试集选择深度。

```bash
bash exp/nodeclass/nodeclass-depth-memory.sh --gpu 0 --runs 5 \
  --depths 4 16 32 64 --epochs 1000 --patience 100 --val-every 2 \
  --output results/depth-quality
```

## 分析和结论判据

输出 `plan.json`（配置及 src 源码 SHA256）、每进程 config/measurement/stdout、
`matrix.json`、`summary.csv`、`trends.json`、`report.md`。逐进程记录实际软件版本和 GPU 型号。
CSV/Markdown 给出全程 peak allocated 均值和总体标准差；JSON 保留 allocated/reserved、
参数大小、训练预算与精度。SF 额外保存各层原始结果。

自动计算同数据集/算子下两种方法都完整成功的深度点上的：

1. 线性拟合 `M(L) = a + bL` 的 b（MiB/层）。这只是描述统计，不是复杂度证明。
2. 相同最浅与最深成功层数之间的 `(M_deep / M_shallow - 1) × 100%`。
3. 是否覆盖完整预设深度范围；BP OOM 后不得把短区间斜率外推至 64 层。

建议事先固定可操作的强假设：在 2→64 层完整区间内，SF 平均峰值增长 ≤25%，
且 SF 斜率 ≤ BP 的 25%（仅 BP 正斜率时定义）。这是本次设计的工程阈值，
不是已有实测结论或通用定理。若未满足，报告真实增幅及失败场景；不能只展示有利子集。
若 BP OOM，只能补充“该配置下 SF 可训练到更深层”的容量结论，不能宣称阈值已通过。

正式论文应画每数据集/算子独立的 L–MiB 曲线、误差条和 M(L)/M(2) 曲线，标出 OOM。
5 个划分的原始配对值可进一步做配对 bootstrap 的斜率差/增长差区间；当前脚本不把
均值趋势视为统计显著性。若短预算与质量预算结论不一致，应同时报告两者。

## 后续机制消融（不混入主实验）

若要将现象归因于逐层训练，另设计统一图/算子/层宽/输出头的全局 BP 与 detach 局部训练。
单独记录活动参数、Adam 状态、当前层激活和层间表示，分别开关 SF 历史优化器释放、
概率在线求和、缓存算子。这些会改变实现与测量边界，应使用独立方法名称；
不能悄悄替换原 SF 来制造平坦曲线。主扫描排除 CachedGCN/CachedSAGE 和 SparseTensor，
以免将逐层缓存或稀疏聚合路径改变与深度效应混合。

## 当前验证状态

本机默认 Python 未安装 torch、torch_geometric、torch_sparse，未执行 GPU 训练。
已验证计划生成、CLI、Python 3.8 语法与人工构造的完整/缺失/OOM 汇总逻辑。
尚无实测显存下降或深度不敏感的结论。
