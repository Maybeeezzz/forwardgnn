# 小规模网络结构 × 学习算法显存矩阵

状态：运行入口与算法实现已完成，已进行离线及 CPU 单元验证，尚未执行 CUDA 实验。2026-10-21。

入口：`exp/network-algorithm-memory/run.sh`；控制器/worker：`src/network_algorithm_memory.py`；训练规则：`src/network_memory_models.py`。只依赖 PyTorch 和匹配版本的 torchvision，不要求 PyG 或 torch_sparse。

## 目标与范围

将现有 GNN 显存研究扩展到常见非图网络，定位不同学习算法的训练显存峰值、峰值阶段和主要 storage。首轮只做机制筛查，不做完整精度复现或大规模调参。公共采集与口径沿用 [显存 utility](../../docs/layer_memory_profile.md) 和 [算法显存归因](../../docs/algorithm_memory_analysis.md)。

首轮使用两个网络、五种学习算法，共 10 个逻辑配置。原 GNN-SF/BP 结果保留在 [原实验](../compare-algorithm-memory/README.md)，作为单独参考；由于数据、输入结构和目标不同，不与本矩阵直接比较绝对 MiB。这里的局部损失或 FF 不命名为 GNN-SF。

## 网络 × 算法

| 网络 | BP | Local-CE | Forward-Forward（FF） | DFA | SPSA |
|---|---|---|---|---|---|
| MLP | ✓ | ✓ | ✓ | ✓ | ✓ |
| 小型 CNN | ✓ | ✓ | ✓ | ✓ | ✓ |

| 方法 | 更新机制 | 是否跨块反向传播 | 是否使用局部导数/反向计算 |
|---|---|---|---|
| BP | 全局交叉熵和端到端反向传播 | 是 | 是 |
| Local-CE | 每块独立分类头、局部交叉熵，块间 detach | 否 | 是，块内 backward |
| FF | 正负输入、逐块 goodness 局部目标 | 否 | 本实现用块内 autograd 更新；不是完全无 backward |
| DFA | 输出误差经固定随机反馈矩阵直接送到各隐藏块 | 无精确跨块链式反传 | 是，块内向量-雅可比乘积 |
| SPSA | 两次正负参数扰动估计更新方向 | 否 | 否，完全不建立 autograd 图 |

不能把所有方法都描述为“不使用反向传播”。报告分别注明端到端 BP、局部反向、随机反馈、零阶估计。

## 小规模配置

- 数据：Fashion-MNIST，固定选取训练集前 4096 个样本与测试集前 1024 个样本，保持原数据划分边界；所有方法使用相同样本顺序。数据准备与下载不进入测量窗口。
- 预处理：图像转换到 float32 的 [0,1]，无数据增强。标签作为整数保留。
- MLP：784 → 256 → 256 → 256，每块 Linear + ReLU；标准分类读出 256 → 10。
- CNN：三块 Conv2d(3×3, padding=1) + ReLU + AvgPool2d(2)，通道 1 → 16 → 32 → 64；全局平均池化后分类读出 64 → 10。不使用 BatchNorm、Dropout，以减少运行状态及随机性干扰。
- 同一骨干的公共权重以独立 RNG、seed=100 初始化，算法特有参数使用另外的 RNG；MLP 与 CNN 之间不声称等参数量。
- batch size=64，1 个 seed，5 个训练宏步预热、20 个宏步正式测量；最后用 no_grad 评估固定测试子集。
- 统一使用无 momentum 的 SGD，初始 lr=0.01；SPSA 使用同样的更新步长，扰动幅度 c=0.001、每宏步一个方向。以上为诊断起始值，不是各算法最佳超参数。
- float32，不启用 AMP、checkpoint、offload、torch.compile 或算子融合改造。首轮不扫深度、batch size、宽度和学习率。
- 宏步：每个方法接收同一个 minibatch，完成自身所有块的更新。各方法宏步的 forward 次数和计算量不同，不能称为等计算预算。

## 方法适配约定

### BP 与 Local-CE

BP 仅使用末端分类头。Local-CE 在每块后增加独立读出头；CNN 的局部头采用全局平均池化接 Linear。按块计算局部目标、反向并更新，将该块更新前产生的表示 detach 后传给下一块。保持这一顺序，避免隐含额外 forward。推理使用最后一块的分类头；其他局部头属于算法专属状态。

### FF

采用顺序逐块的局部 goodness 训练，goodness 为每样本该块激活的平方均值，阈值 2，正负样本等权的 softplus 二分类目标。逐块更新，下一块输入来自更新前表示的 detach；按样本归一化表示后传入下一块。

MLP 将输入前 10 个位置覆盖为候选标签 one-hot。CNN 将图像固定的 10 个像素位置覆盖为候选标签 one-hot，保持输入通道数不变；这是本实验的 CNN 标签注入适配，不声称复现原论文 CNN 协议。正样本用真实标签，负样本用不同于真实标签的随机标签。明确计入正负输入副本和临时表示的显存。

正负分支顺序执行：分别累计当前块的局部梯度，二者完成后仅更新一次。推理顺序枚举 10 个候选标签，累积各块 goodness 并取最大值，不同时堆叠十份输入。FF 不使用标准分类头，报告需列出其实际参数量及这一结构差异。

### DFA

隐藏块之间 detach，用固定的随机矩阵将 softmax-cross-entropy 的输出误差映射到各隐藏块的输出形状，再执行块内 VJP；末端分类头采用其真实梯度。当前宏步的所有梯度计算完成后再更新权重。

CNN 反馈矩阵直接映射到展平后的块输出（C×H×W），不偷偷改用全局池化压缩反馈。矩阵可能占显著显存，必须单列为算法状态。矩阵初始化后固定，不能作为 model baseline 排除，即便实现时注册为 buffer。

### SPSA

每宏步按参数生成一个 Rademacher（±1）扰动方向，在同一个 minibatch 上顺序计算 L(θ+cΔ) 与 L(θ−cΔ)，以 (L+−L−)/(2c)×Δ 估计梯度。恢复原参数后执行更新；整个过程关闭 autograd。

首版保留原参数快照以可靠恢复，保留扰动方向及必要的更新缓冲，全部归入算法专属状态，不用额外参数副本换取表面上的零梯度占用。后续若改为 RNG 重放或减少副本，应作为明确的优化变体。

## 显存测量与公平性

每个逻辑配置使用独立进程；分别运行两个测量模式，因此完整首轮为 20 个短进程：

1. 低侵入 allocator 模式：不安装变量/保存张量追踪，仅在预热结束后重置峰值，分别记录训练和评估的 peak allocated/reserved。原生临时空间保留在这一指标中。
2. compact 归因模式：相同数据和初始化，使用通用 utility，保留阶段汇总与峰值快照，不写逐事件 JSONL。这一模式的耗时不用于速度对比。

预热结束后状态不清空，保证正式训练有真实常驻状态；不在每步使用 empty_cache。实现中预热在归因 context 安装前完成，单独保存 `warmup` allocator 数值和损失；正式训练新建归因汇总器并重置 allocator 峰值。因此预热不会混入训练峰值，也不额外输出预热的逐变量记录。初始化参数和环境记录在 metadata 中，不计入训练窗口。公共 utility 也接受显式 `period='warmup'`，供其他调用者使用。

每块显式标记 forward、objective、local_backward/backward、feedback、perturbation、optimizer_step 等阶段。该实验沿用汇总字段 `epoch_zero_based` 存放宏步编号（默认正式测量为 5–24），不表示完整数据集 epoch。无梯度方法的显存也要枚举，不能因没有 saved_tensors 就报告算法显存为零。

### 已实现的 utility 扩展

原分类器默认优先将 Parameter/buffer 归入模型基线。现在增加 Tensor 元数据 `_memory_algorithm_category` 和可选 `_memory_owner`，显式算法标记优先于默认参数/buffer 基线；JSON 规则也可用 `override_baseline: true` 声明这一优先级。没有标记的原 GNN 行为保持不变；同一 storage 无论有多少别名仍只计一次。

新增类别为 `local_heads`、`feedback_state`、`perturbation_state`、`parameter_backup`、`update_buffer`。局部读出头的参数、DFA 注册的反馈 buffer 均显式标记；SPSA 的方向、备份、估计梯度分别计入对应类别。公共最后分类头仍为模型基线，FF 没有这一分类头。

分类至少覆盖：公共模型参数、原始 minibatch、局部头、固定反馈矩阵、正负数据副本、扰动方向、参数备份、梯度/更新缓冲、saved_for_backward、其他目标和缓存。记录逻辑大小与实际 storage 大小、别名和所属层。新增 warmup 区间，防止预热峰值混入 training。

模型 forward workspace 不进入算法主指标，但保留在 allocator 结果；无法归因的部分明确报告，不能一律归为模型开销。FF/DFA/SPSA 的专有张量必须有归因证据。指标仍叫“算法保留存储观测峰值”，不是精确因果额外内存。

## 输出和验收

默认输出到独立根目录 `results/network-algorithm-memory/`，每个网络/算法/测量模式一个子目录。支持 compact、完整配置校验后 resume、失败不当作零、磁盘写满停止。本次实现没有生成正式实验结果。

总表字段：network、algorithm、mode、实际参数量、算法专属状态量、training/evaluation、算法观测峰值、allocator 峰值、峰值块/阶段、主要 storage、末次 loss、测试准确率、有效完成标记。保留每层各阶段明细和峰值变量快照；各类别峰值不能跨事件相加。

实施验收先进行 CPU 小张量检查：公共初始化与样本一致、Local-CE/FF 块间无梯度、DFA 反馈固定且映射正确、SPSA 不建立计算图且扰动恢复正确、专有状态归因不漏计。再进行 CUDA 短运行；不凭显存数字小就认定实现正确。

20 步只用于检查存储机制和训练过程是否有效，不用于算法收敛排名。准确率是运行健全性记录，不足以比较统计性能。训练损失和预测必须有限且参数确有更新；发现异常先修实现，不扩大规模。

## 后续扩展门槛

只有首轮出现值得验证的趋势后，才选择 1–2 个配置增加到 3 个 seed，或增加一个深度/batch size。Transformer 暂不纳入，避免 FF/DFA 的块划分和标签注入适配扩大首轮范围。

## 方法来源

- [Forward-Forward 原始论文](https://www.cs.toronto.edu/~hinton/FFA13.pdf)：局部正负 goodness 训练；本实验中的 CNN 注入与训练调度是明确约定的实验变体。
- [Direct Feedback Alignment 原始论文](https://arxiv.org/abs/1609.01596)：用随机反馈直接向隐藏层传递输出误差。
- [Spall：SPSA 概述](https://www.jhuapl.edu/spsa/pdf-spsa/spall_an_overview.pdf)：同时扰动与双侧目标测量。

## CUDA 服务器运行命令

以下从仓库根目录运行。沿用已有 CUDA 环境，需安装与 torch 匹配的 torchvision。训练 worker 不自动联网下载数据；数据准备单独执行。

```bash
cd ~/forwardgnn
conda activate ForwardLearningGNN

# 只读计划：无需 torch/torchvision，不写文件、不启动 worker
bash exp/network-algorithm-memory/run.sh \
  --architectures mlp cnn --algorithms bp local ff dfa spsa \
  --modes allocator compact --batch-size 64 --warmup 5 --steps 20 \
  --seed 100 --gpu 0 --output results/network-algorithm-memory --dry-run

# 仅准备数据；已有数据可省略 --download 以验证本地数据
bash exp/network-algorithm-memory/run.sh --prepare-data --download --data-dir data

# 完整小矩阵：10 个配置 × 2 种测量模式，串行独立进程
bash exp/network-algorithm-memory/run.sh \
  --architectures mlp cnn --algorithms bp local ff dfa spsa \
  --modes allocator compact --batch-size 64 --train-samples 4096 --test-samples 1024 \
  --warmup 5 --steps 20 --seed 100 --lr 0.01 --perturbation 0.001 \
  --data-dir data --gpu 0 --output results/network-algorithm-memory
```

中断后，保持训练参数与代码不变，用相同命令追加 `--resume`。默认参数对应以上命令，因此可直接运行：

```bash
bash exp/network-algorithm-memory/run.sh \
  --gpu 0 --data-dir data --output results/network-algorithm-memory --resume
```

先预览续跑则再加 `--dry-run`。完成判断检查配置/源代码签名、完成标记及 compact 汇总；失败或不完整配置从头运行到新 `attempt-*` 目录。签名包含数据路径、训练配置和测量代码，GPU 号不参与签名；改变配置或代码后应使用新的输出目录。不会按 epoch 恢复，也不删除旧日志。

只汇总已完成结果（不导入 torch，不启动训练）：

```bash
bash exp/network-algorithm-memory/run.sh \
  --data-dir data --output results/network-algorithm-memory --summarize-only
cat results/network-algorithm-memory/report.md
less -S results/network-algorithm-memory/matrix-periods.csv
less -S results/network-algorithm-memory/matrix-stages.csv
```

自定义配置的汇总/续跑需传入原配置。缺少结果时仍输出已有行，同时写入 `missing.json` 并返回非零退出码，不把失败当成零显存。

```text
results/network-algorithm-memory/
  plan.json / status.json / missing.json
  report.md / matrix-periods.csv / matrix-stages.csv
  mlp/bp/allocator/attempt-*/
    metadata.json / rules.json / process.log / result.json
  mlp/bp/compact/attempt-*/
    metadata.json / rules.json / process.log / result.json
    trace/
      algorithm-memory-summary.json
      algorithm-memory-stages.csv / algorithm-memory-stages.md
      algorithm-memory-periods.csv
      memory-stage-summary.* / memory-trace-summary.json / memory-trace-report.md
```

`algorithm-memory-summary.json` 中的 `period_peaks.training`、`period_peaks.evaluation` 和 `stage_peaks` 保留具体 storage、类别与变量别名。`matrix-periods.csv` 的算法峰值在 allocator 行留空，不能将空值解释为零。warmup 的独立 allocator 结果在各 `result.json` 中。普通训练日志只打印运行元数据与异常，不输出逐步张量清单。

## 本地验证

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover -s tests -p 'test_*memory*.py' -q
```

测试使用 CPU 小张量和临时目录，不下载 Fashion-MNIST，不执行 CUDA 实验。覆盖共同初始化、十种组合的有限更新、块间 detach、DFA VJP 数值一致性、SPSA 异常恢复、标签注入、显式归因去重、dry-run 和 resume。没有 torch 时 CPU 学习规则测试会跳过；CUDA 真实性能与显存数值需在服务器运行上述实验后确认。
