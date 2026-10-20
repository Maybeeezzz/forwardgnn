# 模型基线与训练算法保留显存

从仓库根目录，在已有 CUDA/PyTorch/PyG 环境运行：

```bash
bash exp/compare-algorithm-memory/run.sh --architectures SAGE --depths 2 \
  --output results/algorithm-memory-matrix
cat results/algorithm-memory-matrix/matrix-report.md
```

脚本现支持矩阵，默认 GCN/SAGE/GAT × 2/4/8/16 层；上面的命令显式选择两层 SAGE 回归配置。hidden=128、Amazon-Photo、seed=100、3 epochs、关闭 early stopping。完整参数与输出结构见 [实验说明](../exp/compare-algorithm-memory/README.md)。SF 是每层 3 epochs，BP 是全模型 3 epochs；它们不是等计算量或等准确率实验。SF 使用增广图、末层输出 128，BP 的分类输出为类别数 8；这些是仓库现有模型差异，本工具没有更改算法或将其伪装成同构对照。

单独重建对比：

```bash
python src/memory_algorithm_summary.py \
  --sf results/algorithm-memory-matrix/Amazon-Photo/SAGE/layers-2/sf \
  --bp results/algorithm-memory-matrix/Amazon-Photo/SAGE/layers-2/bp \
  --output results/algorithm-memory-comparison
cat results/algorithm-memory-comparison/comparison.md
```

传入根目录时选择名称排序最后的 run；若最新 run 未正常结束则报错，不静默回退。也可传入明确的 run 目录。sources.json 记录实际使用的路径。

逐项查看梯度、优化器、保存张量等分类：

```bash
python - <<'PY'
import csv
with open('results/algorithm-memory-comparison/comparison.csv') as f:
    rows = list(csv.DictReader(f))
for r in rows:
    if r['owner'] == 'all':
        print(r['algorithm'], r['active_layer'], r['phase'],
              'peak MiB=', r['algorithm_observed_peak_mib'],
              'grad=', r['gradients_mib'], 'Adam=', r['optimizer_state_mib'],
              'saved=', r['saved_for_backward_mib'],
              'data=', r['algorithm_data_mib'], 'cache=', r['algorithm_cache_mib'])
PY
```

## 测量口径

主指标是**已观测的、按底层 storage 去重的训练保留存储峰值**。不会用训练总峰值减推理总峰值，也不会把 allocator 的未知差额计为算法显存。

- model_baseline：参数、buffer、规则声明的原始图/特征。autograd 再保存其别名不产生第二份计费。
- model_forward_visible：forward/evaluation 阶段可见、没有训练保留证据的普通激活；排除在算法量外。这不是全部模型基线的精确测量。
- gradients：可见参数梯度、梯度回调的 incoming_gradient。
- optimizer_state：Adam 等优化器状态，只计 CUDA storage。
- saved_for_backward：通过 saved_tensors_hooks 观察仍由 autograd 保存的非基线 storage。包括在 Python 局部变量里看不到的保存张量。
- algorithm_data/cache/objective/checkpoint：由规则声明的专有增广数据、缓存、目标计算及早停副本。
- unattributed_visible：可见但没有明确归因证据的 storage。
- allocator_minus_visible：分配器记账减去可见 storage；保留原始正负符号。可能含底层临时张量、分配器取整、未观测训练状态等。

**边界**：saved_for_backward 表示训练持有的存储集合，不能证明每一字节在同一时刻都比无梯度 forward 多占；一个普通激活可能仍用于 forward。此工具不声称测得精确的因果额外峰值，也不能完整观测原生 backward/optimizer 内部临时空间。确定性 SAGE 的约 12.44 GiB 索引/排序 workspace 不会进入算法主指标，但仍保留在进程区间峰值里。研究结论必须使用“训练保留存储观测峰值”这个指标名，不能将其写成所有算法额外显存的精确峰值。

## SF 数据的具体计费方式

configs/memory/sf.json 声明 graph_augmenter 和 aug_graph 的专有张量、forward_train.node_feats 缓存及局部目标。原始 x/y/edge_index 或其它与原始 data 共用 storage 的别名优先算基线。增广图若另行复制整个输入数组，其完整新 storage 计入算法额外分配；不只算新增虚拟节点的逻辑行数，因为完整副本实际与原始数组并存。

规则使用变量路径 glob，可替换或扩展，而不在 tracer 中硬编码 SF。默认 generic 模式只识别参数、梯度、优化器、早停副本和 autograd 保存集合；没有规则的图数据不会自动认定为算法专属。

## 层与阶段

active_layer 是执行层，owner 是存储归属层。layer1 对应 layers.0 或 convs.0；layer2 对应索引 1。共享图、无法唯一归层的保存张量或缓存标为 shared，不平均分摊给每层。梯度和优化器状态通过对应参数归层；保存张量按保存它的模块归层；多个层共用一个 storage 时计入 shared。

owner=all 为当时全模型保留量，不是执行层独占。不同层/阶段/owner 的峰值可能来自不同事件，不能相加；全局峰值单独在 algorithm-memory-summary.json 的 global_peak 中给出。每行各分类来自同一个事件，避免把独立分类峰值相加。阶段含 forward、loss_build、backward、optimizer_step、zero_grad、evaluation 等；无法判断的语句为 other。

算法模式仍使用 Python 边界和参数梯度回调。SF 对 forward 的直接方法调用通过源码帧定位；不要求改成 Module.__call__。BP 独立 optimizer 用弱引用追踪，在原生 autograd 回调没有 Python 父帧时也可观察。默认关闭，无 hook/训练行为修改。

## 输出与验证

每个 run 产生：

- algorithm-memory-stages.csv / .md：阶段与存储归属峰值表。
- algorithm-memory-summary.json：全局峰值、各行完整 storage 分类/别名及规则。
- memory-events.jsonl：每个边界的 algorithm_observed_bytes 和原始进程量。
- memory-variables.jsonl：变量与 storage 别名、出现/不可见的变化、autograd 保存标记。
- memory-run.json：运行参数、版本及归因模式。

saved hook 使用不复制 storage 的 detach 包装，由 autograd 持有；追踪器只持有弱引用。不开启磁盘/CPU offload，不主动延长保存张量寿命。hook 会影响 autograd 保存机制及诊断耗时，因此不能将该运行用于计时基准。已有自定义 saved_tensors_hooks 的程序需另行检查兼容性。

旧的 memory-debug 或 operator-only 日志缺少保存张量生命周期，不能离线补成新指标，必须重跑。不要与 --memory-analysis-operators 或 --memory-analysis-no-variables 同用。

运行测试：

```bash
python -m unittest discover -s tests -p 'test_*memory*.py' -v
```

含真实 autograd 梯度一致性/释放测试及 CUDA 端到端测试；缺少 PyTorch/CUDA 时相应测试跳过。

## 分开统计训练与评估（schema v2）

BP 节点分类和链接预测的 `test()` 均使用 `@torch.no_grad()`，同时保留 `model.eval()`。

新日志将 `period` 独立于具体 `phase` 保存：

- `training`：epoch 循环内的训练过程，以及反向梯度回调。
- `evaluation`：测试/评估函数，或由 `val_`/`test_` 目标变量识别的验证调用；嵌套模型 forward 继承评估归类。
- `other`：初始化、checkpoint、epoch 循环外的 SF 层间转换等。循环结束后残留的局部 `epoch` 变量不作为训练证据。
- `unclassified`：调用方未提供区间标签；不会静默并入训练。自动阶段识别是针对现有训练代码的启发式识别，扩展新训练器时应检查标签。

`algorithm-memory-summary.json` 的 `period_peaks` 保存各区间完整峰值快照；`global_peak` 仅作全程背景。新文件 `algorithm-memory-periods.csv` 和对比文件 `comparison-periods.csv` 汇总各区间；原有阶段表新增 `period` 列。未采集到某区间时输出空值/“未观测”，不会报告为 0。

评估峰值仍包含评估时存活的训练遗留状态（梯度、优化器等），不能解释为评估新增分配；本次修复没有改变这些对象的生命周期。每个峰值的类别分解保留在 JSON 中。

自定义显式 scope 可使用 `memory.stage('forward', period='other')` 标记层间转换。显式 forward/loss/backward/update scope 默认按训练归类，evaluation scope 按评估归类。

旧 schema v1 日志不能可靠区分循环内训练 forward 和层间转换，比较工具会明确拒绝，而非自动猜测；历史结果不被修改。

## 精简输出与矩阵续跑

独立训练入口增加 `--memory-analysis-output compact`；矩阵入口默认使用 `--memory-output compact`。精简模式仍在每个原有边界执行变量枚举、storage 去重和算法归因，保留各层各阶段汇总及选中的峰值快照，但不写 `memory-events.jsonl` 和 `memory-variables.jsonl`。不能用精简结果恢复完整变量时间线或重新做事件归因；需要原始事件时使用 `full`。compact 不与算子 profiler 模式组合。

训练和评估的峰值仍分开，指标仍是观测到的算法保留 storage，非完整原生临时分配峰值。普通训练日志、运行参数、汇总报告仍保留，旧日志不会自动删除。

矩阵 `--resume` 校验已有结果的完成标记、训练参数、归因规则和阶段表，跳过完整配置；不完整配置在新 attempt 目录中从头运行。保持原矩阵参数，先用 `--resume --dry-run` 预览。完整服务器命令见 [矩阵实验说明](../exp/compare-algorithm-memory/README.md#断点续跑已有矩阵只补-sage-16-层和-gat)。
