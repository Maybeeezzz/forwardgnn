# 模型基线与训练算法保留显存

从仓库根目录，在已有 CUDA/PyTorch/PyG 环境运行：

```bash
bash scripts/compare_algorithm_memory.sh
cat results/algorithm-memory-comparison/comparison.md
```

脚本包含完整 SF、BP 命令，不依赖用户预设的环境变量。两层、hidden=128、Amazon-Photo、seed=100、3 epochs、关闭 early stopping。SF 是每层 3 epochs，BP 是全模型 3 epochs；它们不是等计算量或等准确率实验。SF 使用增广图、末层输出 128，BP 的分类输出为类别数 8；这些是仓库现有模型差异，本工具没有更改算法或将其伪装成同构对照。

单独重建对比：

```bash
python src/memory_algorithm_summary.py \
  --sf results/algorithm-memory-sf \
  --bp results/algorithm-memory-bp \
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
