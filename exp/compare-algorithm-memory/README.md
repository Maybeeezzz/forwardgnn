# SF/BP 算法保留显存：骨干与深度矩阵

入口：`exp/compare-algorithm-memory/run.sh`，调度器：`src/compare_algorithm_memory_matrix.py`。
本目录已从固定两层 SAGE 脚本扩展为参数化矩阵；每个配置启动独立训练进程，串行使用同一 GPU。

## 实验目标与默认矩阵

比较 GCN、SAGE、GAT 随层数增加时的算法保留存储，并将训练、评估、其他阶段分别报告。

|维度|默认配置|
|---|---|
|方法|SF、BP|
|任务|节点分类|
|数据集|Amazon-Photo|
|骨干|GCN、SAGE、GAT|
|层数|2、4、8、16|
|隐藏总宽度|128|
|重复次数|1 run/配置|
|种子基数|100，沿用仓库 SeedManager 和 split_i=run_i|
|预算|3 epochs；lr=0.001；patience=-1；val-every=1|
|SF 专有参数|append-label=none、aug-edge-direction=bidirection、temperature=1.0|
|统计|memory-analysis-algorithm；configs/memory/sf.json 与 bp.json|
|输出|默认 compact，仅阶段汇总、峰值快照及少量运行元数据，不写逐事件 JSONL|

默认 24 个训练进程、24 个 run，组成 12 组 SF/BP 对照。`--runs 5` 时仍是 24 个训练进程，每个进程执行 5 次运行，共 120 个 run；每个 run_i 独立配对汇总，不只选择最后一次。

SF 的 epochs 是逐层训练预算，BP 是端到端预算；这不是等计算量实验。SF 输出隐藏表示并使用虚拟节点目标，BP 末层输出类别数，图与目标结构存在差异。GAT 使用 4 heads，hidden 必须被 4 整除；SF 中每头宽度为 hidden/4，BP 使用现有 PyG GAT 配置，末层分类宽度由类别数决定。保留现有确定性设置，不人为统一或改动模型实现。

## CUDA 服务器命令

在已配置 Python 3.8 / PyTorch / PyG 的 CUDA 环境和仓库根目录运行。正式论文对照应提前准备作者提供的节点划分；本入口沿用原数据加载逻辑，不验证划分文件的来源。

仅打印完整计划与参数，不导入 torch、不创建结果目录、不启动训练：

```bash
cd ~/forwardgnn
conda activate ForwardLearningGNN
bash exp/compare-algorithm-memory/run.sh \
  --datasets Amazon-Photo --architectures GCN SAGE GAT \
  --depths 2 4 8 16 --hidden 128 --runs 1 --epochs 3 \
  --gpu 0 --output results/algorithm-memory-matrix --dry-run
```

之后需要实际执行时，去掉 `--dry-run`：

```bash
bash exp/compare-algorithm-memory/run.sh \
  --datasets Amazon-Photo --architectures GCN SAGE GAT \
  --depths 2 4 8 16 --hidden 128 --runs 1 --seed 100 \
  --epochs 3 --lr 0.001 --patience -1 --val-every 1 \
  --gpu 0 --output results/algorithm-memory-matrix
```

先单独验证原两层配置的命令：

```bash
bash exp/compare-algorithm-memory/run.sh \
  --datasets Amazon-Photo --architectures SAGE --depths 2 \
  --runs 1 --epochs 3 --gpu 0 \
  --output results/algorithm-memory-sage2
```

扩展到五个数据集、五个划分/run 的短预算诊断：

```bash
bash exp/compare-algorithm-memory/run.sh \
  --datasets CitationFull-CiteSeer CitationFull-Cora_ML CitationFull-PubMed Amazon-Photo GitHub \
  --architectures GCN SAGE GAT --depths 2 4 8 16 \
  --hidden 128 --runs 5 --seed 100 --epochs 3 \
  --patience -1 --val-every 1 --gpu 0 \
  --output results/algorithm-memory-matrix-five-datasets
```

该扩展为 120 个训练进程、600 个 run。3 epochs 用于显存筛查，不表示准确率收敛，也不表示覆盖完整训练的所有后期峰值。

## 断点续跑：已有矩阵只补 SAGE 16 层和 GAT

在原输出目录上加 `--resume`，保持原矩阵的完整参数。不要将 architectures/depths 改成子集：调度器会自行查找完整结果并跳过，最终统一生成全矩阵对比。

先预览（只读，不启动训练、不写文件）；每个 case 的 `action` 为 `skip` 或 `run`：

```bash
cd ~/forwardgnn
conda activate ForwardLearningGNN
bash exp/compare-algorithm-memory/run.sh \
  --datasets Amazon-Photo --architectures GCN SAGE GAT \
  --depths 2 4 8 16 --hidden 128 --runs 1 --seed 100 \
  --epochs 3 --lr 0.001 --patience -1 --val-every 1 \
  --gpu 0 --memory-output compact --resume \
  --output results/algorithm-memory-matrix --dry-run
```

确认后实际续跑：

```bash
bash exp/compare-algorithm-memory/run.sh \
  --datasets Amazon-Photo --architectures GCN SAGE GAT \
  --depths 2 4 8 16 --hidden 128 --runs 1 --seed 100 \
  --epochs 3 --lr 0.001 --patience -1 --val-every 1 \
  --gpu 0 --memory-output compact --resume \
  --output results/algorithm-memory-matrix
```

已提供结果的只读检查为跳过 14 个配置，运行 10 个：SAGE 16 层 SF/BP，以及 GAT 2/4/8/16 层 SF/BP。服务器实际行为取决于该目录当前完整结果。

完成判断检查所有 run 的 `complete`、schema、训练参数、归因规则、峰值和阶段表，不仅检查目录或进程退出状态。新旧 full/compact 结果可以一起汇总。任一 run 不完整时重跑该配置的全部 runs；不会从某个 epoch 或某一层恢复训练。

续跑日志及结果放入带时间戳的新文件/`attempt-*` 子目录，保留旧失败结果。原 `plan.json` 保留；本次计划写入 `resume-plan-*.json`，旧进程状态备份为 `process-status-before-*.json`。可多次 `--resume`。GPU、解释器、输出所在路径及输出精简程度可以变化，训练配置必须匹配原计划。

## 精简模式的内容和限制

矩阵入口默认 `--memory-output compact`；单独调用训练入口需显式加 `--memory-analysis-output compact`（并启用 `--memory-analysis`）。

- 保留 `memory-stage-summary.{json,csv,md}`、`algorithm-memory-summary.json`、`algorithm-memory-stages.csv`、算法报告以及 `memory-trace-summary.json`/报告；算法统计需启用算法分析。
- 保留训练/评估峰值、各层各阶段分类汇总和选中峰值处的 storage/变量别名。采样、storage 去重、归因口径不变。
- 不生成 `memory-events.jsonl`、`memory-variables.jsonl`；无法事后逐事件查询变量释放，也无法从原始事件重新归因。需要这些功能时选 `--memory-output full`。
- compact 不支持算子 profiler 模式；普通训练日志和运行参数元数据仍保留。输出大小随层数/峰值快照规模增长，并非固定大小；追踪计算开销仍存在。

精简模式只减少新输出，不自动删除旧的巨大 JSONL，也不会立即腾出磁盘空间。运行前仍需保证结果盘有可用空间。

## 与论文预算的关系

本矩阵实现之前计划中的“骨干与深度显存筛查”，不声称完成论文精度复现。可通过 `--epochs 1000 --patience 100 --val-every 2 --runs 5` 配置更长诊断预算，但深度 8/16 仍是扩展实验。
原作者协议中 BP 分别训练 1/2/3/4 层，SF 训练四层并保存前缀；本工具对每个 depth 独立训练 SF/BP，二者不等价。原协议入口见 [nodeclass-paper-memory](../nodeclass-paper-memory/README.md)。无追踪的准确率/计时复现应独立运行，不能使用本诊断的耗时作为性能结论。

## 输出结构

```text
<output>/
  plan.json
  process-status.json
  pair-status.json
  matrix-report.md
  matrix-periods.csv
  matrix-stages.csv
  Amazon-Photo/GCN/layers-2/
    sf-process.log
    bp-process.log
    training-results/{sf,bp}/...
    sf/run0-.../
    bp/run0-.../
    comparison/run0/
      comparison.md
      comparison-periods.csv
      comparison.csv
      sources.json
      compare.log
```

- `matrix-periods.csv`：dataset / architecture / depth / run_i / algorithm / period，以及该区间的算法观测峰值。
- `matrix-stages.csv`：所有配置的逐层、逐阶段、存储归属和类别分解，包含对应事件。
- 每个 run 原始 JSON 保留峰值 storage/变量别名；完整命令及参数在 plan.json，实际训练日志在各进程 log。
- 汇总保留每个 run，不混合不同随机划分，也不将各层独立峰值相加。此版本不自动生成跨 run 均值/标准差。

```bash
cat results/algorithm-memory-matrix/matrix-report.md
```

只查看训练/评估总量：

```bash
python - <<'PY'
import csv
with open('results/algorithm-memory-matrix/matrix-periods.csv') as f:
    for row in csv.DictReader(f):
        if row['period'] in ('training', 'evaluation'):
            print(row['dataset'], row['architecture'], row['depth'], row['run_i'],
                  row['algorithm'], row['period'], row['algorithm_observed_peak_mib'])
PY
```

## 失败、输出与测量边界

首次运行的输出目录必须新建或为空；已有目录必须显式使用 `--resume`。训练结果隔离在 output 内。默认遇到非零退出（包括 OOM）记录日志并继续其余配置；`--fail-fast` 可提前停止。磁盘空间/配额耗尽时停止调度，修复空间后可续跑；状态文件采用临时文件替换，避免写满时截断上次状态。失败会在终端显示日志尾部。失败不伪装成零，缺少任一侧完整结果的配对不进入数值表，见 pair-status.json。任一配对失败，调度器最终退出码为 1；控制器错误为 2；Ctrl-C 保存进程状态并以 130 退出，不启动后续实验。

主指标是按 storage 去重的训练保留存储观测峰值，不是精确因果额外显存。模型参数/原始数据基线排除；未观测的原生临时空间不归因。评估总量仍包含常驻训练状态。即使排除了模型临时空间，实际 CUDA 执行仍会产生该空间，因此深层模型仍可能 OOM。详见[统计口径](../../docs/algorithm_memory_analysis.md)。

包装脚本可从任何目录启动；相对 `--output` 相对于调用者工作目录。`PYTHON` 可选指定解释器，其余所有参数原样转发。未传 `--output` 会报错，不自动开始默认大矩阵。
