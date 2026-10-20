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

输出目录必须新建或为空；不存在静默覆盖或自动续跑。训练结果也隔离在 output 内，不写旧的 algorithm-memory-sf/bp 目录。默认遇到非零退出（包括 OOM）记录日志并继续其余配置；`--fail-fast` 可提前停止。失败不伪装成零，缺少任一侧完整结果的配对不进入数值表，见 pair-status.json。任一配对失败，调度器最终退出码为 1；Ctrl-C 保存进程状态并以 130 退出，不启动后续实验。

主指标是按 storage 去重的训练保留存储观测峰值，不是精确因果额外显存。模型参数/原始数据基线排除；未观测的原生临时空间不归因。评估总量仍包含常驻训练状态。即使排除了模型临时空间，实际 CUDA 执行仍会产生该空间，因此深层模型仍可能 OOM。详见[统计口径](../../docs/algorithm_memory_analysis.md)。

包装脚本可从任何目录启动；相对 `--output` 相对于调用者工作目录。`PYTHON` 可选指定解释器，其余所有参数原样转发。未传 `--output` 会报错，不自动开始默认大矩阵。
