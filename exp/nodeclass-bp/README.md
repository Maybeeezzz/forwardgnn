# 节点分类：BP 性能基线

入口：`exp/nodeclass-bp/run.sh`。

使用普通端到端反向传播训练节点分类模型，作为性能对照。

## 默认实验配置

- 数据集：CitationFull-CiteSeer、CitationFull-Cora_ML、CitationFull-PubMed、Amazon-Photo、GitHub。
- 模型：`GNN-GCN GNN-SAGE GNN-GAT`。
- 层数：1 2 3 4；隐藏宽度 128。
- 每配置 5 runs，seed=100，最多 1000 epochs，学习率 0.001，patience=100，每 2 epochs 验证。
- 前向逐层方法的 epoch 预算按该算法训练循环解释，不能等同于 BP 的计算量。

## 运行

从仓库根目录、激活 ForwardLearningGNN 环境后执行：

```bash
bash exp/nodeclass-bp/run.sh --gpu 0
```

尾部参数原样传给每次 Python 训练命令；脚本默认含 `--overwrite-result`。

## 输出与解读

`results/perf-results/<dataset>/<task>/` 下的结果 JSON 与 stdout 日志。多个脚本共享此结果根，文件名按模型区分；重复运行可能覆盖相同配置。
这是性能实验入口，默认不启用显存分析。使用论文数据划分才能对齐论文协议；此目录本身不表示复现已完成。

## 迁移记录

原入口：`exp/nodeclass/nodeclass-bp.sh`。仅整理路径与启动定位，保留原有实验参数和结果位置。
