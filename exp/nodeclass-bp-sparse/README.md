# 节点分类：GCN 稀疏后端 BP 对比

入口：`exp/nodeclass-bp-sparse/run.sh`。

比较 GCN 的 SparseTensor 聚合后端与 edge_index 后端。

- 数据集：CitationFull-CiteSeer、CitationFull-Cora_ML、CitationFull-PubMed、Amazon-Photo、GitHub。
- GCN；1/2/3/4 层；hidden=128；5 runs；seed=100；1000 epochs；lr=0.001；patience=100；val-every=2。
- 默认 graph-backend=sparse-tensor。需要 CUDA 与匹配版本的 torch_sparse。

```bash
bash exp/nodeclass-bp-sparse/run.sh --gpu 0
bash exp/nodeclass-bp-sparse/run.sh --gpu 0 --graph-backend edge-index
```

结果写入 `results/bp-sparse-results/`。后端结果命名与统计口径参见[详细说明](../../docs/bp_sparse_experiment.md)。参数通过尾部原样转发。

## 迁移记录

原入口：`exp/nodeclass/nodeclass-bp-sparse.sh`。仅整理路径与启动定位，保留原有实验参数和结果位置。
