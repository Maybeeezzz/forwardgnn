# 合成图 CUDA 稀疏后端验证矩阵

入口：`exp/bp-sparse-cuda-matrix/run.sh`。

合成图上比较 edge_index 与 SparseTensor 后端，检查数值一致性和 CUDA 显存。

默认：GCN/SAGE/GIN/GAT；节点分类/链接预测；uniform/hub 拓扑；2048 节点、32768 边；输入/隐藏 64；warmup=2、steps=10、repeats=3。

## 运行

需要已配置 CUDA/PyTorch/PyG 环境，按下文具体实验补齐数据划分或 torch_sparse。所有尾部参数原样转发给 Python 入口。

```bash
bash exp/bp-sparse-cuda-matrix/run.sh --gpu 0 --output results/bp-sparse-cuda-matrix
```

该包装脚本不改变调用者工作目录；相对 `--output` 相对于当前终端目录，建议在仓库根目录运行。输出目录、各报告与参数细节见[详细说明](../../docs/bp_sparse_cuda_matrix.md)。

这些入口统计的是各自定义的进程/边界显存，不应直接等同于算法保留存储；后者使用 [compare-algorithm-memory](../compare-algorithm-memory/README.md)。

## 迁移记录

原入口：`exp/bp-sparse-cuda-matrix.sh`。仅整理路径与启动定位，保留原有实验参数和结果位置。
