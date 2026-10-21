# 公共文档

本目录保留跨实验使用的工具说明与统计口径。实验设计、运行协议和结果分析放在对应的 `exp/<实验名>/` 中，完整实验索引见 [exp/README.md](../exp/README.md)。文档中的命令默认从仓库根目录运行。

## 通用工具与口径

- [通用显存分析 utility](layer_memory_profile.md)：启用方式、变量/storage 追踪、阶段汇总和算子分析。
- [模型基线与训练算法保留显存](algorithm_memory_analysis.md)：公共归因规则、峰值统计、测量边界及精简模式。

## 实验专属文档

| 实验 | 设计与运行说明 | 结果分析 |
|---|---|---|
| SF/BP 深度扩展显存 | [设计说明](../exp/nodeclass-depth-memory/sf_depth_memory.md) | [结果报告](../exp/nodeclass-depth-memory/sf_depth_memory_results.md) |
| 论文预算显存复现 | [运行协议](../exp/nodeclass-paper-memory/paper_memory_reproduction.md) | [统计与差异分析](../exp/nodeclass-paper-memory/paper_memory_results.md) |
| GCN-BP 稀疏后端对照 | [实验说明](../exp/nodeclass-bp-sparse/bp_sparse_experiment.md) | — |
| CUDA 稀疏后端矩阵 | [实验说明](../exp/bp-sparse-cuda-matrix/bp_sparse_cuda_matrix.md) | — |
| SF/BP 算法显存矩阵 | [实验说明](../exp/compare-algorithm-memory/README.md) | [SF 峰值分析](../exp/compare-algorithm-memory/sf-peak-analysis.md) |
| 网络×算法显存矩阵 | [协议与 CUDA 命令](../exp/network-algorithm-memory/README.md) | 尚未执行 CUDA 实验 |
