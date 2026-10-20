# 实验与工具入口

所有 16 个 shell 脚本统一放在本目录。每个子目录包含一份 `run.sh` 和一份实验说明 `README.md`。环境安装和 CUDA 验证作为辅助入口一并归档。

从仓库根目录运行 `bash exp/<名称>/run.sh`；具体参数、输入要求和输出位置见各目录 README。此次整理不运行实验，不改变训练预算。

|目录|用途|
|---|---|
| [bp-sparse-cuda-matrix](bp-sparse-cuda-matrix/README.md) | 合成图 CUDA 稀疏后端验证矩阵 |
| [compare-algorithm-memory](compare-algorithm-memory/README.md) | SF/BP 算法保留显存矩阵：GCN/SAGE/GAT，2/4/8/16 层 |
| [install-packages](install-packages/README.md) | 实验环境安装 |
| [linkpred-bp](linkpred-bp/README.md) | 链接预测：BP 性能基线 |
| [linkpred-fl-topdown](linkpred-fl-topdown/README.md) | 链接预测：TopDown Forward Learning |
| [linkpred-fl](linkpred-fl/README.md) | 链接预测：Forward Learning |
| [nodeclass-bp-sparse](nodeclass-bp-sparse/README.md) | 节点分类：GCN 稀疏后端 BP 对比 |
| [nodeclass-bp](nodeclass-bp/README.md) | 节点分类：BP 性能基线 |
| [nodeclass-depth-memory](nodeclass-depth-memory/README.md) | 深度扩展显存基准 |
| [nodeclass-ff-label-appending](nodeclass-ff-label-appending/README.md) | 节点分类：标签拼接 Forward-Forward |
| [nodeclass-ff-virtual-nodes](nodeclass-ff-virtual-nodes/README.md) | 节点分类：虚拟节点 Forward-Forward |
| [nodeclass-layer-memory](nodeclass-layer-memory/README.md) | 逐层、逐阶段显存诊断 |
| [nodeclass-paper-memory](nodeclass-paper-memory/README.md) | 论文节点分类预算下的 SF/BP 显存监控 |
| [nodeclass-sf-top2input](nodeclass-sf-top2input/README.md) | 节点分类：Top2Input SingleForward |
| [nodeclass-sf-top2loss](nodeclass-sf-top2loss/README.md) | 节点分类：Top2Loss SingleForward |
| [nodeclass-sf](nodeclass-sf/README.md) | 节点分类：SingleForward |

旧入口已迁移，当前文档已更新；历史 results 中的命令/来源路径作为原始记录保留，不重写。
