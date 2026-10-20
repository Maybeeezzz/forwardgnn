# 深度扩展显存基准

入口：`exp/nodeclass-depth-memory/run.sh`。

SF/BP 深度扩展显存测量，观察层数增长时的占用。

默认：Amazon-Photo/GitHub；GCN/SAGE；2/4/8/16/32/64 层；hidden=128；3 runs；20 epochs；patience=-1。

## 运行

需要已配置 CUDA/PyTorch/PyG 环境，按下文具体实验补齐数据划分或 torch_sparse。所有尾部参数原样转发给 Python 入口。

```bash
bash exp/nodeclass-depth-memory/run.sh --gpu 0 --output results/depth-memory
```

该包装脚本不改变调用者工作目录；相对 `--output` 相对于当前终端目录，建议在仓库根目录运行。输出目录、各报告与参数细节见[详细说明](../../docs/sf_depth_memory.md)。

可使用 `--dry-run` 生成计划；这不是已经执行的实验。`PYTHON` 可选，用于指定 Python 可执行文件。

这些入口统计的是各自定义的进程/边界显存，不应直接等同于算法保留存储；后者使用 [compare-algorithm-memory](../compare-algorithm-memory/README.md)。

## 迁移记录

原入口：`exp/nodeclass/nodeclass-depth-memory.sh`。仅整理路径与启动定位，保留原有实验参数和结果位置。
