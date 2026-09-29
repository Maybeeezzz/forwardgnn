# GCN-BP SparseTensor 对照实验

新增入口 `src/train_backprop_sparse.py` 复用作者的 `build_model`、
`NodeClassificationTrainer`、`SeedManager` 和 `ResultManager`。
仅替换传给官方 PyG GCN 的图表示。原 `train_backprop.py` 和原实验脚本不变。

## 改动与协议

- `edge-index`：原 Tensor COO 边索引。
- `sparse-tensor`：`torch_sparse.SparseTensor(row=target, col=source)`，显式设置
  `(num_nodes, num_nodes)` 保留孤立节点。每次 run 在 CPU 构造一次，训练循环中复用。
  GCNConv 根据输入类型选择其官方 fused `message_and_aggregate` / SpMM 路径。
- 使用原 BP 的原始图，不添加 SF 虚拟节点。GCN 输出维度仍为类别数。
- 支持节点分类 GCN；不将这项 SpMM 实验外推至 GAT。
- 默认矩阵与作者 `exp/nodeclass/nodeclass-bp.sh` 的 GCN 子集一致：
  CitationFull-CiteSeer、CitationFull-Cora_ML、CitationFull-PubMed、Amazon-Photo、GitHub；
  1–4 层；每项五个划分；hidden=128；最多 1000 epoch；Adam lr=0.001、
  weight_decay=0.0005；每 2 epoch 验证；patience=100。
- 参数 seed=100 沿用作者 SeedManager，实际五个 seed 为
  10100、10103、10106、10109、10112。
- 必须预先放置作者发布的节点划分。缺少划分时退出，不自动生成替代划分。
  划分来源：https://github.com/NamyongPark/forwardgnn-datasplits 。

## 运行

使用作者 `install/install_packages.sh` 所定义的环境（Python 3.8、PyTorch 1.13.1、
CUDA 11.7、PyG 2.2.0），并确保安装匹配 PyTorch/CUDA 的 `torch_sparse`。
应记录实际 GPU 型号；不同 GPU 的内存和时间不能直接解释为论文复现误差。

从项目根目录运行完整 GCN 矩阵：

```bash
conda activate ForwardLearningGNN
bash exp/nodeclass/nodeclass-bp-sparse.sh --gpu 0
```

在同一环境运行配对的 edge_index 基线（结果文件名区分 backend）：

```bash
bash exp/nodeclass/nodeclass-bp-sparse.sh --gpu 0 --graph-backend edge-index
```

单配置、小预算检查（不能当作论文精度结果）：

```bash
cd src
python train_backprop_sparse.py --model GNN-GCN --dataset CitationFull-CiteSeer \
  --num-layers 2 --num-hidden 128 --num-runs 1 --epochs 10 \
  --seed 100 --lr 0.001 --val-every 2 --patience 100 \
  --exp-setting bp-sparse-smoke --gpu 0
```

同一配置已有结果会复用；更换训练参数需使用新 `--exp-setting`，或显式指定
`--overwrite-result`。默认要求 CUDA；`--gpu -1` 可用于 CPU 正确性检查，但显存字段为 null。
此入口不提供 MPS 的 torch_sparse 后端。

## 结果与测量口径

结果写入 `results/<exp-setting>/<dataset>/node-class/`，保留作者的
`perf`、`train_time`、`train_epochs`、`best_val_epoch`、`run_i`、`run_seed` 和参数字段。
其中 epoch 字段沿用作者零起始编号，`best_val_epoch=-1` 也沿用原 BP 记录方式。
每配置另保存 `summary-<backend>-GNN-GCN-L<depth>.json`，报告 accuracy 均值及
总体标准差（ddof=0），并列出原始逐次运行文件。

新增记录包括训练期间 CUDA peak allocated / reserved（字节及 MiB）、
训练加测试期间峰值、同步后的训练耗时、图构造耗时、软件版本及 GPU 型号。
峰值在每次 run 的 GPU 数据传输前重置，包括训练、验证和早停 checkpoint；
训练加测试峰值另包含作者 test()。CPU 数据加载和 CPU 稀疏图构造不计入 CUDA 峰值。
保留原 trainer 的 train_time，同时记录同步后的 training_wall_seconds。
不注册性能 hooks，不在更新循环清理缓存。

上述协议与作者公开 GCN-BP 脚本对应，但新增 CUDA allocator 显存口径并未证实
与论文显存表的统计边界完全相同，因此应先比较本入口的两种 backend。
默认没有 warm-up，首次 CUDA 调用的开销可能包含在耗时内。CPU/MPS 读数不能替代 CUDA。

本改动尚未产生 GPU 实验结果，也未验证完整训练精度一致性；应先做相同初始化的
小预算输出/梯度检查，再运行完整矩阵评估收敛精度。保持确定性设置与原作者一致；
若某版本的 sparse CUDA 算子不支持确定性，记录报错，不静默放宽设置。
