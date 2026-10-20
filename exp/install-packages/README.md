# 实验环境安装

入口：`exp/install-packages/run.sh`。

这是环境准备入口，不是训练或显存实验。统一归档在 exp 下便于管理。

创建或复用 Conda 环境 `ForwardLearningGNN`，安装 Python 3.8、PyTorch 1.13.1、torchvision 0.14.1、torchaudio 0.13.1、PyG 2.2.0 和 tqdm。Linux 分支指定 CUDA 11.7；macOS 分支不安装 CUDA runtime。需要已安装 Conda 并能访问软件源。

```bash
bash exp/install-packages/run.sh
```

该脚本会修改指定 Conda 环境，不生成实验结果。脚本结束后，在当前终端另行执行 `conda activate ForwardLearningGNN`。不接收实验参数。

## 迁移记录

原入口：`install/install_packages.sh`。仅整理路径与启动定位，保留原有实验参数和结果位置。
