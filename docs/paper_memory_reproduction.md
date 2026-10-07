# 沿用作者脚本条件的显存监控入口

这是新增入口 `src/reproduce_paper_memory.py`，启动脚本为
`exp/nodeclass/nodeclass-paper-memory.sh`。原论文模型、训练脚本和已有实验入口均不修改。

## 保持的行为

直接调用原 `train_backprop.py` / `train_forward.py` 的 parse_args、populate_args 和 main。
参数来自作者 nodeclass-bp.sh 和 nodeclass-sf.sh：

- 五个原始数据集；GCN/SAGE/GAT；隐藏宽度 128。
- 五次运行，seed=100，保留 SeedManager 的划分/种子逻辑。
- 最多 1000 epoch，patience=100，val_every=2，lr=0.001；优化器由原实现创建。
- BP 分别训练 1、2、3、4 层；SF 一次训练 4 层并保存前缀结果，与作者脚本一致。
- 保留原验证、checkpoint、内部测试、最终测试行为，包括原 BP 测试时的梯度模式。
- 不覆盖 TF32 或确定性设置，由原入口与环境决定；不安装梯度 hooks。
- 每个原脚本配置一个进程，在进程内执行原五次运行循环，不清缓存或重建 run 循环。

完整节点分类 SF/BP 矩阵为 75 个进程：60 个 BP 配置和 15 个 SF 配置，每个进程五次运行。
SF 保存前缀精度不等于分别训练独立的 1～4 层模型；这里不将四层进程的显存冒充独立浅层显存。
本入口不包含 FF、top-down 或链接预测实验。

唯一输出行为变更是 exp-setting 指向本次 output 下的 original-results，避免覆盖作者已有结果。
GPU 编号可选；默认不提供训练预算覆盖参数。可选子集不会修改单配置训练参数。
预检要求已有五组 train/val/test 划分，不自动生成替代划分；应使用作者公布的数据划分文件。
仅检查文件存在，文件来源仍须由运行者保证。

## 运行

在仓库根目录、原 ForwardLearningGNN 环境中执行：

```bash
# 仅检查计划，不需要 CUDA
bash exp/nodeclass/nodeclass-paper-memory.sh \
  --output results/paper-memory-plan --dry-run

# 单个数据集/算子的正式预算验证，依然五次运行、1000 epoch 上限和早停
bash exp/nodeclass/nodeclass-paper-memory.sh --gpu 0 \
  --datasets Amazon-Photo --architectures GCN \
  --output results/paper-memory-amazon-gcn

# 全部节点分类 SF/BP 配置
bash exp/nodeclass/nodeclass-paper-memory.sh --gpu 0 \
  --output results/paper-memory

# 同条件关闭监控的对照，可比较精度和原训练输出
bash exp/nodeclass/nodeclass-paper-memory.sh --gpu 0 \
  --datasets Amazon-Photo --architectures GCN --monitor none \
  --output results/paper-control-amazon-gcn
```

使用新输出目录；缺少划分时补齐后换新目录。运行时默认保持服务器现有软件环境，不自动安装或升级。
建议按原版本 PyTorch 1.13.1 / PyG 2.2.0 / CUDA 11.7 配置。

## 显存记录

默认 boundaries 模式跟踪原 main、训练、测试、checkpoint 和卷积 forward 的 Python 边界。
调用 GPU 同步后记录 current allocated/reserved 和区间峰值；每条记录包括 run_i，
可定位五次运行及原始层号。模型到层的映射仅保存整数 id，不保留模型张量。
不会新增 no_grad、empty_cache、gc.collect 或梯度 hook。

每个配置目录有 environment.json、stdout.log、memory-events.jsonl、
memory-trace-summary.json、memory-trace-report.md。后两者汇总的是原五次运行进程的监控范围；
分 run 的判断应依据事件中的 run_i，不可将进程汇总当成某一 run。
original-results 下保留原脚本的精度/epoch 等结果格式与单位，BP 和 SF 的 perf 单位沿用原代码。
plan.json 记录实际参数和源码哈希；status.json 保存各子进程退出码。

## “严格复现”的边界

本入口严格沿用公开节点分类脚本的训练参数和控制流程，而非声称已经复现论文数值。
同步和 Python tracing 本身会改变执行时序；不要用插桩耗时评价速度。none 模式完全不安装
显存 tracer，供同环境精度对照，不生成显存报告。

原论文使用 H100；若运行在 L40，硬件仍不一致。软件版本、驱动、运行环境也须记录和核对。
论文未公开足够细致的显存测量边界，因此本工具的 allocator 区间峰值不保证对应论文显存表。
不能通过删除临时峰值或选择最接近论文的读数来宣称复现。

原有逐层工具与此入口的区别：这里没有梯度就绪回调，因此 backward 内部各层的细分更少；
forward 仍能定位到卷积层和源码行，backward 作为原语句区间统计。这是为保持原执行行为而作的取舍。

开发验证：已检查原参数组合、75 个配置、SF/BP 深度方式、CLI、Python 3.8 语法以及源码未修改。
尚未完成真实 CUDA 五次运行或与无监控对照的精度一致性验证。
