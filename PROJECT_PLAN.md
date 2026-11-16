# ESTR2018 Project Plan — Probability in Deep Learning

**Team:** Chen Diagonal (11155260102), Hau Wing Shing (1155258158)
**Topic:** 用 PyTorch 在 MNIST 上训练神经网络，把随机权重初始化和 mini-batch SGD 的随机性当作概率问题研究。

## 最终交付
- [ ] 代码 + 实验结果（Python / PyTorch / MNIST）
- [ ] 概率分析（初始化分布、SGD 随机过程）
- [ ] Presentation slides：PDF 上传 Gradescope，5 分钟，title slide + 4–5 页

---

## Step 0 — 准备
- [x] 安装 Python + PyTorch，确认能 `import torch`（Python 3.10.0 / torch 2.8.0+cu128 / RTX 5070 Ti 可用）
- [x] 建目录：`train.py`、`results/`、`data/`（`experiments.py`、`plots/` 待建）
- [ ] 两人分工：一人训练/实验代码，一人理论分析与画图，slides 共同审

## Step 1 — 跑通 baseline
- [x] 写 `train.py`：MNIST → MLP (784-256-128-10) → SGD 训练 5 epochs → 打印 test accuracy
- [x] 固定超参（batch 64, lr 0.1），只跑 1 个 seed
- [x] 目标：test accuracy ≥ 97%（实测 **97.76%**，5 epochs）
- **产出：** 能跑通的训练脚本 ✅

## Step 2 — 加入随机种子，看到分布
- [x] 固定 `torch.manual_seed`、`np.random.seed`、dataloader shuffle（已写在 `train.py`）
- [ ] 同一配置跑 5 个 seed（0–4），记录 test accuracy 与 loss 曲线
- **产出：** `seed | accuracy` 表，看到准确率是分布

## Step 3 — 扫 batch size（核心实验）
- [ ] batch size = 16 / 32 / 64 / 128 / 256，每个 5 seeds（共 25 次）
- [ ] 记录：最终准确率、总训练时间、每 epoch loss
- [ ] 图 1：各 batch size 准确率 mean ± std 柱状图
- [ ] 图 2：loss 曲线 mean + 方差阴影带
- **产出：** 项目最核心的两张图

## Step 4 — 扫 learning rate（可选）
- [ ] 固定较好 batch size，lr = 0.1 / 0.01 / 0.001 / 0.0001，各 5 seeds
- [ ] 观察发散 / 收敛慢，用 mean ± std 呈现

## Step 5 — 随机初始化分析（理论部分）
- [ ] 实现四种初始化：uniform、normal、Xavier、He
- [ ] 用前向/反向 hook 记录每层激活与梯度方差
- [ ] 画“方差 vs 层数”曲线，说明梯度消失/爆炸
- [ ] 配公式：方差传播与 fan-in / fan-out 的关系
- **产出：** 各层方差对比图 + 理论解释

## Step 6 — SGD 噪声实验（加分项）
- [ ] 构造有两三个局部极小的简单 loss，用不同 batch size 的 SGD 跑
- [ ] 或直接用 Step 3 数据说明小 batch 方差大但可能最终更好
- **产出：** “噪声帮助跳出坏解”的证据

## Step 7 — 汇总结果
- [ ] 所有实验整理成 CSV + 3–4 张最终图
- [ ] 写 5 条结论，每条对应一张图 + 一句概率解释

## Step 8 — Slides 与提交
- [ ] 5 页 + title：Title → 问题动机 → 方法/实验设置 → 结果图 → 结论
- [ ] 按 5 分钟排练，删掉细节
- [ ] 导出 PDF，上传 Gradescope

---

## 实验矩阵（记录到一张表）

| 实验 | 变量 | 取值 | seeds | 记录 |
|---|---|---|---|---|
| Baseline | — | batch 64, lr 0.1 | 0 | acc, loss |
| Seed 分布 | seed | 0–4 | 5 | acc, loss |
| Batch size | batch | 16/32/64/128/256 | 5 each | acc, time, loss |
| Learning rate | lr | 0.1/0.01/0.001/0.0001 | 5 each | acc, loss |
| 初始化 | init | uniform/normal/xavier/he | 3 each | 各层方差 |

## 原则
1. 每次实验自动存 CSV，不要只留一张图
2. 一次只变一个变量
3. 先能跑再求漂亮

## 排期建议
| 阶段 | 时间 |
|---|---|
| Step 0–2 | 第 1–3 天 |
| Step 3 | 第 4–6 天 |
| Step 5 | 第 7–9 天 |
| Step 4/6 | 第 10–11 天 |
| Step 7–8 | 第 12–14 天 |
