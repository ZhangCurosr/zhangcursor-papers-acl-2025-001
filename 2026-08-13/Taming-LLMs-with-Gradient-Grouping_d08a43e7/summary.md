---
title: "Taming-LLMs-with-Gradient-Grouping"
source: https://aclanthology.org/2025.acl-long.164.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:22:47"
field: "大语言模型训练优化"
keywords: ["LLM 优化器", "梯度分组", "参数高效微调", "AdamW", "LoRA", "自适应学习率"]
innovations: ["提出SGG优化器包装器，通过在线聚类+簇级MDA缩放实现'缩放而非替换'的自适应学习率校准", "首次使低秩预训练(LoRA+SGG)达到与全参数训练相当的C4 perplexity水平", "基于全局MAD的跨层学习率homogenization机制，显著提升大batch/LR下的训练鲁棒性"]
benchmarks: ["C4 pre-training", "GLUE", "Commonsense Reasoning (8 tasks)", "DPO (ultrafeedback_binarized)", "LLaVA-v1.5 MLLM benchmarks"]
---

# 论文速读：Taming-LLMs-with-Gradient-Grouping

## 一句话总结
提出 **SGG（Scaling with Gradient Grouping）**，一种可即插即用的优化器包装器，通过在每一层对梯度统计量进行在线聚类并施加簇级缩放，在保留参数级自适应学习率的同时引入群体约束，从而提升 LLM/MLLM 训练的稳定性、收敛速度和最终性能。

## 研究问题与动机
- **问题**：主流自适应优化器（如 AdamW）依赖参数级一阶/二阶矩统计，显存开销大；在资源受限场景下，参数高效微调（PEFT，如 LoRA）虽能省内存，但相比全参数训练存在明显性能损失（Table 4）。
- **不足一**：现有低资源优化器（如 Adam-mini、GaLore、CAME 等）依赖启发式先验或预定义分组/矩阵分解，容易丢失关键优化信息，跨任务效果不稳定（Table 5）。
- **不足二**：LLM 各层（如 Attention 与 MLP）呈现异构优化行为，但现有方法要么全局共享学习率，要么用静态预定义分组，无法捕获层内与层间动态差异。
- **动机**：通过在线动态分组+簇级缩放，在不替换参数级自适应学习率的前提下引入群体约束，兼顾"个性化适应"与"集体优化行为"，同时实现零架构改动、零额外 GPU 显存（CPU 存储索引即可）的无缝集成。

## 核心贡献（创新点）
1. **提出 SGG 优化器包装器**：通过在线聚类每层动量向量并施加簇级缩放来校准参数级学习率，平衡参数级动态与集体优化行为。与 Adam-mini（预定义静态分组+直接替换为组均值学习率）的本质区别在于"缩放而非替换"，保留层内参数级差异信号。
2. **设计基于 MDA 的簇级缩放策略**：以层内动量对层均值的 Median of Deviation to Average（MDA）为指标，再与全局 MDA 做比值构造缩放因子 $S_l^t[c]$，实现跨层/簇的学习率 homogenization。区别于 CAME 的 NMF 近似、Adam-mini 的组内均值，本工作引入全局视角抑制发散。
3. **验证 SGG 首次使低秩预训练达到全参数水平**：LoRA+SGG 在 C4 预训练中 1B 模型 PPL 从 19.21 降至 14.73（降幅 23.2%），60M 模型从 34.99 降至 30.62，首次弥合低秩与全参数训练的性能鸿沟。
4. **提供 CPU/GPU/Hybrid 多实现**：将聚类索引和缩放因子存储于 CPU，对 GPU 显存无额外开销（Table 3），训练时间增幅仅约 1.8%（GPU）/ 8.2%（CPU），工程友好。

## 方法详解
- **基础流程（Algorithm 1）**：在标准梯度计算→动量估计（EMA $\beta_1$）→自适应学习率估计（EMA $\beta_2$）之后，SGG 额外执行聚类分配和缩放更新，最终参数更新仍为 $\theta_l^t = \theta_l^{t-1} - \hat{\alpha}_l^t \cdot m_l^t$。
- **在线聚类（GradCluster）**：每 $T$ 步（默认 $T=500$，约占总迭代 5%），将当前层 $l$ 的动量向量 $m_l^t$ 展平后做 mini-batch K-means 聚成 $K$ 个簇（默认 $K=3$），输出簇索引 $\mathcal{C}_l^t$。
- **簇级缩放（ScaleUpdate）**：
  - 层内 MDA：$\mathcal{D}_{l,c}^t = \mathrm{median}\big(|m_l^t \cdot \mathcal{C}_l^t[c] - \mathrm{mean}(m_l^t)|\big)$，衡量簇 $c$ 相对于层均值的典型偏离。
  - 全局 MDA：$\mathcal{D}^t = \mathrm{median}_{l,c}(\mathcal{D}_{l,c}^t)$，作为模型尺度的鲁棒参考。
  - 缩放因子：$S_l^t[c] = \frac{\mathcal{D}^t}{\mathcal{D}_{l,c}^t + \epsilon}$，$\epsilon=10^{-8}$；再将 $S_l^t[c]$ 夹到 $[0.1, 10]$。
  - EMA 平滑更新：$S_l^t[c] \leftarrow \beta_3 \cdot S_l^{t-1}[c] + (1-\beta_3) \cdot \frac{\mathcal{D}^t}{\mathcal{D}_{l,c}^t+\epsilon}$，默认 $\beta_3=0.99$。
- **最终有效学习率**：$\hat{\alpha}_l^t = \alpha_l^t \cdot S_l^t[\mathcal{C}_l^t]$，即原始自适应学习率乘以所属簇的缩放因子。
- **实现细节**：不使用 NMF/Low-rank 投影，仅增加 $K \ll mn$ 个整数索引和标量缩放值；对 norm/bias 等标量参数不施加 SGG 缩放（因其无低秩特性且对 scale 敏感）。

## 实验与结果
- **数据集与任务**：C4 预训练（LLaMA 60M~1B）、GLUE SFT（RoBERTa-base）、8 个 Commonsense Reasoning PEFT（LLaMA-7B）、DPO 偏好对齐（Qwen2.5-0.5B）、MLLM 评测（LLaVA-v1.5-7B，GQA/VizWiz/SciVQA/MME/MMBench 等 9 项）。
- **基线**：Adam、NAdam、RAdam、LAMB、Adan、Muon、Adam-mini、Adafactor、Low-Rank、CAME、APOLLO、LoRA、ReLoRA、GaLore、DoRA、Fira 等。
- **核心结果**：
  - **C4 全参数预训练**：Adam+SGG 在 1B 模型上 PPL 14.30（vs Adam 15.56，-8.1%）；在 60M 上 30.31（vs 34.06，-10.9%）。全面超越 Adam-mini、Adafactor、CAME、APOLLO。
  - **C4 低秩预训练**：LoRA+SGG 在 1B 上 PPL 14.73，首次追平 Adam 全参数 14.56；相对 LoRA 基线（19.21）降低 23.2%（最大提升达 10.30% @ 130M）。
  - **GLUE SFT**：AdamW+SGG 平均 +1.00%（Full-rank），rank-4 LoRA 上平均 +1.27%，MRPC +1.35%、MNLI rank-4 +1.36%。
  - **PEFT Commonsense**：LLaMA-7B + LoRA+SGG 平均 +2.9%（OBQA 最高 +4.2%），优于 Prefix/Series/Parallel/DoRA/GaLore/Fira。
  - **DPO**：Qwen2.5-0.5B + LoRA+SGG 准确率 72.02%，超 AdamW+LoRA 基线 +1.80%，且超过 AdamW 全参数（71.85%）。
  - **MLLM**：LLaVA-v1.5-7B AdamW+SGG 平均 +0.9%（VizWiz +2.4%）；LoRA+SGG 平均 +1.0%；Q-LoRA+SGG 平均 +0.6%。
  - **鲁棒性**：图 5 显示，在 batch size 128~4096、LR 1e-1~1e-5 的极端组合下，Adam+SGG 验证 loss 显著稳定，克服 Adam 的 surge 现象。
- **开销**：GPU 训练时间仅增 1.8%（+2h/110h），GPU 显存增 4.3G（CPU 版本显存零增，时间增 8.2%）。

## 相关工作脉络
- **Adam-mini (Zhang et al., 2024)**：按预定义层类型（Attention/MLP）静态分组共享组内均值学习率。SGG 在此基础上引入层内动态聚类 + 非替换式缩放，保留参数级变异信号。
- **GaLore (Zhao et al., 2024a)**：对梯度做低秩投影压缩优化器状态。SGG 不依赖 NMF/低秩投影，而是基于梯度统计分布动态分组并缩放。
- **CAME (Luo et al., 2023)**：用 NMF 近似二阶矩压缩显存。SGG 无矩阵分解，仅增加 $O(K)$ 标量索引，兼容性更强。
- **APOLLO (Zhu et al., 2024a)**：SGD-like 低显存优化器。SGG 可作为其 wrapper，进一步提升 APOLLO+SGG 在 1B C4 上 PPL 至 13.95。
- **DoRA / LoRA+ / LoRA-XS**：参数高效微调的变种。SGG 与这些 PEFT 方法正交可组合（Table 6、7、8 分别验证了 LoRA/DoRA/Q-LoRA+SGG 的一致性增益）。
- **LARS / SignSGD**：层级自适应与符号梯度思路。SGG 的 MDA 缩放受到 LARS（$L_2$-norm 缩放）启发，但以中位数偏差替代norm，并引入全局一致性约束。

## 局限性与未来方向
- **分组策略单一**：目前仅使用 mini-batch K-means 在线聚类，未探索更精细的聚类、启发式静态划分或可学习分组函数。
- **计算开销**：CPU 卸载虽避免 GPU 显存压力，但在线聚类本身仍有开销（尤其是大规模模型的频繁重聚类），在资源受限场景仍需轻量化分组方案。
- **评估范围**：主要覆盖 LLaMA/RoBERTa/Vicuna 等架构；对于 MoE、超大规模模型（>7B 全参数预训练）的泛化性需进一步验证（作者已在 Limitations 中提及）。
- **超参敏感性**：$K$ 和 $T$ 在小模型上有较好鲁棒性，但在更大模型或更长训练时长下的最优设定仍需探索。

## 研究启发与可借鉴点
- **"缩放而非替换"的设计哲学**：不剥夺参数级自适应学习率，而是在其基础上叠加群体约束。这一思想可迁移到任何需要平衡"个体适应"与"集体稳定性"的优化场景（如强化学习、多模态训练）。
- **MDA 跨层级归一化思路**：用全局中位数偏差与层内簇偏差之比构造缩放因子，是一种轻量而有效的训练 homogenization 机制，可推广至其他异构网络架构（如 ResNet、Transformer-MLP 混合结构）。
- **CPU 存储索引的零显存 wrapper 模式**：将额外优化状态（聚类索引、标量因子）放 CPU、GPU 只接收乘子，是通用低开销插件范式，值得复用于其他 optimizer wrapper 开发。
- **SGG × PEFT 的即插即用增益**：LoRA/DoRA/Q-LoRA 搭配 SGG 后均获得额外提升，提示优化器设计可与微调方法正交组合，后续可探索 SGG 与 Fira、Muon 等新方法的结合。
- **鲁棒性 benchmark 设计**：Figure 5 的 LR×Batch-size 热力图分析方式可直接借鉴到团队后续的优化器论文中，作为展示稳定性的标准可视化手段。

## 关键术语表
- **SGG (Scaling with Gradient Grouping)**：一种即插即用的优化器包装器，通过对每层梯度统计量动态聚类并施加簇级缩放来校准参数级学习率。
- **MDA (Median of Deviation to Average)**：以层内均值为中心、取中位数绝对偏差的统计量，用于刻画参数簇的训练动力学偏离程度。
- **PEFT (Parameter-Efficient Fine-Tuning)**：通过仅训练少量参数（如 LoRA 低秩矩阵）实现大模型适配的技术路线。
- **Adam-mini**：按预定义层类型静态分组、组内共享均值为学习率的低显存优化器，是 SGG 的主要对比基线之一。
- **GaLore (Gradient Low-Rank Projection)**：对梯度做低秩投影以压缩优化器状态的显存高效优化器。
- **Surge 现象**：在大 batch size 和高 LR 联合扩展时 Adam 类优化器出现的训练 loss 骤升/不稳定现象。
- **CAME**：基于 NMF 近似二阶矩记忆的高效优化器，常作为低显存基线与 SGG 对比。
- **Q-LoRA**：将 LoRA 与 8-bit 量化结合的参数高效微调方法，SGG 在其上仍可带来小幅增益。

## 可复现要素
- **数据集**：C4 (en subset)、GLUE、8 个 Commonsense Reasoning 数据集、ultrafeedback_binarized（DPO）、LLaVA-v1.5-mix665K、VQA 系列（GQA/VizWiz/TextVQA/SciVQA/VQAv2）、MMBench/MMBench-CN/POPE/SEED/MME 等。论文未明确声明数据集公开状态，但均为公开基准。
- **代码**：论文声称在 PyTorch 中实现并与主流优化器兼容，**论文未提供开源链接**，附录说明了 sklearn MiniBatchKMeans 的使用方式与超参设置。
- **权重**：使用 LLaMA、RoBERTa-base、Vicuna-v1.5-7B、CLIP-L-336px、Qwen2.5-0.5B 等公开预训练权重，论文未提供额外新权重。
- **关键超参**：簇数 $K=3$（可在 $\{2,3\}$ 间调），重聚类间隔 $T=500$（约占总迭代 5%），EMA 衰减 $\beta_3=0.99$，缩放因子夹限 $[0.1, 10]$，$\epsilon=10^{-8}$。
