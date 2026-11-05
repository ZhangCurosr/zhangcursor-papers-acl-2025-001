---
title: "ATLANTIS-Weak-to-Strong-Learning-via-Importance-Sampling"
source: https://aclanthology.org/2025.acl-long.52.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:39:36"
field: "大语言模型微调与对齐"
keywords: ["监督微调", "重要性采样", "弱到强学习", "数据分布偏差", "影响权重", "知识理解", "人类偏好"]
innovations: ["首次将重要性采样引入SFT以bridging训练数据分布与最优分布的差距", "利用小模型与参考模型的概率差估计采样比率，提出影响权重机制", "证明influence weights可在不同模型结构间迁移，具备跨架构适应性"]
benchmarks: ["Open LLM Leaderboard 2", "TruthfulQA", "MT-Bench", "AlpacaEval", "Arena-Hard-Auto"]
---

# 论文速读：ATLANTIS-Weak-to-Strong-Learning-via-Importance-Sampling

## 一句话总结
论文提出了一种名为ATLANTIS的监督微调(SFT)新技巧，通过引入重要性采样机制，利用小模型与参考模型之间的概率差距来估计采样比率，从而弥补训练数据分布与现实世界最优数据分布之间的鸿沟，在知识理解与人类偏好两个维度上均带来稳定且显著的性能提升。

## 研究问题与动机
- **核心问题**：当前SFT训练数据（人工标注或模型生成）的分布$p_d$与理想的最优分布$p^*$存在系统性差距，这种分布偏差限制了大语言模型的容量上限。
- **现有方法不足**：传统数据选择方法（如丢弃低质量样本）虽能缓解问题，但未直接针对分布差距本身进行建模与修正。
- **理论难点**：最优分布$p^*$无法显式采样也无法显式计算概率，使得基于重要性采样的优化目标难以直接实现。
- **直觉来源**：受weak-to-strong generalization启发，假设小模型与参考模型之间的概率变化方向可迁移至大模型的优化过程。

## 核心贡献（创新点）
- **首次将重要性采样引入SFT**：利用重要性采样从可采样的训练数据分布$q$中估计不可直接采样的最优分布$p^*$的期望， bridging 分布鸿沟。
- **提出基于概率差的采样比率估计**：引入小模型$p_b^S$和参考模型$p_r$，通过proxy-tuning假设下的比例关系$\frac{p^*(y|x)}{p_r(y|x)} \propto \frac{p_b^L(y|x)}{p_b^S(y|x)}$间接估计采样比率。
- **定义"影响权重"(influence weight)**：将额外项$\frac{p_r(y|x)}{p_b^S(y|x)}$视为每个训练样本的重要性权重，对SFT优化方向进行动态调整。
- **证明跨模型结构的可迁移性**：实验验证ATLANTIS可灵活适配不同架构的大模型，且influence weights可在不同结构间迁移（ATLANTIS-cross）。
- **与现有方法高度兼容**：可与数据选择方法(IFD、superfiltering)结合使用，在多数情况下带来进一步提升。

## 方法详解
- **问题建模**：标准SFT损失为$\mathcal{L}(\theta) = -\mathbf{E}_{x\sim q, y\sim p_d}\log p_\theta(y|x)$，目标是最小化模型输出分布$p_\theta$与最优分布$p^*$之间的差距。
- **重要性采样转换**：将期望改写为$\mathcal{L}(p_b^L) = -\mathbf{E}_{x\sim q, y\sim p_r}[\frac{p^*(y|x)}{p_r(y|x)}\log p_b^L(y|x)]$，其中$\frac{p^*(y|x)}{p_r(y|x)}$为采样比率。
- **采样比率估计**：基于proxy-tuning假设，假设大小模型的分布变化方向成比例，推导出$\frac{p^*(y|x)}{p_r(y|x)} \propto \frac{p_b^L(y|x)}{p_b^S(y|x)}$。
- **最终损失函数**：$\mathcal{L}(p_b^L) \propto -\mathbf{E}_{x\sim q, y\sim p_r}[\frac{p_b^L(y|x)}{p_b^S(y|x)}\log p_b^L(y|x)]$，相比标准SFT增加了一个衡量大小模型分布差距的修正项。
- **算法流程**：预计算每个样本的影响权重$W_i = \frac{p_b^L(y_i|x_i)}{p_b^S(y_i|x_i)}p_r(y_i|x_i)$，在训练循环中对小批量样本使用加权交叉熵损失进行参数更新。
- **与proxy-tuning的关系**：proxy-tuning直接将logits差值加到大模型上($s_r + s_b^L - s_b^S$)，而ATLANTIS将分布变化转化为样本权重进行SFT训练。

## 实验与结果
- **训练数据集**：OpenHermes-2.5，包含合成对话数据。
- **评测基准**：知识理解方面使用Open LLM Leaderboard 2（含BBH、GPQA、IFEval、MATH-Hard、MMLU-Pro、MUSR六个子任务）和TruthfulQA；偏好方面使用MT-Bench、AlpacaEval、Arena-Hard-Auto。
- **主实验结果（Table 2）**：
  - Llama2-13B：OpenLLM Leaderboard 2从32.37提升至33.50，TruthfulQA MC1从36.23提升至53.36，MT-Bench Pairwise从19.06提升至21.56，AlpacaEval Easy从6.81提升至7.80。
  - Qwen2-7B：OpenLLM Leaderboard 2从36.22提升至38.19，TruthfulQA MC1从35.37提升至51.45，MT-Bench Pairwise从25.63提升至26.25。
  - Gemma2-9B：OpenLLM Leaderboard 2从33.51提升至33.86，TruthfulQA MC1从36.47提升至52.85，MT-Bench Pairwise从20.31大幅提升至27.50。
- **强结果与提升幅度**：Gemma2-9B在MT-Bench Pairwise上提升最为显著（+7.19绝对值）；Llama2-13B在TruthfulQA MC1上提升17.13个百分点。
- **与数据选择方法结合（Table 3）**：ATLANTIS与IFD/superfiltering结合后在Qwen2上进一步提升，但Gemma2的数据选择效果不明显；最优样本率约为0.10。
- **更大规模模型（Table 5）**：Qwen2-72B的MT-Bench Pairwise从31.88提升至36.88（+5.0）；Gemma2-27B从26.56提升至32.50（+5.94）。
- **跨结构实验（Table 4）**：ATLANTIS-cross在不同模型结构间迁移influence weights时，多数情况下仍可获得可观提升，但Qwen2在MT-Bench Pairwise上出现下降。

## 相关工作脉络
- **Supervised Fine-tuning**：从prompt tuning、prefix tuning到instruction tuning的经典工作（Brown et al. 2020; Li & Liang 2021; Wei et al. 2021; Ouyang et al. 2022），本文在此基础上引入重要性采样进行优化方向修正。
- **Proxy-tuning (Liu et al. 2024)**：同样基于"大小模型分布变化方向成比例"的假设，但proxy-tuning直接将logits差值叠加到大模型上，而ATLANTIS将其转化为样本权重用于SFT训练；实验表明ATLANTIS显著优于proxy-tuning。
- **IFD (Li et al. 2023a)**：使用微调后模型与base模型的perplexity比值作为数据选择分数；本文与之形成对比，证明两者可结合使用且互补。
- **Superfiltering (Li et al. 2024a)**：弱到强的数据过滤方法；本文表明ATLANTIS可与该方法兼容，且在数据量较少时优势更明显。
- **Weak-to-Strong Generalization (Burns et al. 2023)**：本文的核心灵感来源，研究如何利用弱监督信号激发强模型能力。
- **Data Selection for Instruction Tuning**：模型无关或模型感知的打分方法（Song et al. 2024; Chen et al. 2023; Bukharin & Zhao 2023; Du et al. 2023），本文从优化方向角度提供了新的数据影响度量方式。

## 局限性与未来方向
- **模型结构依赖**：虽然influence weights可跨结构迁移，但从同一模型系列获取小模型的效果更佳，限制了在非同源模型场景下的应用。
- **额外计算开销**：引入参考模型和小模型带来额外推理成本，尽管参考模型较小、开销可接受，但仍需权衡。
- **模型间差异的影响**：不同模型的参数分布差异可能导致ATLANTIS-cross在某些场景下失效（如Qwen2在MT-Bench上的性能下降）。
- **样本量敏感性**：当训练样本过多(0.15)时可能引入低质量样本噪声，过少(0.05)则不足以有效训练。
- **未来方向**：探索与更多SFT技术的结合方式，优化跨模型结构的影响力权重迁移策略，进一步降低额外计算开销。

## 研究启发与可借鉴点
- **重要性采样视角的SFT改进**：将重要性采样框架引入SFT的思路具有普适性，可迁移到其他对齐或生成任务中，通过重加权解决数据分布偏差问题。
- **基于概率差的样本影响力度量**：利用base模型与参考模型的概率差来衡量样本重要性，提供了一种无需额外标注或复杂计算的轻量级数据评估方法。
- **跨模型结构迁移的可行性验证**：ATLANTIS-cross实验证明了不同架构模型间分布变化规律的相似性，为跨模型的知识迁移和权重共享提供了新思路。
- **与数据选择方法的正交互补**：ATLANTIS与IFD、superfiltering等数据选择方法结合后效果持续提升，提示未来可将分布修正与数据筛选形成联合优化框架。
- **实验设计的全面性**：从不同模型系列(Llama2/Qwen2/Gemma2)、不同规模(7B/13B/27B/72B)、不同评估维度(知识/偏好)多方位验证方法有效性，值得借鉴。

## 关键术语表
**Importance Sampling（重要性采样）**：一种蒙特卡洛估计方法，通过在易采样的分布$q$上采样来估计难采样分布$p$下的函数期望，通过权重$\frac{p(x)}{q(x)}$进行校正。
**Influence Weight（影响权重）**：本文提出的额外项$\frac{p_b^L(y|x)}{p_b^S(y|x)}$，用于衡量训练样本对大模型SFT优化方向的影响程度，概率提升越大的样本获得更高权重。
**Reference Model（参考模型）**：与大模型同系列但已完成SFT的模型$p_r$，用于估计最优分布与训练数据分布之间的差距。
**Proxy-tuning（代理微调）**：Liu et al. (2024)提出的方法，通过将小模型与参考模型的logits差值直接加到大模型上来实现知识迁移，本文与其共享核心假设但实现方式不同。
**Optimal Distribution（最优分布）**：理论上的理想训练数据分布$p^*$，能够完全覆盖真实世界的所有自然语言模式和知识内容。
**Weak-to-Strong Generalization（弱到强泛化）**：Burns et al. (2023)提出的概念，指利用弱监督信号或弱模型的信息来激发强模型潜力的现象。
**OpenHermes-2.5**：由Teknium构建的大规模合成对话数据集，用作本文的SFT训练数据源。
**SFT（Supervised Fine-tuning）**：监督微调，指在预训练模型基础上使用标注数据进行后续训练以对齐人类指令或特定领域知识的技术。

## 可复现要素
- **数据集**：OpenHermes-2.5（公开可用）
- **代码/权重**：论文未明确提及代码开源状态，需查阅作者主页或仓库
- **关键超参数**：
  - 训练步数：≤13B模型60K步，27B模型30K步，72B模型15K步
  - Batch size：≤13B为2，27B/72B为1
  - Learning rate：≤13B为2e-5，27B为5e-6，72B为5e-7
  - Warmup steps：≤13B为6K，27B为3K，72B为1.5K
  - Optimizer：AdamW
  - Scheduler：linear warmup + linear decay
  - 随机种子：所有实验使用同一随机种子
