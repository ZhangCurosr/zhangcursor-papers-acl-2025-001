---
title: "LADM-Long-context-Training-Data-Selection-with-Attention-bas"
source: https://aclanthology.org/2025.acl-long.154.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:10:05"
field: "长上下文大语言模型"
keywords: ["Long-context", "Data Selection", "Attention Mechanism", "Large Language Models", "Pre-training", "Dependency Measurement"]
innovations: ["提出基于注意力分数累积的三级依赖度量体系（PFS/AFS/CDS）", "利用轻量模型作为 Long Attention Calculator 高效筛选高质量长上下文数据", "仅用1B tokens筛选数据即超越随机采样使用2B+ tokens的性能"]
benchmarks: ["LongBench", "Needle-in-the-Haystack", "Proof-Pile Perplexity"]
---

# 论文速读：LADM-Long-context-Training-Data-Selection-with-Attention-bas

## 一句话总结
本文提出了长上下文训练数据选择框架 **LADM**，通过注意力机制的内在检索能力度量长上下文的依赖关系，从而从大规模多域预训练语料中高效筛选出高质量长上下文数据，仅需 1B tokens 的持续训练即可显著提升 LLM 在多类长上下文任务上的性能。

## 研究问题与动机
- **核心问题**：现有大语言模型（LLM）可通过持续训练配备的长上下文数据，但缺乏有效的**质量度量方法**来从海量语料中筛选高质量样本。
- **现有方法不足**：已有方法（如 ProLong）将长上下文切片为独立片段，仅通过片段间的 delta perplexity 衡量依赖强度，忽略了完整上下文中的内在结构与长程依赖关系，导致评估不准确。
- **关键洞察**：拼接的短上下文样本（缺乏跨段依赖）训练的模型在远距离检索任务上表现显著下降（Table 1），而完整长上下文样本能激发注意力机制对远距离 token 的权重，证明**上下文依赖是衡量数据质量的关键指标**。

## 核心贡献（创新点）
1. **提出 LADM 框架**：通过注意力分数累积度量 span 间依赖，实现从大规模预训练语料中高效筛选高质量长上下文数据。与 ProLong 的片段独立评估不同，LADM 在完整上下文中捕获多样性的长程依赖。
2. **设计注意力驱动的依赖度量体系**：引入 Pairwise Focus Score (PFS)、Aggregated Focus Score (AFS) 和 Contextual Dependency Score (CDS) 三级指标，综合考虑距离加权与依赖多样性（标准差），更精准地量化长上下文结构复杂度。
3. **实证高效性**：仅使用 1B tokens 的高质量筛选数据持续训练，即可在 LongBench 等任务上达到甚至超过随机采样使用 2B tokens 的效果，证明数据质量优于数量。

## 方法详解
1. **Long Attention Calculator**：选取轻量模型 **TinyLlama-1.1B-v1.1**（基础上下文 2K），用 5B 随机采样的 32K token 序列进行训练，赋予其基础的长上下文建模能力，用于后续注意力计算。
2. **Pairwise Focus Score (PFS)**：将样本 S 划分为 N 个长度为 l=128 的 span，计算 span $s_j$ 对 preceding span $s_i$ 的注意力权重累积：
   $$
   \mathrm{PFS}(i, j) = \mathrm{Sum}\left(\mathrm{Softmax}\left(\frac{Q_j K_{0:j}^T}{\sqrt{d_k}}\right)[, i]\right)
   $$
   该值量化了 $s_i$ 对 $s_j$ 表示的影响程度。
3. **Aggregated Focus Score (AFS)**：对每个 span $s_j$，聚合其与前面 span 的所有 PFS（跳过前 m 和邻近 n 个 span，以步长 d 采样），并引入长度加权与依赖多样性（标准差 $\sigma_j$）：
   $$
   \mathrm{AFS}(j) = \sigma_j \sum_{i=m}^{j-n-1} \frac{j-i}{N} \cdot \mathrm{PFS}(i, j)
   $$
   标准差鼓励样本呈现更多样的依赖模式，长度加权强调远距离依赖。
4. **Contextual Dependency Score (CDS)**：对所有 span 的 AFS 进行位置加权求和，得到样本级得分：
   $$
   \mathrm{CDS}(S) = \sum_{j=n_0}^{N-1} \frac{j}{N} \cdot \mathrm{AFS}(j)
   $$
   排除初始 span（$j < n_0$）以避免信息不足的测量偏差。最终按 CDS 排序，从每个数据域中选择 Top N 样本以保持原始分布，用于持续训练。

## 实验与结果
- **数据集**：使用 **Pile** 语料库，过滤长度 ≥32K 的样本（Table 7），持续训练 Token 数为 1B。
- **评估基线**：Random Sampling、ProLong（使用相同 TinyLlama 背板）。
- **模型**：OpenLlama-3B-v2、Llama2-7B/13B、Mistral-7B-v0.1。
- **评估任务**：Perplexity（Proof-Pile）、Synthetic（Needle-in-the-Haystack）、Real-world（LongBench：SD-QA、MD-QA、Summarization、Code）。
- **主要结果**：
  - **Perplexity**（Table 2）：LADM 在所有模型和上下文窗口下均获得最低 PPL。
  - **Needle Retrieval**（Figure 3）：L-7B/13B 和 M-7B 在 LADM 训练下接近 100% 检索准确率，远超基线。
  - **LongBench**（Table 3）：LADM 在四个模型上平均提升 **2.16%**（相对 ProLong）；对 Mistral-7B，单文档 QA 提升 **10.09%**，多文档 QA 提升 **4.66%**。
  - **训练效率**（Table 4、8）：LADM 使用 1B tokens 即可超越 Random Sampling 使用 2B/3B/4B tokens 的性能，证明数据选择的高效性。

## 相关工作脉络
1. **ProLong (Chen et al., 2024a)**：通过分割长上下文并计算片段间 delta perplexity 衡量依赖强度；LADM 与之关键区别在于直接在完整上下文中利用注意力分布，能捕捉更深层的全局依赖。
2. **Staniszewski et al. (2023)**：通过整合相关文档构建长上下文训练数据；LADM 侧重于**筛选**而非构造，且依赖度量更精细。
3. **训练增强方法（PI、NTK、YaRN、LongLoRA）**：专注于位置编码或注意力优化的**训练技巧**，而 LADM 关注**数据层面**的质量筛选，两者正交可结合。
4. **Lost-in-the-middle 缓解工作（An et al., 2024b; Xiong et al., 2024）**：针对模型忽视中间信息的现象进行任务设计；LADM 从数据源头提升长程依赖，从根本上减轻该问题。

## 局限性与未来方向
- **额外计算开销**：依赖小型模型（Long Attention Calculator）进行数据选择，引入前置计算成本。
- **模型规模限制**：未在 13B 以上参数规模的模型上验证，效果未知。
- **上下文长度局限**：受限于长上下文数据稀缺，仅实验了 32K 长度，未探索更长上下文的数据选择策略。

## 研究启发与可借鉴点
1. **注意力分数可作为数据质量代理指标**：利用模型内在的检索机制（注意力分布）来量化数据价值，为其他领域的**数据选择**提供了新思路。
2. **分层聚合的依赖度量设计**：PFS → AFS → CDS 的三级指标设计，兼顾局部依赖、全局结构与多样性，可迁移至其他需要评估序列内部关联的任务。
3. **小模型做数据筛选的可行性**：用 TinyLlama 训练出的注意力计算器能有效区分数据质量，证明了**轻量代理模型**在数据工程中的实用价值。
4. **实验设计严谨**：通过 Needle-in-the-Haystack 实验（Table 1）直观证明“依赖强度”对长上下文能力的影响，为后续研究提供了清晰的**归因验证范式**。

## 关键术语表
- **LADM**：Long-context data selection framework with Attention-based Dependency Measurement，长上下文数据选择框架。
- **Pairwise Focus Score (PFS)**：衡量两个 span 间注意力权重累积的成对依赖分数。
- **Aggregated Focus Score (AFS)**：对单个 span 与其所有前序 span 的 PFS 进行加权聚合的得分。
- **Contextual Dependency Score (CDS)**：样本级依赖分数，对所有 span 的 AFS 进行位置加权求和。
- **Long Attention Calculator**：具备基础长上下文建模能力的轻量模型（TinyLlama），用于计算注意力分数。
- **Needle-in-the-Haystack**：合成检索任务，评估模型在长上下文中定位特定信息的能力。
- **ProLong**：基于片段 delta perplexity 的长上下文数据筛选方法。

## 可复现要素
- **数据集**：Pile（已公开），实验使用其中长度≥32K的样本（详细组成见 Table 7）。
- **代码/权重**：论文未明确声明代码开源情况，Long Attention Calculator 使用 TinyLlama-1.1B-v1.1 预训练权重。
- **关键超参**：span 长度 $l=128$，$N=256$；AFS 计算中 $m=1$, $n=d=4$；CDS 计算中 $n_0=16$, $d=4$；训练 5B tokens 用于训练 Long Attention Calculator；持续训练 1B tokens，学习率 $2\times10^{-5}$，RoPE base 从 10,000 增至 500,000。
