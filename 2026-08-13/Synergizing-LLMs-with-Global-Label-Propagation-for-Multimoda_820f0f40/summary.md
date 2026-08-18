---
title: "Synergizing-LLMs-with-Global-Label-Propagation-for-Multimoda"
source: https://aclanthology.org/2025.acl-long.72.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:21:18"
field: "多模态虚假信息检测"
keywords: ["fake news detection", "large language models", "label propagation", "multimodal learning", "graph neural networks", "pseudo labeling"]
innovations: ["将LLM伪标签通过全局标签传播机制融入多模态假新闻检测，而非直接拼接预测", "设计基于Mask的全局标签传播机制防止训练时标签泄露", "提出混合主动标签策略，通过置信度阈值筛选高质量LLM伪标签"]
benchmarks: ["Twitter", "PHEME", "Weibo"]
---

# 论文速读：Synergizing LLMs with Global Label Propagation for Multimodal Fake News Detection

## 一句话总结
本文提出了 GLPN-LLM，将 LLM 生成的伪标签通过全局标签传播（Global Label Propagation）技术融入多模态假新闻检测框架，有效弥补了 LLM 直接预测伪标签效果差的不足，在三个基准数据集上均取得了优于 SOTA 的性能。

## 研究问题与动机
- **LLM 伪标签的直接整合效果差**：GPT-4o 直接生成的伪标签准确率低于现有主流多模态检测方法（如 Table 1 所示），简单地将两者结合无法获得显著提升。
- **标签传播在中等质量伪标签下仍有效**：Label Propagation（LP）技术即使在全局中伪标签准确度不高的情况下也能保持有效性，但需解决训练过程中的标签泄露问题。
- **多模态特征与标签的对齐不足**：现有方法未能充分利用跨模态图结构中的全局信息来对齐文本-图像表征与标签语义。
- **缺少对伪标签质量的过滤机制**：LLM 输出的伪标签存在不确定性，需设计置信度筛选策略以确保传播质量的可靠性。

## 核心贡献（创新点）
1. **提出 GLPN-LLM 框架**：首次将 LLM 生成的伪标签通过全局标签传播机制引入多模态假新闻检测，而非直接拼接预测结果，与现有方法形成本质区别。
2. **设计基于 Mask 的全局标签传播机制**：通过训练时对随机节点子集进行标签掩码（GRM），防止节点将自身标签反向传播给自己（标签泄露），相比 FCN-LP 的固定标签传播更为灵活且避免了信息泄露。
3. **引入混合主动标签策略（Mixed-Initiative Labeling）**：利用结构化 Prompt 引导 LLM 输出二分类标签及置信度分数，并通过置信度阈值（Top 5%）筛选高置信度伪标签，确保传播质量。
4. **跨模态图构建增强**：基于五种相似度度量（拼接特征相似度、图-文、文-图、图-图、文-文）构建跨模态图，阈值 θ=0.95，使标签能在多模态关联节点间有效传播。

## 方法详解

### 整体框架
GLPN-LLM 包含两个核心模块：（1）基于 LLM 的伪标签生成模块；（2）基于 Mask 的全局标签传播模块，二者协同工作。

### 多模态特征提取
使用 CLIP 双编码器分别提取图像特征 $\mathbf{v_i} \in \mathbb{R}^{d_v}$ 和文本特征 $\mathbf{t_i} \in \mathbb{R}^{d_t}$，拼接得到统一表示：$\mathbf{x_i} = \mathbf{t_i} \oplus \mathbf{v_i}$。

### 跨模态图构建
节点为新闻样本，边基于五种相似度度量，任一超过阈值 θ=0.95 即建边。这五种相似度包括：拼接特征余弦相似度、图-文相似度、文-图相似度、图-图相似度、文-文相似度。

### 标签集成模块
将标签特征 $\mathbf{y_i'}$ 与节点特征 $\mathbf{x_i}$ 拼接：$\mathbf{x_i'} = \mathbf{x_i} \oplus \mathbf{y_i'}$。

### 混合主动标签策略
通过结构化 Prompt 引导 LLM 输出检测标签 $\hat{\mathbf{y}}$ 和置信度 $c$。伪标签按公式（5）整合：真实标签节点保留原标签，高置信度无标签节点使用 LLM 伪标签，其余置零向量。

### 全局随机掩码（GRM）
每个训练 epoch 以掩码率 ρ（最优 0.5）随机掩码部分节点的标签嵌入（置零），仅对掩码节点计算交叉熵损失，防止标签泄露。公式（6）：$\mathbf{y_i'} = \tilde{\mathbf{y_i}} \cdot m_i$，其中 $m_i$ 为二进制掩码。

### 分类器
在 augmented 图上使用 GCN 进行分类，以 Adam 优化器（学习率 1e-3）优化交叉熵损失。

## 实验与结果
- **数据集**：Twitter（15,000 训练 / 2,000 测试）、PHEME（1,414 / 608）、Weibo（4,141 / 1,125），均为公开数据集。
- **基线**：EANN、SpotFake、MVAE、SAFE、MCAN、HMCAN、FCN、FCN-LP、FCN-LP + LLM（ naive 方案）。
- **主要结果（F1 分数，取最高值）**：
  - Twitter：GLPN-LLM (CLIP) **89.03%**，较 FCN-LP (CLIP) + LLM（85.97%）提升约 **3.06 pp**；较 FCN-LP (CLIP)（84.50%）提升约 **4.53 pp**。
  - PHEME：GLPN-LLM (CLIP) **90.66%**，较 FCN-LP (CLIP) + LLM（89.21%）提升约 **1.45 pp**。
  - Weibo：GLPN-LLM (CLIP) **91.52%**，较 FCN-LP (CLIP) + LLM（89.85%）提升约 **1.67 pp**。
  - GLPN-LLM (HMCAN) 在 Twitter 达 86.86%、PHEME 达 86.87%、Weibo 达 91.46%。
- **消融实验**（Table 2）：GLPN 优于 FCN-LP；GLPN-LLM 进一步优于 GLPN，验证各模块有效性。
- **关键超参**：掩码率 ρ=0.5 最优；伪标签选取置信度 Top 5% 最优。
- **最强结果**：Weibo 数据集 F1 达 **91.52%**（GLPN-LLM CLIP），为三数据集最高。

## 相关工作脉络
1. **FCN-LP（Zhao et al., 2023）**：本文直接扩展的对象，使用 CLIP+GCN+固定标签传播；本文通过引入 LLM 伪标签和 Mask 机制超越了它。
2. **HMCAN（Qian et al., 2021）**：多层多模态上下文注意力网络，本文以其作为特征提取 backbone 之一进行对比。
3. **SAFE（Zhou et al., 2020）**：基于跨模态相似度的假新闻检测方法，代表早期无图结构的多模态方法。
4. **MCAN（Wu et al., 2021）**：多模态共注意力网络，通过共注意力机制融合文本与图像特征。
5. **Label Propagation（Zhu & Ghahramani, 2002）**：经典半监督标签传播算法，本文将其引入假新闻检测领域并结合 LLM 伪标签。
6. **CLIP（Radford et al., 2021）**：视觉-语言预训练模型，本文用它提取多模态特征并构建跨模态相似度图。

## 局限性与未来方向
- **依赖骨干模型**：框架效果高度依赖 FCN 等骨干的特征提取能力，若骨干模型无法捕捉全面语义关系，则标签传播效果受限。
- **依赖高置信度伪标签**：LLM 生成的伪标签准确性直接影响传播质量，不准确或有偏的伪标签会引入噪声。
- **可扩展性**：作者明确指出未来需探索方法在大规模复杂数据集上的可扩展性。
- **跨平台适配**：目前仅在 Twitter、PHEME、Weibo 三个平台验证，需进一步验证在不同社交媒体和多元内容模态上的泛化能力。

## 研究启发与可借鉴点
1. **LLM 伪标签 + 图传播的协同范式**：直接拼接 LLM 预测效果差，但通过标签传播在全局图结构上平滑扩散可实现有效整合，该思路可迁移至其他半监督/少样本图分类任务。
2. **Mask 防止标签泄露的设计**：训练时对标签嵌入随机掩码（类似语义分割中的 Dropout 思想），可有效防止模型过拟合标签信息而忽略原始特征，适用于所有基于标签注入的图学习框架。
3. **置信度阈值筛选伪标签**：仅选取 Top 5% 高置信度伪标签参与传播，而非全部采纳，这一策略简单且有效，可作为通用伪标签质量控制方案。
4. **跨模态五种相似度构建图**：除拼接特征外，还考虑了跨模态双向相似度（图-文、文-图），为多模态关系建模提供了可复用的图构建模板。
5. **Prompt 设计影响显著**：详细 Prompt（含评估准则和示例）显著优于简单 Prompt，提示工程在多模态 LLM 应用中值得精细优化。

## 关键术语表
- **GLPN-LLM**：Global Label Propagation Network with LLM-based Pseudo Labeling，本文提出的多模态假新闻检测框架。
- **标签传播（Label Propagation, LP）**：在半监督学习中通过图结构将已知标签信息传播到未标注节点的技术。
- **全局随机掩码（Global Random Mask, GRM）**：训练时随机掩码部分节点标签嵌入，防止标签泄露的核心设计。
- **混合主动标签策略（Mixed-Initiative Labeling）**：结合人类定义 Prompt 与 LLM 推理，生成带置信度的伪标签的方法。
- **跨模态图（Cross-modal Graph）**：节点为新闻样本，边基于文本/图像多类型相似度构建的图结构。
- **伪标签（Pseudo Label）**：由模型（此处为 LLM）为无标签数据生成的预测标签。
- **CLIP**：Contrastive Language-Image Pre-training，多模态预训练模型，用于联合提取文本和图像特征。
- **GCN**：Graph Convolutional Network，图卷积网络，本文用于在跨模态图上进行节点分类。

## 可复现要素
- **数据集**：Twitter（MediaEval 基准）、PHEME、Weibo，均为公开数据集；论文代码已开源（链接见 abstract）。
- **代码**：论文声明代码在线可用（footnote 1）。
- **关键超参**：相似度阈值 θ=0.95；掩码率 ρ=0.5；伪标签选取比例 Top 5%；GCN 隐层维度 512；学习率 1e-3；Adam 优化器；训练 5 次取平均。
- **LLM**：使用 GPT-4o API。
- **骨干模型**：CLIP 和 HMCAN 两种特征提取器，论文使用 FCN-LP 的相同设置。
