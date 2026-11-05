---
title: "LADM-Long-context-Training-Data-Selection-with-Attention-bas"
source: https://aclanthology.org/2025.acl-long.154.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 17:03:03"
---

# 论文速读：LADM: Long-context-Training-Data-Selection-with-Attention-bas

## 一句话总结
本文提出LADM框架，通过轻量级模型的注意力分布量化长文本内部的跨段依赖强度，从大规模预训练语料中高效筛选出高质量长上下文数据；仅用1B tokens持续预训练即可在LongBench与长程检索任务上显著超越随机采样与ProLong基线。

## 研究问题与动机
- **核心问题**：如何准确衡量并筛选高质量的长上下文训练数据？现有长上下文构建方法多依赖随机拼接或孤立段落的困惑度差异，忽视了完整上下文中的内在语义结构与长程依赖关系。
- **动机1**：由短样本简单拼接的数据缺乏跨段落的逻辑关联，会导致模型倾向于关注局部信息而忽视远距离关键内容，加剧“lost-in-the-middle”现象。
- **动机2**：注意力机制本身具备隐式检索能力（Wu et al., 2024），其注意力分布能直观反映token/span间的依赖强度，可作为数据质量的直接量化信号。
- **动机3**：已有数据筛选方法（如ProLong）仅在剥离上下文的情况下计算段间delta perplexity，无法捕捉完整序列中的复杂依赖模式，筛选结果存在偏差。

## 核心贡献（创新点）
1. **提出LADM数据筛选框架**：利用小模型注意力分布测量长样本的全局依赖程度，实现从海量多领域语料中高效挖掘高质量长上下文样本；与ProLong等基于孤立段落困惑度差值的方法本质不同，LADM在完整序列上建模依赖关系。
2. **设计三级依赖度量指标（PFS→AFS→CDS）**：结合区间长度加权与注意力方差，同时捕捉长程依赖的强度与多样性，避免位置偏见对评分的干扰。
3. **验证高效性与泛化性**：仅用1B tokens持续预训练，在4个不同架构/规模的模型上于LongBench平均提升2.16%；即使训练token减半（0.5B），性能仍全面超越基线1B结果，且不损害短上下文能力。

## 方法详解
- **Long Attention Calculator**：选用TinyLlama-1.1B-v1.1作为基础模型，使用5B随机采样的32K token序列进行训练，赋予其基础长上下文建模能力，专门用于后续注意力分数计算。
- **Pairwise Focus Score (PFS)**：将长样本$S$均匀划分为$N$个长度为$l$的span，计算span $s_i$与$s_j$ ($j>i$) 之间的注意力权重累积：$\mathrm{PFS}(i,j) = \mathrm{Sum}(\mathrm{Softmax}(\frac{Q_j K_{0:j}^T}{\sqrt{d_k}})[ :, i])$，量化前方span对当前span最终表示的影响程度。
- **Aggregated Focus Score (AFS)**：对每个span $s_j$，聚合其与前方多个span的PFS。为规避首尾位置偏见，排除前$m$个与近邻$n$个span，并按步长$d$采样以提升效率；引入长度权重$\frac{j-i}{N}$鼓励长距离依赖，并乘以PFS的标准差$\sigma_j$以反映依赖模式的复杂性：$\mathrm{AFS}(j) = \sigma_j \sum_{i=m}^{j-n-1} \frac{j-i}{N} \cdot \mathrm{PFS}(i,j)$。
- **Contextual Dependency Score (CDS)**：将所有span的AFS加权求和，权重$\frac{j}{N}$强调靠后span的贡献（因其涉及更远的历史依赖），得到样本级得分：$\mathrm{CDS}(S) = \sum_{j=n_0}^{N-1} \frac{j}{N} \cdot \mathrm{AFS}(j)$。最终按领域分层排名，选取Top-N样本进行持续预训练。

## 实验与结果
- **数据集与评估**：预训练语料为Pile（过滤≥32K样本）；评估涵盖Proof-Pile（Perplexity）、Needle-in-the-Haystack（合成检索）与LongBench（单/多文档QA、摘要、代码补全）。
- **基线**：Random Sampling、ProLong（delta perplexity筛选框架）。
- **主要结果**：
  - **Perplexity**：LADM在OL-3B、L-7B、M-7B、L-13B上于所有上下文窗口尺寸均取得最低PPL。
  - **合成检索**：Needle-in-the-Haystack平均检索准确率显著优于基线，Llama2-7B/13B与Mistral-7B在1B tokens训练下接近100%。
  - **真实任务（LongBench）**：LADM在单/多文档QA、摘要与代码补全上全面领先，相对ProLong平均提升**2.16%**；其中Mistral-7B单文档QA提升达**10.09%**。
  - **训练效率**：仅使用0.5B tokens时仍超越Random/ProLong的1B结果，证明数据质量筛选可显著降低训练成本。
- **最强结果与提升幅度**：Mistral-7B在单文档QA任务上较ProLong提升10.09%；四种模型在LongBench上平均提升2.16%。

## 相关工作脉络
- **ProLong (Chen et al., 2024a)**：基于段落分割与delta perplexity测量依赖强度，未考虑完整上下文结构；LADM在全序列注意力分布上建模，能捕捉更深层次的段落关联。
- **StructPack (Staniszewski et al., 2023)**：通过语义相似性将相关文档拼接构建长上下文数据；LADM侧重从海量混合语料中“筛选”而非“构造”，依赖内在语义连贯性而非表面相似性。
- **Lost-in-the-middle / DCA & SelfExtend (Liu et al., 2024; Xiao et al., 2024; Han et al., 2024)**：关注位置偏见与训练无关的长度泛化；本文从数据侧切入，通过注意力依赖度量主动筛选富含全局关联的样本。
- **Data Engineering for Long Context (Fu et al., 2024; Gao et al., 2024a)**：侧重领域平衡、长度上采样或查询中心合成；LADM聚焦于依赖关系本身的量化，提供通用的数据质量评估视角。
- **Retrieval Head Mechanism (Wu et al., 2024)**：揭示LLM注意力机制内蕴的检索能力；本文受此启发，将注意力分布直接转化为数据筛选的可计算度量信号。

## 局限性与未来方向
- 局限性1：依赖小模型进行注意力计算，引入额外的预处理算力开销。
- 局限性2：未在小模型（>13B参数）规模上验证，难以直接推断对超大语言模型的适用性。
- 局限性3：仅实验32K上下文长度，受限于公开长文本数据资源的匮乏，未探索128K/1M窗口的数据筛选策略。
- 未来方向：扩展至更长上下文长度、探索线性/近似注意力计算以降低筛选成本、将依赖度量框架迁移至多模态长序列数据筛选。

## 研究启发与可借鉴点
- **注意力分布作为数据质量信号**：直接利用模型内部的注意力权重替代人工规则或外围指标，为数据筛选提供了可解释、可计算的物理意义，该思路可迁移至代码/数学/多模态等需强内部结构的序列数据筛选。
- **PFS-AFS-CDS三级聚合设计**：两两交互→跨度聚合→样本打分的递进结构，结合长度加权与标准差方差，兼顾了依赖强度与多样性；该模块化设计便于替换为其他注意力变体（如滑动窗口、sink attention）。
- **小模型代理评估策略**：1.1B模型训练为“注意力计算器”即可服务大模型数据筛选，证明小模型足以捕获长程依赖模式，大幅降低大规模数据预处理的算力门槛，适合工业级数据管线。
- **分层Top-N选取策略**：按领域分布分别排名再合并，避免全局截断破坏预训练语料的领域均衡，是可复用的工程化数据过滤范式。

## 关键术语表
- **LADM**：Long-context data selection framework with Attention-based Dependency Measurement，本文提出的基于注意力依赖度量的长上下文数据筛选框架。
- **Long Attention Calculator**：基于TinyLlama微调的轻量级长上下文建模模型，专门用于计算样本内部的注意力分布分数。
- **Pairwise Focus Score (PFS)**：衡量长样本中任意两个span之间注意力权重累积和的指标，量化前方段落对后方段落的直接影响。
- **Aggregated Focus Score (AFS)**：对单个span与其前方多个span的PFS进行加权聚合，结合区间长度权重与标准差以捕捉长程依赖的强度与多样性。
- **Contextual Dependency Score (CDS)**：样本级综合得分，由所有span的AFS加权求和得出，用于对长上下文数据进行最终质量排序。
- **Needle-in-the-Haystack**：合成检索测试任务，通过在长文本中随机插入关键信息并询问其内容，评估模型跨长距离检索的能力。
- **LongBench**：双语多任务长上下文理解基准，涵盖单/多文档QA、摘要、代码补全等真实场景任务。
- **ProLong**：Chen等(2024)提出的长上下文数据筛选框架，通过计算分割段落间的delta perplexity来衡量依赖强度。

## 可复现要素
- **数据集**：Pile
