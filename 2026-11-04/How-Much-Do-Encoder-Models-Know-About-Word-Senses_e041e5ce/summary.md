---
title: "How-Much-Do-Encoder-Models-Know-About-Word-Senses"
source: https://aclanthology.org/2025.acl-long.113.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 07:34:18"
field: "词义表征与语义理解"
keywords: ["词义消歧", "探测定量分析", "预训练语言模型", "Transformer层位分析", "DeBERTa-v3", "centroid-based probing"]
innovations: ["首次跨模型族逐层探测PLM在WSD任务上的内在语义表征能力，发现DeBERTa-v3中间层(第7-8层)显著优于输出层(~15 F1点)", "揭示词义库例句同质性/多样性是导致ODE与SemCor性能差距的主因，排除粒度和频率因素", "基于最优层位构建轻量WSD模型GlossDeBERTa-small，F1达79.5%且训练时间较基线缩短28%-70.6%"]
benchmarks: ["SemCor(WordNet)", "ODE(Oxford Dictionary of English)", "ALL(all-words WSD benchmark)"]
---

# 论文速读：How-Much-Do-Encoder-Models-Know-About-Word-Senses

## 一句话总结
本文通过无监督的"探测"（probing）方式系统评估了四种主流编码器模型在不进行任务微调的情况下，对词义消歧（WSD）的内在理解能力；发现 DeBERTa-v3 表现最佳，且其第7–8层的效果显著优于输出层（约高出15个百分点）；同时揭示了评测数据集结构特征对模型性能的巨大影响。

## 研究问题与动机
- **核心问题**：预训练语言模型（PLM）在未经 WSD 任务微调前，其潜在表征中是否已编码了足够的词义区分信息？
- **现有方法的盲区**：以往 WSD 研究多将 PLM 当作黑盒直接微调下游，缺乏对模型"内在知识"的系统测量；已有探测工作多局限于单一模型（如仅 BERT）或简化任务（如仅处理同音异义词），未深入分析不同 Transformer 层的语义专长差异。
- **评测基准结构差异未知**：WordNet（SemCor）与 ODE 两类词义库在粒度、频率分布和例句同质性上存在系统性差异，这些差异如何影响 PLM 性能尚无定论。
- **效率诉求**：当前 WSD 系统多依赖大型模型全层微调，计算成本高、碳排放大；若中间层已蕴含最优语义表征，则有望构建更小、更高效的 WSD 模型。

## 核心贡献（创新点）
1. **系统性层析探测框架**：首次在同一研究中对比评估 BERT、RoBERTa、ELECTRA、DeBERTa-v3 共四层模型的全部12层输出，揭示了 MLM 与 RTD 预训练策略在语义表征深度上的本质差异。
2. **发现 DeBERTa-v3 中间层最优现象**：DeBERTa-v3 的第7–8层在 WSD 探测任务中超越其输出层约15个 F1 点，打破"最深层=最优语义"的常规假设。
3. **揭示词义库结构性偏差**：通过内/外相似度分析，证明 SemCor 在 ODE 上性能差距的主要来源是例句的语义同质性与上下文多样性差异，而非词义粒度或频率分布。
4. **构建轻量高效 WSD 模型 GlossDeBERTa-small**：基于最优层位设计，以仅7层 DeBERTa-v3 实现与全12层 GlossBERT/GlossDeBERTa 相当的 F1（79.5%），训练时间分别缩短 28% 与 70.6%，碳足迹显著降低。

## 方法详解
**整体思路**：冻结预训练编码器，利用训练集计算每个词义（sense）的嵌入向量质心，再以余弦相似度将测试实例归类到最近质心对应的词义。

1. **数据过滤**：移除单义词（monosemous words）及词义示例数少于 $K$（$K=5$ 或 $K=10$）的词义，确保质心计算的可靠性。
2. **质心计算**：对词义 $s$ 的 $\mathcal{N}$ 个训练示例，用冻结 Transformer 提取词嵌入 $E=\{e_1,\ldots,e_\mathcal{N}\}$，若词被拆分为多 token 则取平均；质心公式为：
$$
\text{Centroid}(E)=\frac{1}{\mathcal{N}}\sum_{i=1}^{\mathcal{N}} e_i
$$
3. **分类决策**：对测试实例 $t$ 的嵌入 $e$，选择同 lemma 与 POS 下余弦相似度最高的质心：
$$
c^*=\arg\max_{c\in C}\frac{c\cdot e}{\|c\|\|e\|}
$$
4. **逐层评估**：分别用每一层 $\mathcal{M}_L$ 的输出生成质心与测试嵌入，记录 F1 得分，定位最优层位。
5. **结构分析指标**：定义内部相似度（同词义内例句嵌入的平均余弦相似）与外部相似度（不同词义间例句嵌入的平均余弦相似），用以量化词义库的可分性：
$$
\text{IntSim}(S_i)=\frac{1}{n^2-n}\sum_{p\neq q}\cos(e_p,e_q),\quad
\text{ExtSim}(S_i,\hat{S})=\frac{1}{nm}\sum_{e_p\in S_i}\sum_{e_q\in\hat{S}}\cos(e_p,e_q)
$$

## 实验与结果
- **数据集**：SemCor（WordNet 3.0，33,362 词义/226,036 实例）与 ODE（Oxford Dictionary of English，79,004 词义/约 785,000 实例），按 80/20 划分训练/测试集。
- **模型**：BERT-base-cased/uncased、RoBERTa-base、ELECTRA-base-discriminator、DeBERTa-v3-base。
- **最优模型与层位**：
  - **ODE 上**：DeBERTa-v3 第7层 F1 = **90.46%**（K=5）、第7层 F1 = **91.40%**（K=10）；输出层仅 75.94%（K=5）/ 77.33%（K=10），落差约 **15 个百分点**。
  - **SemCor 上**：DeBERTa-v3 第7–8层 F1 ≈ **71%**（K=5/10）；输出层仅 55.15%（K=5）/ 55.56%（K=10）。
  - BERT/RoBERTa 峰值出现在第10–11层；ELECTRA 同样在中层（第5–8层）表现最佳但绝对值低于 BERT/RoBERTa。
- **关键结论**：MLM 预训练模型（BERT/RoBERTa）最优语义层靠后，RTD 预训练模型（ELECTRA/DeBERTa-v3）最优层居中；DeBERTa-v3 在所有设置下全面领先。
- **结构差异解释**：ODE 内部相似度 ~70、外部相似度 ~17（可分性强）；SemCor 内部/外部相似度均偏高（~70 / ~45），词义间边界模糊，是性能差距的主因，而非粒度或频率不平衡。
- **轻量模型**：GlossDeBERTa-small（7层）在 ALL 数据集上 F1 = **79.5%**，与 GlossBERT（77.0%）和 GlossDeBERTa（79.5%）持平，训练时间分别缩短 28% 和 70.6%。

## 相关工作脉络
- **WSD 监督方法**（GlossBERT、SensEmBERT、ARES、SREF）：利用词义定义（gloss）或 WordNet 层级关系增强分类，本文的轻量模型 GlossDeBERTa-small 即沿用 GlossBERT 的 sentence-gloss pair 弱监督框架，但以最优中间层替代全模型。
- **探针研究（Probing）**：Adi et al. (2017)、Conneau et al. (2018) 证明 PLM 预训练阶段已编码句法/语义知识；Tenney et al. (2019a) 揭示不同层擅长不同语言层次，本文将其结论从"句法-语义分层"推进到"词义表征的最优层位"。
- **几何表征探测**：Coenen et al. (2019) 以可视化方式发现 BERT 的几何结构可分离词义；Tripodi (2021) 证明上下文化模型为同一词义生成可聚类表征；本文首次系统化地跨模型族、跨层位定量对比。
- **同音异义探测**：Proietti et al. (2024) 报告 PLM 可在无需微调下以 95% 准确率区分同音异义词，但未涉及多层分析与不同词义库比较；本文填补了这一空白。
- **ELECTRA 架构分析**：He et al. (2021) 指出 ELECTRA 中 generator 与 discriminator 共享 token embedding 导致目标冲突（tug-of-war），本文从 WSD 视角实证验证了这一缺陷，并突显 DeBERTa-v3 解耦注意力机制的优势。

## 局限性与未来方向
- 仅覆盖 base 版本（12层）的四款模型，未纳入 large 版本及 ALBERT、CamemBERT、SpanBERT、DistilBERT 等变体。
- 分析限于英语；未涉及 mBERT、XLM-R、XLM-E 等多语言模型的跨语言词义表征能力。
- 未扩展到 LLM（如 Llama、ChatGPT、Mistral）的 decoder-only 架构探测。
- 词义库结构差异的分析可能受 ODE 例句人工编写、SemCor 语料年代（1960年代文本）等外部因素干扰，结论外推需谨慎。
- 未来计划：扩展至 multilingual PLM 及大型生成式语言模型的多语言词义探测。

## 研究启发与可借鉴点
1. **层位选择先于模型放大**：在部署语义任务前，可通过 probing 定位最优中间层而非盲目使用输出层，以 minimal compute 获得最大收益——对低资源场景尤具价值。
2. **评测基准结构敏感性**：词义库的例句同质性/多样性直接影响探测结果，后续研究应引入更多样化的基准（如跨年代语料、多领域文本）以验证结论稳健性。
3. **RTD vs. MLM 策略对比**：本文对 ELECTRA 与 DeBERTa-v3 的对比为预训练目标设计提供了新的实证依据，可启发后续对 other discriminator-based 预训练范式的语义能力分析。
4. **小型化路径清晰**：GlossDeBERTa-small 证明"最优层位 + 精简层数"策略可在几乎不损失性能的前提下大幅压缩训练成本，此思路可迁移至其他词义相关任务（如实体链接、语义角色标注）。

## 关键术语表
- **Word Sense Disambiguation（WSD）**：词义消歧，根据上下文确定多义词在具体语境中的正确含义，是 NLP 的经典基础任务。
- **Probing（探测）**：通过构造辅助分类任务来量化预训练模型内部表征中所编码的语言知识，而不修改主模型参数。
- **Centroid-based disambiguation**：词义消歧方法，对每个词义的所有训练示例嵌入求均值作为该词义的"中心点"，再以余弦相似度将测试样本归入最近的中心。
- **Masked Language Modeling（MLM）**：掩码语言建模，预训练目标，随机遮蔽输入 token 并令模型预测被遮蔽内容，BERT/RoBERTa 采用。
- **Replaced Token Detection（RTD）**：替换 token 检测，预训练目标，由生成器替换 token 后令判别器判断每个位置是否被替换，ELECTRA/DeBERTa 采用。
- **Disentangled Attention（解耦注意力）**：DeBERTa-v3 的核心创新，将内容表示与位置/掩码信息分离计算，避免 generator 与 discriminator 的梯度冲突。
- **Internal/External Similarity**：内部相似度衡量同一词义内例句嵌入的聚集程度；外部相似度衡量不同词义间例句嵌入的接近程度；两者差值越大表示词义越易区分。

## 可复现要素
- **数据集**：SemCor（公开）、ODE（Chang et al. 2018 版本，公开）；按 80/20 随机划分。
- **代码/权重**：软件已公开于 https://github.com/SapienzaNLP/wsd-probing。
- **关键超参**：最小示例数 $K \in \{5, 10\}$；GlossDeBERTa-small 训练：4 epochs、dropout=0.1、学习率 2e-5、batch size=64、NVIDIA GPU 1080 Ti、SE07 开发集选最优 checkpoint。
- **模型版本**：google-bert/bert-base-cased、google-bert/bert-base-uncased、FacebookAI/roberta-base、google/electra-base-discriminator、microsoft/deberta-v3-base（均来自 Hugging Face）。
