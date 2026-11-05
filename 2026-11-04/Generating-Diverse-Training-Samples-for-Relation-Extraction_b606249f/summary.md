---
title: "Generating-Diverse-Training-Samples-for-Relation-Extraction"
source: https://aclanthology.org/2025.acl-long.35.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 22:03:35"
field: "关系抽取与数据增强"
keywords: ["Relation Extraction", "Large Language Models", "Data Generation", "DPO", "In-Context Learning", "Few-shot Learning"]
innovations: ["提出OBO迭代生成策略提升LLM生成样本多样性", "首次将DPO引入训练样本生成任务并自动化构建偏好数据对", "证实生成样本训练非LLM模型优于直接LLM推理的少样本RE新范式"]
benchmarks: ["TACRED", "TACRED-Revisit", "Re-TACRED", "SemEval 2010 Task 8"]
---

# 论文速读：Generating-Diverse-Training-Samples-for-Relation-Extraction

## 一句话总结
本文针对关系抽取（RE）任务，提出利用大语言模型（LLM）生成多样化训练样本的方法，通过上下文学习（ICL）和直接偏好优化（DPO）提高生成样本的多样性并保持正确性，实验表明生成的样本可有效训练非LLM模型，在少样本场景下达到与人工标注数据相近的性能。

## 研究问题与动机
1. **低资源困境**：关系抽取在不同领域存在大量实体关系类型，实际应用中训练数据稀缺是普遍问题。
2. **LLM生成样本多样性不足**：直接提示LLM生成RE训练样本时，模型倾向于使用有限的关键词或短语表达关系，导致样本结构高度相似。
3. **多样性与正确性的平衡难题**：如何在提高生成样本多样性的同时，确保其语义正确性和标注准确性尚未得到充分研究。
4. **LLM直接推理成本高昂**：相比直接用LLM进行关系抽取，使用LLM生成样本后训练轻量非LLM模型可能更具成本效益。

## 核心贡献（创新点）
1. **提出ICL驱动的两阶段生成策略**：首次系统性地对比"一次性全量生成"（All-at-once, AAO）与"逐步迭代生成"（One-by-one, OBO）两种ICL生成模式，发现OBO通过动态累积示例能更有效地抑制生成样本的结构相似性。
2. **设计基于DPO的多样性微调方法**：将DPO引入RE训练样本生成任务，通过自动构造的不偏好答案（错误标注样本、相似示例、完全重复示例）对LLaMA2-7b-Chat进行LoRA微调，无需人工标注偏好数据即可引导模型兼顾多样性与正确性。
3. **揭示LLM生成样本的可迁移性规律**：实证发现使用LLM生成样本训练的KnowPrompt/RetrievalRE模型性能可与纯人工标注训练相媲美，且混合训练（生成样本+人工样本）在少样本（K=8/16）时显著提升最终F1，但随人工数据增加边际收益递减。

## 方法详解
**ICL Prompt构建**（三段式）：
- **任务描述模块**：说明生成RE训练样本的任务定义（包含文本、头尾实体及位置、关系类别），并在末尾加入多样性指令"生成与示例尽可能不同的样本"。
- **关系解释模块**：为每个关系类别提供自然语言解释，增强LLM对关系语义的理解。
- **样本演示模块**：随机选取人工标注样本作为示范；OBO模式下每生成一个新样本即追加至演示模块，形成累积递增过程。

**DPO微调方法**：
目标函数为：$\max_{\pi} \sum_i \log \sigma\left(\frac{1}{\beta}\log\frac{\pi(y_{i,1}|x_i)}{\pi(y_{i,2}|x_i)}\right)$，其中$y_{i,1}$为偏好输出，$y_{i,2}$为非偏好输出。

**不偏好答案的自动构造策略**（三种）：
1. 将其他关系类别的人工样本改标为目标关系，构造错误标注样本；
2. 基于演示样本替换实体并增删词汇，构造结构相似样本；
3. 直接复制演示样本作为完全重复样本。

**DPO训练设置**：将数据集按关系类别对半划分，一半用于DPO微调、另一半用于样本生成，通过交换角色覆盖所有类别，避免"作弊"风险。

## 实验与结果
**数据集**：SemEval（19类）、TACRED（42类）、TACRED-Revisit（42类）、Re-TACRED（40类）。

**评估设置**：使用KnowPrompt和RetrievalRE（均基于RoBERTa-Large）作为下游训练模型，以Micro F1为指标；LLM采用LLaMA2-7b-Chat（temperature=0.4, top_p=0.9, LoRA微调）。

**主要结果**：
- **Ours (mix-OBO) on TACRED K=32**：KnowPrompt达**36.25%** Micro F1，RetrievalRE达**37.46%**，超越纯人工标注基线（KnowPrompt: 36.33%, RetrievalRE: 37.07%）。
- **Ours (pure) vs Direct-RE**：在所有数据集上，用生成样本训练的非LLM模型均优于直接用LLM进行RE（如TACRED K=32时Ours pure为30.41%，Direct-RE为21.17%，提升约9个百分点）。
- **最优提升幅度**：在Re-TACRED K=16设置下，Ours (mix-OBO) 达62.61%，相比纯人工标注（56.51%）提升**6.1个百分点**；相比另一数据生成方法Xu et al. (2023) 的Data Generation基线（35.03%）提升约**27.6个百分点**。
- **多样性验证**：DPO微调后生成样本的平均余弦相似度降低约0.051，词重复率也显著下降。

## 相关工作脉络
1. **Self-Instruct (Wang et al., 2023)**：利用LLM生成指令微调数据，但未聚焦生成样本的多样性与正确性平衡问题；本文在此基础上引入DPO实现自动化偏好优化。
2. **RelationPrompt (Chia et al., 2022)**：通过提示语言模型生成结构化文本来合成未见过的关系类型；本文聚焦已有关系类型的样本多样性增强而非新关系发现。
3. **Xu et al. (2023) Data Generation**：在RE任务上使用LLM生成数据辅助模型训练，但生成样本缺乏多样性控制机制；本文方法在其基础上显著提升生成质量。
4. **KnowPrompt (Chen et al., 2022b) / RetrievalRE (Chen et al., 2022a)**：作为下游评估模型，本文验证了生成样本对这些先进RE框架的有效性；二者与本文正交，可结合使用。
5. **DPO (Rafailov et al., 2023)**：原始工作应用于对话和摘要任务；本文首次将其引入训练样本生成领域，用于联合优化多样性与正确性。

## 局限性与未来方向
1. **LLM能力上限**：生成样本质量受限于所使用的开源LLM（LLaMA2-7b-Chat）本身的能力，更大规模模型可能产生更高质量样本。
2. **生成数量天花板**：实验表明生成约16-32个样本后性能趋于饱和，受限于LLM训练语料的多样性边界，难以无限扩展。
3. **领域迁移性待验证**：实验主要在通用RE数据集上进行，在垂直领域（如生物医学、法律）的关系抽取中效果未验证。
4. **DPO偏好构造的近似性**：当前不偏好答案通过规则自动生成，可能存在构造偏差，未来可探索人工校验或LLM自一致性筛选。

## 研究启发与可借鉴点
1. **OBO迭代生成范式可迁移**：将历史生成结果动态纳入Prompt的累积式生成策略可推广至其他NLP数据生成任务（如NER、事件抽取），有效缓解LLM的重复生成倾向。
2. **DPO自动化偏好构建思路**：通过规则构造"负样本"（错误标注、结构相似、完全重复）来替代人工偏好标注，显著降低数据标注成本，该策略可迁移至其他生成任务的偏好优化。
3. **多样性量化指标组合**：同时使用余弦相似度与词重复率双重指标评估生成多样性，比单一指标更可靠，可作为后续工作的标准评估协议。
4. **混合训练策略启示**：生成样本与人工样本按适当比例混合训练可实现性能互补，未来可探索自适应混合比例策略以最大化少样本场景下的数据效率。

## 关键术语表
**Relation Extraction (RE)**：从文本中识别并分类实体对之间语义关系的任务，是知识图谱构建等应用的基础。
**In-Context Learning (ICL)**：通过在Prompt中提供少量示例，使LLM无需参数更新即可完成特定任务的学习范式。
**One-by-one (OBO) Generation**：逐步生成训练样本并将新样本追加至演示模块的迭代式生成策略。
**All-at-once (AAO) Generation**：一次性要求LLM生成所有目标样本的批量生成策略。
**Direct Preference Optimization (DPO)**：绕过显式奖励模型，直接在偏好数据对上优化语言模型参数的微调方法。
**Micro F1**：关系抽取任务的主流评估指标，综合考虑所有类别的精确率与召回率。
**KnowPrompt**：结合知识注入与协同优化的Prompt-tuning关系抽取模型。
**RetrievalRE**：结合检索增强与Prompt-tuning的关系抽取模型，提升对困难模式的泛化能力。

## 可复现要素
- **数据集**：SemEval、TACRED、TACRED-Revisit、Re-TACRED（均为公开数据集）
- **代码**：已开源，地址 https://github.com/Lzx-ZBC
- **模型**：LLaMA2-7b-Chat（HuggingFace公开权重），RoBERTa-Large（公开权重）
- **关键超参**：temperature=0.4, top_p=0.9, top_k=20, repetition_penalty=1.15, truncation_length=1024, learning_rate=5e-5, batch_size=4, epoch=20, LoRA微调
