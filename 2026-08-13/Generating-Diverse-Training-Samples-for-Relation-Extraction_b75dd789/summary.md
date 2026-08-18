---
title: "Generating-Diverse-Training-Samples-for-Relation-Extraction"
source: https://aclanthology.org/2025.acl-long.35.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:53:10"
---

# 论文速读：Generating-Diverse-Training-Samples-for-Relation-Extraction

## 一句话总结
本文针对大语言模型直接生成关系抽取（RE）训练样本时句法结构趋同、表达动词重复的问题，提出了基于增量式上下文学习（ICL）的逐条生成策略，并结合直接偏好优化（DPO）对LLM进行免人工标注的微调；实验表明，用此类高多样性生成样本训练非LLM的RE模型，性能可匹敌甚至超越直接调用LLM进行零样本抽取。

## 研究问题与动机
- **低资源RE的数据瓶颈与生成样本同质化：** 关系抽取在垂直领域或新类别下常面临标注稀缺问题，利用LLM合成训练数据是可行路径，但直接Prompt生成的样本在句式骨架与关系表达动词上高度雷同，多样性严重不足，难以有效覆盖长尾分布。
- **既有生成方法的局限：** 现有LLM数据合成工作（如Self-Instruct、RelationPrompt）多关注指令质量或未知关系类型的发现，未针对RE任务的结构化输出特性设计动态多样性约束；且直接让LLM执行RE（Direct-RE）在Few-shot场景下性能有限、调用成本高。
- **偏好数据的获取成本：** 传统的强化学习对齐需要显式奖励模型或人工偏好标注，在低资源NER/RE任务中难以规模化获取，亟需一种无需人工干预的自动化偏好构造机制。

## 核心贡献（创新点）
1. 提出增量式单条生成（OBO）策略，每次仅要求模型生成一条新样本并将其动态追加至Prompt演示池，本质区别在于通过滚动上下文提供实时负反馈，而AAO等一次性生成方法缺乏迭代约束易导致模式坍塌。
2. 设计基于DPO的多样性微调框架，利用自动构造的三类反样本（标签错误、结构相似、完全重复）替代人工偏好标注，本质区别在于将DPO从对话/摘要对齐扩展至“受约束的数据生成”任务，彻底消除人工标注依赖。
3. 验证“LLM生成数据训练非LLM下游模型”范式的低资源有效性，本质区别在于突破了对大模型在线推理的依赖，证明高质量的合成数据可迁移至轻量级编码器模型，大幅降低部署成本。
4. 建立双维度多样性量化评估体系（语义余弦相似度与词重复率），并揭示生成数量存在约K=32的性能天花板，本质区别在于首次对LLM生成RE样本的多样性瓶颈进行系统性实证刻画，为合成数据规模规划提供依据。

## 方法详解
- **Prompt三模块结构：** 包含Task Description（定义RE训练样本需含文本、头尾实体及其位置、关系类别，并要求新样本与Demonstration尽可能不同）、Relation Explanation（为每个目标关系类别提供自然语言释义）、Sample Demonstration（初始随机抽取1条真实标注样本，随生成过程累加）。
- **ICL生成模式：** 对比两种流程。AAO（All-at-Once）单次指令输出K个样本；OBO（One-by-One）每次指令仅生成1个样本，生成后将其移入Demonstration池，下一轮Prompt的演示样本数递增，迫使模型参考历史输出以维持多样性。
- **DPO目标函数：** 优化目标为 $\max_{\pi} \sum_i \log \sigma\left(\frac{1}{\beta}\log\frac{\pi(y_{i,1}|x_i)}{\pi(y_{i,2}|x_i)}\right)$，其中 $\pi(y|x)$ 为优化策略。偏好输出 $y_{i,1}$ 为从真实标注数据中随机采样的目标关系样本；非偏好输出 $y_{i,2}$ 由三類自动构造样本构成：① 标签污染样本（其他关系类别的真实样本修改标签为目标关系）；② 扰动相似样本（保留Demonstration结构，替换头尾实体或增删上下文词汇）；③ 完全复制样本（直接复用已有Demonstration）。
- **分布对齐构造：** 为匹配OBO推理过程，DPO训练集的Instruction中Demonstration数量呈阶梯递增，将上一轮Output的偏好样本依次注入下一轮的Instruction演示模块，确保微调分布与生成推理分布一致。
- **防作弊划分：** 为避免模型在DPO阶段“记忆”生成目标，将各数据集的关系类别均分为两组，一组专用于DPO微调，另一组专用于样本生成，实验时互换轮换以覆盖全量关系。

## 实验与结果
- **数据集：** SemEval 2010 Task 8（19类）、TACRED（42类）、TACRED-Revisit、Re-TACRED（40类）。
- **评估协议：** 将生成样本用于训练 KnowPrompt 与 RetrievalRE（均基于 RoBERTa-LARGE），以Few-shot Micro F1为指标；基线包括 Direct-RE（2-shot LLM直接抽取）、Data Generation（Xu et al., 2023）、纯手动标注数据训练。
- **主要数字：** 在TACRED-K=32下，Ours (mix-OBO) 达36.25%，Ours (mix-AAO) 达35.66%，略优于纯手动标注的KnowPrompt（36.33%）与RetrievalRE（37.07%），且大幅领先Ours (pure)（30.41%）。在Re-TACRED-K=32下，Ours (mix-OBO) 达64.58%，较RetrievalRE基线（55.79%）提升约8.8个百分点。相比Direct-RE，纯生成样本训练的模型在全部数据集上均更优。
- **最强结果与提升：** TACRED-K=32下Ours (mix-OBO) 的36.25%为KnowPrompt设定下的最高分，较纯LLM生成样本（Ours pure 30.41%）提升约5.8绝对点，较Direct-RE（21.17%）提升逾15个百分点。
- **消融结论：** 移除DPO或多样性指令后性能普遍下降；OBO整体优于AAO；生成数量在K=16~32区间触及性能峰值，超32后收益饱和，归因于LLM语料多样性瓶颈。

## 相关工作脉络
- **Xu et al. (2023) Data Generation：** 同样利用LLM生成RE数据辅助训练，但依赖静态详细Prompt描述，未引入动态反馈与多样性优化机制；本文通过OBO累积上下文与DPO反样本明确解决同质化。
- **Chia et al. (2022) RelationPrompt：** 聚焦未见关系类型的结构化文本合成；本文针对已知关系类别的样本多样性增强，面向低资源而非完全零样本设定。
- **Wang et al. (2023) Self-Instruct：** 通过LLM自生成指令微调数据，未考虑关系抽取的结构约束与重复抑制；本文设计了领域特定的关系解释模块与自动反样本构造策略。
- **Rafailov et al. (2023) DPO：** 原始方法应用于对话/摘要偏好对齐；本文将其适配至生成任务，创新性地用规则自动构造的负样本替代人工标注，拓展了DPO的适用范围。
- **Chen et al. (2022a,b) RetrievalRE & KnowPrompt：** 作为下游评测基线；本文证明生成数据可与其知识注入/检索增强范式无缝衔接，形成“生成-训练-评测”闭环。
- **Meng et al. (2022) / Ye et al. (2022) Zerogen：** 早期基于PLM的合成数据方法；本文依托Instruction-Tuned LLMs在Few-shot下的强大涌现能力，实现更高保真度与可控多样性。

## 局限性与未来方向
- **基座模型能力依赖：** 生成样本质量受限于LLaMA2-7b-Chat的语言知识与知识覆盖面，更强基座有望突破当前多样性天花板。
- **单关系生成数量上限：** 受LLM训练语料分布限制，单一关系的合法多样样本数约在16~32条即趋于饱和，难以通过简单增加采样数无限
