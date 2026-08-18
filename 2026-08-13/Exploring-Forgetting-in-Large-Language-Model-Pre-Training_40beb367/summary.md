---
title: "Exploring-Forgetting-in-Large-Language-Model-Pre-Training"
source: https://aclanthology.org/2025.acl-long.105.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:51:20"
field: "大语言模型训练动态与记忆机制"
keywords: ["灾难性遗忘", "预训练遗忘", "实体记忆", "记忆重放", "遗忘曲线", "PPL指标失效"]
innovations: ["首次系统量化LLM预训练阶段的实体遗忘现象", "提出Min/Mex实体相关遗忘指标替代传统PPL/M(f)", "发现LLM遗忘曲线与人类艾宾浩斯曲线高度相似并指导重放设计"]
benchmarks: ["Hellaswag", "MMLU", "Winograd", "The Pile+SlimPajama A+B双数据集"]
---

# 论文速读：Exploring Forgetting in Large Language Model Pre-Training

## 一句话总结
本文首次系统探索大语言模型预训练阶段的**灾难性遗忘**现象，指出传统指标PPL和M(f)难以有效检测遗忘，并提出新的实体相关指标Min/Mex来量化遗忘；同时探索了低成本的记忆重放方法，发现周期性高强度重放可显著缓解遗忘。

## 研究问题与动机
- **预训练遗忘的隐蔽性**：现有研究多关注SFT/微调阶段的遗忘，但预训练阶段同样存在遗忘，且由于数据分布极度多样（含数千种任务），难以用传统任务级指标衡量
- **传统指标的失效**：PPL和M(f)对常见token的高预测准确率产生偏差，会掩盖实体相关信息的遗忘，导致"遗忘被低估"的假象
- **实体信息的脆弱性**：实体相关信息（人名、地点、组织等低频信息）在持续学习中容易被覆盖，直接影响模型事实准确性
- **类人学习规律的启发**：人类记忆研究显示高强度学习可延缓遗忘速率，LLM是否呈现相似规律尚未被探索

## 核心贡献（创新点）
1. **首次系统量化预训练遗忘**：通过A+B双数据集实验（The Pile → SlimPajama），用新指标实证证明预训练阶段存在显著遗忘，区别于微调阶段的任务级遗忘
2. **提出实体相关遗忘指标Min和Mex**：Min衡量"给定上下文能否输出实体细节"，Mex衡量"能否从隐含上下文中召回实体"，二者对遗忘更敏感，不同于PPL/M(f)的概率平均
3. **揭示LLM遗忘曲线的类人特征**：发现初期学习强度越高，长期保留越好，但低强度组会在后期追上——这与人类艾宾浩斯遗忘曲线高度相似
4. **验证周期性高强度重放的有效性**：提出Intensive Focused Stochasticity方法（每100步重放5个epoch），仅增加5%计算成本即可缓解遗忘，且优于基于BM25的相似度重放

## 方法详解
- **A+B双数据集设置**：先训练数据集A（The Pile或OpenWebText），再训练数据集B（SlimPajama），在训练B的过程中持续评估A上的遗忘指标
- **Min指标**：$M_{\text{in}} = \frac{\sum_{s_i \in S}\sum_{i=1}^{32}\mathbb{1}\{o_i = t_i\}}{32|S|}$，给定实体前32个token，贪婪解码32个token，计算与原始后续32个token的逐token准确率
- **Mex指标**：$M_{\text{ex}} = \frac{\sum_{s_i \in S}\text{is\_substring}(c_j, \hat{o})}{|S|}$，给定不包含实体的上下文，解码后检查实体是否出现在生成文本中
- **Entity-focused评估集构建**：从English Wikipedia选取实体，计算其在A、B两数据集的频率差异，选出在A中高频但在B中低频的实体集合C，构建测试集
- **记忆重放策略**：每T步从Elasticsearch存储的样本库中检索replay batch，进行f个epoch的重训；Intensive Focused Stochasticity采用f=5, T=100，计算成本为$T_{\text{replay}} = (1 + f/T)T_0 = 1.05T_0$
- **周期性重放**：每1000步对挑战性实体样本进行5个epoch的密集重放，实验表明该方法可提升遗忘曲线的上下界

## 实验与结果
- **实验设置**：GPT-2（0.1B~1.5B），8×NVIDIA A100（40 GiB VRAM），batch size=576，seq len=1024，BF16精度，cosine LR decay（MaxLr=6×10⁻⁴）
- **数据集**：A = The Pile（13B tokens）或OpenWebText（8B tokens）；B = SlimPajama（49B tokens）
- **关键结果（Table 1）**：
  - Vanilla pre-training：PPL_ent=26.03，M(f)_ent=0.4093，Mex=5.273×10⁻³，Min=3.988×10⁻²
  - Intensive Focused Stochasticity：PPL_ent=25.40，M(f)_ent=0.4121，Mex=5.450×10⁻³，Min=4.003×10⁻²（最佳）
  - BM25相似度重放效果不如随机采样重放，说明均匀性比相关性更重要
- **下游任务验证（Table 2）**：Intensive Focused Stochasticity在Hellaswag（27.75 vs 27.46）、Winograd（55.68 vs 53.47）上优于vanilla，MMLU略有下降（23.00 vs 23.20）
- **遗忘曲线（Figure 4）**：即使后续训练数据集与初始数据集分布相同，Min/Mex仍显著下降；高学习强度组保持优势，且差距在困难样本上更明显
- **LLaMA2-7B扩展（Figure 5）**：较大模型呈现相同遗忘模式，且记忆容量更强，验证结论可迁移性

## 相关工作脉络
1. **McCloskey & Cohen (1989), Ratcliff (1990)**：经典灾难性遗忘理论，本文将其从连接主义网络扩展到LLM预训练场景
2. **Tirumala et al. (2022)**：提出M(f)指标分析LLM训练动态，本文指出M(f)对实体遗忘不敏感，会低估真实遗忘程度
3. **Biderman et al. (2023a)**：研究LLM memorization与过拟合关系，本文借鉴其32-token序列设计思路，转向评估"遗忘"而非"记忆"
4. **de Masson D'Autume et al. (2019)**：在Lifelong Learning中引入episodic memory replay，本文将其适配到预训练阶段并简化为BM25/随机采样
5. **Gupta et al. (2023)**：研究continual pre-training的warmup策略，本文补充指出即使无分布突变，遗忘仍会发生
6. **Luo et al. (2023), Wang et al. (2023b), Wu et al. (2024)**：研究SFT/微调阶段的task-level遗忘，本文首次聚焦pre-training阶段的sample-level实体遗忘，填补空白

## 局限性与未来方向
- **计算成本高**：核心实验需约10,000 GPU小时（8×A100），扩展到更大模型或更多数据集面临算力瓶颈
- **模型规模限制**：主要实验在GPT-2（0.1B~1.5B）上进行，结论在百亿级模型上的外推需谨慎
- **实体覆盖范围有限**：仅评估英文Wikipedia实体，未覆盖数值、代码、多语言等实体类型
- **与SFT遗忘的区分不足**：预训练遗忘机制与SFT阶段遗忘可能存在本质差异，本文未深入对比
- **未来方向**：探索更大模型（如OLMo、Pythia开源checkpoint）上的验证；研究矛盾实体信息对记忆的影响；将replay机制迁移至SFT/RL阶段

## 研究启发与可借鉴点
- **指标设计思路**：Min/Mex将"遗忘检测"从概率平均转向二元判别，对低频/困难样本更敏感——这一思路可迁移至其他需要量化模型能力退化的场景
- **A+B双数据集范式**：通过构造分布相似但实体频率差异大的A、B数据集，有效放大遗忘信号，可作为预训练遗忘研究的通用实验框架
- **类人遗忘曲线的验证**：发现LLM遗忘规律与人类艾宾浩斯曲线高度相似，提示可借鉴认知科学中的复习策略（如spaced repetition）优化pre-training调度
- **低成本replay方法**：Intensive Focused Stochasticity仅增加5%成本即显著缓解遗忘，为工业界预训练流程提供了可直接集成的优化方案
- **Difficulty-aware分析**：按学习难度分层分析遗忘曲线，发现困难样本更依赖高强度学习——这一分层方法可指导 curriculum learning 设计

## 关键术语表
- **Catastrophic Forgetting（灾难性遗忘）**：神经网络在学习新任务/数据时，对旧知识的记忆急剧下降甚至完全丢失的现象
- **M(f) Metric**：Tirumala et al.提出的记忆分数指标，衡量模型对上下文的下一个token预测准确率，但对实体遗忘不敏感
- **Pre-training Forgetting（预训练遗忘）**：指模型在持续预训练过程中，对早期训练数据中实体信息的记忆逐渐衰退的现象
- **M_in（记忆输入指标）**：本文提出的新指标，衡量给定实体前文时模型能否逐token正确输出实体细节
- **M_ex（记忆提取指标）**：本文提出的新指标，衡量给定隐含实体的上下文时模型能否在生成文本中召回该实体
- **Memory Replay（记忆重放）**：在训练新数据时，周期性重新训练已存储的旧样本，以缓解遗忘的方法
- **Intensive Focused Stochasticity**：本文提出的最优重放策略，每100步随机采样replay batch并进行5个epoch的重训
- **Forgetting Curve（遗忘曲线）**：描述记忆保留率随时间（或后续学习量）衰减的曲线，本文发现LLM遗忘曲线与人类曲线形态相似

## 可复现要素
- **数据集**：The Pile（开源，800GB）、SlimPajama（开源，627B tokens）、OpenWebText（开源）、English Wikipedia（开源）
- **代码**：论文未公开代码，但详细描述了实验设置（batch size=576, seq len=1024, BF16, ZeRO Stage 2, cosine LR decay）
- **模型**：GPT-2（0.1B~1.5B，开源权重可复现）、LLaMA2-7B（开源）
- **关键超参**：MaxLr=6×10⁻⁴, MinLr=0.1×MaxLr, replay interval T=100或1000, replay epochs f=5, replay rate=1%
- **硬件**：8×NVIDIA A100（40 GiB VRAM），估算10,000 GPU小时
