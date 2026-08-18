---
title: "Do-Large-Language-Models-Have-an-English-Accent-Evaluating-a"
source: https://aclanthology.org/2025.acl-long.193.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:49:34"
field: "多语言大语言模型评估"
keywords: ["多语言LLM", "语言自然度评估", "词汇分布", "句法自然度", "DPO对齐", "翻译腔", "跨语言评测"]
innovations: ["提出基于JSD的词汇自然度与基于WL Kernel+MMD的句法自然度语料库级指标，无需外部嵌入模型", "构建Wikipedia跨语言自然度基准，首次系统揭示多语言LLM的非英语输出存在词汇/句法层面的英语口音现象", "基于DPO与回译构造的不自然偏好数据，实现多语言LLM的目标语言自然度对齐，且不损害通用能力"]
benchmarks: ["Wikipedia Description Generation", "CMMLU"]
---

# 论文速读：Do-Large-Language-Models-Have-an-English-Accent-Evaluating-a

## 一句话总结
本文针对多语言大语言模型在非英语语言中输出不够自然的问题，提出了语料库级的词汇自然度（基于JSD）与句法自然度（基于WL Kernel + MMD）评估指标，并在英语、法语、中文上系统评测了主流多语言LLM；同时提出一种基于DPO的简单对齐方法，利用人工构造的不自然偏好数据对模型进行风格微调，在不损害通用能力的前提下显著提升了目标语言的输出自然度。

## 研究问题与动机
1. **英语中心主义训练导致非英语输出不自然**：主流LLM（如Llama 3.1）训练数据中仅8%为非英语，模型在法语、中文等语言中常表现出英语-centric的词汇选择与句法结构，类似于二语学习者的"英语口音"。
2. **现有评估体系忽视自然度维度**：当前多语言LLM评测主要聚焦任务解决能力（如Helpfulness、Factual Accuracy、Safety），缺乏对输出语言形式自然度的系统性度量。
3. **翻译文本引入的"翻译腔"污染**：多语言LLM在预训练或后训练中常接触机器翻译或人工翻译文本，这类文本存在翻译腔（translationese） artifacts，导致模型模仿此类不自然表达。
4. **语言代表性不公平的社会影响**：若LLM在低资源语言中持续产出不够自然的输出，将加剧这些语言社群的技术使用不平等。

## 核心贡献（创新点）
1. **提出首个语料库级词汇自然度指标**：基于Jensen-Shannon Divergence（JSD）比较LLM生成文本与人类文本的词汇分布差异，避免了样本级参考指标（BLEU/ROUGE）对开放生成任务的局限性。
2. **提出基于图核的句法自然度指标**：利用Universal Dependencies依存树结合Weisfeiler-Lehman图核（WL Kernel）与Maximum Mean Discrepancy（MMD），直接量化句法结构分布差异，且无需外部嵌入模型，避免英语中心偏差。
3. **构建跨语言自然度基准并揭示英语口音现象**：基于Wikipedia构建3,722条多语言对齐数据集，评测Llama、Qwen、Mistral系列模型，首次系统展示LLM在非英语输出中存在的词汇/句法层面英语倾向。
4. **提出基于DPO的自然度对齐方法**：通过回译/改写构造不自然拒绝响应，构建偏好数据集，使用LoRA+DPO对模型进行轻量微调，在中文任务上实现词汇与句法自然度的一致性提升，且不降低CMMLU通用能力。

## 方法详解
**词汇自然度（Lexical Naturalness）**：
- 在词级别（而非subword token级别）统计LLM生成文本与人类文本的词汇分布 $P$ 与 $Q$。
- 使用Jensen-Shannon Divergence（JSD）衡量两个分布的差异：
  $$\mathrm{JSD}(P \| Q) = \frac{1}{2}\left(D_{\mathrm{KL}}(P \| M) + D_{\mathrm{KL}}(Q \| M)\right), \quad M = \frac{1}{2}(P + Q)$$
- 值越低表示模型词汇分布越接近人类，自然度越高。

**句法自然度（Syntactic Naturalness）**：
- 使用Stanza工具基于UD语法将句子解析为依存树，节点标注POS tag。
- 计算句子对之间的结构相似度：采用Weisfeiler-Lehman图核（迭代深度H=2），通过邻居标签聚合生成层次化编码：
  $$K_{\mathrm{WL}}(T_1, T_2) = \sum_{h=0}^{H} \sum_{(v_1,v_2)\in(V_1,V_2)} \delta(\ell_h(v_1), \ell_h(v_2))$$
- 构造人类句子集 $\{s_i^h\}$ 与模型句子集 $\{s_j^m\}$ 的核矩阵 $\mathbf{K}$，使用MMD²度量分布差异：
  $$\mathrm{MMD}^2 = \frac{1}{N_h^2}\sum_{i,i'} K_{ii'} + \frac{1}{N_m^2}\sum_{j,j'} K_{jj'} - \frac{2}{N_h N_m}\sum_{i,j} K_{ij}$$

**偏好数据集构建**：
- 保留SFT数据中的原始指令作为preferred response。
- 对rejected response采用**回译**（中文→英语→中文）或**改写**引入翻译腔伪影。
- 过滤条件：$0.15 < \mathrm{BLEU}(\mathrm{Chosen}, \mathrm{Rejected}) < 0.9$，且响应长度≥10词。
- 使用DPO + LoRA进行对齐微调（学习率5e-6, β=0.5, r=256, lora_alpha=128）。

## 实验与结果
**数据集**：3,722条Wikipedia条目（多语言对齐），覆盖英语、中文、法语；每条目生成描述，生成60K词用于词汇评估、3K句用于句法评估。

**评估模型**：Llama-3/3.1-8B、Qwen1.5/2-7B、Mistral-v0.3-Nemo-12B（instruction-tuned版本）。

**主要结果（Table 1）**：
- 英语：词汇分歧最低（23.07人类基准），Llama-3.1词汇分歧26.79，句法分歧16.80。
- 中文：人类基准词汇分歧25.91，Llama-3.1为33.29（最低），Qwen1.5达41.00（最高）；句法分歧人类基准2.93，Mistral-Nemo 12.84最优，Qwen1.5达23.33最差。
- 法语：人类基准词汇分歧24.25，Mistral-v0.3在词汇上最优（28.73），句法上Mistral-Nemo 11.27最优。
- **关键发现**：中文/法语的非英语输出分歧显著高于英语；Llama系列整体优于Qwen系列；新模型版本较旧版本有稳定提升。

**分析实验**：
- 提示语言影响：中文输出使用中文提示优于英文提示；法语输出提示语言效应因模型而异。
- 解码温度：温度升高对已有自然模型进一步提升句法自然度，对不自然模型则恶化；词汇自然度随温度升高普遍改善。
- DPO对齐效果（Table 3）：Llama-3.1在问答任务上词汇分歧从28.05降至28.01，句法从12.19降至9.44；Qwen2词汇从28.09降至26.92，句法从12.25降至11.19；CMMLU准确率均无下降（Llama: 55.38→55.47，Qwen: 80.08→80.49）。

## 相关工作脉络
1. **翻译腔检测**（Translationese detection）： Freitag et al. (2022) 使用型次比、句法树高、困惑度等特征训练分类器区分机器翻译与自然文本，但依赖预训练分类器且过拟合风险高；本文避免此限制，采用自动构造的不自然文本与偏好学习。
2. **语言多样性评估**： Tevet & Berant (2021)、Guo et al. (2024) 指出合成文本的语言多样性下降是不自然的重要信号；本文在此基础上直接对比词汇/句法分布差异，而非仅评估离散度。
3. **多语言LLM内部语言分析**： Wendler et al. (2024) 证明LLaMA-2概念空间与英语更对齐；Papadimitriou et al. (2023) 发现Multilingual BERT存在英语句法偏向；本文首次系统研究此类英语中心倾向如何影响下游开放生成的语言自然度。
4. **自然度评估的已有方法**： 传统MT领域依赖人工评分（Chen et al., 2024）或BERT分类器（Liu et al., 2021）；本文首次在无参照的开放生成场景中提出语料库级自动指标。

## 局限性与未来方向
1. **未覆盖低资源语言**：方法依赖原生人类文本的地面真值分布，而低资源语言此类数据稀缺；且缺乏可靠的词法分词器与依存解析工具。
2. **仅限Wikipedia领域**：跨语言基准仅覆盖Wikipedia摘要生成任务，结论可能不适用于其他领域（如对话、创意写作、专业领域）。
3. **对齐实验仅限中文**：法语等非英语语言缺乏足够的原生SFT数据（如Aya数据集法语过滤后仅958条），未进行多语言对齐实验。
4. **未考虑社会偏见**：当前评估仅关注语言形式自然度，未涉及输出内容的社会偏见或公平性。
5. **知识库密集型任务的风险**：自然度对齐在开放性创作任务上有效，但在知识密集型任务中可能引入意外的知识编辑，增加幻觉风险。

## 研究启发与可借鉴点
1. **语料库级分布对比思路可迁移**：JSD+WL Kernel的组合范式可推广至其他语言（需适配分词器与解析器），也可用于评估多语言大模型的跨语言一致性。
2. **偏好数据构造策略可直接复用**：回译引入翻译腔、配合BLEU阈值过滤的方法，可低成本构建任意语言的不自然偏好对，适用于多领域风格对齐。
3. **解码温度与自然度关系的发现具有实用价值**：提示自然度较差的模型可适当降低温度以提升词汇自然度，而句法自然度优化需结合模型基线水平选择温度。
4. **可作为多语言模型评测的补充维度**：当前主流评测（MMLU、CMMLU等）只衡量任务能力，本文提出的词汇/句法分歧指标可作为自然度基准纳入多语言LLM的标准评测流程。

## 关键术语表
**Lexical Naturalness（词汇自然度）**：通过JSD度量LLM生成文本与人类文本的词汇分布差异，越低越自然。
**Syntactic Naturalness（句法自然度）**：基于UD依存树与WL图核+MMD，量化模型生成句子的句法结构分布与人类分布的偏离程度。
**Translationese（翻译腔）**：翻译文本中区别于原生文本的语言特征，如过度使用被动语态、介词短语等。
**Weisfeiler-Lehman Kernel（WL图核）**：通过迭代聚合邻居标签对图结构进行层次化编码，用于计算依存树之间的结构相似度。
**Maximum Mean Discrepancy（MMD）**：基于核函数的两样本距离度量，用于比较人类与模型生成句子的句法分布差异。
**DPO（Direct Preference Optimization）**：无需显式奖励模型的偏好优化方法，直接利用偏好对更新策略。
**Cross-lingual Naturalness Benchmark（跨语言自然度基准）**：本文构建的基于Wikipedia的多语言对齐数据集，包含3,722条英语/中文/法语条目。
**Self-alignment vs. Cross-model alignment**：前者用同一模型生成偏好对进行自训练，后者用A模型生成数据微调B模型。

## 可复现要素
- **数据集**：Wikipedia Description Generation数据集，3,722条多语言条目（公开可获取），预处理脚本见附录A/B。
- **代码**：论文未明确声明代码开源；指标实现引用了Stanza、GraKeL、Jieba、NLTK等外部库；DPO实验使用trl与PEFT库。
- **关键超参**：Temperature=0.6（主实验），repetition_penalty=1.02；DPO学习率5e-6，β=0.5，LoRA r=256，lora_alpha=128，lora_dropout=0.05，warmup_ratio=0.1，batch_size=6，bf16精度，1 epoch；WL Kernel迭代次数H=2；BLEU阈值0.15~0.9过滤偏好对。
