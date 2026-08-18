---
title: "Token-Prepending-A-Training-Free-Approach-for-Eliciting-Bett"
source: https://aclanthology.org/2025.acl-long.159.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:23:10"
field: "句子表示学习"
keywords: ["句子嵌入", "大语言模型", "零样本", "Token Prepending", "因果注意力", "Prompt工程"]
innovations: ["提出无需训练的Token Preparing即插即用技术，通过层间预置句向量弥补因果注意力反向依赖", "发现仅前7-8层执行TP操作即可最优，并提出中间层Early-Exit策略", "在多个LLM和多种prompt基线上均显著提升句子embedding质量且推理开销几乎不变"]
benchmarks: ["STS 2012-2016", "STS-B", "SICK-R", "SentEval (MR, CR, SUBJ, MPQA, SST-2, TREC, MRPC)"]
---

# 论文速读：Token-Prepending-A-Training-Free-Approach-for-Eliciting-Bett

## 一句话总结
提出了一种无需训练的**Token Prepending (TP)** 即插即用技术，通过在LLM各Transformer层之间将上一层解码的句子嵌入重新预置到下一层输入开头，使因果注意力机制下的前面token也能感知完整句子语义，从而在不引入额外参数、几乎不增加推理开销的前提下，显著提升从大语言模型中提取的句子表示质量。

## 研究问题与动机
- **核心问题**：LLM多为decoder-only架构，采用causal attention，前面token无法attend到后面token，导致句子信息编码存在系统性偏差，进而影响最终抽取的句子embedding质量。
- **现有方法不足**：PromptEOL等方法仅依赖最后一个token（<SET>）来聚合句子信息，但前面token仍缺乏对后置语义的感知能力。
- **已有方案缺陷**：Echo embeddings（Springer et al., 2024）通过重复输入两次实现"反向依赖"，但会使序列长度翻倍，显著增加推理成本并改变句子结构。
- **Fine-tuning路径代价高**：如BeLLM等方法将因果注意力替换为双向注意力并进行对比学习微调，训练成本高且容易损失LLM原有通用能力。

## 核心贡献（创新点）
1. **提出TP即插即用技术**：通过在中间层替换<PST>为上一层解码的<SET>，使所有token均可感知完整句子信息；本质区别在于无需微调、不改变模型参数，仅干预层间输入。
2. **设计Early-Exit策略**：从中间层（而非最后一层）输出句向量；与基于最后层的PromptEOL等方法本质不同，更契合LLM各层角色差异（末层主要用于token生成，语义信息较弱）。
3. **系统揭示TP有效操作边界**：发现TP只需在模型前7–8层执行即可达到最优，超后期层无需重复操作，显著降低实现复杂度。
4. **证明方法广泛适用性**：在LLaMA2、LLaMA3、Qwen2、Gemma2等多个decoder-only模型上均有效提升，且与多种prompt基线（PromptEOL、MetaEOL、Pretended CoT、Knowledge Enhancement）可无缝组合。

## 方法详解
- **初始Token Prepending**：在输入模板中加入自定义特殊token `<PST>`（不在LLM词表中），随机初始化其embedding，置于输入序列最前方，作为后续层句向量占位符。
- **中间层Token Prepending**：对于第 $l \in [2, k]$ 层，将上一层输出 $h^{l-1}$ 中的`<PST>`位置替换为上一层最后一个token的隐藏状态（即当前层解码出的句子嵌入），形成新输入：
  $$\mathbf{h}^l = \text{LLM}^{l-1}(f(\mathbf{h}^{l-1})), \quad f(\mathbf{h}^{l-1}) \text{将}<\text{PST}>替换为\text{SET}\text{隐藏状态}$$
- **Layer Scope截断**：在第 $k$ 层之后停止TP操作，直接进入标准Transformer前向传播。经验表明$k=7$或$8$时效果最优。
- **Early-Exit策略**：不取最后一层输出，而是选取中间某层（验证集选定，如PromptEOL/MetaEOL/Pretended CoT取第27层，Knowledge Enhancement取倒数第二层）作为最终句向量输出。

## 实验与结果
- **数据集**：7个STS基准（STS 2012–2016、STS-B、SICK-R）及SentEval下游分类任务（MR、CR、SUBJ、MPQA、SST-2、TREC、MRPC）。
- **评估指标**：Spearman相关系数（STS）、Accuracy（分类）；句向量相似度用cosine计算。
- **主要结果（LLaMA2-7B）**：
  - PromptEOL + TP：平均STS提升 **+7.16**（最高单任务STS-B提升+9.01，达80.67）；时间开销仅 **1.04×**。
  - MetaEOL + TP：平均提升 **+1.95**。
  - Pretended CoT + TP：平均提升 **+0.68**。
  - Knowledge + TP：平均提升 **+0.40**。
- **跨模型泛化**：在Qwen2-7B上+2.17、LLaMA2-13B上+1.28、LLaMA3-8B上+0.69，均保持正向提升。
- **迁移任务**：21个分类任务中20个提升，平均约+0.48~0.64。
- **最强结果**：PromptEOL+TP在STS-B上80.67，显著超越所有非微调baseline。

## 相关工作脉络
- **PromptEOL (Jiang et al., 2023)**：首次系统提出用prompt将LLM用于句子embedding抽取；TP与其差异在于TP不依赖prompt设计，而是从模型内部结构干预，且普适性更强。
- **Echo embeddings (Springer et al., 2024)**：通过重复输入两次实现反向依赖建模；TP在同等"信息补充"目标下仅需增加1个token，避免序列翻倍。
- **MetaEOL (Lei et al., 2024)** / **Pretended CoT (Zhang et al., 2024)** / **Knowledge Enhancement (Zhang et al., 2024)**：各类prompt增强方法；TP可作为后处理插件与之组合，提升幅度因prompt复杂度不同而各异（越简单prompt受益越大）。
- **BeLLM (Li & Li, 2024)**：通过替换为双向注意力并微调；TP完全不修改模型结构与参数，属inference-time干预策略。
- **Sentence-T5 (Ni et al., 2022)**：Encoder架构的对比学习微调方法；TP面向decoder-only LLM，且无需任何训练数据。
- **PromptBERT (Jiang et al., 2022)**：针对BERT的prompt方法；本文关注LLM场景，解决因果注意力特有的反向依赖问题。

## 局限性与未来方向
- **超参需手动调优**：TP的结束层$k$与early-exit层$M$需根据模型、数据集和prompt在验证集上确定，增加了新场景适配成本。
- **对复杂prompt增益有限**：当已有prompt本身注入强先验知识时，TP提升幅度较小（如Knowledge仅+0.40），说明二者存在互补而非替代关系。
- **未来方向**：可探索自动选择最佳$k$与$M$的策略（如基于验证集loss或信息量估计）；可将TP思想推广至其他因果模型任务（如长文本摘要、多轮对话表征）；可与轻量级微调方法结合探索trade-off。

## 研究启发与可借鉴点
1. **"层间信息注入"思路可迁移**：将后验层输出反向喂给前层输入，是一种通用的"打破因果attention信息瓶颈"的技术范式，可适配其他因果模型任务。
2. **Early-Exit策略的实证价值**：本文清晰论证了LLM末层不适合做语义表示，这对后续句子embedding抽取工作具有普遍指导意义。
3. **即用型插件设计哲学**：不修改模型权重、仅干预输入序列的流程，极大降低了工程落地门槛，值得在更多LLM应用（如RAG检索编码）中推广。
4. **消融实验设计严谨**：对`<PST>`位置、初始化方式、层数边界、token数量等多维度进行系统ablation，为后续工作提供了完整参考基线。

## 关键术语表
- **Token Prepending (TP)**：将每层解码出的句子嵌入预置到下一层输入起始位置的技术，用于弥补causal attention中的反向依赖缺失。
- **<PST> token**：Placeholder Sentence Token，自定义占位符，初始随机embedding，用于在输入序列最前方承载句子语义信息。
- **<SET> token**：Sentence Embedding Token，由prompt引导模型将句子信息编码进其对应位置的隐藏状态（通常为最后一个token）。
- **Causal Attention**：自回归模型使用的单向注意力机制，token只能看到自身及之前位置的输入，无法感知后置信息。
- **Early-Exit Strategy**：从LLM的中间层而非最后一层提取句向量，避免末层因主要用于token预测而导致语义信息衰减。
- **PromptEOL**：一种通过特定prompt模板（如"This sentence: '[Text]' means in one word: '<SET>'"）引导LLM提取句子embedding的方法。
- **STS (Semantic Textual Similarity)**：语义文本相似度任务，衡量两句话在语义层面的相似程度，常用Spearman相关系数评估。
- **Backward Dependency**：句子中前面token对后面token的语义依赖关系，是causal attention模型抽取句子表示的主要障碍。

## 可复现要素
- **数据集**：STS 2012–2016、STS-B、SICK-R、SentEval标准任务；均为公开基准。
- **代码**：已开源（https://github.com/fuyuchenIfyw/token_prepending.git）。
- **权重**：未引入新参数，使用开源LLM（LLaMA2-7B/13B、LLaMA3-8B、Qwen2-7B、Gemma2-9B）原生权重。
- **关键超参**：
  - TP操作结束层 $k = 8$（LLaMA2）或 $k = 7$（Qwen2/Gemma2）。
  - Early-exit层 $M = 27$（PromptEOL/MetaEOL/Pretended CoT）或倒数第二层（Knowledge Enhancement）。
  - `<PST>` 初始化：论文测试了全0、全1、均匀分布、高斯分布及space token embedding，差异<0.01，实践中建议任选。
  - `<PST>` 位置：统一置于prompt冒号后、输入文本前。
