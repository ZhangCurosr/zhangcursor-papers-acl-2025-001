---
title: "Contrastive-Prompting-Enhances-Sentence-Embeddings-in-LLMs-t"
source: https://aclanthology.org/2025.acl-long.174.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:53:36"
field: "LLM表征学习"
keywords: ["句子嵌入", "大语言模型", "对比提示", "推理时干预", "激活导向", "零样本", "语义相似度"]
innovations: ["提出对比提示方法，通过辅助提示与正常提示的激活差分化干预最后一token隐藏状态", "首次将推理时激活干预应用于句子嵌入提取任务，无需训练或额外数据", "设计两种范数调整策略（NS/NR）并证明中间层提取优于最后一层"]
benchmarks: ["STS12-16", "STS-B", "SICK-R", "SentEval", "MTEB"]
---

# 论文速读：Contrastive-Prompting-Enhances-Sentence-Embeddings-in-LLMs-t

## 一句话总结
本文提出**对比提示（Contrastive Prompting, CP）**方法，通过在推理时引入辅助提示与正常提示进行对比，对LLM中间层激活值进行干预，引导模型将句子核心语义编码到最后一个token的嵌入中，过滤stop words等非必要信息，无需任何训练或额外数据即可显著提升从LLM提取的句子嵌入质量。

## 研究问题与动机
1. 从LLM直接提取句子嵌入无需微调/额外数据，具有实践价值，但现有方法（如PromptEOL、Knowledge等）通过提示工程间接引导最后一个token编码语义时，仍会过度编码stop words等非必要信息。
2. 不同提示方法的语义聚焦程度存在差异，简单提示（如PromptEOL）效果受限，需要更普适的改进手段。
3. 推理时干预（inference-time steering）是一种无需训练的即用型方法，但在句子嵌入提取场景中尚未被系统探索。
4. 现有方法多从最后一层提取嵌入，而研究表明中间层可能更适合捕捉语义信息，但缺乏有效的提取机制。

## 核心贡献（创新点）
1. **首次提出推理时激活干预提升句子嵌入质量**：与已有激活导向（activation steering）依赖监督数据不同，CP仅需通过两个不同提示生成激活向量差分，无需额外训练数据。
2. **对比提示（CP）框架**：通过引入辅助提示（编码非核心信息）与正常提示进行对比，直接修改正常提示最后一token的隐藏状态，使其聚焦核心语义——与已有方法的本质区别在于从"提示工程设计"转向"推理时表征干预"。
3. **两种范数调整策略**（Norm Scaling / Norm Recovering）：解决干预后嵌入范数变化问题，NS引入超参数α控制干预强度，NR保持干预前后范数一致，避免了额外超参引入。
4. **插件式通用性**：CP可无缝结合任意现有基于提示的方法（PromptEOL、Pretended CoT、Knowledge等），且辅助提示仅需传播到LLM较低层，额外开销极小。

## 方法详解
CP方法分为三步，是一种即插即用的推理时干预算法：

**Step 1 - 辅助提示传播**：使用辅助提示模板"The irrelevant information of this sentence: "[TEXT]" means in one word: "包裹输入文本，前向传播至LLM第ℓ层多头注意力层，提取上下文值向量 $\mathbf{v}^{\mathrm{aux},(\ell)}$，该向量编码句子的非必要信息。

**Step 2 - 对比激活干预**：用正常提示（如PromptEOL、Knowledge等）包裹文本，传播至同一层得到 $\mathbf{v}^{\mathrm{nor},(\ell)}$，计算语义激活向量：
$$\Delta \mathbf{v}^{\ell} = \mathbf{v}_{N_{\mathrm{nor}}}^{\mathrm{nor},(\ell)} - \mathbf{v}_{N_{\mathrm{aux}}}^{\mathrm{aux},(\ell)}$$
该差分操作去除非必要信息，聚焦核心语义。仅对最后一个token的向量进行干预。

**Step 3 - 范数调整与前向传播**：
- **Norm Scaling (NS)**：$\hat{\mathbf{v}} = \alpha \cdot \Delta \mathbf{v}^{\ell}$，引入缩放因子α控制干预强度。
- **Norm Recovering (NR)**：$\hat{\mathbf{v}} = \Delta \mathbf{v}^{\ell} \cdot \frac{\|\mathbf{v}_{N_{\mathrm{nor}}}^{\mathrm{nor},(\ell)}\|_2}{\|\Delta \mathbf{v}^{\ell}\|_2}$，保持干预前后范数一致。
将调整后的向量替换回序列，继续前向传播，从指定输出层提取最后一个token的隐藏状态作为句子嵌入。

**中间层提取**：实验发现从中间层（非最后一层）提取嵌入效果更好，可通过验证集搜索最优层。

## 实验与结果
**数据集**：STSbenchmark（STS12-16、STS-B、SICK-R）共7个语义文本相似度数据集；SentEval迁移学习任务（MR、CR、SUBJ、MPQA、SST-2、TREC、MRPC）。

**评估指标**：Spearman相关系数（STS任务）、准确率（分类任务）。

**主要结果（LLaMA2-7B骨干）**：
- **PromptEOL + CP-NS**：平均75.27分，较PromptEOL提升**+5.24分**，STS-B上提升**+7.05分**（70.60→78.71）。
- **Pretended CoT + CP-NS**：平均77.45分，提升**+0.59分**。
- **Knowledge + CP-NS**：平均77.56分，提升**+0.42分**。
- **CK（CoT+Knowledge平均）+ CP-NS**：平均78.68分，超越所有单一提示方法，提升**+0.45分**。

**跨模型泛化**：在LLaMA2-13B和LLaMA3.1-8B上均取得一致提升，LLaMA3.1-8B上Pretended CoT+CP-NS提升**+1.15分**。

**迁移学习任务**：PromptEOL+CP-NS平均准确率91.73分，超越4.8B参数、经过监督对比学习的ST5-Enc（91.63分），无需任何训练。

**最强结果**：PromptEOL+CP-NS在STS-B上达到78.71分，较基线提升7.05分，为所有实验中最大提升幅度。

## 相关工作脉络
1. **PromptEOL**（Jiang et al., 2023）：首个将LLM最后一token隐藏状态用于句子嵌入的工作，本文在其基础上引入推理时干预进一步提升。
2. **MetaEOL/Pretended CoT/Knowledge**（Lei et al., 2024; Zhang et al., 2024）：通过复杂提示设计引导语义聚焦，本文从"提示工程"转向"激活干预"，不依赖特定提示设计。
3. **Activation Steering**（Zou et al., 2023; Rimsky et al., 2024）：基于正负样本差分生成定向向量，需要监督数据；CP无需监督数据，仅通过两个提示的激活差分实现目标。
4. **LLM2Vec/Generative Representation Instruction Tuning**（BehnamGhader et al., 2024; Muennighoff et al., 2024）：需微调LLM获得嵌入；CP完全零样本、零训练。
5. **SimCSE/ST5-Enc**（Gao et al., 2021; Ni et al., 2022）：基于对比学习的专用编码器，需大量训练数据；CP在无需训练的情况下可匹敌甚至超越这些监督方法。

## 局限性与未来方向
1. **辅助提示设计待优化**：当前使用固定模板，未来需探索如何为每个正常提示自动生成最优辅助提示。
2. **超参数搜索需求**：NS需搜索干预层和缩放因子α，NR需搜索干预层，虽然实验表明第5/7层通常表现良好，但仍需一定调参。
3. **仅限英文评估**：实验仅在英文数据集上进行，多语言泛化性有待验证。
4. **干预层选择的普适性**：不同提示方法的最优干预层不同（PromptEOL为第5层，Pretended CoT/Knowledge为第7层），需针对不同场景自适应选择。

## 研究启发与可借鉴点
1. **"对比干预"范式可迁移**：利用两个不同视角的输入产生表征差分，再对目标表征进行干预，这一思路可推广至其他LLM表征任务（如段落嵌入、对话嵌入）。
2. **推理时干预无需训练的实用价值**：对于无法微调的大规模LLM，CP提供了一种即插即用的质量提升方案，适合生产环境部署。
3. **中间层提取嵌入的实验设计**：通过验证集搜索最优输出层，这一轻量级策略可被其他嵌入提取工作借鉴。
4. **与多提示平均（CK）结合**：CP不仅能提升单一提示，还能进一步提升多提示平均的效果，说明多提示融合仍含冗余信息，CP的过滤机制具有补充价值。
5. **可扩展到In-Context Learning场景**：附录B证明CP在ICL设置下同样有效，尽管提升幅度较小（因上下文已提供部分引导）。

## 关键术语表
**Contrastive Prompting (CP)**：一种推理时干预方法，通过引入辅助提示提取非核心信息向量，与正常提示的向量做差，对最后一token的隐藏状态进行修正，引导模型聚焦核心语义。

**Inference-time Steering**：在模型推理过程中直接修改中间层激活值，无需训练即可改变模型输出行为的技术。

**Norm Scaling (NS)**：CP中的范数调整策略之一，引入缩放因子α对干预后的激活向量进行缩放，控制干预强度。

**Norm Recovering (NR)**：CP中的范数调整策略之二，通过归一化使干预前后向量L2范数保持一致，避免引入额外超参。

**Contextualized Value Vector**：多头注意力层中每个token的上下文值向量 $\mathbf{v}^{\ell,h}$，是信息交互的关键表征，CP在此层进行干预。

**STS (Semantic Textual Similarity)**：语义文本相似度任务，评估两个句子语义相似程度的基准数据集系列。

## 可复现要素
- **数据集**：STS12-16、STS-B、SICK-R（公开）；SentEval任务（公开）；MTEB子集（公开）
- **代码**：论文声明代码将在 https://github.com/zifengcheng/CP 开源
- **权重**：使用开源LLaMA2-7B/13B、LLaMA3.1-8B
- **关键超参**：干预层ℓ∈{3,4,5,6,7}；NS的缩放因子α∈{0.5,1,2,3,4}；PromptEOL最优干预层为第5层、α=2；Pretended CoT/Knowledge最优干预层为第7层、α=3
