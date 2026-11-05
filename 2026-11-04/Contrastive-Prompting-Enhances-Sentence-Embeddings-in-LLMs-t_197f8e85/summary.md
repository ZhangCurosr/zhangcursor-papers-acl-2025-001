---
title: "Contrastive-Prompting-Enhances-Sentence-Embeddings-in-LLMs-t"
source: https://aclanthology.org/2025.acl-long.174.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:42:19"
field: "LLM表示学习"
keywords: ["句子嵌入", "大语言模型", "推理时干预", "对比学习", "提示工程", "激活引导"]
innovations: ["首次提出推理时双提示对比激活引导方法提升句子嵌入质量", "设计辅助提示提取非本质信息并与正常提示做隐状态差值对比", "提出NS/NR两种范数调整策略控制干预强度"]
benchmarks: ["STS 2012-2016", "STS-B", "SICK-R", "SentEval", "MTEB"]
---

# 论文速读：Contrastive-Prompting-Enhances-Sentence-Embeddings-in-LLMs-t

## 一句话总结
本文提出**对比提示（Contrastive Prompting, CP）**方法，通过在推理时引入辅助提示来对比正常提示，直接从LLM中间层隐状态中提取高质量句子嵌入，无需额外数据或微调。CP可插拔地提升多种现有提示方法在STS任务和下游分类任务上的性能。

## 研究问题与动机
1. **现有提示方法仍编码过多非本质信息**：PromptEOL、Pretended CoT、Knowledge等提示工程方法虽引导LLM将句子语义压缩到最后token，但解码实验显示最高概率token仍为"it"、"a"、"the"等stop words，语义聚焦能力受限。
2. **微调方法成本高且损失通用能力**：直接微调LLM获取句子嵌入（如LLM2Vec、NV-Embed）需要大量数据和计算资源，且会损害模型其他通用能力，零样本场景更实用。
3. **现有激活引导方法依赖监督数据**：Activation steering通常需正负样本对训练，无法直接应用于句子嵌入提取这一无监督/零样本场景。
4. **多层输出并非最优**：研究表明LLM不同层承担不同功能，最后一层嵌入主要用于预测，未必是句子表示的最佳来源。

## 核心贡献（创新点）
1. **首次通过推理时激活引导提升句子嵌入质量**：区别于监督式activation steering，CP利用两种提示的隐状态差异生成sentence-dependent激活向量，无需任何训练数据。
2. **提出对比提示框架与多种辅助提示设计**：引入"The irrelevant/redundant/descriptive information of this sentence..."作为辅助提示，经验证对辅助提示选择鲁棒（前五种语义相关提示均有效）。
3. **提出两种范数调整策略以控制干预幅度**：Norm Scaling（引入缩放因子α）和Norm Recovering（保持干预前后L2范数一致），实验表明NS策略整体优于NR策略。
4. **系统验证CP在STS和下游任务上的泛化性**：在LLaMA2-7B/13B、LLaMA3.1-8B上验证，CP+PromptEOL平均提升5.24分，超越48亿参数 supervised model ST5-Enc。
5. **揭示多层输出的优势**：证明CP不仅改善单一层嵌入，还能提升LLM中间及深层所有层的表示质量，且最优输出层因prompt而异（如PromptEOL为27层，Knowledge为31层）。

## 方法详解
**整体流程**：CP为三步推理时干预算法，需对正常提示（normal prompt）和辅助提示（auxiliary prompt）各做一次前向传播。

**Step 1 — 辅助提示提取非本质信息向量**：
- 辅助提示模板：`The irrelevant information of this sentence: "[TEXT]" means in one word: "`
- 输入LLM前ℓ层（多头注意力层），提取最后一位置的contextualized value vector $\mathbf{v}_{N_{aux}}^{aux,(\ell)}$

**Step 2 — 对比激活引导**：
- 正常提示（任意existing prompt如PromptEOL/Pretended CoT/Knowledge）前向传播至同层ℓ，提取$\mathbf{v}_{N_{nor}}^{nor,(\ell)}$
- 计算语义激活向量：$\Delta\mathbf{v}^{\ell} = \mathbf{v}_{N_{nor}}^{nor,(\ell)} - \mathbf{v}_{N_{aux}}^{aux,(\ell)}$（公式1）
- 该差值向量剔除非本质信息，聚焦核心语义

**Step 3 — 范数调整与继续前向传播**：
- **Norm Scaling (NS)**：$\hat{\mathbf{v}}_{N_{nor}}^{nor,(\ell)} = \alpha \cdot \Delta\mathbf{v}^{\ell}$（公式2），引入超参α控制干预强度
- **Norm Recovering (NR)**：$\hat{\mathbf{v}}_{N_{nor}}^{nor,(\ell)} = \Delta\mathbf{v}^{\ell} \cdot \frac{\|\mathbf{v}_{N_{nor}}^{nor,(\ell)}\|_2}{\|\Delta\mathbf{v}^{\ell}\|_2}$（公式3），保持干预前后范数一致
- 用调整后的向量替换原value vector，继续标准前向传播，最终在指定输出层提取句子嵌入

**关键设计细节**：
- 干预位置：多头注意力层（Multi-head Attention）效果最佳（Table 5，+0.78 vs FFN +0.10）
- 干预层搜索空间：$\ell \in \{3,4,5,6,7\}$，PromptEOL最优为第5层，Pretended CoT/Knowledge为第7层
- 输出层：采用中间层而非最后一层（因最后一层嵌入主要用于预测任务）

## 实验与结果
**数据集**：7个STS基准（STS 2012–2016、STS-B、SICK-R）+ SentEval下游任务（MR/CR/SUBJ/MPQA/SST-2/TREC/MRPC）+ MTEB子集（分类/聚类/重排序/pair classification）

**基线**：BERT avg/prompt、ST5-Enc、LLaMA2 avg/echo、MetaEOL、PromptEOL、Pretended CoT、Knowledge、CK（CoT+Knowledge平均）

**主要结果（LLaMA2-7B）**：
- **PromptEOL + CP-NS**：平均Spearman提升**+5.24**（75.27 vs 70.03），STS-B提升+7.05，覆盖48/56场景
- **Pretended CoT + CP-NS**：平均提升+0.59（77.45 vs 76.86）
- **Knowledge + CP-NS**：平均提升+0.42（77.56 vs 77.14）
- **CK + CP-NS**：平均提升+0.45（78.68 vs 78.23）
- **NS策略整体优于NR策略**

**跨模型泛化**：
- LLaMA2-13B：Pretended CoT + CP-NS平均+0.57
- LLaMA3.1-8B：Pretended CoT + CP-NS平均**+1.15**（提升最大）

**下游任务**：PromptEOL + CP-NS平均准确率**91.73**，超越48亿参数监督模型ST5-Enc（91.63），且无任何微调。

**效率**：辅助提示仅需传播至第5/7层，相比多prompt方法CK（54层）开销极小（Table 3）

## 相关工作脉络
1. **PromptEOL（Jiang et al., 2023）**：首个将句子语义压缩到单一token的提示方法，CP直接集成并增强其效果
2. **MetaEOL（Lei et al., 2024）**：通过meta-task prompt多角度捕捉句子表示，CP可与其组合进一步提升
3. **Pretended CoT / Knowledge（Zhang et al., 2024）**：分别利用CoT和知识引导聚焦主谓结构，CP通过对比去噪强化其效果
4. **LLM2Vec（BehnamGhader et al., 2024）/ NV-Embed（Lee et al., 2024）**：监督微调LLM获取嵌入，CP实现同等甚至更优性能而无需训练
5. **Activation Steering（Zou et al., 2023; Rimsky et al., 2024）**：依赖正负样本对学习steering vector，CP无需监督数据且针对句子嵌入定制
6. **Echo Embeddings（Springer et al., 2024）**：通过重复输入让后续token访问早期信息，CP提供另一条独立增强路径

## 局限性与未来方向
1. **辅助提示优化空间**：当前使用固定模板，未探索如何为每种正常提示自动生成最优辅助提示
2. **超参搜索成本**：NS需搜索干预层ℓ和缩放因子α两个超参，NR需搜索ℓ；虽经验证5/7层通常有效，但缺乏理论指导
3. **仅英文评估**：实验仅在英文数据集上进行，跨语言泛化性未验证
4. **ICL场景增益有限**：在in-context learning设置下CP仍有提升（+0.51~0.79），但幅度低于zero-shot场景，因上下文已提供部分语义引导
5. **简单任务提升有限**：对情感分类等简单任务（MR/CR/MPQA）提升较小，因PromptEOL已能充分表达

## 研究启发与可借鉴点
1. **推理时干预范式可迁移**：CP的"双提示对比→隐状态差值→范数调整"框架可推广至其他LLM应用（如对话生成、代码补全），用于抑制噪声、聚焦关键信息
2. **辅助提示的语义定位原则**：有效的辅助提示应聚焦"非本质信息"（irrelevant/redundant/descriptive），而非实体或情感等特定属性；该原则可为其他对比学习场景提供参考
3. **中间层嵌入提取策略**：结合CP验证了"非最后一层更优"的经验发现，可在团队项目中探索不同任务的最优截取层位
4. **多prompt融合增强**：CK基线验证多prompt平均仍含噪声，CP可直接作用于融合结果，提示可在集成学习中应用对比引导
5. **低成本高效益权衡**：辅助提示仅传播至底层即能显著增强高层表示，为资源受限场景提供轻量化增强方案

## 关键术语表
**Contrastive Prompting (CP)**：通过引入辅助提示与正常提示进行隐状态对比，推理时直接引导LLM聚焦句子核心语义的方法。

**Contextualized Value Vector**：LLM多头注意力层中，经过注意力加权后每个position的隐状态表示（即$V$投影后的结果）。

**Activation Steering**：通过修改LLM中间层的激活值来调控模型行为的推理时干预技术，通常依赖监督信号学习steering vector。

**Norm Scaling (NS)**：CP的范数调整策略之一，引入缩放因子α对对比差值向量进行放大/缩小。

**Norm Recovering (NR)**：CP的范数调整策略之二，将干预后向量重新缩放至与干预前相同的L2范数。

**PromptEOL**：经典提示模板"This sentence: "[TEXT]" means in one word: "，将句子语义压缩至最后一个token的嵌入。

**STS (Semantic Textual Similarity)**：语义文本相似度任务，评估两句话语义相似程度，常用Spearman相关系数衡量。

**SentEval**：标准化句子嵌入下游评估工具包，包含MR/CR/SUBJ等7个分类任务。

## 可复现要素
- **数据集**：STS 2012–2016、STS-B、SICK-R、SentEval七任务、MTEB子集（均为公开基准）
- **代码**：GitHub开源链接 https://github.com/zifengcheng/CP（论文声明将发布）
- **模型权重**：LLaMA2-7B/13B、LLaMA3.1-8B（需自行获取）
- **关键超参**：干预层ℓ∈{5,7}、NS缩放因子α∈{2,3}、输出层取中间层（非最后一层）
- **提示模板**：附录A完整给出PromptEOL、Pretended CoT、Knowledge模板
