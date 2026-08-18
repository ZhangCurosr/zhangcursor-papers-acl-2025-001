---
title: "Retrieve-to-Explain-Evidence-driven-Predictions-for-Explaina"
source: https://aclanthology.org/2025.acl-long.167.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:07:12"
---

# 论文速读：Retrieve-to-Explain-Evidence-driven-Predictions-for-Explaina

## 一句话总结
提出 Retrieve to Explain (R2E)，一种将候选答案完全 mask 并仅通过检索到的支持性证据进行表征的架构，结合 Shapley 值实现忠实、定量的证据级解释；在药物靶点识别任务上，R2E 不仅匹敌非可解释文献模型，更显著超越制药工业广泛使用的遗传学基线与 GPT-4 RAG 基线。

## 研究问题与动机
- 复杂科学发现问题（如药物靶点筛选）常存在多个合理假设，各假设支持证据强度各异，现有语言模型缺乏定量、忠实比较答案合理性的能力。
- 高风险 human-in-the-loop 场景要求模型不仅能输出预测，还能提供可追溯、可检验的证据链，以便专家排查模型缺陷或系统性偏差。
- 传统生成式 LLM 的 RAG 解释多为自然语言文本，缺乏定量归因且易产生幻觉；全参数模型难以在不重训的前提下灵活融入新证据（如非文本模态）。
- 药物靶点识别失败成本极高，临床试验成功预测是验证靶点价值的核心标准，但单一证据源（如纯遗传学关联）泛化覆盖有限。

## 核心贡献（创新点）
1. **证据驱动的可解释架构**：将特征空间直接定义为检索证据，答案实体在查询与证据中均被 mask，使 Shapley 值能忠实归因至每条证据。与现有 RAG/LLM 的本质区别在于解释从“生成式文本”变为“定量特征贡献”。
2. **高 stakes 科学发现突破**：在药物靶点识别任务上，R2E 不仅匹配非可解释文献基线，更显著超越制药业广泛使用的遗传学靶点预测方法。
3. **事后 PMI 频率偏差校正**：引入可调节的 $c \in [0,1]$ 校正因子，等效于在 logit 上叠加点互信息项，灵活缓解文献语料中长尾实体出现的严重先验偏差。
4. **非文本模态的自然语言模板化融合**：将结构化遗传关联数据模板化为句子并接入同一检索管道，无需重训即可实现多模态推理，且显著提升临床结局预测性能。
5. **开源三个靶点识别基准**：填补公开高质量数据集空白，促进可解释科学 AI 的标准化评测。

## 方法详解
- **Masked Entity-Linked Corpus**：构建 1.6 亿句生物医学文献语料，对 19,176 个蛋白编码基因进行实体链接，将每次实体提及替换为 `[MASK]`，形成 `(a, d)` 训练对。
- **Retriever**：基于随机初始化的 scaled-down PubMedBERT（约 1000 万参数），以 MLM 目标训练 10 epoch。推理时为每个候选答案维护独立 FAISS 索引，检索与 cloze-style 查询余弦相似度最高的 $k=64$ 条证据。
- **Reasoner**：采用二进制交叉熵损失学习 $p(L=1|a_i, q)$。包含两部分：(1) Query-Evidence Encoder：查询与证据拼接后经两层 Conv1D 编码维度级交互；(2) Evidence Combiner：使用 Set Transformer（4 heads, 2 ISABS, 32 inducing points）处理无序证据集，输出 sigmoid 概率。总参数量约 200 万。
- **Shapley Value 解释**：将 64 个 query-evidence 对视为特征，采用蒙特卡洛排列采样（$M=100$，配合对偶采样降方差）近似计算每条证据的边际贡献，所有特征 Shapley 值之和等于最终分数。训练时引入 NULL embedding 随机 dropout（采样自 Uniform(0,1)）以增强鲁棒性。
- **Frequency Bias Correction**：调整负采样分布 $P_c(A=a_i|L=0) = C(a_i)^c / \sum C(a_i)^c$，等效于在输出 logit 上叠加 PMI 校正项。$c=0$ 保留原始概率，$c=1$ 完全转为 PMI；文中取 $c=0.5$（R2E-cor）在验证集上取得最佳 MRR。

## 实验与结果
- **数据集**：Held-out Biomedical Literature（2020-2022 文献）、Gene Description Facts（UniProt 人工描述）、Clinical Trial Outcomes（PharmaProjects，2005 年后临床 II/III 期，1449 成功 vs 4222 失败）。
- **基线**：FREQ、MCS、MLM、Genetics Baseline（制药业标准）、GPT-4 Few-shot RAG-CoT。
- **Gene Description Facts**：R2E-cor MRR 0.260，显著优于 MCS（0.176）与 MLM（0.167）；Shapley 值与 GPT-4 相关性标注 AUROC 达 0.824，与人类专家一致率 71.5%。
- **Clinical Trial Outcomes**：R2E-cor (BOTH) AUROC 0.643，显著优于 Genetics Baseline（0.545，$p<0.001$）；即使仅使用纯遗传学模板数据，R2E-uncor (Genetic) 亦显著超越传统遗传基线（$p<0.001$），得益于模型对软语义关联的捕捉。结合文献后性能进一步提升。
- **LLM Auditing**：用 GPT-4 审核 Shapley 值最高的 20,000 条证据并过滤无关项，R2E-AUDIT (BOTH) AUROC 升至 0.647（$p=0.004$）。
- **最强结果**：R2E-AUDIT 在临床实验预测上取得 0.647 AUROC，相对成功率（Relative Success）全面领先遗传基线与 GPT-4 RAG 基线，且解释可审计、可迭代。

## 相关工作脉络
- **检索增强语言模型（kNN-LM / FiD / RAG）**：R2E 不同于仅用检索辅助 next-token 预测的 kNN-LM
