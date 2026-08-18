---
title: "ReSCORE-Label-free-Iterative-Retriever-Training-for-Multi-ho"
source: https://aclanthology.org/2025.acl-long.16.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:19:14"
field: "多跳问答与检索器训练"
keywords: ["多跳问答", "检索器训练", "检索增强生成", "大语言模型监督", "无标注学习", "迭代推理"]
innovations: ["提出ReSCORE无标注检索器训练方法，联合建模文档相关性与答案一致性", "在迭代RAG框架中直接训练密集检索器，实现多跳场景下的渐进式检索优化", "证明伪标签在迭代多跳检索中优于GT标签，揭示对比学习在多跳场景的centroid对齐缺陷"]
benchmarks: ["MuSiQue", "HotpotQA", "2WikiMHQA"]
---

# 论文速读：ReSCORE-Label-free-Iterative-Retriever-Training-for-Multi-ho

## 一句话总结
论文提出 **ReSCORE**，一种无需标注文档即可训练密集检索器的方法，通过 LLM 联合评估文档对问题的**相关性**与对答案的**一致性**生成伪真值（pseudo-GT），并在迭代式 RAG 框架中进行检索器微调，最终在三个多跳问答基准上达到 SOTA。

## 研究问题与动机
1. **密集检索器需要标注数据**：BM25 等稀疏检索器无需训练，但密集检索器（如 Contriever）需要在目标域上进行微调才能发挥优势；多跳问答（MHQA）中每个推理步的查询（reformulated question）各不相同，导致标注成本极高。
2. **既有 LLM 监督方法仅关注一致性**：ATLAS、LLM-Embedder 等工作利用 LLM 监督训练检索器，但只考虑文档与答案的一致性，忽略了文档与问题的相关性，且主要针对单跳问答设计。
3. **迭代检索器训练尚未被充分探索**：现有迭代 RAG 方法（IRCoT、Self-RAG 等）均依赖未微调的稀疏/密集检索器，缺乏在迭代过程中动态训练检索器以适配多跳推理的研究。

## 核心贡献（创新点）
1. **提出 ReSCORE 无标注检索器训练方法**：利用 LLM 联合建模文档相关性 $P_{LM}(q|d)$ 与答案一致性 $P_{LM}(a|q,d)$ 生成伪真值分布，通过 KL 散度损失训练检索器，无需人工标注文档。
2. **在迭代 RAG 框架中直接训练检索器**：将 ReSCORE 嵌入 IQATR 迭代问答流程，使检索器在每一步推理后持续适应当前查询的语义空间，而非仅做静态预训练检索。
3. **揭示"相关性+一致性"联合作用机制**：通过消融实验证明单独使用相关性或一致性均会显著损害性能，只有两者相乘才能得到高质量伪标签，避免无关文档因词频对齐而获得虚假高分。
4. **实现三个 MHQA 基准的 SOTA**：IQATR（Llama-3.1-8B + ReSCORE 微调 Contriever）在 MuSiQue、HotpotQA、2WikiMHQA 上分别达到 EM 23.4 / 47.2 / 50.0，超越既有方法。

## 方法详解
1. **迭代 RAG 框架（IQATR）**：
   - 初始查询 $q^{(1)} = q$，每轮检索 top-$k$ 个文档 $\mathcal{D}^{(i)}$，送入 LLM 判断是否已知答案；
   - 若答案为 "unknown"，LLM 生成一句**thought** $t^{(i)}$ 压缩已检索信息，并重构下一轮查询 $q^{(i+1)}$（支持两种策略：LLM-rewrite 和 Thought-concat）；
   - 迭代最多 $\eta_n$ 轮，直至 LLM 给出非 unknown 答案。

2. **伪真值分布生成**：
   - 对每个候选文档 $d_j^{(i)}$，计算 $Q_{LM}^{(i)}(d_j^{(i)}|q^{(i)}) \propto P_{LM}(q^{(i)}|d_j^{(i)}) \cdot P_{LM}(a|q^{(i)}, d_j^{(i)})$，前者衡量文档与问题的**话题相关性**，后者衡量文档对答案的**支持一致性**。
   - 使用固定温度 $T=0.1$ 的 LLM 计算上述概率，GT 答案 $a_n$ 用于构造伪标签。

3. **检索器训练损失**：
   - 检索器输出分布 $P_R^{(i)}(d_j^{(i)}|q_n^{(i)}) = \text{Softmax}(\mathbf{d}_j^{(i)} \cdot \mathbf{q}_n^{(i)})$，其中 $\mathbf{q}$ 可训练、$\mathbf{d}$ 冻结；
   - 总损失为所有样本和所有迭代轮次的 KL 散度之和：$\sum_n \sum_i D_{KL}(Q_{LM}^{(i)} || P_R^{(i)})$；
   - 为降低计算开销，每轮仅在检索器 top-$M$（$M=32$）个文档上计算伪真值分布。

4. **查询重构策略对比**：
   - **Thought-concat**：$q^{(i+1)} = [t^{(i)}; q^{(i)}]$，将压缩 thought 拼接到原查询后，保留完整问题上下文，在复杂数据集上效果更优。

## 实验与结果
- **数据集**：MuSiQue（4跳）、2WikiMHQA（2-3跳）、HotpotQA（2跳），使用标准子采样验证/测试集，GT 文档标注仅用于评估不使用于训练。
- **基线**：ReAcT、FLARE、Self-RAG、Adaptive-Note、IRCoT、Adaptive-RAG；自建 Baseline 为 Llama-3.1-8B + BM25 / Contriever。
- **主要结果**（EM / F1）：

| 方法 | MuSiQue | HotpotQA | 2WikiMHQA |
|---|---|---|---|
| Our Baseline (Contriever) | 15.2 / 23.8 | 39.4 / 52.3 | 32.8 / 41.6 |
| **IQATR (ReSCORE)** | **23.4 / 32.7** | **47.2 / 59.3** | **50.0 / 59.7** |
| Adaptive-RAG | 23.6 / 31.8 | 42.0 / 53.8 | 40.6 / 49.8 |
| IRCoT | 22.0 / 31.8 | 44.4 / 56.2 | 49.7 / 54.9 |

- **最强结果**：2WikiMHQA EM 达 50.0，较 Contriever Baseline 提升 +17.2 点；HotpotQA EM 47.2，较 IRCoT 提升 +2.8 点。
- **ReSCORE 通用性**：在 Self-RAG、FLARE、Adaptive-Note 上均带来稳定提升（Table 2）。
- **MHR@8 分析**：ReSCORE 训练的检索器在多轮迭代中 MHR 持续上升（MuSiQue i=2: 63.0% vs GT 训练 54.8%），说明能逐步召回互补文档；GT 训练的检索器在 i≥2 后停滞。

## 相关工作脉络
1. **Dense Retrieval（DPR/Contriever）**：依赖 supervised contrastive loss 在 MS-MARCO 等数据上预训练，但未针对 MHQA 迭代推理场景微调；本文在其预训练权重基础上用 ReSCORE 在目标域上针对性训练。
2. **LLM-as-Teacher 检索训练（ATLAS/LLM-Embedder）**：仅利用 $P_{LM}(a|q,d)$ 一致性信号，忽视相关性；本文公式 (2) 将两者相乘，避免无关文档因词对齐而获虚假高分。
3. **Iterative RAG（IRCoT/FLARE/Self-RAG/Adaptive-RAG）**：聚焦生成策略或自适应步数，检索器均为固定 pretrained 模型；本文首次在迭代框架内端到端训练检索器。
4. **Supervised Dense Retrieval Fine-tuning**：直接用 GT 文档做 InfoNCE 训练；本文证明伪标签在迭代多跳场景下优于 GT 标签，因为 GT 标签使 query encoder 向多个 distant 文档 centroid 对齐，损害后续轮的检索质量。
5. **Prompt-based Ranking（RankVicuna/Promptagator）**：用 LLM 生成 ranking list 作为软标签；本文扩展至迭代多跳场景，同时引入相关性+一致性联合信号，且针对 MHQA 设计伪标签构造方式。

## 局限性与未来方向
1. **泛化能力受限**：模型在 MuSiQue/2WikiMHQA/HotpotQA 上微调，对推理模式不同或 hop 数差异较大的 OOD 数据集泛化有限。
2. **计算开销与延迟**：迭代检索过程（最多 6 轮）带来显著推理延迟，在 hop 数高的问题上尤为突出。
3. **未来方向**：探索跨域迁移与少样本适应；优化迭代效率（如自适应停止、课程学习）；将 ReSCORE 框架推广至其他需要多步检索的知识密集型任务。

## 研究启发与可借鉴点
1. **LLM 联合生成伪标签的思路可迁移**：将 $P(q|d) \cdot P(a|q,d)$ 的分解方式应用于任何需要检索辅助的迭代推理任务，可在无标注场景下实现检索器自适应。
2. **Thought-concat 作为 query reformulation 策略**：相比纯 LLM rewrite，保留原始问题上下文有助于复杂问题的错误恢复，该设计可直接复用至其他迭代检索系统。
3. **GT 标签在迭代场景未必最优**：本文揭示了对比学习在多跳场景下"向 centroid 对齐"的缺陷，提示后续研究在训练多步检索器时应考虑渐进式、分步监督而非一次性全局对齐。
4. **消融实验设计值得借鉴**：分别拆解 $P(q|d)$、$P(a|q,d)$ 及两者的乘积，清晰验证各信号的作用，为后续伪标签设计提供方法学参考。

## 关键术语表
- **Multi-hop Question Answering (MHQA)**：需要跨越多个文档进行逻辑推理才能回答的复杂问答任务。
- **ReSCORE**：Retriever Supervision with Consistency and Relevance，本文提出的无标注检索器训练方法。
- **Pseudo-GT（伪真值）**：由 LLM 生成的软标签分布，替代人工标注文档相关性标签用于检索器训练。
- **Iterative RAG**：在每轮推理中动态检索文档、逐步逼近答案的检索增强生成框架。
- **Thought（思维摘要）**：LLM 对已检索文档的压缩表示，用于指导下一步查询重构和答案生成。
- **MHR@k（Multi-hop Recall at k）**：多跳检索评估指标，衡量截至第 i 轮累计召回的 GT 支持文档比例。
- **KL Divergence Loss**：用于最小化检索器输出分布与 LLM 伪真值分布之间差异的损失函数。
- **Reformulated Query**：基于已检索信息和 thought 生成的下一轮查询，用于引导检索器寻找互补文档。

## 可复现要素
- **数据集**：MuSiQue、2WikiMHQA、HotpotQA（均公开）；代码/数据：https://leeds1219.github.io/ReSCORE
- **模型权重**：Contriever（MS-MARCO 预训练权重）、Llama-3.1-8B-Instruct（开源）；ReSCORE 训练代码开源
- **关键超参**：温度 T=0.1；top-M=32（训练）、top-k=8（推理）；最大迭代次数 $\eta_n$=6；batch size=16；初始学习率 $1\times10^{-6}$，每 100 步衰减 0.9 倍；仅训练 query embedder，document embedder 冻结；AdamW 优化器；2× NVIDIA A100（40GB）；最小迭代次数设为 2。
