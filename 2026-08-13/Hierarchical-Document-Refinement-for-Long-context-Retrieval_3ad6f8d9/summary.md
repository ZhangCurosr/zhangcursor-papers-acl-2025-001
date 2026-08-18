---
title: "Hierarchical-Document-Refinement-for-Long-context-Retrieval"
source: https://aclanthology.org/2025.acl-long.176.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:54:00"
field: "检索增强生成（RAG）与长上下文处理"
keywords: ["RAG", "长文本精炼", "层次化文档建模", "Prompt压缩", "检索增强生成"]
innovations: ["基于层次化文档结构（章节-段落树）的长文本精炼框架，取代传统 chunk/困惑度方案", "查询双级分析（Local/Global）结合自适应 LS+GS 多视角评分融合", "XML 扁平化结构表示配合离线/在线解耦推理，实现 10x token 压缩与 4x 延迟降低"]
benchmarks: ["NQ", "TriviaQA", "HotpotQA", "2Wiki-MultiHopQA", "ASQA", "ELI5", "PopQA"]
---

# 论文速读：Hierarchical-Document-Refinement-for-Long-context-Retrieval

## 一句话总结
本文提出 **LongRefiner**，一个即插即用的长文档级精炼框架，通过查询双级分析、层次化文档结构建模和自适应精炼三阶段协同，利用 LLM 挖掘长文档内在结构信息，在七种 QA 数据集上显著优于现有基线，同时仅使用 10% 的 token 预算并将在线延迟降至 ~25%。

## 研究问题与动机
1. **信号噪声比低**：搜索引擎返回的长文档往往包含大量无关内容，模型难以聚焦于查询相关的关键信息。
2. **计算开销高**：完整输入长文档会大幅增加输入上下文长度，带来推理成本高、线上延迟高的瓶颈。
3. **现有精炼方法局限**：当前文本精炼方法大多适合短文本块（如 perplexity-based 方法），或依赖粗糙指标（如困惑度）评估 token 相关性，缺乏对完整文档的全局上下文理解，反而造成性能下降。
4. **结构信息未被利用**：完整文档天然包含章节标题、段落组织等层次化结构信息，传统 chunk-based 方法忽视了这一可用信号。

## 核心贡献（创新点）
1. **层次化文档级精炼框架 LongRefiner**：在文档层面而非 chunk 层面进行精炼，通过挖掘文档内部层次结构实现高效、低延迟的长文本处理；与现有方法本质区别在于利用"章节-段落"全局组织结构，而非逐 token 或固定 chunk 粒度筛选。
2. **查询双级分析（Local/Global）与自适应精炼**：设计信息范围量化机制，根据查询类型动态加权局部（LS）与全局（GS）评分；与以往固定规则（如仅按困惑度截断）的本质区别在于"按查询意图自适应调整精炼策略"。
3. **XML 扁平化层次文档表示（D_xml）**：设计轻量 XML 语法（含 `<section>`、`<subsection>`、`<skip>`、`<br>` 标签）将树形文档压缩为平铺文本，使输出 token 约为原文的 1/10；本质创新在于以少量标记代价保留完整层次结构，便于模型学习生成。
4. **多任务 LoRA + 离线/在线解耦推理范式**：单一 backbone（Qwen2.5-3B-Instruct）通过 task-specific LoRA 模块并行训练三项任务（每模块仅占模型参数 0.03%），文档结构构建离线完成，在线仅处理查询理解与精炼打分；与串行多模型 pipeline 的本质区别在于资源复用与低在线延迟。

## 方法详解

### 3.1 双级查询分析（Dual-Level Query Analysis）
- 定义两个信息等级：**Local Level**（答案局限于特定段落，知识范围窄）和 **Global Level**（需要全面综合文档内容）。
- 教师 LLM（Llama3.1-70B-Instruct）标注训练集查询的 `[Local]`/`[Global]` 二值标签，将其作为特殊 token 微调到 refiner 模型。
- 推理时通过生成概率经 softmax 得到连续信息范围表征 $r_q$：
  $$P_l = P_M(\text{[Local]}|\text{query}),\quad P_g = P_M(\text{[Global]}|\text{query}),\quad r_q = \text{Softmax}(P_l, P_g)_g$$
- $r_q$ 在后续作为全局评分的加权因子。

### 3.2 层次化文档结构建模（Hierarchical Document Structuring）
- 将文档建模为文档树 $D_{\text{str}} = (\mathcal{N}, \mathcal{R})$，节点表示章节/段落，关系表示层级与蕴含。
- **XML 扁平化表示 $D_{\text{xml}}$**：设计 4 种标签（见下表），将树形结构编码为紧凑文本，token 约为原文 1/10：

| 标签格式 | 定义 |
|---|---|
| `<section: {title}>` | 章节开始 |
| `<subsection: {title}>` | 子章节开始 |
| `<skip>` | 省略中间内容（仅保留首尾 k 个 token） |
| `<br>` | 段落分隔 |

- 模型生成过程分为两阶段迭代：先生成层次结构 S（自动推断章节标题），再在每个结构下填充内容并动态使用 `<skip>` 压缩。
- **训练数据构造**：基于 Wikipedia 全文 dump，提取原始网页的章节/段落结构作为 golden $D_{\text{str}}$，去除结构信息得到纯文本 $D$，再转换为 $(D, D_{\text{xml}})$ 训练对。

### 3.3 自适应文档精炼（Adaptive Document Refinement）
- **局部评分（LS）**：从叶子节点（段落）开始，用通用评分模型 $M$ 计算 query 与段落相似度得分，然后自底向上对父节点取子节点平均值：
  $$\text{LS}(n_i) = \begin{cases} M(\text{query}, n_i) & n_i \in \mathcal{N}_L \\ \frac{1}{|\mathcal{C}(n_i)|}\sum_{n_j \in \mathcal{C}(n_i)} \text{LS}(n_j) & \text{otherwise} \end{cases}$$
- **全局评分（GS）**：基于文档大纲（摘要+所有章节标题），微调模型选择与查询相关的 section 节点，通过指示函数 $\mathbb{I}$ 给出初始分，再均匀分配至子节点：
  $$\text{GS}(n_i) = \begin{cases} \mathbb{I}(n_i \in M(q, \text{outline})) & n_i \in \mathcal{N}_S \\ \text{GS}(\text{Pa}(n_i)) & \text{otherwise} \end{cases}$$
- **合并评分**：$Score(n_i) = LS(n_i) + r_q \cdot GS(n_i)$，按得分排序选节点直到满足 token budget（若父节点被选则其子节点一并选中）。

### 3.4 训练与推理
- **训练**：三项任务（query analysis、document structuring、global selection）在 **Qwen2.5-3B-Instruct** 上以 task-specific LoRA 模块分别训练（最大序列长度 2k/32k/4k），各模块仅占总参数 0.03%。
- **推理**：文档结构化在**离线**阶段完成，在线阶段仅执行查询分析与精炼打分（处理数百 token 输入，生成数十 token 输出），整体在线延迟约标准设置的 **25%**。

## 实验与结果

- **数据集**：7 个开放域 QA 数据集，涵盖三种类型：Single-hop（NQ、TriviaQA、PopQA）、Multi-hop（HotpotQA、2Wiki-MultiHopQA）、Long-form（ASQA、ELI5）。
- **生成器**：Llama3.1-8B-Instruct（附录 Table 5 补充实验使用 Qwen2.5-7B-Instruct）。
- **检索策略**：Wikipedia 2018 dump，每个 query 检索 top-8 完整文档（MaxP 设置）。
- **关键结果**（Table 2，token budget 2k）：

| 方法 | NQ Acc | TriviaQA Acc | HotpotQA Acc | 2Wiki Acc | ASQA F1 | ELI5 F1 | PopQA F1 | Tokens | Latency |
|---|---|---|---|---|---|---|---|---|---|
| Full Content | 53.8 | 70.8 | 36.0 | 35.7 | 34.1 | 23.8 | 64.1 | 19567 | 40.6 |
| LongLLMLingua | 45.4 | 67.6 | 34.7 | 33.1 | 33.6 | 23.7 | 56.8 | 1976 | 496.6 |
| **LongRefiner (Ours)** | **54.4** | **71.7** | **39.3** | **36.1** | **35.8** | **23.9** | **59.9** | **1933** | **10.8** |

- **最强结果**：在所有 7 个数据集上取得最佳性能；相对 Full Content 节省 **10x token** 和 **4x 延迟**；相对最佳基线 LongLLMLingua 在 NQ/HotpotQA/2Wiki 上分别提升 **+9.0/+4.6/+3.0** 个百分点（Acc）。
- **消融**（Table 3）：移除 Hierarchical Structuring 模块导致性能骤降约 **20%**（如 Single-hop EM 从 62.3 降至 45.7），是最关键的组件。
- **评分模型对比**（Table 4）：bge-reranker 作为 LS 评分模型表现最佳（Single-hop F1=55.9），嵌入模型（E5/SBERT）次之但效率更高。

## 相关工作脉络
1. **RAG 知识精炼方法分类**：本文将现有工作归为 Hard Prompt Refinement（perplexity-based 截断/摘要/块选择）和 Soft Prompt Refinement（向量编码），指出前者在长文本场景缺乏全局理解，后者需额外训练；LongRefiner 定位于结构化文档建模路线。
2. **LongLLMLingua (Jiang et al., 2023a)**：基于困惑度的 prompt 压缩方法，通过 token 级重要性评分压缩上下文；本文与之核心区别在于 LongRefiner 利用"章节结构"而非"逐 token 困惑度"判断信息价值。
3. **LLMLingua-2 (Pan et al., 2024)**：数据蒸馏驱动的 prompt 压缩方法；本文强调对长文档层次结构的显式建模是更有效的路径。
4. **Selective-Context (Li et al., 2023)**：最早 perplexity-based 方法之一；本文实验表明其在长文档场景显著落后于结构化的 LongRefiner（Latency 100.6 vs 10.8）。
5. **Semantic Chunking (Jina-Segment, Meta-Chunking)**：基于语义边界的智能分块；本文认为固定 chunk 粒度无法利用文档天然的逻辑层次，缺乏"章节→段落"的全局感知。
6. **BM25 / BGE-Reranker / SBERT 检索基线**：传统检索式 chunk 选择方法；本文证明基于层次结构的精炼方法（LongRefiner）在同等 token 预算下显著优于 chunk 重排方案。

## 局限性与未来方向
1. **复杂数据类型支持有限**：当前方法仅处理纯文本，现实场景中检索文档常含表格、图片、超链接等结构化/多模态内容，XML 语法的扩展方向尚待探索。
2. **通用域知识依赖**：训练数据完全基于 Wikipedia 通用领域文档，直接迁移至企业、金融等垂直领域效果未知，需要领域适配或 teacher LLM 辅助标注。
3. **短文档场景略受影响**：在噪声极低的短文档场景（如 PopQA）中，由于精炼过程中的信息损失，性能略低于 Full Content 设置。
4. **结构解析误差**：XML 格式的 `<skip>` 机制与解析算法在小 $k$ 值时可能引入少量解析错误。

## 研究启发与可借鉴点
1. **"离线预处理 + 在线轻量推理"的解耦范式**可推广至其他长文本处理任务（如长文档问答、代码生成），大幅降低线上延迟。
2. **XML 扁平化层次表示**是一种简洁且可训练的文档结构编码方式，可迁移到文档理解、大纲生成等下游任务。
3. **查询意图驱动的多视角评分融合**（Local/Global 加权）思路可推广至多任务 RAG、复杂推理问答等需要"按需检索深度"的场景。
4. **task-specific LoRA 多任务共享 backbone**的设计兼具参数经济性与任务隔离性，是高性价比 RAG 系统工程化的有效实践。
5. **Wikipedia 结构提取作为无监督/弱监督训练信号**的思路，可用于其他有固有章节结构的数据集（arXiv 论文、技术手册等），降低训练数据标注成本。

## 关键术语表
**LongRefiner**：本文提出的长文档精炼系统，通过层次化建模实现高效 RAG 上下文压缩。
**Dual-Level Query Analysis**：将查询信息需求分为 Local（局部精确）和 Global（全局综合）两级，用以指导后续精炼策略的权重分配。
**Hierarchical Document Structuring**：将无序长文本建模为包含 section/subsection 层级的树状结构，并通过 XML 标记实现压缩表示。
**Local Score (LS)**：基于 query-段落相似度计算的细粒度评分，自底向上聚合至高层节点。
**Global Score (GS)**：基于文档大纲（摘要+章节标题）的粗粒度评分，自上而下均匀传播，捕捉全局上下文相关性。
**Adaptive Refinement**：综合 LS 与 GS（按查询信息范围加权）对文档树节点打分并选定压缩后的内容片段。
**Task-specific LoRA**：不同任务使用独立低秩适配器模块（各仅占模型参数 0.03%），共享同一 backbone，实现多任务学习与无干扰推理。
**$\text{<skip>}$ tag**：XML 表示中的省略占位符，用于压缩段落中间内容，仅保留首尾 $k$ 个 token 以减少输出长度。

## 可复现要素
- **数据集**：Wikipedia 2018 dump（KILT 版本，已公开）；训练数据为 NQ/TriviaQA/HotpotQA/2Wiki/ASQA/ELI5 各取前 10,000 条训练样本；PopQA 作为无训练集的 out-of-domain 测试集。
- **代码/权重**：代码开源 — https://github.com/ignorejjj/LongRefiner（论文声明）。
- **关键超参**：
  - 基座模型：Qwen2.5-3B-Instruct
  - LoRA rank 参数占比：0.03%（每任务独立模块）
  - 训练序列长度：query analysis 2k，document structuring 32k，global selection 4k
  - 学习率：$3 \times 10^{-5}$，warmup ratio：0.1，batch size per device：1，gradient accumulation：8
  - 训练轮次：每个任务 1 epoch
  - 推理：greedy decoding（temperature=0），输出 max tokens=500
  - 本地评分模型：bge-reranker-v2-m3
  - 教师 LLM（标注生成）：Llama3.1-70B-Instruct
