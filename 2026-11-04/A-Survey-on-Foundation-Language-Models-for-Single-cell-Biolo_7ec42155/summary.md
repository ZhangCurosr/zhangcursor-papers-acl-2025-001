---
title: "A-Survey-on-Foundation-Language-Models-for-Single-cell-Biolo"
source: https://aclanthology.org/2025.acl-long.26.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:39:43"
field: "单细胞基础语言模型综述"
keywords: ["单细胞基础模型", "语言模型", "预训练语言模型", "大语言模型", "单细胞转录组", "数据分词", "掩码语言建模"]
innovations: ["首次从语言建模视角系统综述单细胞基础模型，提出 PLMs 与 LLMs 分类框架", "提炼数据分词、预训练范式、下游任务三大技术主线", "系统诊断数据质量、模型设计、评估协议三方面挑战并提出未来方向"]
benchmarks: ["Cell type annotation", "Gene network analysis", "Drug sensitivity prediction", "Spatial transcriptomics tasks"]
---

# 论文速读：A-Survey-on-Foundation-Language-Models-for-Single-cell-Biolo

## 一句话总结
本文是首篇系统性综述，将单细胞基础语言模型划分为**单细胞预训练语言模型（PLMs）**与**单细胞大语言模型（LLMs）**两大类，从数据分词策略、预训练/微调范式及下游任务三个维度全面梳理现有方法，并指出当前在数据质量、模型设计与评估协议方面面临的挑战与未来方向。

## 研究问题与动机
1. **语言模型与单细胞生物学的交叉缺乏系统梳理**：近年来 Transformer 类模型被应用于单细胞分析，但已有综述多聚焦 Transformer 架构本身，未从"语言建模"视角统一分析单细胞 foundation model 的演变。
2. **单细胞数据天然具有类文本结构**：细胞可视为"句子"、基因视为"词"，但如何在保持生物学语义的同时解决高维稀疏性、批次效应等问题，尚无统一范式。
3. **模型规模与应用场景差异巨大**：从零预训练的 PLMs 与基于预训练 LLMs 微调的工作并存，两者的技术路线、适用任务与性能边界需要系统对比。
4. **缺乏标准化 benchmark 导致比较困难**：各工作使用私有数据集与不同指标，难以公平评估单细胞 foundation model 的真实能力。

## 核心贡献（创新点）
1. **首创单细胞基础语言模型综述**：首次从语言建模视角（PLMs vs. LLMs）系统梳理单细胞 foundation model 的技术体系与应用场景。
2. **提出新颖的分类框架**：基于"如何将细胞转化为语言模型可理解的表示"这一核心问题，将现有模型划分为单细胞 PLMs（从头预训练）与单细胞 LLMs（基于已有 LLMs 微调）。
3. **提炼出三大技术主线**：数据分词策略（离散/连续/侧信息）、预训练范式（MLM/Next Token Prediction/多任务）、下游任务（细胞级/基因级/药物/空间等），为后续研究提供清晰的技术地图。
4. **系统诊断领域挑战并提出研究方向**：从数据质量（稀疏性、批次效应、多组学资源不足）、模型设计（缺乏统一 tokenizer、scaling law 尚未涌现）、评估协议（benchmark 缺失）三个维度指明未来路径。

## 方法详解

**1. 单细胞 PLMs 的数据分词策略**
- **离散 Token**：scBERT、CellLM 采用分箱（Binning）将 log 转换后的连续基因表达值转为离散整数；GeneFormer、LangCell、sc-Cello 采用秩值编码（Rank Value Encoding），按基因频率排序后映射到基因词汇表，最大长度设为 2048。
- **连续 Embedding**：UCE、scPRINT 利用大型蛋白语言模型获取基因连续嵌入；CellPLM 将基因与随机初始化可学习层相乘；scFoundation 采用层级贝叶斯降采样策略；CellFM 使用 MLP 投影对齐大规模数据；Nicheformer、GeneCompass 结合排序与元数据生成连续嵌入；scGPT 在分箱后融入元数据实现离散→连续转换。
- **侧信息融合**：Nicheformer、scGPT、scMulan、GeneCompass 直接引入细胞状态、类型、器官来源、供体年龄、测序平台等元数据；UCE、scPRINT 借助蛋白语言模型作为桥梁。

**2. 预训练范式**
- **掩码语言建模（MLM）**：主流方式，随机遮蔽 15%–30% 的基因（scBERT、UCE、GeneFormer、scFoundation、Nicheformer）；CellPLM 以高斯混合分布为先验进行遮蔽。
- **下一词预测（Next Token Prediction）**：tGPT、scGPT 采用此范式；作者指出该范式在单细胞领域普及率低的原因在于数据量相对小且细胞数据天然稀疏导致多数真实值为零。
- **多任务预训练**：以 MLM 为基础，额外引入细胞生成、对比学习、分类、细胞-文本匹配、瓶颈学习、元数据预测、去噪、细胞本体对齐等辅助任务。

**3. 单细胞 LLMs 的转换与微调**
- **Cell-to-Sentence**：Cell2Sentence、CHATCELL、CELLama 将细胞表达值排序后选取 Top-100 基因名组成文本句子；scChat 甚至无需转换，直接将原始数据作为输入。
- **Text-level Gene Embeddings**：GenePT、scELMo、scInterpreter 通过提示 LLM 获取每个基因的文本嵌入，再以表达值为权重加权聚合。
- **微调范式**：指令微调（Instruction-based）通过<问题,答案>格式对齐任务；嵌入微调（Embedding-based）直接利用句子/基因嵌入进行监督微调；无微调（Tuning Free）将 LLM 作为 Agent 生成 Python 代码完成分析任务（scChat）。

**4. 下游任务体系**
- 细胞级：细胞类型注释、新细胞类型发现、批次效应校正、聚类、多组学整合、细胞生成。
- 基因级：基因网络分析（共表达网络、调控网络推断、in silico 处理分析）、基因扰动预测、基因预测（标记基因、表达预测、功能预测）。
- 药物相关：药物敏感性预测、药物反应预测。
- 空间相关：空间转录组补全、空间标签预测与组成分析、空间上下文分析。

## 实验与结果

本文是综述，未提出新模型，但汇总了现有 PLMs 与 LLMs 的实验数据集规模与主要成果：

**预训练数据规模对比**
| 模型 | 预训练数据量 |
|------|-------------|
| scBERT | 1M scRNA-seq |
| UCE | 36M Cells |
| GeneFormer | 27.4M Transcriptomics |
| scFoundation | 50M scRNA-seq |
| Nicheformer | 57M Cells (SpatialCorpus-110M) |
| CellFM | 100M Cells |
| GeneCompass | 126M Transcriptomics (人类+小鼠) |
| LangCell | 27.5M Cells + Texts |

**关键结论**
- 细胞类型注释是 most fundamental 任务，几乎所有模型均在此任务上验证；
- scGPT 是唯一同时验证多组学整合（RNA+蛋白）的 PLM；
- Nicheformer、scFoundation、GeneCompass 在基因网络分析与基因扰动预测上表现突出；
- LLMs（如 CHATCELL、scInterpreter）在细胞类型注释与药物敏感性预测上展现出 comparable 甚至 superior 的零样本/少样本能力；
- 空间相关任务主要由 Nicheformer 和 CELLama 覆盖。

**最强结果提示**：GeneCompass 使用最大规模数据（126M），在跨物种基因调控机制推断上表现最佳；CellFM 在 100M 细胞规模下实现了高效的基因功能预测与细胞注释。

## 相关工作脉络

1. **Lan et al. (2024), Szałata et al. (2024), Bian et al. (2024b)**：聚焦 Transformer 架构的单细胞模型综述，但未从语言建模视角系统分析 tokenization 策略与预训练范式的多样性。
2. **Theodoris et al. (2023) - GeneFormer**：开创性地将基因视为 tokens 并采用 MLM 预训练，提出秩值编码方案，本文将其作为离散/连续分词的代表案例对比。
3. **Cui et al. (2024) - scGPT**：首个采用 Next Token Prediction 范式并从 scratch 预训练的单细胞 foundation model，也是少数支持多组学整合的模型，本文重点讨论其方法设计。
4. **Levine et al. (2023) - Cell2Sentence**：首篇将 LLM（GPT-2）应用于单细胞分析的开创性工作，提出 cell-to-sentence 的转换策略，启发了后续 CHATCELL、CELLama 等 LLM-based 方法。
5. **Hao et al. (2024) - scFoundation**：在 50M 细胞规模上进行 MLM 预训练，强调层级贝叶斯降采样策略，代表大尺度 PLM 的设计思路。
6. **Zeng et al. (2024) - CellFM**：在 100M 细胞上训练的最大规模 PLM 之一，展示了 scaling 趋势，与 GeneCompass 一起代表数据驱动路线。

## 局限性与未来方向

**自述局限**
1. 尽管努力覆盖投稿前的相关文献，仍可能有少量工作被遗漏。
2. 本文主要聚焦技术层面，未深入分析各模型设计背后的生物学动机与意义。
3. 未详细探讨单细胞 foundation model 如何具体提升药物敏感性预测、多组学整合等下游应用，留待未来工作。

**可合理推断的不足**
1. **数据偏差**：现有模型预训练数据主要来自 scRNA-seq，对 DNA 甲基化、蛋白质组等其他组学覆盖不足。
2. **模型规模受限**：最大 PLM 仅 <1B 参数，与 NLP 领域的千亿参数模型差距巨大，scaling law 尚未显现。
3. **评估体系不统一**：缺乏公开、标准化的 benchmark，各工作使用私有数据集与不同指标，难以公平比较。
4. **生物学解释性不足**：语言模型作为"黑盒"，其在单细胞生物学中的可解释性仍有待提升。

**未来方向**
- 构建统一且可扩展的 cell tokenizer；
- 探索 scaling law 在单细胞领域的应用；
- 建立公开、标准化的评估 benchmark；
- 将复杂任务转化为用户友好的问答格式，降低生物学背景门槛；
- 深化药物敏感性预测、多组学整合等应用场景的研究。

## 研究启发与可借鉴点

1. **数据分词策略的多样化设计值得迁移**：离散分箱 vs. 秩值编码 vs. 连续嵌入的选择直接影响模型性能，可根据任务特性灵活组合，例如在标记基因预测任务中采用秩值编码可能更具优势。
2. **多任务预训练的有效性已得到验证**：MLM 结合对比学习、细胞-文本匹配等辅助任务能显著提升下游泛化能力，这一思路可迁移到其他生物模态（如空间转录组、蛋白质结构）。
3. **LLM-based 的 instruction tuning 路径为低资源场景提供新思路**：对于数据量有限的细分任务，直接微调预训练 LLM 比从头训练 PLM 更具性价比，可作为后续研究的 baseline。
4. **侧信息（metadata）的融合是提升模型泛化的关键**：引入细胞类型、器官来源、供体信息等辅助特征可显著改善跨批次泛化，这一策略值得在空间转录组分析中推广。
5. **评估协议的标准化是领域发展的前提**：本文揭示的 benchmark 缺失问题提示，构建开放、统一、可复现的评估框架是推动领域进步的关键基础设施。

## 关键术语表

**Single-cell Foundation Language Model**：将语言建模思想引入单细胞生物学，通过学习大规模单细胞数据构建通用细胞表示的基础模型。

**Tokenization（分词）**：将连续的基因表达矩阵转换为语言模型可理解的离散或连续表示的过程，是构建单细胞语言模型的第一步。

**Masked Language Modeling（MLM）**：随机遮蔽部分基因表达值，让模型学习从上下文重建被遮蔽值，是单细胞 PLMs 最主流的预训练目标。

**Next Token Prediction**：自回归式地预测下一个基因表达值，tGPT 和 scGPT 采用此范式，但在单细胞领域因数据稀疏性而应用较少。

**Cell-to-Sentence**：将细胞表示为基因名序列的文本形式，使预训练 LLMs 可以直接处理单细胞数据。

**Batch Effect（批次效应）**：不同实验平台、实验室或测序技术导致的系统性技术偏差，是单细胞数据整合的主要挑战之一。

**In Silico Treatment**：在计算机模拟中对特定基因进行扰动（如敲除），预测其对细胞转录组的影响，用于理解基因功能。

**Scaling Law**：描述模型性能随参数量、数据量增大而系统性提升的经验规律，目前单细胞领域尚未明确涌现 scaling law。

## 可复现要素

- **数据集**：多个公开数据集被使用（如 CELL GENE、Panglao、GEO、HCA、Tabula Sapiens 等），但各工作使用的预处理流程与数据子集存在差异；部分模型构建了专用数据集（如 GeneCompass 的 scCompass-126M、CellFM 的 100M 细胞数据集、Nicheformer 的 SpatialCorpus-110M）。
- **代码/权重**：论文未逐一声明各模型的开源情况，但根据原文引用，scGPT、GeneFormer、scFoundation、Nicheformer 等均有公开代码或权重；Cell2Sentence、CHATCELL 等为 preprint 阶段，开源状态需另行确认。
- **关键超参**：遮蔽概率通常为 15%–30%；最大序列长度多为 2048； GeneCompass 使用跨物种数据（人类+小鼠）；scFoundation 使用层级贝叶斯降采样策略。论文未详尽列出所有超参，需参考原论文。
