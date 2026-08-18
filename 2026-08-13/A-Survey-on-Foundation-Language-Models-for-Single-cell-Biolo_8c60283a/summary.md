---
title: "A-Survey-on-Foundation-Language-Models-for-Single-cell-Biolo"
source: https://aclanthology.org/2025.acl-long.26.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:49:18"
---

# 论文速读：A-Survey-on-Foundation-Language-Models-for-Single-cell-Biology

## 一句话总结
本文是首篇系统综述，聚焦将语言建模范式引入单细胞生物学的基座语言模型（Foundation Language Models），从单细胞预训练语言模型（PLMs）与大型语言模型（LLMs）双路径出发，全面梳理分词策略、预训练/微调范式、下游任务应用，并指出当前在数据质量、模型规模与评测协议方面的核心瓶颈与未来方向。

## 研究问题与动机
- 现有基于 Transformer 的单细胞模型综述缺乏从“语言建模”视角的系统剖析，难以统一指导研究者如何将细胞/基因序列类比为文本并构建通用基座。
- 单细胞数据具有高度稀疏性（<10% 基因可测表达）、无序性及严重批次效应，传统专用模型泛化能力受限，亟需统一、可迁移的跨数据集表征。
- PLMs（从零大规模预训练）与 LLMs（利用现成大模型能力进行轻量转换与微调）两条技术路线在单细胞领域并行发展，但缺乏横向对比与融合视角的综述。
- 领域内缺乏公开、标准化的评测基准与低门槛的任务交互格式，阻碍模型公平比较与跨学科落地应用。

## 核心贡献（创新点）
- **首篇专项综述**：首次全面覆盖单细胞基座语言模型的技术细节、应用场景与发展脉络，填补交叉领域系统性综述空白。
- **提出新颖双轨分类框架**：突破传统架构分类，依据“细胞解读方式”将现有工作明确划分为单细胞 PLMs 与单细胞 LLMs，并分别深入剖析其分词与训练范式。
- **提炼三维挑战与前沿路径**：从数据质量（稀疏性/批次效应/多组学资源不均）、模型设计（统一 Tokenizer 缺失/Scaling Law 未显现）与评测协议（基准匮乏/领域门槛高）指出核心瓶颈，为构建通用单细胞基座指明研究方向。

## 方法详解
- **单细胞 PLMs 构建流程**：
  1. **数据分词（Tokenization）**：分为离散型（Binning 将 log 变换表达值转为整数；Rank Value Encoding 按基因表达频次排序后映射至词汇表，序列长度通常截断至 2048）、连续型（利用蛋白质大模型生成基因 embedding、MLP 投影、层级贝叶斯降采样或 padding+排名编码）以及侧信息融合（注入细胞状态/器官/测序平台等元数据，或通过蛋白质语言模型 bridging）。
  2. **预训练范式**：主流为 **Masked Language Modeling (MLM)**（随机遮蔽 15%-30% 基因或基于高斯混合分布遮蔽）；**Next Token Prediction** 因细胞数据稀疏性易陷入平凡解而应用较少（仅 tGPT、scGPT 采用）；多任务预训练在 MLM 基础上联合对比学习、细胞生成、分类、细胞-文本匹配、去噪与细胞谱系对齐等任务。
- **单细胞 LLMs 构建流程**：
  1. **细胞到文本的转换**：**Cell-to-Sentence**（将归一化后 Top-K 高表达基因名拼接为句子，如 Cell2Sentence、CHATCELL）；**Text-level Gene Embeddings**（提示 LLM 查询基因功能文本，再按表达值加权/排序聚合为细胞向量）。
  2. **微调范式**：主流为 **Embedding-based Tuning**（利用 Sentence Transformer 提取嵌入后监督微调）；辅以 **Instruction-based Tuning**（构造 <question, answer> 配对数据，如细胞类型注释问答）与 **Tuning Free**（将 LLM 视为 Agent 直接生成 Python 代码分析原始数据，如 scChat）。

## 实验与结果
- 本文为综述论文，不单独提供新实验结果，而是系统性汇总已发表模型的评测体系与性能表现。
- **数据规模演进**：PLM 预训练数据已从百万级（scBERT 1M、CellLM 2M）扩展至亿级（scFoundation 50M、scPRINT 50M、Nicheformer 57M、CellFM 100M、GeneCompass 126M），LLM 侧主要依赖外部高质量文本-细胞配对或指令数据。
- **下游任务覆盖**：细胞水平（类型注释、新类型发现、批次校正、聚类、多组学整合、条件/无条件细胞生成）；基因水平（基因网络推断、扰动预测、标记基因/表达量预测）；药物相关（敏感性/反应预测）；空间转录组（空间插补、标签/组成预测、空间上下文分析）及其他（去噪、发育谱系推断、细胞-文本检索）。
- **核心结论**：大规模 PLM（如 GeneFormer、scGPT、scFoundation）在跨数据集泛化、零样本推理与复杂基因网络推断上表现稳健；LLM 路径（如 Cell2Sentence、CHATCELL、GenePT）在低资源场景下凭借涌现能力与交互友好性展现潜力；但统一基准缺失导致模型间难以直接横向对比，不同任务的最佳实践仍需进一步收敛。

## 相关工作脉络
- 区别于 Lan et al. (2024)、Szałata et al. (2024)、Bian et al. (2024b) 等侧重 Transformer 架构泛化应用的综述，本文**专攻语言建模范式**，首次将 PLM 与 LLM 两条技术路线并列对比，突出“细胞即文本”的表征思想。
- **scBERT / UCE / GeneFormer** 代表从零预训练 PLM 的早期探索，侧重 MLM 与离散/连续分词；本文将其纳入统一框架，并与后续引入多任务/元数据/蛋白质嵌入的工作（scGPT、Nicheformer、GeneCompass、CellFM）形成演进脉络。
- **Cell2Sentence / CHATCELL / scInterpreter / GenePT** 展示直接利用现成 LLM（GPT-2/3.5/T5/LLaMA）进行细胞表征的轻量路径，本文揭示了其与 PLM 在训练成本、泛化上限与交互能力上的 trade-off。
- 在评测层面，本文梳理了从早期专用小数据集到当前百万/亿级公共库（Panglao、GEO、CELL GENE、HCA、Tabula Sapiens）的基准演变，指出现有评测缺乏标准化与开放性，为未来建立统一 Benchmark 提供文献坐标。
- 方法设计上，本文提炼出“侧信息融合（元数据/蛋白质 LLM）”与“多任务联合预
