---
title: "RELATIONALCODER-Rethinking-Complex-Tables-via-Programmatic-R"
source: https://aclanthology.org/2025.acl-long.89.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:16:18"
field: "表格理解与问答"
keywords: ["表格转换", "关系型数据", "半结构化表格", "程序化推理", "循环引用解码"]
innovations: ["提出 RELATIONALCODER 通过 SQL 将半结构化表格转换为关系型数据，解锁数据分析工具生态", "设计 Loop Reference Decoding (LRD) 技术，通过循环引用实现极致压缩（最高 160,000×）", "引入 SFT-AP 主动学习标注策略，利用 LLM 筛选难例降低标注成本"]
benchmarks: ["HiTab", "MULTIHIERTT"]
---

# 论文速读：RELATIONALCODER-Rethinking-Complex-Tables-via-Programmatic-R

## 一句话总结
本文提出 RELATIONALCODER，通过 SQL 代码将半结构化复杂表格统一转换为关系型数据， enabling 与现有数据分析工具生态无缝集成；引入 Loop Reference Decoding (LRD) 技术实现输入输出的极致压缩（最高达 160,000×），并在 HiTab 和 MULTIHIERTT 基准上显著提升 QA 准确率，尤其对小型模型增益超过 20%。

## 研究问题与动机
1. **半结构化表格的机器处理障碍**：现实文档中的表格存在合并单元格、多级表头、空白行、跨表依赖等复杂格式，难以直接被关系型数据库工具处理。
2. **现有方法未解锁关系型工具生态**：已有工作多聚焦于让 LM 直接理解复杂表格，而非将其转换为标准关系型格式，限制了与 SQL/Pandas 等工具的结合。
3. **大表格生成的可扩展性瓶颈**：直接逐单元格生成输出表格会导致输出长度随表格行列数线性增长（O(N×M)），引发幻觉且效率低下。
4. **标注数据稀缺且成本高**：缺乏高质量的表格转换训练数据，且人工标注复杂表格成本高昂。

## 核心贡献（创新点）
1. **SQL-based 关系型表格生成框架**：通过 SQL 的 CREATE/INSERT/UPDATE 语句生成规范化关系表，避免 Markdown/HTML 中常见的 schema 错误（如列数不一致、重复表头）。
2. **Loop Reference Decoding (LRD)**：识别输入中的"可扩展组"(expandable groups)，用 FOR/WHERE 循环引用单元格地址批量复制数据，将输出长度从 O(N×M) 压缩至 O(K)，平均压缩率 68.4%，极端情况达 160,000×。
3. **主动参与监督微调 (SFT-AP)**：利用 LLM 自动评估规范化质量和推理准确率，筛选易错样本进行人工标注，显著降低标注成本。
4. **人类标注数据集**：构建包含 850 个表格转换样本的数据集，涵盖 HiTab 和 MULTIHIERTT 来源。
5. **端到端验证**：在 HiTab 和 MULTIHIERTT 上，结合 Chain-of-table 推理框架，使 GPT-4o 达到新 SOTA，小型模型（Llama-2、Mistral）准确率提升超 20%。

## 方法详解
### 1. 表格编码与压缩（Table Encoding & Compression）
- 采用 **SPREADSHEETLLM** 的编码方式，为每个单元格附加显式地址（如 A7:H16），支持后续引用。
- 对空单元格或重复单元格采用逆索引 JSON 压缩。
- 将相邻同类型数值单元格聚类为矩形区域。
- 对超大表格使用 Table Split Algorithm 分割为带传播表头的子段。

### 2. SQL-based 数据转换
- 生成 `CREATE TABLE` 定义列名。
- 通过 `INSERT`/`UPDATE` 语句插入数据，支持任意顺序构建，克服 Markdown/HTML 逐行生成的限制。
- 消除可推导的统计单元格（如总计行/列），但保留具领域意义的计算（如密度、聚合招生数）。

### 3. Loop Reference Decoding (LRD)
- **核心思想**：检测到输入中存在重复结构的"可扩展组"（expandable groups）后，生成 `FOR` 循环（Oracle 风格）或 `WHILE` 循环（SQL Server 风格），通过单元格地址引用批量复制数据。
- **压缩效果**：输出长度从枚举每个单元格的 O(N×M) 降至仅表示唯一值集合的 O(K)。
- **示例**：5,000 只股票 × 252 个交易日的表格，传统 SQL 需 ~18.6M tokens，LRD 仅需 115 tokens，压缩比 160,000×。

### 4. 主动参与监督微调 (SFT-AP)
- **流程**：
  1. 使用当前模型评估训练样本的规范化质量（LLM 判断）和推理准确率（QA 模型验证）。
  2. 筛选四类样本：无违规但错误、规范化违规、计算依赖违规、QA 错误，按比例 (0.4, 0.15, 0.15, 0.3) 采样。
  3. 人工标注选定样本（每表最多标注 32 行 × 8 列）。
  4. 更新数据集并微调模型，迭代至精度提升低于阈值或达到最大轮次。
- **标注协议**：要求标注者避免过度拆分表格，保留分析可解性。

## 实验与结果
### 数据集
- **HiTab**：3,597 张来自政府报告和 Wikipedia 的复杂表格，10,672 个 QA 对。测试集 1,584 样本。
- **MULTIHIERTT**：2,513 份 PDF 金融报告中的 10,440 个 QA 对。验证集 338 个表格样本。

### 评估指标
- **规范化质量**：人类标注 + GPT-4o 自动评估（判断是否符合关系型结构、支持 SQL 查询）。
- **推理准确率**：执行准确率（execution accuracy）。

### 主要结果
| 模型 | HiTab (原始表格) | HiTab (RELATIONALCODER) | MulHi (原始) | MulHi (RELATIONALCODER) |
|------|------------------|------------------------|--------------|------------------------|
| Llama-2 | 30.3% | **55.9%** (+25.6%) | 25.3% | **47.6%** (+22.3%) |
| Mistral | 25.1% | **52.9%** (+27.8%) | 20.2% | **46.7%** (+26.5%) |
| GPT-3.5 | 43.3% | **66.7%** (+23.4%) | 35.0% | **57.0%** (+22.0%) |
| GPT-4o | 80.7% | **84.8%** (+4.1%) | 68.1% | **73.6%** (+5.5%) |

- **LRD 消融**：相比 Markdown 输出，LRD 在 HiTab 上提升约 2% 准确率，输出长度平均减少 68.4%。
- **SFT-AP 效果**：GPT-3.5-SFT-AP 版本超越 zero-shot GPT-4o，微调后的 Llama-2/Mistral 接近 GPT-4o-zero-shot 水平。
- **规范化质量评估**：GPT-3.5 在 HiTab 上人类评估达 79.4%，自动评估 82.3%。

## 相关工作脉络
1. **表格转换传统方法**：基于用户示例的合成变换（Harris & Gulwani, 2011）、分类方法识别表层次结构（Chen & Cafarella, 2014; Wang et al., 2021）——依赖规则/分类，难以应对野外表格的多样性。
2. **表格式归一化工作**：NormTab（Nahid & Rafiei, 2024）仅使用单一转置算子，灵活性受限且未建模表间关系。
3. **自动关系化**：Auto-tables（Li et al., 2024）通过预定义算子合成训练数据，依赖分类识别结构特征，对复杂格式适应性差。
4. **表格编码方法**：Markdown/HTML/XML/LaTeX 等标记语言缺乏显式单元格地址，难以支持精确定位和引用（见 Appendix B 详细对比）。
5. **表格 QA 基线**：EEDP（Srivastava et al., 2024）作为 prompt 策略 baseline；Chain-of-table（Wang et al., 2024）用于关系化后的多跳推理。

## 局限性与未来方向
1. **不适用于需保留原文档格式的场景**：如文档操作任务需保留原始内容和格式，此方法不适用。
2. **未处理非结构化文本信息**：当前仅关注表格，未整合文档中嵌入的有价值文本信息。
3. **转换与推理需两阶段模型**：需要更强模型进行表格转换，未来可探索端到端的"转换-处理-推理"单模型管道。
4. **依赖高质量标注数据**：当前缺乏高质量表格转换标注数据，未来可探索基于强化学习（RL）的弱监督训练。
5. **伦理风险**：依赖 GPT 模型可能继承偏见和不当内容生成风险。

## 研究启发与可借鉴点
1. **程序化输出范式**：将代码（SQL）作为中间表示而非原始表格，可利用外部执行器验证正确性，提升可靠性——可迁移至其他结构化数据转换任务。
2. **循环引用压缩技术**：LRD 的"识别重复模式+地址引用"思想可推广至时间序列、矩阵数据等具有周期性/规律性结构的场景。
3. **主动学习标注策略**：SFT-AP 利用 LLM 自动评估筛选难例的策略，可应用于其他数据标注成本高的任务（如代码生成、公式解析）。
4. **规范化质量的双评机制**：结合 LLM 自动评估与人类校验，既能规模化又能保证质量，可作为表格相关任务的评估模板。
5. **小型模型的增益放大**：关系化转换对小型模型增益最大（>20%），提示在资源受限场景下，数据预处理的重要性可能被低估。

## 关键术语表
**Semi-structured Tables**：半结构化表格，具有层级、合并单元格、多重表头等复杂格式的表格数据。
**Expandable Groups**：可扩展组，输入表格中具有统一 schema 和语义的连续块，可通过循环批量处理。
**Loop Reference Decoding (LRD)**：循环引用解码，通过 FOR/WHILE 循环引用单元格地址批量生成输出数据的技术。
**Relational Normalization**：关系规范化，将数据组织为符合关系模型的多张表，消除冗余和异常。
**SFT-AP**：Supervised Fine-tuning with Active Participation，主动参与监督微调，利用 LLM 筛选难例进行人工标注的训练策略。
**Chain-of-table**：表格链式推理，将复杂 QA 分解为多个关系子查询逐步执行的推理框架。
**Execution Accuracy**：执行准确率，生成 SQL 查询与实际数据库执行结果匹配的比例。
**Table Split Algorithm**：表格分割算法，将超大表格切分为带传播表头的子段以适配上下文长度限制。

## 可复现要素
- **数据集**：HiTab（公开）、MULTIHIERTT（公开）；本文构建的 850 样本转换数据集（论文未明确是否公开，但项目页面提供）
- **代码**：已开源，项目页面 https://github.com/haoyudong/RelationalCoder
- **关键超参**：
  - SFT-AP：m=32, n=8, (p_rnd, p_norm, p_calc, p_qa)=(0.4, 0.15, 0.15, 0.3), q=0.1, l=5
  - Llama-2 微调：lr=5e-5, epochs=40, batch_size=4, gradient_accumulation=8, LoRA rank=32, alpha=64, dropout=0.01
  - GPT-3.5 微调：LoRA dimension=32
  - 推理：temperature=0, top-p=0（确定性解码）
- **硬件**：8×A100 GPU
