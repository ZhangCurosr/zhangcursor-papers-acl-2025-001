---
title: "Behind-Closed-Words-Creating-and-Investigating-the-forePLay"
source: https://aclanthology.org/2025.acl-long.120.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:41:25"
field: "多语言内容安全与审查"
keywords: ["情色内容检测", "多语言 NLP", "内容审查", "波兰语", "标注数据集", "human label variation", "content moderation"]
innovations: ["首个波兰语情色内容多维标注数据集 forePLay（24,768 句，五维分类）", "系统性对比专用波兰语模型与多语言 LLM 在情色检测上的性能差异", "揭示 encoder 模型在不平衡多分类场景下优于 LLM 的发现，并提出 LeWiDi 框架的应用方向"]
benchmarks: ["forePLay (Basic/Core/Extended/Full)", "KLEJ Benchmark (模型选型参考)"]
---

# 论文速读：Behind Closed Words: Creating and Investigating the forePLay Annotated Dataset for Polish Erotic Discourse

## 一句话总结
本文构建了首个波兰语情色内容检测人工标注数据集 **forePLay**（24,768 句，五维分类体系），并通过系统性实验证明：**针对波兰语微调的专用 transformer 模型（如 RoBERTa、PLLuM 系列）在情色内容检测上显著优于通用多语言大模型**，同时揭示了随着分类粒度增加带来的性能衰减与精度-召回权衡规律。

## 研究问题与动机
1. **多语言内容审查的空白**：现有情色/有害内容检测工具高度依赖英语训练数据，在波兰语等形态复杂语言上表现不佳，缺乏语言特定的检测资源。
2. **二元分类的局限性**：已有数据集多采用简化的二元划分（情色 vs 非情色），无法捕捉暧昧、暴力、社会禁忌等细粒度语义差异，难以支撑现实平台的多层次审核策略。
3. **LLM 安全护栏的需求**：大型语言模型生成内容的输入/输出防护需要可扩展、语言适配的检测机制，而现有通用 LLM 易产生过度审查（self-censorship）。
4. **标注主观性与人类变异**：情色文本的理解具有高度语境依赖性和文化特异性，需系统研究 human label variation (HLV) 对模型训练的影响。

## 核心贡献（创新点）
1. **首个波兰语情色内容多维标注数据集**：forePLay 包含 24,768 句，覆盖在线小说（69%）与专业文学作品（31%），并系统纳入 LGBTQ+ 叙事内容。
2. **创新五维分类体系**：提出暧昧（ambiguous）、暴力相关（violence-related）、社会不可接受行为（socially unacceptable）与情色（erotic）、中性（neutral）并行的精细标签，其中社会不可接受类别优先于暴力类别。
3. **系统性模型评测基准**：对比评估了专用波兰语 encoder 模型（HerBERT、Polish RoBERTa）、波兰语 LLM（PLLuM 系列、Bielik）、开源多语言模型（Mixtral、Llama 3.1、Mistral）及商用模型（GPT-4o），涵盖 0-shot 至 5-shot 及 SFT 多种设置。
4. **揭示架构与数据分布的交互效应**：发现 BERT 架构在类别不平衡场景下优于 LLM，且随分类复杂度上升，Type I 错误率显著下降而 Type II 错误率上升，呈现精度-召回的根本权衡。
5. **方法论层面的标注质量分析**：报告了 Cohen's Kappa 和 Krippendorff's Alpha，识别出标注者个体差异（如 Fem1 过度使用暧昧标签），为后续 LeWiDi 等分歧学习框架提供实证基础。

## 方法详解
- **数据采集**：从 6 个波兰语公开小说网站爬取用户生成内容（n=24,768 句，342,546 tokens），每作者最多收录 2 篇故事；专业文学部分选取 22 部波兰及翻译作品，含 LGBT 主题文本。
- **文本预处理**：使用 NLTK 进行句子切分，保留非标准拼写与正字法变异以增强生态效度。
- **标注协议**：6 人标注团队（3 男 3 女，年龄 20–40），盲标无元数据；每句独立由 3 人评分，多数票决定；830 例（3.35%）全分歧时由超级标注员裁决。
- **标签层级**：
  - **e（erotic）**：性行为描述、暧昧挑逗、露骨幻想；排除单纯生殖器名词或中性生理讨论。
  - **v（violence）**：性骚扰、强奸、非自愿暴力；自愿 BDSM 归入 erotic。
  - **u（unacceptable）**：恋童、乱伦、兽交等违法/禁忌行为，优先级高于 v。
  - **a（ambiguous）**：依赖语境才可能产生情色联想的句子。
  - **n（neutral）**：非性相关内容或中性性知识陈述。
- **实验设置**：四种评估配置——Basic（2类：neutral/erotic）、Core（3类：+ambiguous）、Extended（4类：合并 v+u）、Full（5类）；encoder 模型 fine-tune 5 小时（Base）/10 小时（Large）；LLM 采用 few-shot 与 SFT 两种范式。
- **评估指标**：macro-F1 为主，辅以 Type I（假阳性率）和 Type II（假阴性率）误差分析。

## 实验与结果
- **数据集规模**：Total n=24,768；Neutral 68.14%、Erotic 25.68%、Ambiguous 5.43%、Unacceptable 0.47%、Violence 0.28%。
- **Encoder 模型最优**：RoBERTa Base 在 Basic 设置下取得 **macro-F1 = 0.944**（HerBERT Base 0.929）；Full 五类时 RoBERTa Large 仅 0.664，显示粒度增加导致显著性能衰减。
- **LLM 最优**：**PLLuM-Mistral-12B (SFT) 在 Basic 设置达 macro-F1 = 0.946**，PLLuM-Mixtral-8x7B 5-shot 达 0.921。
- **多语言模型局限**：GPT-4o 在 Basic 0.888 尚可，但在 Full 设置降至 0.349；Mixtral 8x22B 在 Basic 0.872 但 Full 仅 0.335。
- **错误模式**：Type I 错误率在 Basic 约 60–80%，Full 降至 40–50%，表明简单任务更易过度审查，复杂任务更易漏检。
- **拒绝响应**：Llama 3.1 8B-Instruct 产生 848 条 refusal（最多），GPT-4o 和 Mixtral 8x22B 几乎无拒绝。

## 相关工作脉络
1. **Triplex**（Achour, 2016）：27,000 部 Archive of Our Own 作品（1.62B tokens），面向英文情色文学分析，无细粒度语义标签。
2. **BeaverTails**（Ji et al., 2023）：LLM 对齐研究数据集，含成人内容类别，但为英文且分类粗糙。
3. **CENSORCHAT**（Qiu et al., 2024）：基于知识蒸馏的对话系统情色检测，英文，聚焦 LLM 监控而非多语言泛化。
4. **Llama Guard**（Inan et al., 2023）：Meta 的 LLM 安全分类器，英文优先，对波兰语等低资源语言支持有限。
5. **Legilimens**（Wu et al., 2024）：利用 LLM 推理的概念特征进行内容审查，发现 LLM 存在自我审查倾向。
6. **KLEJ Benchmark**（Rybak et al., 2020）：波兰语 NLP 综合评测基准，为本文选用 HerBERT/RoBERTa 提供选型依据。

## 局限性与未来方向
1. **样本偏差**：69% 内容来自匿名在线小说，可能无法充分覆盖其他语境（如正式文学、社交媒体、即时通讯）中的情色话语变体。
2. **极端类别的简化处理**：暴力和不可接受行为共仅 185 例（0.75%），合并为单一类别可能掩盖了重要语义区分。
3. **标注主观性未完全解决**：Krippendorff's Alpha 仅 0.387（排除异常标注者后升至 0.716），暧昧类别被部分标注者用作"不确定兜底"。
4. **评估框架的不足**：未在 prompt 中显式提供标签定义，可能人为压低 LLM 性能；未深入测试对抗样本与边缘案例。
5. **跨架构比较的公平性**：参数规模差异巨大（8B vs 70B vs encoder），直接比较存在固有限制。

## 研究启发与可借鉴点
1. **多语言内容审查的资源建设路径**：通过跨学科团队（文学学者 + 语言学家 + NLP 专家）构建领域适配数据集，可有效提升低资源语言上的检测效果，值得在非英语社区复制。
2. **人类标注变异（HLV）的制度化处理**：本文对异常标注者的识别与移除策略，以及提出的 LeWiDi / Multiple Ground Truth 框架，为后续研究如何处理主观标签提供了方法论参考。
3. **架构选择与数据分布的匹配规律**：编码器模型在不平衡多分类场景中优于 LLM 的结论，提示在实际部署中应根据类别稀疏度灵活选择架构，而非盲目追求大模型。
4. **Type I/II 错误权衡的量化评估**：本文引入误报/漏报比例分析而非仅依赖 macro-F1，为内容审查系统的实际部署提供了更贴近业务需求的评估维度。

## 关键术语表
- **forePLay**：首个波兰语情色内容多维度人工标注数据集，含 24,768 句五类标签。
- **Human Label Variation (HLV)**：不同标注者对同一文本给出不同标签的现象，反映主观判断的内在不确定性。
- **Learning with Disagreements (LeWiDi)**：利用多标签标注分歧进行训练的框架，替代传统的单 ground truth 聚合方式。
- **Macro-F1**：按类别分别计算 F1 后取均值，对少数类别更敏感的评估指标。
- **Type I 错误（假阳性）**：将非情色内容误判为情色，导致过度审查。
- **Type II 错误（假阴性）**：将情色内容误判为非情色，导致漏检。
- **PLLuM**：波兰大型语言模型系列（Polish Large Language Model），面向公共 administration 场景微调。
- **Krippendorff's Alpha**：衡量多标注者一致性的统计量，值越高表示标注可靠性越强。

## 可复现要素
- **数据集**：部分公开——forePLay Dataset Release 1.0（3,704 样本：2,728 情色 + 976 暧昧）已上传至作者 GitHub，中性数据因可从公开语料获取而未发布；暴力/不可接受类别因伦理考量未公开。
- **代码**：论文未提及代码仓库链接；实验使用 NLTK、Hugging Face Transformers 等标准库。
- **关键超参**：encoder 模型 batch_size=8，早停；Base 训练 5h、Large 10h（A100 40GB）；LLM SFT 使用 2 节点 × 4×H100 96GB，Llama-3.1-8B 训练 3h，Mistral-12B 训练 4h。
