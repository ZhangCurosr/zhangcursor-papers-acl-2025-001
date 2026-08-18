---
title: "How-Much-Do-Encoder-Models-Know-About-Word-Senses"
source: https://aclanthology.org/2025.acl-long.113.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:54:11"
field: "词义消解与语义表示分析"
keywords: ["Word Sense Disambiguation", "Probing", "Pretrained Language Models", "DeBERTa", "Semantic Representations"]
innovations: ["首次系统评估四种PLM逐层WSD能力，发现DeBERTa-v3中间层最优", "揭示词义库示例同质性是造成ODE/SemCor性能差距的主因", "基于研究发现设计轻量WSD模型GlossDeBERTa-small，降低70%训练时间"]
benchmarks: ["SemCor", "ODE", "ALL"]
---

# 论文速读：How-Much-Do-Encoder-Models-Know-About-Word-Senses

## 一句话总结
论文通过计算词义质心与余弦相似度，系统评估了多种编码器型PLM在不经过微调的情况下区分词义的能力，发现DeBERTa-v3表现最佳且其最优层为中间层（第7-8层），而非通常使用的输出层；同时深入分析了词义库存结构对模型性能的显著影响，并据此设计了一个小型高效WSD模型GlossDeBERTa-small。

## 研究问题与动机
1. **核心问题**：预训练编码器模型在未微调的情况下，其隐式表示能在多大程度上区分同一词语的不同词义？这是理解PLM内在语义能力的关键，也是当前"黑盒"使用方式的盲区。
2. **现有方法的局限**：绝大多数WSD方法将PLM作为特征提取器或微调基础，忽略了预训练阶段本身已编码的语义知识；且已有probing研究大多仅针对单一模型或简化任务，缺乏跨模型、跨层级的系统比较。
3. **词义库存差异的影响未知**：SemCor（WordNet）与ODE的评估结果存在约18个F1点的差距，但两者差异的来源（粒度、频率、示例同质性）尚未被系统分析。
4. **层级选择问题**：不同预训练策略的模型其最优语义层位置可能不同，实践中通常默认使用最后一层，但这一假设可能并非最优。

## 核心贡献（创新点）
1. **首次系统评估四种主流编码器PLM在WSD上的逐层表现**：发现RTD预训练模型（ELECTRA、DeBERTa-v3）的最优层位于中间层（第5-8层），而MLM模型（BERT、RoBERTa）的最优层位于最后一两层，与DeBERTa-v3最优层相比输出层性能下降约15个百分点。
2. **揭示词义库存结构对模型性能的因果影响**：通过控制实验排除粒度差异和频率不平衡，证明ODE与SemCor之间的性能差距主要来源于示例的同质性——ODE的示例更同质、词义边界更清晰，使WSD任务更易。
3. **提出GlossDeBERTa-small：基于研究发现的小模型设计**：利用DeBERTa-v3第7层作为分类器输入，在保持79.5 F1（ALL数据集）的同时，相比GlossBERT提速28%、相比完整GlossDeBERTa提速70.6%，显著降低碳足迹。
4. **开源代码与数据**：公开全部实验代码与数据（https://github.com/SapienzaNLP/wsd-probing），促进可复现研究。

## 方法详解
- **词义质心计算**：对训练集$\mathcal{C}_{train}$中每个词义$s$，使用冻结的Transformer编码器提取所有含该词的上下文嵌入$\{e_1, ..., e_\mathcal{N}\}$，若词被拆分为多个token则取平均，得到该词义的质心：$C_e n t r o i d(E) = \frac{1}{\mathcal{N}}\sum_{i=1}^{\mathcal{N}} e_i$。
- **词义预测**：对测试实例$t$，提取其嵌入$e$后，与同lemma和POS的所有词义质心计算余弦相似度，选择得分最高的质心对应词义：$c^* = \arg\max_{c \in C} \frac{c \cdot e}{\|c\| \|e\|}$。
- **层级分析**：对每个模型的所有12层重复上述流程，分别评估每层的表现，从而定位最优层。
- **数据过滤**：剔除单义词；设置最小示例数阈值$K \in \{5, 10\}$，确保词义有足够样本构建可靠质心；按80%-20%划分训练/测试集。
- **内部/外部相似度度量**：定义内部相似度为同类词义实例嵌入间的平均余弦相似度，外部相似度为不同词义簇间实例的平均余弦相似度，差值越大说明词义分离越清晰。

## 实验与结果
- **数据集**：SemCor（WordNet 3.0，33,362词义，226,036实例）和ODE（Oxford Dictionary of English，79,004词义，约785,000实例）。
- **评估模型**：BERT-base-cased/uncased、RoBERTa-base、ELECTRA-base-discriminator、DeBERTa-v3-base。
- **核心发现**：
  - DeBERTa-v3整体最优：在ODE（K=5）上第7-8层达**90.46 F1**，在SemCor（K=5）上第7-8层达**71.05 F1**，均显著优于其他模型。
  - DeBERTa-v3最优层（第7-8层）相比输出层（第12层）提升约**15 F1点**（ODE上：90.46 vs 75.94）。
  - MLM模型（BERT、RoBERTa）的最优层在最后一两层（第10-11层）。
  - ELECTRA因生成器与判别器共享token嵌入导致训练目标冲突（pull在相反方向），性能普遍低于BERT/RoBERTa。
  - ODE与SemCor间约**18个F1点**的差距，在控制$K=5$后仍存在，排除了频率因素后归因于**示例同质性**：ODE内部分类层（第7-8层）内部相似度约70、外部相似度约20，而SemCor两项均较高（内部约70、外部约45-60），表明SemCor词义间语义边界更模糊。
- **GlossDeBERTa-small结果**（ALL数据集，Fine-grained English all-words WSD）：F1=**79.5**，训练时间比GlossBERT缩短28%、比完整GlossDeBERTa缩短70.6%。

## 相关工作脉络
1. **传统WSD方法**（Lesk, 1986；Raganato et al., 2017）：从基于词典的知识方法演进到以SemCor+WordNet为核心的监督学习范式，本文延续了监督WSD框架但聚焦于预训练表示的内在能力而非任务微调。
2. **Gloss-enhanced WSD**（GlossBERT, SenseBERT, ARES, SREF）：这些工作将词义定义（gloss）作为额外信息注入模型，本文不引入gloss，而是直接探测纯上下文嵌入的语义分离能力，形成对照视角。
3. **PLM probing研究**（Adi et al., 2017；Tenney et al., 2019a）：确立了"不同层编码不同语言知识"的洞见，本文将这一范式从句法/词性 probing延伸到词义消解这一lexical-semantic任务，并系统覆盖四层模型。
4. **几何表示 probing**（Coenen et al., 2019；Tripodi, 2021；Proietti et al., 2024）：这些研究证明了BERT等模型能区分词义，但多数仅分析单一模型或未覆盖层间差异，本文的系统性跨模型跨层分析填补了这一空白。
5. **WSD基准与评估**（SemCor+WordNet vs ODE）：以往研究多单一使用SemCor，本文首次在同一实验框架下系统比较两种词义库存，揭示了评估资源选择对结果的显著影响。

## 局限性与未来方向
1. 仅评估了四种base规模模型（12层），未涵盖ALBERT、CamemBERT、SpanBERT、DistilBERT等其他架构及large版本（24层），结论的外推性有限。
2. 仅针对英语，未扩展到多语言场景；多语言PLM（mBERT、XLM-R等）的WSD能力值得探究。
3. 词义库差异分析虽排除了粒度和频率因素，但ODE的示例来自词典定义而非真实语境（SemCor来自1960年代文本），示例生态差异对模型学习的影响仍需进一步控制。
4. 未来计划扩展至多语言PLM和大语言模型（ChatGPT、Llama、Mistral），探索其在跨语言WSD任务中的表现。

## 研究启发与可借鉴点
1. **"最优层不在最后一层"的启示**：对于RTD预训练模型（如DeBERTa系列），下游任务不一定从最后一层取值，中间层可能携带更合适的语义表征，这在设计轻量级模型时具有重要参考价值。
2. **质心相似度法作为轻量级WSD探针**：无需微调即可评估模型内在语义能力，该方法计算成本低、可解释性强，适用于对PLM进行快速语义能力审计。
3. **控制变量法分析评估数据质量**：通过$K=5$控制示例数量后仍观察到显著性能差异，结合内部/外部相似度指标定位问题根源，这种"排除法+量化指标"的分析范式可迁移到其他NLP任务的benchmark分析中。
4. **基于发现设计小模型**：研究结论（DeBERTa-v3第7层最优）直接指导了GlossDeBERTa-small的架构选择，验证了"理解模型→优化模型"的研究闭环价值，适合在资源受限场景下复用。

## 关键术语表
**Word Sense Disambiguation (WSD)**：词义消解，根据上下文确定多义词在特定语境中的正确含义，通常建模为从固定词义库中选择标签的多分类问题。
**Pretrained Language Models (PLMs)**：预训练语言模型，如BERT、RoBERTa、DeBERTa等，在大规模语料上通过自监督目标预训练后用于下游任务的编码器模型。
**Centroid-based Probing**：质心底层探测方法，通过计算同一词义所有训练实例嵌入的均值作为该词义的"质心"，再用余弦相似度匹配测试实例与质心来评估模型区分词义的能力。
**Sense Inventory**：词义库存/词义本体，如WordNet和ODE，提供词汇所有合法含义的封闭集合，是WSD任务的标签体系来源。
**Internal/External Similarity**：内部相似度（同词义实例间的平均余弦相似度）与外部相似度（不同词义簇实例间的平均余弦相似度），二者差值反映词义分离的可辨别程度。
**Replaced Token Detection (RTD)**：被替换词检测，ELECTRA提出的预训练目标，通过判别器检测生成器替换的token，样本效率高于MLM但与MLM存在训练目标冲突。
**GlossDeBERTa-small**：本文提出的小型WSD模型，在DeBERTa-v3第7层之上添加分类头并在SemCor上微调，以较少参数和训练成本实现竞争性性能。

## 可复现要素
- **数据集**：SemCor（开源，https://wordnet.princeton.edu/）和ODE（Chang et al., 2018版本，论文标注为openly available）。
- **代码**：已公开，地址为https://github.com/SapienzaNLP/wsd-probing。
- **模型**：Hugging Face官方权重（bert-base-cased/uncased、roberta-base、electra-base-discriminator、deberta-v3-base）。
- **关键超参**：$K \in \{5, 10\}$（最小示例数阈值），训练/测试集按80%-20%划分；GlossDeBERTa-small：4 epochs、dropout=0.1、学习率=2e-5、batch size=64，使用SE07开发集选择最优checkpoint，在NVIDIA GPU 1080 Ti上训练。
