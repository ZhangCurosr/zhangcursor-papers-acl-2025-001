---
title: "Centurio-On-Drivers-of-Multilingual-Ability-of-Large-Vision"
source: https://aclanthology.org/2025.acl-long.143.pdf
model: agnes-2.5-flash
chunks: 5
summarized_at: "2026-08-18 15:50:37"
field: "多语言多模态学习"
keywords: ["多语言视觉语言模型", "LVLM", "多语言训练策略", "OCR", "语言保真度", "多模态"]
innovations: ["系统性揭示语言数量与比例对多语言LVLM性能的驱动规律", "提出SMPQA多语言图表OCR评测数据集", "实证发现预训练阶段比指令微调更需要多语言数据"]
benchmarks: ["XM3600", "xGQA", "M3Exam", "CVQA", "SMPQA", "MTVQA", "xMMMU"]
---

# 论文速读：Centurio: On Drivers of Multilingual Ability of Large Vision-Language Models

## 一句话总结
本文系统探究了多语言视觉-语言模型（LVLM）的训练策略，揭示了语言数量、语言比例分布及OCR数据对多语言能力的驱动作用，并在此基础上训练了覆盖100种语言的高效多语言LVLM——Centurio。

## 研究问题与动机
- **英语数据垄断**：多数现有LVLM以英语训练数据为主，导致非英语指令理解能力差、文本-图像理解弱、输出语言保真度低。
- **缺乏系统性研究**：关于语言数量、语言分布比例、预训练/指令微调阶段的数据配比等关键超参对多语言能力的影响，尚无系统性实证分析。
- **多语诅咒的疑问**：直觉上增加训练语言数量可能导致"多语诅咒"（multilingual curse），即已有语言性能下降；论文旨在验证此现象是否存在。
- **非拉丁脚本的困难**：合成OCR数据能否帮助非拉丁脚本语言是未知的，因为现有数据严重偏向拉丁脚本。

## 核心贡献（创新点）
- **系统性消融研究**：首次系统评估训练语言数量、语言分布比例（预训练 vs 指令微调阶段）、OCR数据增强对多语言LVLM性能的影响，揭示"多语诅咒"并不显著存在。
- **关键训练策略发现**：证明语言暴露次数比数据总量更重要，25%–50%的非英语训练数据即可大幅提升多语言能力，且预训练阶段比指令微调阶段更需要多语言数据。
- **新数据集SMPQA**：提出Synthetic Multilingual Plot Question Answering数据集，覆盖11种语言、7种书写系统，同时测试多语言OCR的"读取"与"定位"两大能力。
- **Centurio模型**：基于上述发现训练了覆盖100种语言的LVLM（使用Aya-Expanse或Qwen 2.5作为LLM主干），在多个多语言基准上达到SOTA。

## 方法详解
### 架构设计
- **基础架构**：基于LLaVA架构，图像编码器采用SigLIP SO400（Zhai et al., 2023），LLM主干为Phi 3.5（3.8B参数，Abdin et al., 2024b）；子集实验使用Llama 3（8B）验证泛化性。
- **训练策略**：冻结图像编码器，仅用LoRA更新MLP与LLM权重；后续实验解冻图像编码器以提升多脚本性能。
- **Centurio最终版本**：LLM主干选用Aya-Expanse或Qwen 2.5，图像分块采用Shi et al. (2024)方法（沿特征维度拼接全图与分块token，避免过长序列）。

### 训练数据
- **预训练数据**：ShareGPT4V的1.3M密集图像描述（Chen et al., 2024b）+ ALLaVA的0.7M数据。
- **指令微调数据**：源自LLaVA-Next的0.77M样本。
- **多语言翻译**：通过NLLB机器翻译（Costa-jussà et al., 2022）生成多语言数据，精确控制语言比例。
- **合成OCR数据**：使用Synthdog代码生成500k合成OCR数据（Kim et al., 2022），每种语言5k（拉丁脚本）/10k（非拉丁脚本）进入预训练，50k进入指令微调。

### 训练超参
- **优化器**：AdamW + cosine learning rate schedule + 3% linear warmup。
- **LoRA配置**：rank=256, α=512，应用于LLM所有矩阵；LLM其余参数冻结。
- **学习率**：图像编码器1e-6；LoRA & MLP 1e-4（预训练用5e-5，指令微调用3e-5）。
- **其他**：Weight decay=0，Batch size=32（梯度累积），Loss为causal language modeling，同时mask图像token和prompt token。
- **关键技巧**：pretraining→instruct tuning阶段继续训练同一LoRA adapter，不merge权重重新初始化。
- **硬件与时长**：4×H100 GPU，Centurio共训练6天（预训练与指令微调各半）。

### 评估体系
- **13个下游任务，覆盖43种语言**（扩展至66种语言），按Joshi et al. (2020)资源等级分为T1–T5五层。
- **任务类型**：判别式（二分类/多选）+ 开放式生成（评估语言保真度）。
- **SMPQA数据集**：每语言100张图（50饼图+50柱图），每图5道reading问题+8道grounding问题；使用精确匹配评估（放弃编辑距离，因语言词长差异大）。

## 实验与结果
### 数据集
- **训练数据**：自然图像（LLaVA Instruct 160k, VQAv2 83k, GQA 72k等）、多图任务（NLVR 86k, Spot-the-difference 8k）、OCR文本密集数据（OCRVQA 50k, DocVQA 10k等）、合成OCR数据（500k Synthdog）。
- **评测基准**：BabelImageNet-MC、M3Exam、CVQA（10,000问题，31种语言，39个国家-语言对）、M5B-VGR、M5B-VLOD、MTVQA（6,778问答对，2,116张图像）、SMPQA、XM3600（36语言）、xGQA、xMMMU、XVNLI、MaRVL、MaXM。

### 关键发现与数字
**RQ1：训练语言数量**
- L100（99种非英语语言）vs 仅T5（6种）：T1从14.4提升至19.3，T2从24.4提升至32.6，英语仅从53.6降至52.6（损失可忽略）。
- 语言保真度：英语100%，T5从6.2%→98–99%，T1从0.2%→72.9%。

**RQ2：指令微调语言分布**
- 英语占比E∈{1%, 10%, 25%, 50%, 75%, 90%}，峰值出现在E=25%–75%区间，E=50%为最稳健选择。
- 低资源语言（T1/T2）受益更多于多语言数据；高资源语言（T5）和英语受益更多于英语数据。

**RQ3：预训练语言分布**
- 固定指令微调为L100 + E_IT=50%，E_PT=50%为最优：T1从19.3→22.8，T2从32.6→39.5，英语52.6→54.9。
- 纯英语预训练对非英语任务几乎无益；极低英语比例（E_PT=1%）效果接近E_PT=50%，但多语言性能提升更有限。

**RQ4：多语言文本图像理解（OCR增强）**
- 加入500k Synthdog合成OCR数据 + 解冻图像编码器。
- 拉丁脚本语言提升显著，但非拉丁脚本仍存在巨大性能差距（Read: en=54.8 vs other=8.0 @ 1%英语）。

**Centurio模型表现**
- 在13个基线模型（参数2B–13B）中，Centurio Qwen和Centurio Aya在多数基准上达到最优或接近最优。
- xGQA：Centurio Qwen avg.=54.8，de=57.0，zh=55.6；Centurio Aya avg.=53.2。
- XM3600语言保真度：Centurio Qwen avg.=70.2，Centurio Aya avg.=62.4。
- M3Exam：Centurio Qwen en=87.5，avg=73.1（最高平均分）。

## 相关工作脉络
- **LLaVA系工作**：本文基于LLaVA架构扩展多语言能力，区别于LLaVA原始单语设计。
- **多语言LLM前作**：Aya-Expanse（Dang et al., 2024）、NLLB翻译方案（Costa-jussà et al., 2022）为本文明多语言数据生成提供基础。
- **视觉-语言模型基线**：对比Qwen2-VL、Phi-3.5-vision、Llama-3.2-Vision、InternVL2.5等13个模型，定位为系统性训练策略研究而非单一模型改进。
- **多语言评测基准**：使用XM3600、xGQA、M3Exam等已有基准，同时提出SMPQA填补多语言图表OCR评测空白。
- **多语诅咒讨论**：与之前关于多语言训练可能导致性能下降的担忧形成对比，本文实证表明该现象不明显。

## 局限性与未来方向
- **非拉丁脚本数据不足**：合成OCR数据对非拉丁脚本提升有限，假设需数量级级更多的非拉丁脚本数据。
- **低资源语言仍薄弱**：T1 Tier语言性能虽提升但仍较低（如quz仅0.2），需要更多高质量数据。
- **语言保真度有待提高**：尽管有显著改善，但部分语言的输出保真度仍有提升空间。
- **未来方向**：收集更多非拉丁脚本数据、探索更优的语言比例分配策略、扩展至更多语言和任务。

## 研究启发与可借鉴点
- **语言比例经验法则**：25%–50%非英语数据为 optimal range，可直接迁移到其他多语言模型训练。
- **预训练阶段更需多语言数据**：指令微调阶段英语占比可适当提高，节省多语言数据成本。
- **LoRA持续训练技巧**：pretraining→instruct tuning阶段不merge权重而继续训练同一adapter，此技巧可复用于其他多阶段训练场景。
- **SMPQA评测设计**：reading+grounding双任务设计、精确匹配评估方式，可作为多语言OCR评测的参考模板。
- **非拉丁脚本数据瓶颈意识**：提醒团队在构建多语言模型时需特别关注非拉丁脚本的数据收集策略。

## 关键术语表
- **LVLM**：Large Vision-Language Model，融合视觉与语言理解的大规模多模态模型。
- **多语诅咒**：Multilingual Curse，指增加训练语言数量可能导致已有语言性能下降的现象。
- **SMPQA**：Synthetic Multilingual Plot Question Answering，本文提出的多语言图表OCR评测数据集。
- **NLLB**：No Language Left Behind，Meta开发的多语言机器翻译模型，用于本文多语言数据生成。
- **语言保真度**：Language Fidelity，模型输出与其目标语言语法、拼写的一致性程度。
- **Joshi Tier分级**：按语言资源丰富程度将语言分为T1（极低资源）至T5（高资源）五个层级。
- **LoRA**：Low-Rank Adaptation，低秩适配器，本文用于高效微调LLM权重。
- **SigLIP SO400**：Google开发的视觉编码器，本文作为图像特征提取 backbone。

## 可复现要素
- **数据集**：训练数据基于ShareGPT4V、LLaVA-Next等公开数据集；SMPQA为本文提出，合成OCR数据使用Synthdog代码生成。
- **代码/权重**：论文未明确提及代码开源状态；Centurio模型基于Aya-Expanse和Qwen 2.5训练，需自行复现。
- **关键超参**：LoRA rank=256, α=512；学习率1e-4（LoRA/MLP）或1e-6（图像编码器）；Batch size=32；训练6天（4×H100）。
