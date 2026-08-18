---
title: "Evaluating-Lexical-Proficiency-in-Neural-Language-Models"
source: https://aclanthology.org/2025.acl-long.64.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:51:02"
field: "语言模型词汇能力评估"
keywords: ["lexical proficiency", "reverse dictionary", "definition modeling", "nonce words", "language model evaluation", "Italian NLP", "morphological creativity"]
innovations: ["首个统一框架同时评测RD/DM/EM三任务 across 词典词/新词/临时造词三层级", "引入Optimal Innovation Hypothesis的novelty-adhesion二维人类创造力评估", "构建37万条目级意大利语开放词汇资源支持开放词汇生成式评测"]
benchmarks: ["Dictionary setting (IT5-large RD SBERT=0.73, Acc@10=0.56)", "Neologism setting (IT5-large RD Acc@10=0.27)", "Nonce words human evaluation (IT5-large novelty=3.11, adhesion=3.37)"]
---

# 论文速读：Evaluating-Lexical-Proficiency-in-Neural-Language-Models

## 一句话总结
本文提出了一种统一的评估框架，通过逆向词典（RD）、定义建模（DM）和例句建模（EM）三个任务，系统评测 Transformer 语言模型在词典词、新词（neologisms）和临时造词（nonce words）上的词汇能力与语言学创造力；同时构建了首个面向意大利语的大规模词汇资源数据集。

## 研究问题与动机
- 现有 LM 评估多关注下游任务性能，对词汇能力（lexical proficiency）这一核心语言能力缺乏系统评测，仅 Zheng et al. (2024) 的 Neo-Bench 涉及新词场景，但未覆盖完整词汇生成链条。
- 词汇化/非词汇化词的生成、定义与使用需要复杂的形态理解、语言学创造力与常识推理，是检验 LM 泛化能力的理想试验场——例如根据定义生成"seismophony"要求模型掌握后缀-phony的含义漂移并正确拼接词根。
- 模型规模（small/base/large）与预训练语言类型（单语 vs 多语）如何影响词汇能力，尚缺乏可控对比实验。
- 现有逆词典研究多为封闭词汇表排序方法，缺乏开放词汇生成式评测，难以测度模型对未登录词（OOV）的形态生成能力。

## 核心贡献（创新点）
1. **首个统一框架同时评测 RD/DM/EM 三类词汇任务**：覆盖"生成词→解释词→使用词"的完整词汇能力链条，区别于仅评测单一任务的已有工作。
2. **构建三个层级的意大利语词汇数据集**：词典词（37万+条目）、新词（100条）、临时造词（100条，由 GPT-4o 生成并经人工校验），填补了意大利语开放词汇资源的空白。
3. **引入基于 Optimal Innovation Hypothesis 的人脸创造力评估**：要求被试对 novelty 与 adhesion 两个维度分别打分，使创造力评测量化可比，超越单纯 ROUGE/SBERT 自动指标。
4. **揭示单语大模型在多词汇任务上显著优于多语模型的规律**：MT5-base 仅在含外来借词的新词 RD 任务上表现更强，其余场景全面落后于同规模 IT5-base/large。

## 方法详解
- **任务形式**：采用 text-to-text 多任务微调，所有输入输出均构造为自然语言对：
  - **RD（逆向词典）**：输入 = PoS + labels + definition [+ etymology(以1/5概率附加)]，输出 = 目标词。使用 diverse beam search（beam=100, penalty=0.8）生成100个候选并按概率排序，取 Acc@1/10/100。
  - **DM（定义建模）**：输入 = labels + word + PoS [+ usage example]，输出 = 定义释义。
  - **EM（例句建模）**：输入 = word + PoS + labels + definition，输出 = 用法例句。
- **评估指标**：
  - RD/DM 共用 ROUGE-N（subtoken级）、CER（编辑距离）、SBERT 余弦相似度。
  - 自定义 SBERT 修正分：$m = Z \cdot sim(\mathbf{t}, \mathbf{p})$，其中 $Z = 1 - \max(0, sim(\mathbf{s},\mathbf{t}) - sim(\mathbf{s},\mathbf{p}))$，惩罚偏离源输入的语义漂移预测。
  - EM 使用 Minerva-1B 计算的预测例句中位数 perplexity。
  - Nonce RD 任务由5名意大利母语者进行 Likert 5点打分（novelty + adhesion），报告 Krippendorff's α。
- **模型与训练**：使用 IT5 系列（small 60M / base 220M / large 738M）及 MT5-base（580M），仅在词典+新词数据上微调（15 epoch early stopping，patience=3），label smoothing + Adafactor 优化器，学习率 1e-3/7e-4/1e-4。

## 实验与结果
- **数据集**：Wikizionario（2024年4月dump，370,786条目）+ ONLI新词库（2,986条）作为训练源；测试集共10,517条（见 Table 1）。
- **字典设定（Dictionary setting）**：
  - RD：IT5-large 在所有指标上最优，Acc@10=0.56，SBERT=0.73；IT5-base 次之（SBERT=0.71）；MT5-base 全面落后于同规模 IT5-base。
  - DM：IT5-large 与 IT5-base SBERT均为0.65；EM：IT5-large PPL=112.66 最优。
  - PoS 分析：形容词最容易（Acc@10=0.58），动词最难（0.51）；首字母缩略词生成能力随模型增大急剧提升（IT5-large Acc@10=0.76）。
  - 词频效应：RD 准确率随词频下降呈非线性的中间峰形态（ρ=-0.09）；DM 则与词频正相关（ρ=0.06）；两任务在频率维度呈反向关系。双子词（subtoken）词表现最优峰值，说明模型可利用词组成分性。
- **新词设定（Neologism setting）**：所有指标显著下降（RD Acc@10从0.56→0.16，DM SBERT从0.65→0.53），但模型排序保持一致；MT5-base 在 CER（61.47）上因外来借词优势领先。
- **临时造词设定（Nonce words setting）**：DM SBERT回升（IT5-large=0.58 vs Neo 0.53），EM PPL也改善；GPT-4o 作为上限参考，novelty=3.32、adhesion=3.86，α=0.17/0.07。
- **最强结果**：IT5-large 在字典设定 RD SBERT=0.73、Acc@10=0.56；在临时造词 Human Novelty=3.11、Adhesion=3.37，是本文模型中最接近 Optimal Innovation 理想区（高adhesion、中等novelty）的。

## 相关工作脉络
- **Zheng et al. (2024) Neo-Bench**：仅评估新词场景下的 LLM 表现，任务覆盖定义、翻译等但不含逆向词典与例句生成；本文扩展至三任务+三层级词汇设定并加入创造力人类评估。
- **Xu et al. (2024) / Aljaafari et al. (2024)**：用逆词典任务探测 LLM 概念推理能力，但未涉及新词/临时词扩展；本文验证了词汇能力与下游任务的相关性。
- **Lencione et al. (2022) Nameling**：将新词生成视为 extreme summarization，仅覆盖词生成单一任务；本文提供完整 RD+DM+EM 评测链。
- **Pinter et al. (2020) / Malkin et al. (2021)**：前者研究 BERT 分解 blend 的困难，后者发现 GPT-3 定义临时词有时优于人类；本文在此基础上系统量化多模型创造力维度。
- **Bevilacqua et al. (2020) Generationary**：使用编码器-解码器加例句输入的 DM 方法；本文将其扩展为三项任务的统一框架。
- **Sarti & Nissim (2024) IT5**：本团队使用的底層预训练模型来源，本文验证其词汇能力与下游任务表现的一致性。

## 局限性与未来方向
- 仅使用 T5 系 encoder-decoder 架构，未评测 decoder-only 大模型（如 GPT-4o 仅作人类评估上限参考）及 instruction-tuned 零样本/少样本设置。
- 限定意大利语，跨语言泛化能力未知；框架可扩展至其他语言但尚未验证。
- 人类评估仅5名母语者（≥本科），样本量小且未考察职业/教育水平差异；每个实例评分数不足。
- 未深入分析形态 productive 的具体机制（如 derivational vs compounding 的区别处理）。
- Diverse beam search 的保守性可能低估创造力上限；sampling-based 解码可能产生更有趣候选。
- 未来可引入 partial correlation 分析解耦词频与子词数之间的共线性，精确量化 subtoken 组成性对词汇泛化的贡献。

## 研究启发与可借鉴点
1. **三任务统一评测框架的设计思路可直接迁移**：RD→DM→EM 构成词汇能力"识别-解释-使用"的完整链路，可适配至其他语言及不同 LM 架构，作为标准化的词汇能力探针。
2. **SBERT 修正分 m = Z·sim(t,p) 的设计值得复用**：通过源-目标相似度 Z 惩罚语义漂移，比单纯 ROUGE/SBERT 更能捕捉"贴近源语境的正确定义"，可用于任何 gloss/explanation 生成评测。
3. **Optimal Innovation Hypothesis 的人类评估范式具有跨任务迁移价值**：novelty + adhesion 二维 Likert 评分可推广至 blend 生成、poetry generation 等其他创造性语言任务。
4. **"词频×子词数"交叉分析揭示任务反向相关性**：RD 与 DM 在频率维度呈相反趋势，提示未来工作可设计解耦实验分离频率效应与形态组成性效应。
5. **首字母缩略词生成作为字符级能力的代理指标**：IT5-large 在虚构定义下可生成正确缩写（Table 9），为研究大模型字符知识（Spelling Miracle）提供新评测维度。

## 关键术语表
- **Reverse Dictionary (RD)**：给定词语定义/描述，要求模型生成匹配的目标词汇，属"从义到形"的逆向映射任务。
- **Definition Modeling (DM)**：给定词汇及其词性标签，生成准确释义，属"从形到义"的解释任务。
- **Exemplification Modeling (EM)**：给定词汇+定义，生成符合语境的例句，评估模型的上下文词汇使用能力。
- **Nonce words（临时造词）**：无先例的、一次性使用的语言创造物（H-Creative，Boden 2004），用于测试模型的真实形态生成能力而非记忆。
- **Optimal Innovation Hypothesis**：Giora et al. (2004) 提出，最优创新刺激带来的愉悦感取决于新颖度与可恢复性（adhesion）的平衡。
- **Acc@k**：逆词典任务中，目标词出现在前 k 个候选中的准确率，衡量检索质量。
- **SBERT 修正分**：本文提出的评估指标，通过源-目标相似度假因子 Z 校准预测的语义忠实度。
- **ETYM（词源）**：作为输入辅助信息以1/5概率附加，引导模型理解词素构成与语义来源。

## 可复现要素
- **数据集**：基于 Wikizionario 2024年4月 dump 与 ONLI 新词库自建；GPT-4o 生成临时造词；**论文未声明开源仓库链接，但附录含完整数据处理流程与示例**。
- **代码/权重**：IT5 系列模型可从 HuggingFace 获取（Sarti & Nissim, 2024）；MT5-base 来自 Google；**论文未明确声明自定义代码开源，附录 B 提供完整超参表（Table 8）**。
- **关键超参**：max_input=128 / max_output=64，15 epoch early stopping（patience=3），label smoothing + Adafactor，lr=1e-3/7e-4/1e-4；RD 推理 diverse beam（100 beams, penalty=0.8）；DM/EM 推理 nucleus sampling（top_k=50, top_p=0.9, repetition_penalty=1.3）。
- **硬件**：2× NVIDIA RTX 4090。
- **SBERT 模型**：paraphrase-multilingualmpnet-base-v2（278M）。
