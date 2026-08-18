---
title: "Measuring-Social-Biases-in-Masked-Language-Models-by-Proxy-o"
source: https://aclanthology.org/2025.acl-long.68.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:11:35"
field: "语言模型偏见评估与去偏"
keywords: ["social bias", "masked language model", "bias evaluation", "CrowS-Pairs", "StereoSet", "retraining bias", "prediction quality proxy"]
innovations: ["提出注意力加权的 ∆PA/CRRA 代理度量以更高灵敏度检测 MLM 偏见", "设计 BSRT 比较指标量化 MLMO 再训练引入的偏见增量", "证明新度量在对再训练后偏见估计上比 AUL/AULA/CSPS 更准确且更少低估"]
benchmarks: ["CrowS-Pairs (CPS)", "StereoSet (SS)"]
---

# 论文速读：Measuring-Social-Biases-in-Masked-Language-Models-by-Proxy-of-Prediction-Quality

## 一句话总结
本文提出了基于预测质量的代理度量（∆PA 和 CRRA），通过迭代掩码实验评估掩码语言模型（MLM）在 masked language modeling 目标下编码的社会偏见，并证明了这些度量相比现有方法能更准确、敏感地检测偏见——尤其是对重新训练后引入的偏见。

## 研究问题与动机
- 当前主流 MLM（如 BERT、RoBERTa）虽然预训练效果好，但隐式编码了对弱势群体的不利社会偏见，亟需更准确的偏见评估方法。
- 已有评测方法（CSPS、SSS、AUL、AULA）基于伪似然分数或固定掩码子集，无法充分捕捉 MLM 在 iteratively masked token 预测中的上下文感知偏好差异。
- 缺乏系统评估在 MLMO 目标下对模型进行重新训练（retraining）后会如何引入或改变偏见的方法体系。
- 需要一种能直接对应 MLM 核心预训练目标（masked token 预测质量）的度量，并与人类标注偏见对齐。

## 核心贡献（创新点）
- **提出 ∆PA 与 CRRA 两个注意力加权的代理度量**：利用平均多头注意力权重对预测概率差/互补倒数排名进行加权，比未加权的 ∆P/CRR 更能反映 token 重要性。
- **设计模型比较指标 BSRT**：量化同一模型类中重新训练前后偏见相对变化，用于估计因 MLMO 再训练引入的偏见增量。
- **证明新度量在重新训练场景下显著更准确、更敏感**：ΔPA、ΔP、CRRA 在 McNemar 检验下对全部偏见类别均给出显著差异结果，而 AUL/AULA/CSPS 出现系统性低估（偏向下限）。
- **开放度量计算工具包**：将代码打包为 HuggingFace 可集成模块，支持用户在自有或基准数据集上快速计算偏差分数。

## 方法详解
- **迭代掩码实验（IME）**：对每句话逐 token 掩码，用其余 token 作上下文进行 fill-mask 预测，记录 ground-truth token 的预测概率 $P(t_m|s_{\setminus t_m})$ 和排名 $\rho(t_m|s_{\setminus t_m})$。
- **∆P（改进版）**：将原版概率差改为 log 形式
  $$\Delta P(t|s_{\setminus t_m}; \theta) = \log P(t_p|s_{\setminus t_m}; \theta) - \log P(t_m|s_{\setminus t_m}; \theta)$$
  对低概率 token 差异更敏感。
- **CRRA（注意力加权互补倒数排名）**：
  $$\text{CRRA}(t|s_{\setminus t_m};\theta) = a_m(1 - \log \rho(t_m|s_{\setminus t_m};\theta)^{-1})$$
  其中 $a_m$ 为 ground-truth token 在所有多头注意力上的均值。
- **∆PA（注意力加权概率差）**：同理引入 $a_m$ 加权 $\log$ 概率差。
- **句子级聚合**：对句中所有 token 求均值得到 $\text{CRRA}(s)$、$\Delta\text{PA}(s)$、$\Delta P(s)$、$\text{CRR}(s)$。
- **BSPT（预训练模型偏见分数）**：统计 $f(S_{adv}) - f(S_{dis}) > 0$ 的配对比例，值 > 50 表示对弱势群体的偏见更强。
- **BSRT（再训练偏见相对增量）**：比较 $T_1$（再训练后）与 $T_2$（预训练基座）在各句子对上的 $\Delta f$ 差值分布，值 > 50 表示再训练加剧了对弱势群体的偏见。
- **度量集合划分**：$M_1 = \{\text{CRR, CRRA, ∆P, ∆PA}\}$，$M_2 = \{\text{AUL, AULA, CSPS, SSS}\}$，两者的 $\Delta f$ 方向相反（$M_1$ 直接相减，$M_2$ 反序相减）以保证一致的解释语义。

## 实验与结果
- **数据集**：CrowS-Pairs（CPS，9 类偏见，含 race/religion/gender/disability 等）、StereoSet（SS，4 类偏见）。
- **模型**：bert-base-uncased、roberta-base、distilbert-base-uncased、distilroberta-base；每种预训练基座均分别用 $S_{dis}$ 和 $S_{adv}$ 句子集在 PyTorch + P100/T4 GPU 上按 mlm_probability=0.15、30 epochs 重新训练。
- **主要结果**：
  - 所有度量在四种模型上均检测到 > 50 的整体偏见分数，确认 MLB 普遍存在对弱势群体偏见。
  - BSRT 实验表明：$\Delta$PA、$\Delta$P、CRRA、CRR 在全部 4 类模型 × 2 类再训练方向上，对所有偏见类别均给出显著（McNemar p < 0.05）且方向正确的结果；AUL/AULA/CSPS 分别出现 5/6/11 项不显著（$S_{dis}$ 方向）及系统性低估。
  - 误差率对比（Table 4）：CRR/ΔP/ΔPA 达成 100% 准确率，CRRA 达 99%；AUL/AULA/CSPS 仅 93%/88%/94%。
  - 与人类标注对齐（AU-ROC）：在 RoBERTa 和 BERT 上，$\Delta$PA/CRRA 及其变体均优于 AUL/AULA。
- **最强结果**：$\Delta$PA 和 CRRA 在再训练偏见估计上最稳定、最敏感，且与人类标注一致性最高。

## 相关工作脉络
- **CSPS / SSS（Nangia et al., 2020; Nadeem et al., 2021）**：基于伪似然分数，分别以 unmodified 和 modified token 为条件，只依赖单点似然；本文的 ∆PA/CRRA 在 IME 迭代掩码框架下考虑所有 token 的上下文贡献。
- **AUL / AULA（Kaneko & Bollegala, 2022）**：对全未掩码句子的 token 预测似然（带/不带注意力加权）；作者指出其评估对象是下游 embedding bias 而非 MLMO 本身，与本文的“直接反映模型预训练目标”立场不同。
- **Salutari et al. (2023)**：提出 CRR 和 ∆P 作为 fill-mask 场景下的代理；本文在其基础上引入 log 变换与注意力加权形成 ∆PA/CRRA，提升对低概率 token 和对再训练响应的敏感度。
- **WEAT / SEAT（Caliskan et al., 2017; May et al., 2019）**：基于静态 embedding 余弦相似度；本文聚焦于 contextualized MLM，不依赖向量空间分解。
- **RIPA（Ethayarajh et al., 2019）**：用于缓解 WEAT 高估偏差的子空间投影法；本文不讨论去偏，仅提供更敏感的测量基准。

## 局限性与未来方向
- 实验局限于英语 CPS/SS 数据集的二分类配对，未覆盖其他语言或多维非对称偏见。
- 只关注 MLM（掩码预测）这一核心预训练目标，未涉及 next sentence prediction 等其他任务维度的偏见。
- 再训练实验仅考虑在同一模型类内、相同 MLMO 目标下对整束句子的重新训练，未考察混合偏见分布、课程学习、或不同超参的影响。
- 仍依赖 CPS/SS 的人工标注作为对齐基准，未探索无需人工标注的自监督或因果推断式偏差估计。
- 未涉及去偏方法本身的评估，后续需与梯度裁剪、对抗训练、数据重采样等去偏技术联合检验。

## 研究启发与可借鉴点
- **注意力加权思想可直接迁移**：将多头注意力均值作为 token 重要性代理用于任意 token-level 预测分数（如 perplexity、contrastive loss）的加权，可提升对其他预训练目标的敏感性测量。
- **log 变换改进低概率敏感性的设计**：对长尾分布的概率差做 log 化是通用技巧，可复用于模型幻觉探测、罕见 token 预测可靠性评估等场景。
- **BSRT 比较范式适合任何再训练/微调消融**：将同一模型的不同训练阶段以 $\Delta f$ 的分布差形式评估，可用于评测不同去偏数据配比、课程难度、域适应对偏见增减的因果影响。
- **结合团队方向的创新机会**：若团队关注中文 MLM 或低资源语言，可把 CPS/SS 替换为中文社会偏见语料（如性别/地域/职业刻板印象对），并用本文度量体系做跨语言偏见比较；同时可将 BSRT 推广到 LoRA/PEFT 参数-efficient 微调场景下的增量偏见检测。

## 关键术语表
- **IME（Iterative Masking Experiment）**：逐 token 掩码填充的探测实验，用于观察 MLM 在不同掩码位置下对 ground-truth token 的预测偏好。
- **∆PA（Probability Difference with Attention）**：注意力加权的 log 预测概率差，用于衡量模型对弱势/优势群体偏好的相对大小。
- **CRRA（Complementary Reciprocal Rank with Attention）**：注意力加权的互补倒数排名度量，反映模型将 ground-truth token 排在前列的倾向。
- **BSPT（Bias Score for Pretrained Transformer）**：衡量预训练模型整体偏见方向的比例指标，> 50 表示对弱势群体更不友好。
- **BSRT（Bias Score for MLM Re-training）**：衡量同一模型在再训练前后偏见变化的比例指标，> 50 表示再训练加剧了偏见。
- **CSPS（CrowS-Pairs Score）**：基于未修改 token 似然的偏见评估分数，源自 Nangia et al. (2020)。
- **AUL/AULA**：基于全部未掩码 token 预测似然的评估指标，其中 AULA 额外引入注意力权重。
- **Socioeconomic bias**：社会经济地位相关的刻板印象偏见（如贫富对立的语言特征）。

## 可复现要素
- **数据集**：CrowS-Pairs、StereoSet（公开可用）；再训练语料基于 CPS 构建。
- **代码/权重**：论文声明已发布测量包，可与 HuggingFace Transformers 集成；模型权重来自公开版本（HuggingFace library）。
- **关键超参**：mlm_probability=0.15；训练 30 epochs；80/20 train/val 划分；GPU：P100 与 T4；PyTorch 实现。
