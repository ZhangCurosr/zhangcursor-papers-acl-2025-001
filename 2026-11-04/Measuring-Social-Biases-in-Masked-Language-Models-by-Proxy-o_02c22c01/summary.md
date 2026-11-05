---
title: "Measuring-Social-Biases-in-Masked-Language-Models-by-Proxy-o"
source: https://aclanthology.org/2025.acl-long.68.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 17:04:11"
---

# 论文速读：Measuring-Social-Biases-in-Masked-Language-Models-by-Proxy-o

## 一句话总结
本文针对Transformer型MLM中编码的社会偏见，提出基于迭代掩码实验（IME）与注意力加权预测质量的代理指标（ΔPA、CRRA），并引入模型比较函数BSRT量化重新训练后的偏见变化，证明现有伪似然方法在弱势群体偏向数据重训练时会严重低估偏见，而新指标在灵敏度与准确率上全面占优。

## 研究问题与动机
- **核心问题**：预训练MLM（如BERT、RoBERTa）在文本中内嵌了对弱势群体的社会偏见，亟需更准确、更灵敏的评估方法以支撑后续去偏工作。
- **现有方法不足**：主流指标（CSPS、SSS、AUL、AULA）依赖伪对数似然或固定掩码/未掩码子集，隐含“掩码token统计独立”假设，且未紧扣MLM的核心预训练目标（MLMO）。
- **重训练评估缺失**：当前评测多聚焦静态预训练模型，缺乏能灵敏捕捉“模型经特定偏向数据重训后偏见相对变化”的统一度量标准。
- **灵敏度瓶颈**：当模型在弱势群体偏向句子上重训练时，AUL/AULA/CSPS会产生低于50的错误结果，导致对偏见引入的严重低估。

## 核心贡献（创新点）
- **IME+注意力加权代理指标体系**：提出ΔPA与CRRA，通过对数变换与多头注意力权重融合，提升对低概率token偏好差异的捕捉能力；本质区别在于直接针对MLMO目标设计，而非像AUL/AULA那样试图消除掩码token偏差。
- **模型比较指标BSRT**：定义Δf指示函数与BSRT，实现同架构下预训练与重训练模型的相对偏见量化；填补了“重训练引入偏见”的系统性评估空白。
- **实证 superiority**：在CPS与SS上验证，ΔPA、CRRA、改进ΔP与CRR在人类标注对齐（AU-ROC）与重训练误差率（100%/99%）上显著优于CSPS、AUL、AULA（94%/93%/88%），后者在 $S_{dis}$ 重训练下普遍低估偏见。

## 方法详解
- **迭代掩码实验（IME）**：对输入句 $s$ 逐个掩码token，直至覆盖全句或获得 $n$ 个logit预测，获取模型在上下文 $s_{\setminus t_m}$ 下的完整预测分布。
- **基础代理指标**：
  - 改进版 $\Delta P$：采用对数概率差 $\log P(t_p|c)-\log P(t_m|c)$，增强低概率区间的相对差异灵敏度。
  - $\text{CRR}$：互补逆序排名 $1-\rho(t_m|c)^{-1}$，衡量模型选中真实掩码token的倾向。
- **注意力加权扩展**：
  - $\text{CRRA}(w)=a_w(1-\log \rho(t_m|c)^{-1})$，$\Delta\text{PA}(w)=a_w(\log P(t_p|c)-\log P(t_m|c))$，其中 $a_w$ 为ground truth token的平均多头注意力权重。
  - 句子级聚合：$\text{CRR A}(s)=\frac{1}{l_s}\sum_{w\in s}\text{CRRA}(w)$，$\Delta\text{PA}(s)$ 同理。
- **预训练偏见分数 BSPT**（Eq. 9）：统计 $\Delta f_T(i)>0$ 的句对比例，>50表示模型相对更偏好弱势群体偏向句。
- **重训练偏见分数 BSRT**（Eq. 10）：比较 $T_1$（重训）与 $T_2$（预训练）的 $\Delta f$ 大小，>50表示重训练引入了针对弱势群体的额外偏见。

## 实验与结果
- **数据集与模型**：CPS（9类偏见，含race/religion/socioeconomic等）、SS（4类偏见）；模型为 $\mathrm{BERT_{unc}}$、RoBERTa、$\mathrm{distilBERT_{unc}}$、distilRoBERTa。
- **预训练偏见评分（Table 2）**：所有方法均显示 >50 的偏见分数。RoBERTa/distilRoBERTa在多数指标上偏见高于BERT系模型；宗教偏见在所有模型中普遍偏高，种族偏见偏低。
- **人类标注对齐（Section 4.4.1）**：在CPS上以5人标注为ground truth进行二分类AU-ROC评估，$\Delta$PA、CRRA、$\Delta$P、CRR的AUC优于AUL/AULA，对RoBERTa与 $\mathrm{BERT_{unc}}$ 表现一致。
- **重训练实验（Table 3-4）**：在 $S_{dis}$ 上重训练后，$\Delta$P/$\Delta$PA/CRR对所有偏置类别均给出显著 >50 的高分；CSPS/AUL/AULA却出现4~6个 <50 的结果（如 $\mathrm{distilRoBERTa}$ 重训后CSPS在年龄类别仅45.98），严重低估偏见引入。
- **准确率对比**：CRR、$\Delta$P、$\Delta$PA 达 100%，CRRA 达 99%；AUL、AULA、CSPS 分别为 93%、88%、94%。
- **核心结论**：本文指标对重训练引起的相对偏见变化更具统计显著性（McNemar's test p<0.05覆盖全类别），且更符合人类偏见认知。

## 相关工作脉络
- **WEAT/RIPA（静态词向量）**：基于余弦相似度与子空间投影，无法捕捉上下文敏感偏见；本文转向Transformer级句子表征评估。
- **SEAT / Liang et al.（模板句对数似然）**：仅考察代词或单一token的对数似然差；本文利用全句上下文与注意力权重，保留MLMO特性。
- **CSPS (Nangia et al.) / SSS (Nadeem et al.)**：依赖修改/未修改token子集的伪似然；本文指出其独立性假设与选择偏差问题，并证明其在重训练场景下的高误判率。
- **AUL/AULA (Kaneko & Bollegala, 2022)**：要求模型同时预测全句token以“消除掩码偏差”；本文认为这偏离了MLMO核心目标，且对相对偏见变化不敏感。
- **CRR (Salutari et al., 2023)**：提出互补逆序排名；本文在其基础上引入注意力加权与对数变换，显著提升低概率区间的判别力。

## 局限性与未来方向
- 数据集仅限英语与二元句子对，社会偏见维度单一，未来可扩展至多语言、连续谱系偏见及跨文化语料。
- 仅围绕MLMO目标评估，未涵盖NSP、因果语言建模等其他预训练目标下的偏见表现。
- 仅测试四种基础/蒸馏架构，未涉及更大参数规模模型或不同训练语料（如Common Crawl vs WebText）的差异。
- 对齐评估依赖人工标注基准，未来可结合反事实生成或大模型辅助标注降低人力成本。

## 研究启发与可借鉴点
- **通用探针设计**：IME+注意力加权机制可直接迁移至其他基于token预测的语言模型（如T5、GPT-style decoder）的偏好探测任务。
- **前后对比范式**：BSRT的“同架构重训练对比”逻辑适用于评估数据清洗、去偏增强、领域适应等干预措施的真实副作用。
- **工程集成价值**：作者已开源测量包，团队可将其嵌入现有模型评估流水线，快速获得可复现的偏见基线分数。
- **闭环去偏机会**：将CRRA/ΔPA作为反馈信号与对抗训练或梯度裁剪结合，有望构建实时监测偏见漂移的微调框架。

## 关键术语表
- **IME (Iterative Masking Experiment)**：逐词掩码并记录模型预测分布的实验范式，用于探测MLM在完整上下文下的token偏好。
- **CPS (Crowdsourced Stereotype Pairs)**：包含种族、宗教、社会经济等9类社会偏见的双句对比基准，每句对含刻板与反刻板变体。
- **SS (StereoSet)**：评估句子内与句子间刻板印象的benchmark，分为intra-sentence与inter-sentence任务。
- **BSPT / BSRT**：分别用于计算预训练模型绝对偏见比例与重训练前后相对偏见变化的指标函数。
- **CRRA / ΔPA**：引入多头注意力权重与对数变换的新型代理指标，增强对低概率token偏好差异的灵敏度。
- **MLMO (Masked Language Modeling Objective)**：MLM的核心预训练目标，本文所有评估均严格围绕该目标设计。

## 可复现要素
- **数据集**：CPS与SS均为公开基准，重训练数据取自CPS的 $S_{dis}$ 与 $S_{adv}$ 子集。
- **代码**：作者声明已发布专用测量包，支持用户自定义或基准数据集上的BSPT/BSRT计算。
- **模型权重**：通过HuggingFace `Transformers` 库获取（`bert-base-uncased`, `roberta-base`, `distilbert-base-uncased`, `distilroberta-base`）。
- **关键超参**：重训练使用 `mlm_probability=0.15`，80%/20% train/val切分，训练30 epochs；硬件为NVIDIA P100/T4 GPU。

<!--META
{"keywords": ["Social Bias Measurement", "Masked Language Models", "Iterative Masking", "Bias Evaluation Metrics", "Model Retraining", "NLP Fairness"], "field": "大
