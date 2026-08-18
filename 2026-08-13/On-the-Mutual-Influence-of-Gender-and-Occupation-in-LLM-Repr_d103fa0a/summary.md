---
title: "On-the-Mutual-Influence-of-Gender-and-Occupation-in-LLM-Repr"
source: https://aclanthology.org/2025.acl-long.83.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:13:32"
field: "大语言模型公平性与可解释性"
keywords: ["LLM性别偏见", "性别-职业刻板印象", "嵌入空间分析", "内部表征解释性", "反事实评估", "职业预测偏差"]
innovations: ["提出Internal Coefficient度量内部性别表征与外部偏差行为的中度关联", "系统验证性别方向近似在现代LLM中的有效性并揭示职业语境对性别表征的动态调制", "在1200万条提示规模上建立首名性别表征、现实统计与下游偏见行为的三重关联"]
benchmarks: ["Bias in Bios", "SSA Social Security", "Llama-3.1-8B-Instruct", "Mistral-7B-Instruct-v0.3", "OLMo-7B-0724-hf", "Phi-3.5-mini-instruct"]
---

# 论文速读：On-the-Mutual-Influence-of-Gender-and-Occupation-in-LLM-Repr

## 一句话总结
本文系统研究了大语言模型中**首姓名代表的女性/男性性别表征**如何与职业语境相互影响：一方面，模型对首名性别的内部表征与真实世界性别统计显著相关，并随职业语境发生偏移；另一方面，这些内部表征与下游职业预测任务中的刻板偏见行为存在中度关联，揭示了"内部表征—外部行为"的相互作用机制。

## 研究问题与动机
- **黑盒偏差研究的解释不足**：现有工作大多以黑盒方式检测LLM的首名性别偏见，缺乏对模型内部为何产生此类偏见的机制理解。
- **职业与性别表征的交互尚未系统研究**：社会科学研究表明首名会触发职业刻板印象，但LLM中职业语境如何动态影响首名性别表征尚不清楚。
- **内部表征能否作为偏差检测指标存疑**：尽管已有研究在词向量中发现了性别子空间，但将内部性别表征与下游偏见行为关联的系统评估仍属空白。
- **二元性别框架的局限性**：多数研究仅关注强二元性别关联的首名，忽略了性别模糊名称所揭示的上下文敏感性。

## 核心贡献（创新点）
- **首个针对4个开源LLM的系统性性别方向验证**：将Bolukbasi等的工作适配到现代LLM，通过二进制性别分类任务验证了第一主成分（PC）是有效的性别方向近似，区别于此前仅在Word2Vec等静态嵌入上的验证。
- **发现内部性别表征与现实统计的强相关性**：证明LLM对首名的性别感知与真实世界性别分布呈显著线性相关（Pearson相关），这是首次在三变量间建立系统性相关证据。
- **揭示职业语境对性别表征的动态调制效应**：展示了 feminine-dominated 职业（如nurse，90.9%女性）使首名表征更偏向女性化，masculine-dominated 职业（如comedian，21.1%女性）使其更偏向男性化，且性别模糊名称受影响更大。
- **提出"内部系数（Internal Coefficient）"作为偏差预测指标**：用Spearman相关衡量内部性别表征与occupation logits之间的关联，发现其与外部Bias Coefficient中度相关（Llama-3.1-8B为0.61，Mistral-7B为0.76），为内在-外在偏差度量的一致性提供了新证据。
- **超大规模反事实评估实验设计**：在4个LLM上执行超过1200万条提示（470个首名 × 28个职业 × 270篇传记），为LLM偏见研究树立了规模基准。

## 方法详解
- **性别方向近似（Gender Direction Approximation）**：
  - 选取9对性别对立词对（如 she–he、woman–man，排除Mary/John避免过拟合），在每个LLM中计算其contextualized embedding的平均值。
  - 对配对差值矩阵 $\vec{w} - \vec{c}_j$（其中 $\vec{c}_j = \frac{1}{2}\sum_{\vec{w} \in \mathcal{G}_j}\vec{w}$）执行PCA，取第一主成分 $\vec{g}_{1st}$ 作为性别方向（公式1）。
  - 验证：对10对随机词做相同PCA，方差解释率分布更均匀，确认第一PC确实捕获性别子空间（图2）。

- **上下文首名嵌入的获取**：
  - 从English Wikipedia提取24个首名各10句，通过counterfactual substitution（将原名替换为470个目标名）获取每个名的平均contextualized embedding $\vec{n}_{wiki}$。

- **二分类性别预测验证**：
  - 用 $\vec{n}_{wiki}$ 及其与 $\vec{g}_{1st}$ 的点积 $DOT(\vec{n}_{wiki}, \vec{g}_{1st})$ 作为特征，训练logistic regression和Naive Bayes预测首名性别（50%阈值为二分类标签）。
  - OLMo-7B表现最佳：logistic regression达 **80.57%**（对比原始嵌入76.60%），证明第一PC有效保留了性别信息。

- **职业语境下的性别偏移测量**：
  - 模板：`Question: {NAME} is {ARTICLE} {OCC.}. Is {NAME} male or female? Answer: {NAME} is ...`
  - 使用28个来自Bias in Bios的职业（按女性传记百分比量化性别主导程度），计算提及职业前后首名embedding与性别方向的点积变化，以及female token的softmax概率变化。

- **下游职业预测偏差实验**：
  - 零样本职业预测模板（Prompt 5.1）：给定去标识化传记，让LLM预测职业。
  - 共470个首名 × 28职业 × 270篇传记 = **~1200万条提示**。
  - **Bias Coefficient**：对每个职业，计算首名 femininity（% Female）与TPR的Pearson相关，正值表示女性名在女性主导职业中TPR更高。
  - **Internal Coefficient**：计算 $DOT(\vec{n}_{bios}, \vec{g})$ 与 P(ground-truth occupation) 的Spearman相关，衡量内部表征与输出logits的关联。

## 实验与结果
- **数据集**：
  - Bias in Bios（De-Arteaga et al., 2019）：28个职业、男女各135篇传记。
  - SSA Social Security数据集：470个首名的真实世界性别分布。
  - 美国选民注册数据集（Rosenman et al., 2023）：补充种族/族裔分布。
- **模型**：Llama-3.1-8B-Instruct、Mistral-7B-Instruct-v0.3、OLMo-7B-0724-hf、Phi-3.5-mini-instruct（均在NVIDIA RTX A5000上运行，约2000 GPU小时）。
- **主要结果**：
  - **三变量强相关**：真实世界% Female、模型先验概率 $P_{prior}(Female)$、嵌入点积 $DOT(\vec{n}_{wiki}, \vec{g})$ 三者间呈统计显著的线性相关（图3）。
  - **职业语境偏移**：所有LLM中，feminine职业使同性别桶内首名点积更正（更女性化），masculine职业使其更负；性别模糊名称受影响最大，强二元名称相对稳定。
  - **职业预测偏差**：Llama-3.1-8B在pastor（24.09%女性）上Bias Coefficient为 **-0.55**，在dietitian（92.80%女性）上为 **+0.68**（均 p < 0.001）。
  - **内部-外部关联**：Internal Coefficient与Bias Coefficient的Spearman相关：Llama-3.1-8B为 **0.61**，Mistral-7B为 **0.76**；OLMo-7B为0.86，Phi-3.5-mini为0.90（附录C）。
  - **最强结果**：OLMo-7B的二分类性别预测准确率最高（80.57%），Phi-3.5-mini的内部-外部关联最强（Spearman 0.90）。
- **关键局限信号**：内部系数在某些职业（如nurse、journalist）上未能捕捉外部偏差，在physician、accountant上产生假阳性。

## 相关工作脉络
- **Bolukbasi et al. (2016)**：在Word2Vec静态嵌入中发现性别子空间，本文将其扩展到现代LLM的contextualized嵌入并严格验证。
- **Basta et al. (2019)**：在BERT类嵌入中近似性别方向，本文在其基础上增加了性别模糊名称分析和职业语境交互实验。
- **You et al. (2024)**：研究了gender-neutral名称预测中的偏见，本文扩展了跨职业语境的动态偏移分析。
- **Sancheti et al. (2024)**：研究了性别与种族对浪漫关系预测的影响，本文聚焦职业预测并建立了内部表征—外部行为的关联框架。
- **De-Arteaga et al. (2019) / Bias in Bios**：本文的核心评测数据集和偏差系数定义均来自此工作。
- **Goldfarb-Tarrant et al. (2021)**：指出内在偏差度量与外在应用偏差不一定一致，本文的实验结果部分印证了这一发现。

## 局限性与未来方向
- **二元性别框架不足**：性别方向仅近似female-male轴，无法覆盖非二元性别认同，论文明确承认此局限。
- **人口统计覆盖有限**：首名仅选自美国语境下的4种种族/族裔（White/Black/Hispanic/Asian），且受数据门槛（种族分布>90%，频率>200）限制，性别模糊名称样本较少。
- **模型规模受限**：仅测试~7B参数级别的开源LLM，未探索更大规模模型中偏见趋势是否增强或减弱。
- **内部-外部关联不够稳定**：Internal Coefficient在部分职业上出现假阳/假阴，说明仅凭内部性别表征不足以可靠检测外部偏差。
- **未提出缓解方案**：论文定位为 interpretability 研究，未设计任何debiasing方法。

## 研究启发与可借鉴点
- **投影式性别方向验证流程可迁移**：通过PCA+二分类验证的性别方向提取 pipeline 可直接复用于其他语言/模型，为嵌入空间分析提供标准化方法。
- **反事实替换获取稳定嵌入的策略**：使用固定上下文集+counterfactual substitution平均嵌入，有效控制了上下文噪声，适用于其他属性表征研究。
- **双系数评估框架（Bias Coefficient + Internal Coefficient）**：将外部行为偏差与内部表征关联结合评估，为"内在偏差是否预示外在危害"这一开放问题提供了可量化的分析路径。
- **性别模糊名称的诊断价值**：强二元名称对上下文不敏感，而gender-ambiguous名称更易受职业语境调制——这提示后续工作应优先关注中间分布样本以捕捉微妙的偏见机制。
- **结合大规模提示的统计检验设计**：1200万条提示+Holm-Bonferroni校正的显著性检验，为LLM偏见研究的实验规模设计提供了参考标杆。

## 关键术语表
- **Gender Direction**：通过PCA从性别对立词对嵌入差值中提取的第一主成分向量，近似LLM嵌入空间中的女性-男性轴。
- **Bias Coefficient**：首名 femininity（真实女性比例）与职业预测TPR之间的Pearson相关系数，正值表示模型对符合性别刻板印象的首名给予更高准确率。
- **Internal Coefficient**：内部性别表征（$DOT(\vec{n}_{bios}, \vec{g})$）与目标职业token预测概率之间的Spearman相关系数，用于衡量内部表征对偏差行为的部分解释力。
- **Counterfactual Substitution**：在固定语境句子中将原名替换为目标名，获取可比对的contextualized embedding的方法。
- **True Positive Rate (TPR)**：在职业预测任务中，模型给出正确职业标签的比率，用于衡量不同性别首名在各职业上的预测性能差异。
- **Occupational Context Shift**：职业提及导致首名性别表征沿性别方向发生偏移的现象，体现职业与性别的相互调制。
- **Gender-ambiguous Names**：真实世界中女性/男性比例接近50%的首名，在模型中受职业语境影响最显著。

## 可复现要素
- **数据集**：Bias in Bios（公开）、SSA Social Security数据集（公开）、Rosenman et al. 美国选民注册数据集（公开）。
- **代码/权重**：4个LLM均为Hugging Face开源模型；论文未声明额外代码开源。
- **关键超参**：PCA维度 k=1（除Mistral-7B可考虑k=2）；性别分类阈值50%；Wikipedia上下文24个首名×每名10句；1200万条提示；Holm-Bonferroni多重检验校正。
- **硬件**：NVIDIA RTX A5000，约2000 GPU小时。
