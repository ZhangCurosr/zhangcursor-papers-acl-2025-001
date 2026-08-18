---
title: "MultiSocial-Multilingual-Benchmark-of-Machine-Generated-Text"
source: https://aclanthology.org/2025.acl-long.36.pdf
model: agnes-2.5-flash
chunks: 8
summarized_at: "2026-08-18 16:16:25"
---

# 论文速读：MultiSocial-Multilingual-Benchmark-of-Machine-Generated-Text

## 一句话总结
本文提出 MultiSocial，一个覆盖 22 种语言与 5 大主流社交平台的机器生成文本（MGT）多语言评测基准，系统对比统计类、预训练类及微调类检测器在跨语言、跨平台、跨域及分布外场景下的性能边界与泛化规律。

## 研究问题与动机
- 现有 MGT 检测基准多局限于单一语言或正式文体（如新闻、对话），缺乏对真实社交媒体非正式、多语言、跨平台文本的系统性覆盖。
- 不同生成器（闭源 GPT/Gemini vs. 开源 Mistral/Vicuna/OPT）的文本质量与检测难度差异显著，现有评测未能充分刻画这一生成器分布偏移。
- 单语微调检测器在跨语言场景下的泛化能力有限，多语联合微调的收益及其与模型架构（自回归 vs. 非自回归）的交互关系尚未厘清。
- 真实部署中检测器常面临域外数据（如从社交平台迁移至新闻）与未见生成器的挑战，缺乏标准化跨域与 OOD 评测协议。

## 核心贡献（创新点）
- **构建 MultiSocial 多语言多平台数据集**：同步收集 22 种语言的人类社交文本与 7 款 LLM 生成文本，覆盖 Discord/Twitter/WhatsApp/Telegram/Gab 五大平台。
- **建立三层检测器评测体系**：系统性对比统计类、预训练类与微调类检测器，揭示不同类别在跨语言/跨平台场景下的性能分化。
- **揭示架构与微调策略的交互规律**：发现自回归模型多语微调显著优于单语，而非自回归模型因预训练覆盖广，单语/多语微调差异不显著。
- **提供跨域与分布外评测基准**：引入 MULTITuDE 新闻数据集进行域外测试，并评估未见生成器（Llama-2-70b）的泛化能力，填补相关空白。

## 方法详解
- **数据偏见与质量筛查**：使用 multilingual toxicity detector 检测有毒文本（占比约 8%，Twitter 10%、WhatsApp 5%）；结合 social media text topic detector 与 multilingual text genre detector 验证主题与体裁分布均衡性。
- **生成文本质量元评估**：采用 3 位标注员对每组合 10 个样本进行多数投票评分；Gemini（1.83）与 Aya-101（1.73）生成质量最优，OPT-IML-Max-30b（1.23）最差，人类文本均分 1.10。
- **检测器微调协议**：基于 QLoRA 高效微调，`target_modules` 设为 `query_key_value`，`r = 4`，优化器为 AdamW（学习率 `2E-4`），batch size=2，梯度累积=8；固定训练 1 epoch（小样本子集 7 epoch），每 20% epoch 保存检查点并限制总时长 48 小时；类别不平衡采用多数类下采样策略。
- **评估指标与切片维度**：主指标为 AUC ROC，辅指标包括 MacroF1 与 AUC ROC @5%FPR；按生成器、语言、平台、模型架构（自回归/非自回归）进行多维度切片分析。

## 实验与结果
- **数据集与基线**：MultiSocial 主数据集；MULTITuDE 跨域新闻测试集；基线涵盖 IL-ion、D-R、S、Fast-Detect-GPT、Binoculars 等统计类检测器，XLM-RoBERTa-large、mDeBERTa-v3-base 等预训练类检测器，以及 Llama-3
