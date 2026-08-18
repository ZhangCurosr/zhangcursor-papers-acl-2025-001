---
title: "Wait-that-s-not-an-option-LLMs-Robustness-with-Incorrect-Mul"
source: https://aclanthology.org/2025.acl-long.75.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:07:40"
field: "大语言模型评测与安全"
keywords: ["Reflective Judgment", "LLM Robustness", "Multiple-Choice Questions", "RLHF", "Instruction Following", "Model Alignment"]
innovations: ["提出并量化\"反思性判断\"概念，区分传统二元拒绝机制", "系统揭示对齐训练对批判性推理能力的潜在损害", "揭示人类标注偏见可通过RLHF数据集传播到模型"]
benchmarks: ["BAD (Basic Addition Dataset)", "MMLU", "MedMCQA", "HH-RLHF"]
---

# 论文速读：Wait-that's-not-an-option-LLMs-Robustness-with-Incorrect-Mul

## 一句话总结
本文首次系统评估大语言模型在面对**所有选项均错误**的多选题时的"反思性判断"能力，发现后训练对齐模型（如GPT-4o、Claude 3 Opus）往往盲从指令选择无效选项，而基础模型（如Llama 3.1-405B、Qwen2.5-32B）在拒绝错误选项方面表现显著更强。

## 研究问题与动机
1. **有用性与批判性推理的张力**：RLHF/DPO等对齐技术旨在提升模型"有用性"，但当用户指令包含错误或有害前提时，过度追求有用性可能导致模型丧失批判性推理能力。
2. **缺乏对无效选项场景的系统评估**：现有工作多关注安全拒绝机制（Abstention Ability）或含"None of the above"选项的MCQ评测，**刻意不含**拒绝选项，才能暴露模型对误导性指令的盲从。
3. **人类反馈数据集的质量隐患**：作者假设人类标注者在RLHF-style数据集中同样存在指令盲从偏见，可能导致错误数据传入模型对齐训练。

## 核心贡献（创新点）
1. **引入"反思性判断（Reflective Judgment）"概念**：区别于传统拒绝机制（二元answer/not answer），反思性判断强调模型对问题本身有效性的批判性评估，能在指令与逻辑矛盾时优先选择后者。
2. **系统揭示训练阶段对反思性判断的影响**：通过对比Qwen2-Math-7B、DeepSeek-Math-7B、Qwen2.5系列在Base/Instruct/RLHF各阶段的性能，发现**指令微调普遍降低反思性判断，而后续对齐可在一定程度上恢复**。
3. **发现规模效应与CoT增强效应**：模型参数量越大（Llama 3.1 8B→405B、Qwen 2.5 7B→32B），反思性判断越强；引入Chain-of-Thought可使反射性判断分数提升超85%。
4. **揭示人类偏见向模型传递的风险**：50人小规模人类实验显示80%参与者在陷阱题上失败；HH-RLHF数据集抽样分析发现约40%的"chosen"答案存在事实错误，暗示人类偏见可通过标注传播到模型。

## 方法详解
- **数据集构建**：
  - **Basic Addition Dataset (BAD)**：三层算术题（Easy: 0-9加法；Medium: 10-99；Hard: 100-999），每层30题，选项均为错误数值。
  - **MMLU子集**：400题，均衡覆盖STEM、人文、社科等八大领域。
  - **MedMCQA**：200题医疗决策问题（麻醉/病理/放射/外科）。
- **三种反射条件（Table 3）**：
  - **Easy**：提示"正确答案可能不在选项中"。
  - **Standard**：无任何额外指导。
  - **Hard**：强制要求"必须从给定选项中选一个"。
- **反射性判断分数（RJ_score）**：
  $$RJ_{score} = \frac{\text{Total reflective actions}}{\text{Total questions}}$$
  其中"reflective action"包括：指出无正确答案、提供未列出的正确解。
- **位置偏差控制**：对每个问题同时评估原始顺序和随机打乱顺序，取平均准确率。
- **对照基线（Baseline）**：每道题含一个正确选项+一个错误选项，测量标准选择题准确率。
- **额外实验设计**：
  - **不合理选项实验**：将数值选项替换为完全无关的名词（如Elephant, Dolphin），测试纯指令跟随与逻辑一致性的区别。
  - **安全性评估**：使用50个明显有害的选项测试反射性判断与安全性能的关联。

## 实验与结果
**BAD数据集表现（Table 1）**：
- 最强：**Llama 3.1-405B** Easy条件100%、**DeepSeekMath-7B RLHF** 全条件100%、**Qwen2-Math-7B Base** 95.5%-100%。
- 最差：GPT-4o Easy仅0.9%、Standard 0%、Claude 3 Sonnet全0%、Qwen2.5-7B-Instruct Easy仅1.8%。

**MMLU数据集表现（Figure 2）**：
- 最强：Gemini 1.5 Pro Easy 97.27%、Llama 3.1-70B Easy 86.36%。
- 最差：GPT-4o/Google Gemini Flash均为0%（Easy）。

**核心发现**：
1. **基础模型 > 对齐模型**：大部分Base模型在算术和MMLU上优于其Instruct/RLHF版本，存在"有用性-批判性权衡"。
2. **规模效应显著**：Llama 3.1系列从8B到405B，Easy条件从0%升至100%；Qwen 2.5从7B到32B同样呈单调上升。
3. **高利害场景未改善**：MedMCQA医疗任务的反射性判断表现与简单算术相似（均较低），说明高 stakes 本身不触发反思。
4. **CoT显著增强**：引入Chain-of-Thought后反射性判断分数提升超85%（Figure 6）。
5. **不合理选项测试**：GPT-4o-mini和Claude 3 Haiku面对随机名词选项仍100%遵循指令选择，而Llama 3.1-405B和Qwen2-Math-7B 100%拒绝。
6. **安全性关联**：反射性判断强的模型（Llama 3.1-405B）在有害选项拒绝上也更强（Hard条件82% vs GPT-4o-mini 60%）。

**人类实验（Section 5.1）**：
- 标准题：平均26.5/27正确。
- 陷阱题：平均仅2.02/3正确，14/50参与者全部失败。
- 结论：超过80%人类在无效选项前倾向遵循指令而非批判性推理。

**HH-RLHF数据质量（Section 5.2）**：
- 经3人独立标注，约**40%**的"chosen"列答案存在事实错误。

## 相关工作脉络
1. **Refusal Mechanisms**（Xu et al., 2024; Cao, 2024）：现有工作聚焦于基于安全/知识边界的二元拒绝（answer/not answer），本文提出更精细的"对问题有效性批判评估"概念。
2. **Multiple-Choice Benchmarks**（MMLU, BIG-Bench）：主流评测假设存在正确答案，本文刻意排除"None of the above"，填补对**无效选项鲁棒性**的评估空白。
3. **Positional Bias**（Pezeshkpour & Hruschka, 2023; Zhang et al., 2024b）：本文控制位置偏差，通过洗牌重复实验确保结果可靠性。
4. **Model Alignment**（RLHF/DPO）：本文直接比较Base/Instruct/RLHF三阶段性能，揭示对齐优化对批判性推理能力的潜在损害，填补对齐副作用研究的空白。
5. **Chain-of-Thought**（Wei et al., 2023）：本文验证CoT对反思性判断的增强效果，同时指出其对小模型可能因容量不足而失效。

## 局限性与未来方向
- **数据集覆盖有限**：BAD仅覆盖加法运算，MMLU/MedMCQA子集未能涵盖LLM可能遇到的全部复杂场景。
- **人类实验样本量小**：仅50名参与者，统计效力不足，无法做细粒度的人口学差异分析。
- **封闭模型的不确定性**：API模型结果可能受供应商更新影响，可复现性受限。
- **未来方向**：①开发更全面的反思性判断评测基准；②探索既能保留有用性又能维持批判性推理的对齐方法；③研究模型架构（如MoE vs Dense）对反思性判断的影响。

## 研究启发与可借鉴点
1. **评测设计值得借鉴**：通过"移除'None of the above'选项"来测试模型真实判断力的思路，可迁移到知识密集型任务（如法律、医学）的鲁棒性评测中。
2. **对齐副作用的系统性研究**：建议团队在后续工作中对Base/Instruct/RLHF不同阶段模型进行对比评测，量化各训练阶段对特定能力的损耗/增益。
3. **人机偏见对比框架**：人类-模型并行评测的方法论（相同任务、相同陷阱设计）可有效验证"人类偏见通过RLHF数据传播"的假设。
4. **不安全选项作为安全测试**：Reflection Judgment与Safety Performance的正相关发现，提示可将反思性判断能力作为补充安全评测指标。
5. **Prompt敏感性分析**：论文展示不同prompt变体对RJ分数的显著影响（Table 4），说明评测设计本身可能成为噪声源，需在后续研究中建立prompt鲁棒性评测协议。

## 关键术语表
**Reflective Judgment（反思性判断）**：模型在面对无效或误导性指令时，能够批判性评估选项有效性并拒绝作答或提供正确答案的能力。
**RLHF（Reinforcement Learning from Human Feedback）**：通过人类偏好数据训练奖励模型，再以此指导策略优化的对齐技术。
**DPO（Direct Preference Optimization）**：无需显式奖励模型，直接从偏好数据优化策略的直接偏好优化方法。
**Chain of Thought（CoT）**：通过引导模型分步推理来增强复杂任务表现的提示技术。
**Positional Bias（位置偏差）**：模型倾向于选择选项中位置靠前（如A）而非内容更优的答案的现象。
**Abstention Ability（拒绝作答能力）**：模型识别自身知识边界或问题超出能力范围时选择不回答的能力。
**Sycophancy（阿谀倾向）**：模型倾向于无条件同意用户观点或偏好，即使与事实相悖的行为。

## 可复现要素
- **数据集**：BAD自建（论文未公开代码链接中提到"anonymous"仓库）、MMLU公开、MedMCQA公开。
- **代码/权重**：评估代码见 https://anonymous.4open.science/r/When-All-Options-Are-Wrong-4C05（论文附录H声明）；开源模型权重在HuggingFace可下载。
- **关键超参**：temperature=0，max_tokens=128，无system prompt（附录A.1）。
- **硬件**：NVIDIA A100 Ampere 40GB。
- **评测时间**：2024年8月。
