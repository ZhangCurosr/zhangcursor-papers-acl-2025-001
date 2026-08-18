---
title: "PunchBench-Benchmarking-MLLMs-in-Multimodal-Punchline-Compre"
source: https://aclanthology.org/2025.acl-long.49.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:15:52"
field: "多模态大模型评测与推理"
keywords: ["Multimodal Punchline Comprehension", "PunchBench", "SC-CoQ", "Shortcut Mitigation", "MLLM Evaluation", "Humor and Sarcasm"]
innovations: ["提出PunchBench基准，首次针对多模态笑点理解提供准确且全面的评测", "生成同义/反义Caption以消除模型对文本捷径的依赖", "设计SC-CoQ由简到繁的链式提问策略，超越ICL与CoT"]
benchmarks: ["PunchBench"]
---

# 论文速读：PunchBench-Benchmarking-MLLMs-in-Multimodal-Punchline-Compre

## 一句话总结
本文提出了 **PunchBench**，首个专为准确、全面评估多模态大语言模型（MLLMs）“图文笑点/讽刺理解”能力设计的基准。同时设计了 **SC-CoQ（Simple-to-Complex Chain-of-Question）** 链式提问策略，通过由浅入深的问题排序显著提升模型性能，并揭示了当前 SOTA MLLMs 与人类在复杂幽默理解上仍存在显著差距。

## 研究问题与动机
1. **文本捷径依赖（Shortcut Reliance）**：现有评测中，模型常利用 Caption 中的情感词偏差（如 "enjoy", "plenty of"）或纯文本矛盾即可作答，无需真正融合图文信息，导致评测结果虚高。
2. **问题格式单一**：既往数据集多仅采用单一问答格式（如仅有 Yes/No QA），无法衡量模型在不同用户提问方式下的鲁棒性。
3. **领域覆盖狭窄**：已有工作多聚焦于单一内容类型（如仅漫画或仅帖子）或单一笑点类型（仅讽刺或仅幽默），缺乏对真实多媒体场景的广泛覆盖。
4. **缺乏深层理解评测**：现有基准多停留在“检测是否存在笑点”的浅层任务，缺少要求模型解释“为何构成笑点”的深层推理评测。

## 核心贡献（创新点）
1. **提出 PunchBench 基准**：构建包含 6,000 图文对与 54,000 QA 对的综合性评测集，首次系统评估 MLLMs 在多领域、多格式、多笑点类型下的理解能力。
2. **同义/反义 Caption 扰动机制**：通过 GPT 生成保留或破坏原有幽默语义的 Caption，有效剥离模型对原文本捷径的依赖，迫使模型真正进行图文对齐推理。
3. **设计 SC-CoQ 链式提问策略**：基于问题复杂度递进规律，提出 Intra-task 与 Inter-task 两种链式结构，使模型先掌握简单问题再攻克复杂问题，优于传统 3-shot ICL 与 CoT。
4. **发布全面评测与人类基线**：评测 10 款主流开源/闭源 MLLMs 并引入人工基线，量化揭示当前模型在幽默理解上与人类的显著差距及跨格式性能波动。

## 方法详解
- **数据收集与筛选**：从历史数据集（MTSD, MORE, HUB）及社交媒体/漫画网站（X, Instagram, YouTube, CartoonMovement, CartoonStock）原始采集，经 MLLM 初筛 + 人工校验 + 众包投票（>10票且单选项占比>80%）确定标签。最终保留 6,000 对图文，其中约一半含笑点，并由 3 名人工标注员撰写中文/英文推理句。
- **同义/反义 Caption 生成**：
  - 使用 `gpt-3.5-turbo-0125` 对原 Caption 进行词义替换（synonym）与反义反转（antonym）。
  - 针对存在语义冲突的 Caption（如 "I am so glad today! What a disgusting rainy day!"），先由 GPT 识别并隔离冲突部分，再对两部分分别执行替换/反转，若新 Caption 仍保留笑点则记为同义，若破坏笑点则记为反义。
- **指令构造（两类任务 × 三种格式）**：
  - **Punchline Perception（浅层感知）**：Yes/No QA（判断是否含笑点）、Matching QA（二选一辨别哪个 Caption 配合图像传达笑点）、Multi-option QA（四选一）。
  - **Punchline Reasoning（深层推理）**：Yes/No QA（判断给定的推理句是否解释到位）、Matching QA（二选一选择正确推理句）、Generation QA（要求模型直接生成推理句）。
  - 干扰项（Distractor）由 `gpt-4o` 自动生成，位置顺序随机打乱以避免偏差。
- **SC-CoQ 策略**：
  - **Intra-task**：在同一任务内按复杂度升序串联，如 `<Yes/No QA → Matching QA → Multi-option/Generation QA>`。
  - **Inter-task**：跨任务串联相同格式，如 `<Yes/No QA_m (Perception) → Yes/No QA_n (Reasoning) → Matching QA_m → Matching QA_n>`。
  - 每步提示将前序问题的模型回答作为上下文输入，形成递进式推理链。

## 实验与结果
- **评测对象**：8 款开源 MLLMs（LLaVA, GLM-4V, Qwen2-VL 2B/7B/72B, CogVLM2, LLaVA-OneVision, InternVL2.5, MiniCPM-o 2.6, Aria）+ 2 款闭源（GPT-4V, GPT-4o）+ 人工基线（3名本科外聘标注员）。
- **提示方式**：Zero-shot、3-shot ICL、CoT、SC-CoQ。
- **评估指标**：离散题采用 Accuracy；Generation QA 采用 `gpt-3.5-turbo-0125` 语义匹配二分判定，并与人工 pairwise 评估交叉验证（Gwet's γ > 70%）。
- **核心结果**：
  - **整体差距**：Zero-shot 下所有 MLLMs 准确率均低于 80%，GPT-4o 以 SC-CoQ 达 Perception 80.7% / Reasoning 77.4%，仍大幅落后于人类（98.3% / 96.0%）。
  - **任务难度**：Reasoning 显著难于是 Perception；格式难度排序为 Yes/No QA < Matching QA < Multi-option/Generation QA。
  - **SC-CoQ 优势**：在所有模型、所有格式及所有 Caption 类型（Original / Synonymous / Antonymous）下均显著优于 3-shot 与 CoT（P-value < 0.01）。
  - **捷径消除有效性**：替换为同义/反义 Caption 后模型性能普遍下降，证明原有评测存在捷径依赖；而 SC-CoQ 能在各类 Caption 上恢复并提升性能。

## 相关工作脉络
1. **MTSD (Cai et al., 2019)**：早期 Twitter 多模态讽刺检测数据集，仅含单一分类任务与帖子领域，无深度推理评测。
2. **MORE (Desai et al., 2022)**：聚焦多模态讽刺解释的 benchmark，但仅限讽刺类型、单一 Matching 格式，且未考虑文本捷径干扰。
3. **HUB (Hessel et al., 2023)**：来自《纽约客》配图文库的幽默理解基准，局限于漫画/幽默领域，问题形式单一。
4. **现有幽默/讽刺检测工作 (Qiao et al., 2023 等)**：多为传统多模态小模型架构，缺乏针对现代 MLLMs 的通用性评测框架。
5. **定位差异**：PunchBench 同时覆盖幽默与讽刺、四大内容域、两层理解任务与四种问答格式，并首创同义/反义扰动机制与递进式 SC-CoQ 评测/推理方案，填补了高精度、高全面性评测的空白。

## 局限性与未来方向
- **静态内容局限**：当前基准仅包含静态图文对，未涉及视频等多媒体形式，而真实场景（如喜剧短片、短视频）中的笑点常嵌入动态时序信息。
- **未来方向**：将 PunchBench 扩展至视频模态，评估 MLLMs 在跨帧时序动态与上下文流中的多模态笑点理解能力。

## 研究启发与可借鉴点
1. **扰动去捷径范式**：同义/反义 Caption 生成方法可作为通用“去偏扰动”工具，迁移至视觉推理、情感分析等其他易受文本捷径污染的多模态评测中。
2. **SC-CoQ 结构可复用**：基于“由简到繁”的链式提问设计具有普适性，可适配至其他复杂多模态推理基准（如视觉叙事、因果推理）的测试时增强（Test-time Scaling）。
3. **双层级任务设计**：Perception（检测）→ Reasoning（解释）的分层架构为构建多模态理解评测提供了清晰的方法论模板，便于横向对比模型的浅层感知与深层推理能力。
4. **GPT 辅助干扰项生成+人工质检流水线**：采用大模型批量生成干扰选项/推理句，再结合人工抽样校验，兼顾了数据集构建效率与质量可控性，值得后续基准建设借鉴。

## 关键术语表
- **Multimodal Punchline**：通过图像与文本的强烈反差或巧妙呼应而产生的幽默/讽刺内容，常见于社交媒体图文。
- **PunchBench**：本文提出的首个针对多模态笑点理解能力的综合性评测基准，包含 6,000 图文对与 54,000 QA 对。
- **Synonymous/Antonymous Caption**：经词义替换或反义反转生成的 Caption，前者保留原笑点、后者破坏原笑点，用于剥离文本捷径。
- **Punchline Perception**：浅层理解任务，要求模型判断给定图文对是否包含笑点/讽刺。
- **Punchline Reasoning**：深层理解任务，要求模型生成或选择能够解释该图文对为何构成笑点的推理句。
- **SC-CoQ (Simple-to-Complex Chain-of-Question)**：由作者提出的递进式链式提问策略，按问题复杂度升序串联提示，分为任务内（Intra-task）与跨任务（Inter-task）两种变体。

## 可复现要素
- **数据集**：PunchBench（6,000 image-caption pairs, 54,000 QA pairs），公开地址：https://github.com/OuyangKun10/PunchBench
- **开源状态**：代码与数据集均已开源；测试集未公开 ground truth，另提供含标注的验证集。
- **许可证**：CC BY-NC 4.0
- **关键超参/设置**：解码策略见原文 Table 2（LLaVA/GLM-4V/CogVLM2 采用 Random T=0.7；Qwen2-VL 采用 Top-p p=0.7；其余模型 Greedy）；3-shot ICL 与 CoT 作为默认对比；SC-CoQ 提示模板见 Appendix B。
- **评估工具**：Generation QA 自动评分使用 `gpt-3.5-turbo-0125`，已通过人工 pairwise 与 Gwet's γ 验证一致性。
