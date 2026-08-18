---
title: "Tree-of-Evolution-Tree-Structured-Instruction-Evolution-for"
source: https://aclanthology.org/2025.acl-long.14.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:24:46"
---

# 论文速读：Tree-of-Evolution-Tree-Structured-Instruction-Evolution-for

## 一句话总结
本文提出 Tree-of-Evolution (ToE) 框架，将代码指令合成过程建模为树结构以并行探索多条进化路径，并结合“挑战性+多样性”质量评分与 Beam Search 优化驱动迭代，仅用 75k 合成数据微调开源基座模型，即可在五大代码生成基准上与甚至超越使用百万级数据训练的 SOTA 开源 Code LLM（如 Qwen2.5-Coder-Instruct）。

## 研究问题与动机
- 代码指令数据人工标注成本极高，需借助 LLM 自动化合成，但现有主流方法（Code Evol-Instruct、OSS-Instruct）依赖单向扩展或随机采样，难以系统性地保障数据质量。
- 单向合成（Unidirectional synthesis）仅沿单条轨迹扩展种子，未充分探索多元进化方向，导致生成数据多样性不足，制约模型泛化能力。
- 随机驱动生成（Randomness-driven generation）缺乏质量控制闭环，低质量样本会显著拉低指令微调效果（LIMA 已明确指出数据质量优于数据规模）。
- 现有范式难以在“任务难度（Challenge）”与“样本差异度（Diversity）”之间建立可量化、可优化的联合评估标准。

## 核心贡献（创新点）
1. 提出 Tree-of-Evolution (ToE) 框架，将代码指令合成建模为树结构，从同一代码种子出发并行探索多条进化路径，打破传统单向生成的局限。
2. 设计基于挑战性与多样性的质量评估函数 $V(s) = V_{comp}(s) + V_{div}(s)$，利用 LLM-as-a-Judge 量化复杂度，并通过跨树 embedding 余弦距离保障全局多样性。
3. 引入优化驱动进化（Optimization-driven evolution）机制，结合 Beam Search 保留 top-n 节点，并将父节点评分拆解为显式优化目标反馈给生成模型，确保子节点质量严格优于父节点。
4. 实验表明仅用 75k 合成指令微调 1.5B~14B 模型，即可在 HumanEval、MBPP、EvalPlus、LiveCodeBench 和 BigCodeBench 上与 SOTA 开源 Code LLM 持平或超越，且开放权重模型替代商业 API 后效果更优。

## 方法详解
- **树结构合成（Tree-Based Synthesis）：** 以随机 GitHub 代码片段为根节点 $s_0$，构建有向树 $T=(S,E)$。每轮演化 $r$ 中，合成模型 $G(p_\theta, s_r, k)$ 为每个活跃节点生成 $k$ 个候选指令作为子节点，直至达到最大深度 $d_{max}$ 或质量不达标。
- **质量评估函数（Quality Evaluation）：** 
  - 挑战性得分 $V_{comp}(s) = C(p_\theta, s)$，由 LLM 评估代理按 1-10 分制打分，反映指令推理复杂度。
  - 多样性得分 $V_{div}(s) = \min_{s_e \in DB} D(s, s_e)$，其中 DB 为当前轮次其他树中选出的最具挑战性指令集合，$D$ 为基于 `gte-large-en-v1.5` 的余弦距离；排除同树节点以避免同源相似性扭曲评估。
  - 总分 $V(s) = V_{comp}(s) + V_{div}(s)$，两者等权相加。
- **优化驱动进化（Optimization-Driven Evolution）：** 采用 Beam Search 策略，每轮保留 top-n 节点 $S_{next} = \{s \in S_{active} \mid V(s) \geq V_{(n)}\}$。同时将父节点的 $V_{comp}$ 与 $V_{div}$ 作为 prompt 中的显式目标传入，要求生成的子节点满足 $V(s_{child}) \geq V(s_{parent})$，不满足则剪枝终止该分支，形成“生成-评估-递进优化”闭环。
- **数据收集与训练（Data Collection & Training）：** 按轮次对节点按质量排序筛选，计算同轮同树内候选节点与已选节点的相似度，仅保留距离大于阈值（6.0）且质量高于父节点的节点，最终截断至 75k 样本。使用因果语言建模损失（mask instruction tokens）在 Qwen2.5-Coder-Base (1.5B/7B/14B) 上进行 SFT。

## 实验与结果
- **数据集与基线：** 5k 初始种子取自 Stack v1 数据集；主合成模型为 `gpt-4o-2024-08-06`；基线涵盖 10 个开源 Code LLM 与 5 个商业闭源模型；评测基准包括 HumanEval (HE)、MBPP、EvalPlus (HE/MBPP-Plus)、LiveCodeBench、BigCodeBench。
- **主要结果：** 
  - 1.5B 模型在 HE 上达到 75.0 pass@1，较 Qwen2.5-Coder-Instruct (1.5B) 提升 4.3 分。
  - 14B 模型在 MBPP-Plus 上达到 74.9，较基线提升 2.0 分，并超越 GPT-4o 在 MBPP/MBPP-Plus 上的表现。
  - LiveCodeBench (n=10, temp=0.2) 与 BigCodeBench Full/Hard 上，1.5B/7B/14B 均持平或超越 Qwen2.5-Coder-Instruct，14B 模型在 BigCodeBench Hard 上与 o1-mini 相当。
- **关键结论：** 仅用 75k 高质量合成数据即可达到甚至超越使用百万级数据训练的 SOTA 开源 Code LLM，验证了优化驱动进化在数据质量把控上的显著优势。

## 相关工作脉络
- **Code Evol-Instruct (WizardCoder)：** 启发式提示迭代生成更复杂指令，但为单向
