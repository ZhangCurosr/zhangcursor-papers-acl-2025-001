---
title: "VReST-Enhancing-Reasoning-in-Large-Vision-Language-Models-th"
source: https://aclanthology.org/2025.acl-long.199.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:07:41"
field: "多模态大模型推理"
keywords: ["Vision-Language Model", "Chain-of-Thought", "Monte Carlo Tree Search", "Self-Reward", "Test-time Scaling", "Multimodal Reasoning"]
innovations: ["免训练MCTS树搜索框架增强LVLM推理深度", "多模态自奖励机制（子问题效用+答案正确性）零样本评估推理轨迹", "验证多模态测试时扩展定律并实现三基准SOTA"]
benchmarks: ["MathVista", "MathVision", "CharXiv"]
---

# 论文速读：VReST-Enhancing-Reasoning-in-Large-Vision-Language-Models-th

## 一句话总结
论文提出 **VReST**，一种免训练的蒙特卡洛树搜索（MCTS）与自奖励机制相结合的方法，用于系统化探索大型视觉语言模型（LVLM）的推理空间；在 MathVista、MathVision 和 CharXiv 三个多模态数学推理基准上均取得 SOTA 性能，并验证了多模态任务中的测试时扩展定律。

## 研究问题与动机
1. **LVLM 复杂视觉推理能力受限**：现有 CoT 提示方法在 LVLM 中生成的中间推理步骤有限，且缺乏对推理轨迹的评估与细化能力，导致在复杂视觉数学任务上甚至不如直接问答（Direct QA）。
2. **训练扩展成本高**：构建大规模 LVLM 推理数据集并进行训练的方式昂贵且难以扩展，需探索免训练（training-free）的推理增强路径。
3. **树搜索在多模态领域应用空白**：MCTS 等树搜索方法已在 LLM 推理中证明有效，但尚未被系统地引入多模态 CoT 推理场景，且缺乏适配视觉-文本联合评估的奖励机制。

## 核心贡献（创新点）
1. **免训练 MCTS 推理框架**：将蒙特卡洛树搜索引入 LVLM，通过 Selection-Expansion-Rewarding-Backpropagation 迭代系统化探索推理空间，与贪婪式 CoT 方法形成本质区别。
2. **多模态自奖励机制（Self-Reward）**：仅依赖 LVLM 自身，无需额外 reward model，通过子问题效用（R₁）与最终答案正确性（R₂）的几何均值评估每个推理步骤的质量，融入了视觉线索判断。
3. **多策略最终轨迹选择**：提出 Greedy Trace、Best Trace 与 Trace Vote 三种选优策略，其中 Vote 聚合多个高奖励轨迹，显著提升鲁棒性。
4. **SOTA 性能与测试时扩展验证**：在三个基准上全面超越现有 prompting 方法，并证明性能随 MCTS 迭代次数增加而持续提升，呈现更优的测试时扩展定律。

## 方法详解
- **问题形式化**：给定问题 $Q$ 与图像 $I$，目标寻找最优推理轨迹 $\mathcal{P}^* = \{Q, S_1, S_2, ..., S_n\}$，其中每步 $S_i = (Q_i, A_i)$。
- **MCTS 四阶段迭代**（共 $K$ 轮）：
  1. **Selection**：基于 UCT 公式 $UCT(v) = R(v) + c\sqrt{\frac{\ln N(p(v))}{N(v)}}$ 递归选择叶节点，平衡探索与利用。
  2. **Expansion**：以当前轨迹为 prompt，提高温度参数生成 $w$ 个候选推理步骤 $\{S_{t,j}\}$，并选用最高奖励节点继续扩展，直至深度 $D_{\max}$ 或到达终止节点（包含 "Now we can answer the question"）。
  3. **Rewarding**：对新生成步骤 $S_t$，构造 prompt $\mathcal{P}_t = [Q, S_1, ..., S_t]$，分别计算：
     - $R_1 = P(\text{"Yes"} | [\mathcal{P}_t, \mathcal{P}_Q], I)$，其中 $\mathcal{P}_Q$ 为 "Are questions $Q_1,...,Q_t$ useful?"
     - $R_2 = P(\text{"Yes"} | [\mathcal{P}_t, \mathcal{P}_A], I)$，其中 $\mathcal{P}_A$ 为 "Is the answer $A_t$ correct?"
     - 最终奖励 $R = \sqrt{R_1 R_2}$。
  4. **Backpropagation**：沿路径更新节点统计量：$R(S_t) = \text{Avg}(\{R(S_i)\}_{i=t}^T)$，$N(S_t) = N(S_t) + 1$。
- **最终轨迹选择**：
  - **Greedy Trace**：每步选最高奖励节点。
  - **Best Trace**：计算整条轨迹平均奖励 $R(\mathcal{P}) = \text{Avg}(\{R(S_t)|S_t \in \mathcal{P}\})$，选最大值。
  - **Trace Vote**：选 $n$ 条高奖励轨迹， majority vote 决定最终答案。

## 实验与结果
- **数据集**：MathVista（testmini，1000 样本）、MathVision（testmini，304 样本）、CharXiv（validation，1000 样本），均以准确率评估。
- **模型**：主实验使用 Qwen2-VL-7B-Instruct；消融与小模型实验使用 Qwen2.5-VL-3B-Instruct。
- **基线**：QA、CoT、CoT-Vote、Best-of-N、Cantor、ToT。
- **主要结果**：
  - **MathVista**：VReST 整体准确率 **64.50%**，VReST-Vote 达 **65.40%**；在 MWP（72.04%→75.81%）、SCI（67.21%→68.03%）、STA（75.75%→77.74%）等子任务上显著提升。
  - **MathVision**：VReST 整体 **26.64%**，VReST-Vote 达 **28.29%**；在 GrphT（42.11%）、Topo（52.63%）等几何/拓扑任务上领先。
  - **CharXiv**：VReST 整体 **33.10%**，VReST-Vote 达 **38.10%**；在 Text in General（54.55%→61.62%）、Num in Chart（33.62%→39.22%）等图表理解任务表现突出。
- **最强结果**：VReST-Vote 在 CharXiv 上提升幅度最大（+8.60% vs ToT 的 32.10%）。

## 相关工作脉络
1. **CoT for LVLMs**（Zhang et al., 2023; Mitra et al., 2024; Shao et al., 2024; Zheng et al., 2023; Gao et al., 2024）：现有方法多为两阶段或分解子问题的线性 CoT，缺乏对推理轨迹的全局评估与回溯优化；VReST 引入树搜索与奖励反馈实现动态 refinement。
2. **Tree-based Reasoning with LLMs**（Wang et al., 2022; Lightman et al., 2023; Yao et al., 2024; Hao et al., 2023）：Self-Consistency、Best-of-N、ToT、MCTS 已在纯文本 LLM 中验证；本文首次将 MCTS 适配至多模态场景，并设计视觉感知的自奖励机制。
3. **Process Reward Model（PRM）**（Lightman et al., 2023）：传统 PRM 需额外训练；本文 Self-Reward 完全零样本调用 LVLM 自身完成中间步骤评估，保持 training-free。
4. **Multimodal Mathematical Reasoning Benchmarks**（Lu et al., 2023; Wang et al., 2024a; Wang et al., 2024c）：MathVista、MathVision、CharXiv 为本文评测基准，覆盖几何、代数、图表理解等多类视觉数学推理。
5. **Test-time Scaling**（OpenAI, 2024; Zhang et al., 2024b）：本文验证 MCTS 迭代次数增加可持续提升性能，拓展了测试时扩展定律至多模态领域。

## 局限性与未来方向
1. **自奖励机制的偏差传播风险**：依赖 LVLM 自身判断，模型固有偏见或错误可能在奖励过程中放大；未来可引入独立训练的 reward model 辅助评估。
2. **计算开销较高**：MCTS 多轮迭代与树扩展带来显著推理延迟（单样本平均耗时约 108–157 秒，见原文 Table 5）；可通过剪枝或早停策略优化。
3. **模型依赖性**：仅在 Qwen2-VL-7B-Instruct 上验证，未测试不同架构、规模或训练策略的 LVLM，泛化性待进一步评估。
4. **数据集覆盖有限**：聚焦数学推理，未探索图表分析、科学图示理解等其他多模态复杂推理任务。

## 研究启发与可借鉴点
1. **免训练树搜索范式可迁移**：VReST 证明无需微调即可通过搜索+自评估提升 LVLM 推理深度，该思路可复用于其他多模态任务（如科学图表解读、多跳视觉问答）。
2. **多模态自奖励设计**：将视觉-文本联合 prompt 融入奖励计算（$R_1$ 与 $R_2$ 均 conditioned on $I$），为多模态轨迹评估提供了轻量且可解释的替代方案。
3. **测试时扩展定律验证方法**：通过控制迭代次数/采样宽度绘制性能曲线，可有效对比不同推理方法的扩展效率，建议作为后续 work 的标准评估流程。
4. **Trace Vote 策略的工程价值**：在 Best Trace 基础上引入多数投票，以小幅额外计算换取鲁棒性提升，适合对可靠性要求高的部署场景。

## 关键术语表
- **LVLM（Large Vision-Language Model）**：融合视觉与语言理解的超大参数多模态模型，如 Qwen2-VL、LLaVA 等。
- **MCTS（Monte Carlo Tree Search）**：通过随机模拟与回溯更新节点价值，在搜索空间中高效寻找最优路径的算法。
- **CoT（Chain-of-Thought）**：引导模型逐步推导的提示技术，将复杂问题分解为中间推理步骤。
- **Self-Reward**：无需外部奖励模型，由生成模型自身对推理轨迹进行质量评估的机制。
- **UCT（Upper Confidence Bound applied to Trees）**：MCTS 中平衡探索与利用的节点选择策略公式。
- **Test-time Scaling Law**：推理阶段通过增加计算预算（如更多采样或迭代）持续提升模型性能的经验规律。
- **Terminal Node**：推理树中包含 "Now we can answer the question" 的节点，表示推理结束。
- **Trace Vote**：聚合多条高奖励推理轨迹的最终答案，通过多数投票确定输出。

## 可复现要素
- **代码**：已开源，链接 https://github.com/GaryJiajia/VReST
- **数据集**：MathVista、MathVision、CharXiv 均为公开基准，测试集为 testmini/validation。
- **模型权重**：使用 Qwen2-VL-7B-Instruct 与 Qwen2.5-7B-Instruct（开源权重）。
- **关键超参**：$D_{\max} = 8$，$K = 10$，$w = 5$，$c = 1$，temperature = 0.7，top_p = 0.95。
- **Prompt 模板**：论文附录 G 提供了 Reasoning Step Generation、R1/R2 Rewarding、Answer Evaluation 的完整 zero-shot prompt。
