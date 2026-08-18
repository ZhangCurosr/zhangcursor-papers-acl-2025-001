---
title: "A-Survey-of-Post-Training-Scaling-in-Large-Language-Models"
source: https://aclanthology.org/2025.acl-long.140.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:49:17"
field: "大语言模型后训练与对齐"
keywords: ["后训练扩展", "监督微调", "强化学习", "测试时计算", "大语言模型", "Scaling Law", "RLHF", "合成数据"]
innovations: ["系统提出SFT/RL/TTC三分法后训练扩展框架", "聚焦各类方法的可扩展变体而非传统方法", "识别六大约束并指向未来研究方向"]
benchmarks: ["RewardBench", "MMLU", "HumanEval", "AIME", "GAIA"]
---

# 论文速读：A-Survey-of-Post-Training-Scaling-in-Large-Language-Models

## 一句话总结
本文系统综述了大语言模型（LLM）的**后训练扩展**（Post-Training Scaling）新范式，将其划分为监督微调（SFT）、反馈强化学习（RLxF）和测试时计算（TTC）三大类方法，旨在应对预训练数据枯竭与计算成本攀升的挑战，探索通过扩展后训练阶段而非继续扩大预训练规模来提升模型能力的新路径。

## 研究问题与动机
1. **预训练扩展面临瓶颈**：高质量互联网语料趋于耗尽（Villalobos et al.），合成数据的有效性和多样性存疑，且预训练计算成本极高，边际收益递减。
2. **后训练阶段计算占比过低**：传统后训练（对齐阶段）仅占总训练计算的不到1%，存在巨大的可扩展潜力未被充分挖掘。
3. **缺乏系统性分类与综述**：尽管OpenAI o1等模型展示了后训练扩展的潜力，但相关方法（SFT、RL、TTC）分散发展，亟需统一框架进行梳理与比较。
4. **AGI需要新的扩展定律**：仅靠预训练阶段的scaling law已不足以推动模型向通用人工智能（AGI）迈进，需要建立后训练阶段的scaling law。

## 核心贡献（创新点）
1. **提出后训练扩展的三分法框架**：将后训练扩展系统划分为SFT、RLxF和TTC三类，每类进一步区分"传统方法"与"可扩展方法"，构建了清晰的研究谱系（Figure 2）。
2. **深入剖析各类方法的可扩展性机制**：不仅综述方法本身，更聚焦于其可扩展变体（如Evol-Instruct、RLAIF、Tree-of-Thought等），揭示数据规模、反馈信号和推理计算如何驱动性能提升。
3. **识别应用场景与未解问题**：梳理了数学、代码生成、自主Agent三大潜力应用领域，并系统讨论理论基石、合成数据、持续学习、主动探索、超对齐、评估基准等六大局限与未来方向。
4. **提供方法论对比与取舍指南**：针对每类方法指出其本质权衡（如合成奖励可能存在偏差、环境反馈构建成本高、自反馈受限于模型初始能力），为研究者选择合适路径提供参考。

## 方法详解

### 一、监督微调（SFT）扩展

**数据获取视角分类**：

1. **指令生成（Instruction Generation）**
   - **基于上下文的方法**：从外部知识库检索并构造指令，如Web-Instruct（从网页数据库检索文档）、Backtranslation（利用ClueWeb语料迭代微调种子模型）、Ditto（基于角色知识模拟对话）、SOLID（多意图指令自动生成）。
   - **进化式方法**：模仿自然选择迭代优化指令，如Evol-Instruct（深度/广度进化策略生成不同难度指令）、Promptbreeder（基于突变提示的遗传算法）、DiverseEvol（K-Center采样增强多样性）、Self-instruct（从种子数据集自主生成并验证新指令）。

2. **响应生成（Response Generation）**
   - **采样策略**：RFT（拒绝采样过滤推理路径）、RAFT（奖励排序微调迭代生成最优响应）、LMSI（CoT提示+多数投票）、STaR/Quiet-STaR（采样推理过程，Quiet-STaR引入并行采样）。
   - **自博弈（Self-Play）**：SPIN（无标注数据自博弈）、AMIE（医疗领域内外自博弈）、Self-Talk（角色模拟生成对话）、Sotopia-π（社会场景生成）。
   - **自我精炼（Self-Refinement）**：SCORE（用小模型基于正确解反馈进行自我修正）、SELF（元技能学习框架）、SELF-ALIGN（基于原则与示范的持续精炼）、ISR-LLM（LLM作为自验证器迭代优化计划）。
   - **弱监督**：利用弱模型输出训练强模型，关注强-弱泛化（PGR指标）、计算效率权衡。

### 二、强化学习（RLxF）扩展

**按反馈信号来源分类**：

1. **传统方法**
   - **人工标注**：DPO（单阶段策略优化最大化奖励）、KTO（人类感知损失函数无需偏好数据）、RRHF（基于排名的对齐）、SimPO（长度归一化奖励+目标奖励差值）。
   - **奖励建模**：RLHF（InstructGPT的PPO框架）、ReMax（REINFORCE算法更高效）、ORM/PRM（数学任务步骤监督）、WARM（多奖励模型加权平均缓解reward hacking）、UNA（统一RLHF为监督学习问题）。

2. **可扩展方法**
   - **合成奖励建模（Synthetic Reward）**：RLAIF（基于宪法的AI反馈训练偏好模型）、RLSF（利用LLM响应质量差异训练奖励模型）、Q*（历史/未来奖励优化状态优先）、IterAlign（红队对抗+宪法发现）、Easy-to-Hard Generator Generalization。
   - **环境反馈（Environment Feedback）**：DigiRL（GUI学习并行环境）、ENVISIONS（LLM生成轨迹与仿真交互）、SANDBOX/RLTF（交互反馈+单元测试）、SPAG/Prover-Verifier Games（语言规则环境对抗训练）。
   - **自反馈（Self-Feedback）**：RLCD（对比蒸馏）、Agent Q（MCTS引导）、Self-Rewarding（自我评估）、Meta-Rewarding（模型评估自身评估）、Self-Taught Evaluator（迭代修改输入提示）。

### 三、测试时计算（TTC）扩展

**推理阶段增加计算以提升性能**：

1. **采样（Sampling）**：Self-Consistency（多推理路径多数投票）、验证引导的加权多数投票、奖励模型选择对齐响应。
2. **验证链式思维（Verified CoT）**：PRM（验证每步推理）、SelfCheck（步骤级自检与重生成）、DiVeRSe（多样提示+加权投票）、Logi-CoT（集成符号逻辑验证）。
3. **搜索（Searching）**：Tree-of-Thought（ToT，回溯多路径）、Pathfinder/Cumulative Reasoning/Graph of Thoughts/Diagram of Thought（多样化搜索结构）、Algorithm of Thoughts（优化ToT效率）、MCTS（蒙特卡洛树搜索）。
4. **长上下文学习（Long ICL）**：Many-Shot ICL（自动构造多达2048个示例）、束搜索优化长提示。
5. **自验证（Self-Verification）**：Self-Refine（迭代自我反馈）、Reflexion（环境反馈减少幻觉）、自检查范式、Self-Verification（逆向验证打分）、Self-Contrast（生成对比方案避免偏差）。

## 实验与结果

本文为**综述论文**，未开展独立实验，而是系统总结已有工作的实验结果与结论：

- **SFT扩展**：Web-Instruct、Evol-Instruct等方法已通过百万级指令数据验证了可扩展性；Self-instruct、RAFT等方法证明采样与自我精炼可显著提升复杂推理任务表现。
- **RL扩展**：RLAIF、Self-Rewarding等工作表明合成/自反馈信号可替代大量人工标注；Math-Shepherd、DeepSeek-Math在数学推理任务上验证了PRM/PPO扩展的有效性。
- **TTC扩展**：Self-Consistency、Tree-of-Thought、MCTS等方法在数学与代码任务上展示测试时计算与性能的正相关关系；Snell et al. (2024) 提出"compute-optimal"策略证明扩展推理计算可优于扩展模型参数。
- **最强结果参考**：OpenAI o1模型通过扩展TTC（推理时计算）在复杂推理任务上达到接近人类水平；DeepSeek-R1等模型在数学和代码任务上通过RL扩展显著超越基线。

## 相关工作脉络

1. **Pre-training Scaling Law**（Brown, 2020; Hoffmann et al., 2022）：本文与其本质区别在于研究对象从"预训练阶段"转向"后训练阶段"，关注对齐与推理能力而非纯语言建模能力。
2. **InstructGPT/RLHF**（Ouyang et al., 2022）：经典后训练方法，但计算占比小、依赖大量人工标注；本文聚焦其可扩展变体（如RLAIF、DPO系列）如何解决规模化问题。
3. **Self-Instruct/Evol-Instruct**（Wang et al., 2023; Xu et al., 2023）：SFT数据构建的代表性工作，本文将其纳入更系统的分类框架，并扩展至Web-Instruct、Backtranslation等新方法。
4. **Chain-of-Thought/Tree-of-Thought**（Wei et al., 2023; Yao et al., 2023）：TTC的经典方法，本文进一步纳入Self-Consistency、MCTS、Many-Shot ICL等最新扩展技术，形成完整的测试时计算谱系。
5. **Synthetic Data & Model Collapse**（Shumailov et al., 2024）：本文在局限部分明确指出合成数据的双面性，与预训练合成数据研究形成呼应与对比。
6. **Weak-to-Strong Generalization**（Burns et al., 2023）：本文在SFT和RL部分多次引用此概念，讨论如何用弱监督/弱反馈实现强模型对齐，是后训练扩展的核心挑战之一。

## 局限性与未来方向

1. **理论基石薄弱**：现有方法高度依赖经验，虽有少量理论分析（如TTC增强Transformer序列计算能力、RLHF缓解分布偏移），但缺乏统一的后训练扩展理论框架。
2. **合成数据风险**：递归生成的合成数据可能导致模型坍缩（model collapse），且无干预的数据多样性难以超越种子集分布。
3. **持续学习挑战**：后训练需从动态环境持续收集数据，同时避免灾难性遗忘和对齐税（alignment tax）。
4. **主动探索不足**：当前方法依赖人工或模型增强数据，模型主动发现自身薄弱环节并生成针对性训练样本的能力有限，自评估偏差问题突出。
5. **超对齐（Superalignment）难题**：基础模型能力极强后，如何有效后训练以充分利用其能力（弱到强泛化），以及确保真正安全而非模拟安全的后训练方法尚待探索。
6. **评估基准局限**：静态基准存在数据泄露和排行榜饱和问题，需开发动态自动化排行榜和新评估指标。

## 研究启发与可借鉴点

1. **三分法框架可迁移**：SFT/RL/TTC的分类逻辑可用于其他模态（如多模态、视频生成）的后训练扩展研究，提供清晰的方法论地图。
2. **合成数据+验证机制**：Web-Instruct、Evol-Instruct等"生成-验证-筛选"范式值得借鉴，尤其结合PRM/自我验证的反馈闭环设计。
3. **测试时计算可扩展性**：Snell et al.的compute-optimal思想启示我们，在参数扩展受限时，通过扩展推理计算（如MCTS、Self-Consistency）是性价比更高的路径。
4. **弱监督/自反馈的潜力**：Self-Rewarding、RLCD等方法证明模型可自我改进，为低资源场景下的高质量训练数据构建提供新思路。
5. **领域适配的通用模式**：数学（PRM+PPO）、代码（执行反馈+RL）、Agent（环境交互+Hierarchical RL）三领域的成功模式可作为其他垂直领域（如医疗、法律）后训练的参考模板。

## 关键术语表

**Post-Training Scaling（后训练扩展）**：在预训练之后，通过对齐、推理增强等阶段扩展计算与数据规模以提升LLM性能的范式。

**SFT（Supervised Fine-tuning，监督微调）**：使用标注指令-响应数据对预训练模型进行微调，使其学会遵循人类指令的训练技术。

**RLxF（Reinforcement Learning from Feedback）**：利用环境、人工或AI反馈信号通过强化学习对齐模型行为的方法统称，包括RLHF、RLAIF等。

**TTC（Test-time Compute，测试时计算）**：在推理阶段增加计算资源（如采样、搜索、验证）以提升模型性能的技术，而非修改模型参数。

**Evol-Instruct**：通过深度和广度进化策略自动生成分层难度指令数据的方法，实现SFT数据的规模化构建。

**RLAIF（Reinforcement Learning from AI Feedback）**：使用AI生成的反馈（如基于宪法原则）替代人工反馈进行强化学习对齐的方法。

**PRM（Process Reward Model，过程奖励模型）**：对推理过程的每个步骤进行评分的奖励模型，用于指导LLM的逐步推理优化。

**Self-Consistency（自一致性）**：通过生成多个推理路径并取多数投票结果来提升LLM推理准确性的测试时计算技术。

## 可复现要素

- **数据集**：论文未提出新数据集，引用的数据集包括SQuAD、HellaSwag、DROP、FLAN、ClueWeb、OpenAssistant等（多为公开数据集，具体开源状态需查阅原论文）。
- **代码/权重**：论文未开源代码或模型权重，作为综述论文不提供可复现实现；引用工作的代码开源状态各异（如Self-Instruct、Evol-Instruct、RLAIF等均有开源实现）。
- **关键超参**：论文未设定统一超参，各子方法超参依原始论文而定（如PPO的clip range、温度参数、采样数量、搜索深度等）。
