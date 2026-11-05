---
title: "Battling-against-Tough-Resister-Strategy-Planning-with-Adver"
source: https://aclanthology.org/2025.acl-long.184.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:41:34"
field: "非协作对话策略规划"
keywords: ["non-collaborative dialogue", "strategy planning", "adversarial game", "reinforcement learning", "Nash equilibrium", "Chain-of-Mind"]
innovations: ["构建说服者与多样化抵抗者的零和博弈并通过对抗自我Play训练逼近ϵ-NE", "设计Chain-of-Mind提示逐步推理抵抗者心理状态以增强策略感知", "证明多样化抵抗者训练可提升说服者最优下界"]
benchmarks: ["Craigslist-Bargain", "DIALOCONAN", "Charity Persuasion"]
---

# 论文速读：Battling-against-Tough-Resister-Strategy-Planning-with-Adver

## 一句话总结
本文提出GAIA，一种基于博弈的对抗性自我Play交互式训练范式，将说服者与多样化抵抗者建模为两人零和博弈，通过强化学习逼近ϵ-纳什均衡，显著提升非协作对话中针对强硬抵抗者的策略规划能力。

## 研究问题与动机
1. **核心问题**：非协作对话中双方利益冲突，需通过多轮策略规划引导达成共识。
2. **刺激提示方法的不足**：基于LLM的提示方法依赖预训练知识进行零/少样本策略生成，但LLM预训练并未显式学习非协作策略规划，缺乏任务特定的序列规划能力。
3. **外部策略规划器的局限**：现有微调规划器的方法未充分考虑对抗性交互，仅优化当前对话状态，忽略对手潜在响应及后续对话走向。
4. **静态对抗训练的缺陷**：基于固定提示模拟对手的对抗训练无法动态适应双方能力提升，导致鲁棒性受限。

## 核心贡献（创新点）
1. **提出GAIA博弈训练范式**：将说服者与多样化抵抗者构建为零和博弈并通过RL逼近纳什均衡，区别于仅依赖静态提示或单次监督微调的方法。
2. **设计Chain-of-Mind提示**：逐步推理抵抗者的情绪、未来行动和对话目标以规划说服策略，使策略感知超越传统基于对话历史的预测。
3. **多样化抵抗规划器构造与理论证明**：构造性能分布多样的抵抗者集合，理论证明该训练能提升说服者的最优下界，相比单一抵抗者训练显著增强泛化性。
4. **ϵ-NE迭代验证算法**：设计交替验证说服者与抵抗者策略的ϵ-NE收敛判据，解决严格纳什均衡难以求解的实际问题。

## 方法详解
- **模型架构**：GAIA包含五个模块：Resister-aware persuasive strategy planner、Diverse resistant strategy planners、Strategic response generator、Adversarial self-play interaction via game、Approximation of Nash Equilibrium。
- **Resister-aware说服规划器**：采用Chain-of-Mind (CoM) 提示，基于LLM的心智理论能力，分五步逐步推理抵抗者的情绪、未来行动和对话目标；随后通过监督微调（SFT）初始化规划器$\text{SP}_{per}$，最小化交叉熵损失$\mathcal{L}_c^p = -\frac{1}{|\mathcal{D}|}\sum_{d\in\mathcal{D}}\frac{1}{T_d}\sum_{t=1}^{T_d} a_t^p \log y_t^p$。
- **多样化抵抗规划器**：在SFT阶段保存不同性能的抵抗规划器$\text{SP}_{res}$，确保对抗样本多样性；理论证明训练集多样性（性能方差更大）可提高信息增益，从而提升说服者的最优下界$L(\pi^m) \geq L(\pi^l)$。
- **战略响应生成器**：通过映射$\mathcal{M}$将策略转化为指令，结合角色提示$p_{sys}$由LLM生成 utterance；为抵抗生成器分配Big-Five人格以进一步多样化响应风格。
- **对抗自我Play交互博弈**：构建两人零和马尔可夫博弈$\mathcal{G}=(S, \mathcal{A}^p, \mathcal{A}^r, \mathcal{P}, \mathcal{R}, \gamma)$，双方交替选择策略并交互；设计目标导向奖励（CB采用SL%、DC采用Δℋ、CP采用二元成功奖励）及每轮轻微负奖励以惩罚冗长对话；采用REINFORCE算法更新策略参数$\theta \leftarrow \theta - \alpha \nabla \log \pi_\theta R_t$。
- **ϵ-NE迭代验证**：交替优化说服者策略$\pi$和抵抗者策略$\mu$，并验证是否满足$\epsilon$-最佳响应条件（变化后期望奖励增益不超过$\epsilon$），直至收敛。

## 实验与结果
- **数据集**：Craigslist-Bargain（CB，议价谈判）、DIALOCONAN（DC，反驳仇恨言论）、Charity Persuasion（CP，慈善劝说）。
- **基线方法**：Prompt-based（Standard、ProCoT、ICL-AIF）；External planner（MCTS、PPDPP、TRIP）。
- **主要结果**：
  - **CB**：SR 0.693（↑5.2% vs TRIP），AT 6.03（↓6.7%），SL% 0.413（↑6.1%）。
  - **DC**：SR 0.643（↑6.9% vs TRIP），AT 6.92（↓6.7%），Δℋ 0.412（↑10.7%）。
  - **CP**：SR 0.562（↑6.4% vs TRIP），AT 8.12（↓4.4%）。
- **对抗性分析**：面对强硬抵抗者时，GAIA较PPDPP/TRIP提升约5%（弱抵抗者提升约2.8%）；在高对抗强度区间（[0.6,0.8]）SR提升8%。
- **人格泛化**：在Big-Five各人格抵抗者上均取得最高成功率。
- **人类评估**：PSR 0.59，Coherent 3.68/5，显著优于PPDPP（0.48/3.54）和TRIP（0.54/3.57）。

## 相关工作脉络
1. **传统深度学习方法**（Zhou et al., 2020; Joshi et al., 2021; Deng et al., 2024）：依赖对话行为预测进行监督训练，需大量标注数据且仅考虑下一步反馈，忽视长期目标。
2. **LLM提示方法**（Deng et al., 2023b; Fu et al., 2023）：利用预训练LLM的内在知识通过CoT/AI反馈进行零样本策略规划，但缺乏显式任务特定优化。
3. **外部策略规划器**（PPDPP、TRIP）：将可微调规划器与LLM结合，通过SFT或RL预测下一轮策略；但未充分建模对手动态响应，对抗agent由静态提示模拟。
4. **GAIA定位**：通过构建可动态增强的多样化抵抗者池并进行零和博弈对抗自我Play训练，克服静态提示局限，实现更强鲁棒性与泛化能力。

## 局限性与未来方向
1. **计算开销大**：依赖强化学习与多轮博弈交互，训练成本高，难以直接扩展至更复杂长对话场景。
2. **现实应用验证不足**：仅在三个基准数据集上评估，未测试于多样化真实商业、法律或客服场景。
3. **依赖底层LLM质量**：性能受限于所用LLM的能力与偏见，可能影响对细微交互的处理。
4. **未来方向**：探索高效训练策略（如离线RL、课程学习）；在更多样化现实任务中验证；结合对齐技术减少偏见；将博弈框架推广至多智能体协作/竞争场景。

## 研究启发与可借鉴点
1. **博弈对抗训练范式迁移**：将非协作对话建模为零和博弈并通过自我Play优化的思路可复用于辩论系统、谈判助手、客服对抗训练等需要策略规划的任务。
2. **Chain-of-Mind提示设计**：逐步推理对手心理状态（情绪、意图、目标）以提升策略感知的方法，可推广至需要心智理论的其他多轮交互系统。
3. **多样化对抗者池构建**：通过性能方差控制训练对手多样性，并利用信息论证明提升最优下界，该方法适用于任何需要增强模型鲁棒性的对抗性学习场景。
4. **ϵ-NE收敛判据**：提供实用的博弈训练停止标准，避免严格NE求解的计算难题，可借鉴至其他多智能体强化学习任务。

## 关键术语表
- **Non-collaborative dialogue**：利益冲突双方通过多轮对话达成各自目标的交互形式。
- **Strategy planning**：在对话过程中选择并执行一系列策略以引导对手走向预期结果。
- **Chain-of-Mind (CoM)**：基于心智理论的分步推理提示，用于推断对手的情绪、未来行动与对话目标。
- **Nash Equilibrium (NE)**：博弈中双方策略均无单边改进动机的稳定状态。
- **Zero-sum game**：一方收益等于另一方损失的博弈，双方目标完全对立。
- **Adversarial self-play**：双方智能体通过交替训练相互对抗以提升能力的强化学习方法。
- **ϵ-NE**：允许策略偏离最佳响应不超过ϵ的近似纳什均衡，用于降低严格NE求解难度。
- **Success Rate (SR)**：在最大轮次内成功实现对话目标的样本比例。

## 可复现要素
- **数据集**：Craigslist-Bargain、DIALOCONAN、Charity Persuasion，均为公开数据集。
- **代码/权重**：论文未明确声明代码与预训练权重开源情况。
- **关键超参**：SFT阶段 batch size=32，learning rate=1e-5，max sequence length=512；RL阶段 learning rate=1e-6，max conversation turn=12；规划器使用RoBERTa，生成器使用GPT-3.5（DC仇恨言论任务使用uncensored LLM）。
