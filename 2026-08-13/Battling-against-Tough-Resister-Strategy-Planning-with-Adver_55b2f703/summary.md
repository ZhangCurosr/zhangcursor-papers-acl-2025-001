---
title: "Battling-against-Tough-Resister-Strategy-Planning-with-Adver"
source: https://aclanthology.org/2025.acl-long.184.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:51:36"
---

# 论文速读：Battling-against-Tough-Resister-Strategy-Planning-with-Adver

## 一句话总结
本文提出GAIA（Game-based Adversarial self-play InterActive training paradigm），通过构建劝说者与抵抗者之间的两玩家零和博弈，利用强化学习迭代优化双方策略并逼近Nash均衡，有效解决了非合作对话中大模型策略规划能力不足的问题，尤其在对抗"强硬抵抗者"和"高对抗强度"场景下显著优于现有方法。

## 研究问题与动机
1. **非合作对话策略规划的核心难题**：两个利益冲突的参与者通过多轮对话争取各自目标，策略规划需同时兼顾短期对话状态与长期目标达成，而现有方法难以有效处理对手的智能对抗行为。
2. **传统深度学习方法依赖标注且忽视长期目标**：基于对话行为预测的监督学习方法严重依赖高质量标注数据，且仅基于下一步反馈规划策略，容易陷入局部最优而忽视全局目标。
3. **LLM刺激提示方法未显式学习策略规划**：尽管ProCoT、ICL-AIF等方法通过提示激发LLM的自我思考和反思能力，但大模型预训练阶段并未显式学习非合作策略规划，直接应用于序列策略规划时效果受限。
4. **现有对抗训练的对手能力静态不变**：PPDPP、TRIP等方法虽引入了外部策略规划器或模拟对手进行对抗训练，但对手能力固定，无法随劝说者能力提升而动态增强，导致训练出的策略在遇到"强硬抵抗者"时鲁棒性不足。

## 核心贡献（创新点）
1. **提出GAIA博弈训练范式**：构建劝说者与抵抗者的两玩家零和马尔可夫博弈，通过对抗性自我对弈交互式RL迭代优化双方策略并逼近ε-Nash均衡，区别于PPDPP/TRIP的静态对抗训练，实现对手能力的动态增强。
2. **设计Chain-of-Mind (CoM) 推理机制**：利用LLM的Theory-of-Mind能力，通过五步链式推理（历史分析→心理状态→情绪→未来行动→对话目标）感知抵抗者内在状态，使劝说策略更具针对性和情感共鸣，突破了传统方法仅依赖对话历史的局限。
3. **多样化抵抗规划器构建的理论证明**：从信息论角度严格证明对抗更多样化的抵抗者训练可提升劝说者的最优下界$L(\pi^m) \geq L(\pi^l)$，为多样化对手构建提供了理论基础，相比单一抵抗者策略显著提升泛化能力。
4. **ε-NE迭代验证算法**：设计交替验证劝说者与抵抗者策略的迭代算法，在无法严格收敛到Nash均衡时找到近似均衡点，兼顾训练可行性与策略质量，避免传统self-play训练中的震荡问题。

## 方法详解

### 整体架构
GAIA包含五个核心模块：**Resister-aware Persuasive Strategy Planner**（抵抗者感知的劝说策略规划器）、**Diverse Resistant Strategy Planners**（多样化抵抗策略规划器集合）、**Strategic Response Generator**（策略感知响应生成器）、**Adversarial Self-play Interaction via Game**（对抗性自我对弈交互模块）、**Approximation of Nash Equilibrium**（Nash均衡逼近模块）。

### Chain-of-Mind (CoM) 提示设计
CoM利用LLM的Theory-of-Mind能力，通过In-Context Learning实现五步链式推理：
- **Step 1**：分析对话历史
- **Step 2**：基于分析推理心理状态
- **Step 3**：从心理状态推断抵抗者的情绪
- **Step 4**：基于情绪和心理状态预测抵抗者可能的未来行动
- **Step 5**：推理抵抗者的对话目标

CoM prompt模板包含Task definition、Guideline instruction、CoM exemplars和Dialogue context四个部分，使劝说规划器能全面感知抵抗者的情绪、预期行动和隐藏目标，从而制定更具针对性的劝说策略。

### 劝说策略规划器初始化（SFT）
对预训练语言模型进行监督微调，最小化交叉熵损失：
$$\mathcal{L}_c^p = -\frac{1}{|\mathcal{D}|}\sum_{d \in \mathcal{D}}\frac{1}{T_d}\sum_{t=1}^{T_d} a_t^p \log y_t^p$$
其中输入包含推理得到的抵抗者情绪$\hat{e}^r$、未来行动$\hat{a}_t^r$、对话目标$\hat{g}^r$及对话历史$d$，输出为劝说策略$a_t^p$。SFT仅获得次优策略，需进一步通过RL优化。

### 多样化抵抗规划器构建
在SFT训练过程中保存多个性能各异的抵抗策略规划器，形成策略池$S_r$。论文通过理论证明（Appendix B）：当$S_r^l \subseteq S_r^m$时，更多样化的抵抗者集合提供更大的信息增益（Information Gain = $H(S_r^m) - H(S_r^l) \geq 0$），从而提升劝说者的最优下界$L(\pi^m) \geq L(\pi^l)$。

### 策略感知响应生成
将策略规划与文本生成解耦：策略规划器输出离散策略$a \in \mathcal{A}$，通过映射$\mathcal{M}$转换为自然语言指令，由大语言模型根据角色prompt生成对话轮次。劝说者和抵抗者分别使用不同的角色prompt $p_{sys}^{per}$和$p_{sys}^{res}$，并对抵抗者生成器分配Big-Five Personality人格以增强多样性。

### 对抗性自我对弈交互式RL
构建两玩家零和马尔可夫博弈$\mathcal{G} = (S, \mathcal{A}^p, \mathcal{A}^r, \mathcal{P}, \mathcal{R}, \gamma)$，其中：
- **状态空间**$S$：对话历史
- **策略空间**$\mathcal{A}^p, \mathcal{A}^r$：预定义的劝说/抵抗策略集合
- **奖励函数**$\mathcal{R}$（目标导向）：
  - CB数据集：$SL\% = P_{current}/P_{buyer}$（成交价与买家目标价比率）
  - DC数据集：$\Delta\mathcal{H} = \mathcal{H}(u_0) - \mathcal{H}(u_t)$（仇恨强度降低量）
  - CP数据集：目标达成得1.0，未达成得-1.0
- **终止惩罚**：每轮额外-0.1鼓励高效对话

总回报$R_t = \sum_{t'=t}^{T} \gamma^{T-t'} r_{t'}$，使用REINFORCE算法优化策略参数：
$$\theta \leftarrow \theta - \alpha \nabla \log \pi_\theta R_t$$

### ε-Nash Equilibrium迭代验证
定义ε-最佳响应：$V(s; \pi^{\epsilon b}, \mu) \geq V(s; \pi^b, \mu) - \epsilon$。ε-NE要求双方策略互为ε-最佳响应，即单方面偏离的收益变化不超过ε。设计交替验证算法（Algorithm 1）：每轮依次优化劝说者策略并验证条件C1、优化抵抗者策略并验证条件C2，两者均满足时终止训练。

## 实验与结果

### 数据集
- **Craigslist-Bargain (CB)**：买家与卖家商品价格谈判，3,090/188/188条训练/验证/测试
- **DIALOCONAN (DC)**：反仇恨言论者说服仇恨言论者改变态度，2,805/246/246条
- **Charity Persuasion (CP)**：劝说者鼓励受劝者慈善捐赠，300/50/50条

### 评估指标
- **Success Rate (SR)**：最大轮次内达成目标的百分比
- **Average Turn (AT)**：达成目标所需的平均轮数
- **数据集特定指标**：CB的SL%、DC的ΔH、CP的目标达成率

### 基线方法
- **提示类**：Standard、ProCoT、ICL-AIF
- **外部策略规划器类**：MCTS、PPDPP、TRIP

### 主要结果
| 数据集 | 指标 | GAIA | 最佳基线(TRIP) | 提升幅度 |
|--------|------|------|----------------|----------|
| CB | SR↑ | **0.693** | 0.659 | +5.2% |
| CB | AT↓ | **6.03** | 6.46 | -6.7% |
| CB | SL%↑ | **0.41
