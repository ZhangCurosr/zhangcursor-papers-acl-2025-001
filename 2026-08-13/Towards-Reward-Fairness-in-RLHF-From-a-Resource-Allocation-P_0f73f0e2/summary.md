---
title: "Towards-Reward-Fairness-in-RLHF-From-a-Resource-Allocation-P"
source: https://aclanthology.org/2025.acl-long.163.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:23:23"
field: "大语言模型对齐"
keywords: ["RLHF", "Reward Fairness", "Resource Allocation", "DPO", "Length Bias", "Reward Model"]
innovations: ["从资源分配视角统一建模reward unfairness，提出Fairness Regularization和Fairness Coefficient两种通用方法", "在BT-RM和DPO框架下实现公平性奖励分配，同时缓解长度偏置、类别偏置和社会偏置", "提出统一的公平度量函数族（含Jain's index推广形式），支持τ参数灵活调节公平偏好强度"]
benchmarks: ["Reward Bench", "HH-RLHF", "AlpacaEval2", "MT-Bench", "CrowS-Pairs"]
---

# 论文速读：Towards-Reward-Fairness-in-RLHF-From-a-Resource-Allocation-P

## 一句话总结
本文从资源分配视角统一建模RLHF中的各类reward bias（如长度偏置、类别偏置、社会偏置），提出Fairness Regularization与Fairness Coefficient两种公平性奖励方法，分别应用于Reward Model训练和DPO策略优化，在不牺牲utility的前提下实现更公平的reward分布。

## 研究问题与动机
- **Reward bias的多态性**：RLHF中reward model存在长度偏置（length bias）、类别偏置（category bias）、社会偏置（social bias）等多种不公平现象，导致策略模型输出偏离人类真实偏好。
- **现有方法缺乏统一视角**：Park et al. (2024)、Chen et al. (2024)等针对特定bias分别设计解法（如length regularization），但方案不可迁移，难以覆盖多种bias的协同效应。
- **BT模型的reward分布不均匀**：在HH-RLHF上，Helpful与Harmless数据的reward分布存在显著差异（Helpful平均reward约-1.39/-2.26，Harmless约-4.15/-5.23），导致下游RL阶段偏向某一类别。
- **公平性定义的混淆**：LLM领域"fairness"多指社会公平（social fairness），本文将其重新定义为资源分配中的reward分配公平性，更具数学可操作性。

## 核心贡献（创新点）
- **统一视角**：将length bias、category bias、social bias等统称为"reward unfairness"，为后续通用建模奠定基础。
- **资源分配框架**：将偏好学习建模为资源分配问题，以Jain's index的统一公平度量形式（含参数τ）刻画reward分配公平性，兼顾utility与fairness。
- **双路径方法**：提出Fairness Regularization（加法）与Fairness Coefficient（乘法）两种损失构造方式，分别适配Verification和RL场景。
- **跨场景验证**：在Reward Model训练（验证公平性）和DPO/KTO策略学习（隐式公平reward）两个阶段均验证方法有效性。

## 方法详解
- **统一公平度量函数**：采用Lan et al. (2010)提出的公平函数族 $f_\tau(\mathbf{a}) = \text{sign}(1-\tau) \cdot \left[\sum_i (a_i / \sum_j a_j)^{1-\tau}\right]^{1/\tau}$，当τ=-1时退化为Jain's index $J(\mathbf{a}) = (\sum a_i)^2 / \sum a_i^2$。
- **Utility定义**：对RRM，$U(\mathbf{a}) = \mathbb{E}[\log\sigma(a_i)]$，其中$a_i = r_\phi(y_w) - r_\phi(y_l)$；对DPO，$a_i = \beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}$。
- **FR RM损失**：$\mathcal{L}_{FR\ RM} = -\mathbb{E}[\log\sigma(a_i)] - \alpha F(\mathbf{a})$，通过加性正则项注入公平性约束。
- **FC RM损失**：$\mathcal{L}_{FC\ RM} = -\mathbb{E}[\log\sigma(a_i)] \cdot F(\mathbf{a})^\gamma$，通过乘性系数耦合utility与公平性。
- **Fair DPO等价形式**：将DPO的隐式reward差值作为分配向量$\mathbf{a}$，直接套用上述FR/FC框架，无需额外修改DPO结构。
- **公平性公理**：要求公平度量满足连续性（continuous）、齐次性（homogeneous degree 0）、单调性（monotonicity），确保不同τ值下的公平函数均可稳定优化。

## 实验与结果
- **数据集**：验证阶段使用HH-RLHF（ID）和Reward Bench（OOD）；RL阶段使用UltraFeedback Binarized + SHP训练策略模型，评估用AlpacaEval2和MT-Bench。
- **公平性验证结果**：FR RM和FC RM在Reward Bench（Avg: 78.38/77.50）和HH-RLHF（Avg: 73.55/73.96）上与BT RM（78.11/73.81）性能持平，但reward分布在Helpful/Harmless类别间更均匀。
- **RL结果**（LLaMA3-SFT为基座）：FR DPO在AlpacaEval2取得LC WR=20.48、WR=15.74，优于DPO（16.71/14.23）；MT-Bench得分6.70，优于DPO（6.46）。
- **长度偏置缓解**：相同输出长度下，Fair DPO性能优于DPO；同等性能下输出更短，表明公平性约束有效抑制了长度膨胀。
- **数据选择效率**：使用FR RM/FC RM进行样本筛选时，相同样本数可选中更高质量样本，达到相同性能所需样本更少。
- **社交偏见验证**：在CrowS-Pairs上，BT RM对stereotypical句子的reward显著偏高，FR/FC RM的"sent more"与"sent less"分布差异更小。
- **超参消融**：τ∈[-5,10]范围内FR DPO始终优于DPO，α≈0.1、γ≈0.5为较优取值。

## 相关工作脉络
- **Length Regularization (Park et al. 2024, R-DPO)**：仅针对长度偏置设计，本文方法在同等设置下仍有效，且能同时缓解类别与社会偏置。
- **Odin (Chen et al. 2024)**：解耦reward以减轻reward hacking，本文从资源分配视角统一处理，不依赖task-specific假设。
- **Fast RL (Li et al. 2024)**：通过ensemble不同RM提升公平性，属于方法层面改进；本文聚焦单RM的内在公平性建模。
- **Learning Diverse Preferences (Yang et al. 2024, Padmakumar et al. 2024)**：间接缓解类别偏置，本文显式构造公平度量，机制更透明。
- **DPO/KTO/SimPO**：作为隐式reward学习框架，本文可直接嵌入其公平性约束，拓展了这些方法的应用边界。
- **Social Bias在LLM中的研究 (Li et al. 2023, Gallegos et al. 2024)**：本文将其纳入统一框架，证明社会偏置同样是reward分配不公平的表现形式。

## 局限性与未来方向
- **验证范围有限**：主要验证category bias和length bias，对reward hacking等更广泛的不公平现象仅初步探讨。
- **基线算法局限**：仅在BT-RM和DPO上验证，未扩展至PPO、SimPO、KTO等更多RL框架（论文声明可扩展）。
- **公平度量的选择依赖τ**：不同τ值对应不同公平偏好，缺乏自动选择机制，需手动调参。
- **跨语言/跨领域泛化未验证**：实验主要在英文benchmark上进行，其他语言和领域的公平性提升有待验证。

## 研究启发与可借鉴点
- **资源分配视角的统一建模**：将多种bias统一为reward分配不公平问题，为后续研究提供了可复用的理论框架，可迁移至多目标偏好学习场景。
- **Jain's index及其推广形式可直接复用**：作为fairness function的标准组件，只需替换$a_i$的定义即可适配不同模型架构。
- **公平性对长度的隐性约束**：实验发现公平性训练自然抑制了长度膨胀，提示可在不额外增加长度正则项的情况下缓解此问题。
- **数据筛选效率的提升**：公平reward模型在sample selection任务上表现更优，可结合本团队的偏好数据清洗流程使用。
- **消融实验中τ的鲁棒性**：τ在[-5,10]范围内方法均有效，说明该方法对公平度量形式具有较强的泛化性，可作为默认设置直接使用。

## 关键术语表
- **Reward Unfairness**：reward在不同数据类型（长度、类别、社会属性）间分布不均匀的现象，是length bias、category bias等的统一表述。
- **Fairness Regularization (FR)**：将utility与fairness以加法形式结合的优化目标，$\mathcal{L} = -U + \alpha F$。
- **Fairness Coefficient (FC)**：将utility与fairness以乘法形式结合的优化目标，$\mathcal{L} = -U \cdot F^\gamma$。
- **Jain's Index**：衡量分配公平性的经典指标 $J(\mathbf{a}) = (\sum a_i)^2 / \sum a_i^2$，取值范围[0,1]，1表示完全公平。
- **Length-controlled Win Rate (LC WR)**：AlpacaEval2上的长度控制胜率指标，用于消除长度差异对评测结果的影响。
- **Bradley-Terry (BT) Model**：基于配对比较的reward model基础架构，假设$P(y_w \succ y_l) = \sigma(r(y_w) - r(y_l))$。
- **Direct Preference Optimization (DPO)**：无需显式reward model的直接偏好优化方法，隐式拟合reward函数以对齐人类偏好。
- **Utility-Fairness Trade-off**：reward分配中偏好对齐能力（utility）与跨类别分布一致性（fairness）之间的平衡关系。

## 可复现要素
- **数据集**：HH-RLHF（开源）、Reward Bench（开源）、AlpacaEval2（开源）、MT-Bench（开源）、UltraFeedback Binarized（开源）、SHP（开源）
- **代码开源**：https://github.com/shoyua/Towards-Reward-Fairness（论文已声明）
- **关键超参**：学习率（RM: 2e-6，Policy: 5e-6）、batch size=256、β=0.1、α=0.1、γ=0.5、τ=-1（默认）
- **训练配置**：8×H800、1 epoch、temperature=1
- **基座模型**：LLaMA3-SFT、Qwen2.5-SFT（自行训练）
