---
title: "TRACT-Regression-Aware-Fine-tuning-Meets-Chain-of-Thought-Re"
source: https://aclanthology.org/2025.acl-long.147.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:22:04"
field: "大语言模型评估与对齐"
keywords: ["LLM-as-a-Judge", "回归感知微调", "链式思维推理", "RAFT", "分布对齐", "数值预测", "自生成数据"]
innovations: ["提出CoT-RAFT联合训练目标，融合CE损失与回归损失", "两阶段自生成CoT策略缓解训练-推理分布偏移", "单点评分模型在有限推理预算下超越同规模SOTA"]
benchmarks: ["Feedback Bench", "FLASK", "Vicuna Bench", "MT Bench", "RewardBench"]
---

# 论文速读：TRACT-Regression-Aware-Fine-tuning-Meets-Chain-of-Thought-Re

## 一句话总结
本文提出TRACT方法，将链式思维（CoT）推理与回归感知微调（RAFT）相结合，通过两阶段训练策略解决LLM-as-a-judge中数值评分任务的分布偏移问题，显著提升了开源模型在自动文本评分任务上的性能。

## 研究问题与动机
- **CE损失的数值预测缺陷**：传统LLM-as-a-judge使用交叉熵（CE）损失微调，但CE损失对相近错误（如预测"2"而非"1"）与远离错误（如预测"5"而非"1"）同等惩罚，无法对齐平方误差等数值评估指标。
- **RAFT缺乏CoT能力**：RAIF仅适用于直接数值回归，未利用CoT推理，而CoT对LLM-as-a-judge的评分质量至关重要。
- **训练-推理分布偏移**：使用外部模型（如GPT-4）生成的CoT进行微调后，模型在推理时生成自CoT与训练CoT存在显著分布差异，损害预测性能。

## 核心贡献（创新点）
- **CoT-RAFT联合训练目标**：首次将CE损失与RAFT损失结合，分别优化CoT生成质量与数值预测准确性。
- **两阶段自生成CoT策略**：第一阶段用GPT-4生成的CoT微调种子模型，第二阶段用该模型生成的CoT重新训练，缓解训练-推理分布偏移。
- **性能超越同规模SOTA**：在有限推理计算下，TRACT在Mistral-7B上超越Prometheus-2-7B（同规模最强开源模型）平均0.059 Pearson相关系数，且无需额外配对数据。
- **单点评分模型迁移至奖励建模**：验证了点wise LLM-as-a-judge模型可在RewardBench配对排序任务上取得合理表现（平均0.736准确率）。

## 方法详解
**CoT-RAFT训练目标**（公式4）：
$$\ell_{\mathrm{CoT-RAFT}}^{\lambda}(y^*, p_{\mathrm{t}}, p) = \lambda\left(\sum_{y \in \mathcal{Y}} p(\mathrm{str}(y)|[x, \hat{s}]) \cdot y - y^*\right)^2 - \log p([\hat{s}, y^*]|x)$$
其中$\hat{s} \sim p_{\mathrm{t}}(\cdot|x)$为生成CoT，第一项为RAFT回归损失，第二项为CE CoT损失，$\lambda$为权重系数（实验默认$\lambda=1$）。

**两阶段训练流程**：
1. **Stage 1**：以种子LLM$p_0$初始化，用GPT-4生成的CoT与真实分数微调，得到$p_s$。
2. **Stage 2**：冻结$p_s$，为每个输入$x$生成自CoT $\hat{s}_s(x) \sim p_s(\cdot|x)$，丢弃分数预测，构建新数据集$D_{\mathrm{self}} = \{(x, \hat{s}_s(x), y^*)\}$，以$p_0$初始化重新微调，得到最终模型$p_{\mathrm{tract}}$。

**推理方式（CoT-RAIL）**：从微调模型$p$采样CoT $\hat{s} \sim p(\cdot|x)$，计算期望得分$\hat{y}_{\mathrm{CR}}(x) = \sum_{y \in \mathcal{Y}} p(\mathrm{str}(y)|[x, \hat{s}]) \cdot y$。

**技术细节**：CoT以字符串"d = So the overall score is"作为结束标记，确保与评分预测位置对齐。

## 实验与结果
**数据集**：
- **Feedback Bench**：官方测试集，1K响应。
- **FLASK**：200提示、2000响应（Alpaca-7B/Vicuna-13B/Bard/GPT-3.5）。
- **Vicuna Bench**：80用户指令、320响应。
- **MT Bench**：多轮对话、320响应。

**基线**：标准CE微调+解码（无/有CoT）、零样本RAIL、RAFT、Prometheus-2-7B（同规模SOTA）、CLoud奖励模型。

**主要结果（Mistral-7B）**：
- TRACT平均Pearson r=**0.650**，超越CE+CoT基线（0.557，提升0.093）、超越Prometheus-2-7B（0.591，提升0.059）、超越RAFT（0.623）。
- **FB Bench**上Pearson r=0.931，与RAFT（0.932）持平；**Vic. Bench**提升最显著（0.593 vs. 基线0.463）。
- **Llama-3.1-8B**上同样全面超越所有基线。

**消融结论**：
- 移除自生成CoT（A.1）：平均Pearson r下降0.094。
- 用CE替代CoT-RAFT（A.2）：下降0.033。
- 仅CE微调自生成CoT（A.3）：反而劣于GPT-4 CoT基线，证明CoT-RAFT必要性。
- Stage 2从$p_s$初始化（A.4）：性能降至0.515，验证种子初始化关键性。
- RMSE分布偏移分析（Table 3）：Stage 1模型用训练CoT与自生成CoT的RMSE差距0.51，Stage 2降至0，证实分布对齐效果。

## 相关工作脉络
- **LLM-as-a-Judge（Kim et al., 2024a,b）**：本文主要对比对象，使用CE损失+CoT微调；TRACT在同等训练数据下显著超越。
- **RAFT（Lukasik et al., 2025）**：回归感知微调，仅用平方误差损失预测分数，无CoT；TRACT扩展其为带推理能力的联合优化。
- **RAIL（Lukasik et al., 2024）**：贝叶斯最优推理框架，通过期望得分降低回归误差；TRACT将其嵌入CoT条件化推理。
- **Prometheus-2（Kim et al., 2024b）**：同规模最强开源LLM-as-a-judge，合并反馈收集与偏好收集数据训练；TRACT仅用单点数据在有限推理预算下实现超越。
- **Self-Distillation（Yang et al., 2024）**：用自生成数据微调可降低困惑度；本文确认此现象但指出CE损失无效，需配合CoT-RAFT。
- **CLoud（Ankner et al., 2024）**：使用自生成CoT+回归头的奖励模型；本文证明点wise评分模型与奖励模型能力不可互换。

## 局限性与未来方向
- **依赖模型概率输出**：需要访问模型输出分布，不适用于黑盒API模型（如GPT-4）。
- **推理计算可扩展性有限**：多CoT采样对TRACT性能提升不明显，而标准解码可通过增加采样提升性能。
- **数据规模限制**：当前训练数据为~100K样本，更大规模训练效果待验证。
- **泛化性待检验**：主要在5分制评分任务验证，对连续分数或其他评分范围的有效性未知。
- **未来方向**：探索多CoT采样与RAFT的结合、跨语言/跨领域泛化、与人类反馈直接对齐。

## 研究启发与可借鉴点
- **两阶段自生成策略可迁移**：任何需要CoT推理且输出为连续/离散数值的任务（如评分、预测、估计）均可借鉴此"外部CoT→自生成CoT"两阶段范式。
- **联合损失设计思路**：CE损失保障生成质量、回归损失对齐数值预测的联合优化策略，可应用于数值型生成任务（如时间序列预测、回归式文本生成）。
- **分布对齐的实证价值**：Table 3的RMSE对比实验设计简洁有力地验证了分布偏移假设，可作为后续工作的标准验证手段。
- **单点模型的双用潜力**：证明点wise评分模型可迁移至配对比较任务，为多任务统一建模提供启发。
- **超参鲁棒性**：λ在0.2-10范围内均有效，表明方法对超参不敏感，便于实际部署。

## 关键术语表
**LLM-as-a-Judge**：利用大语言模型根据细粒度评分标准自动对文本进行评估并输出分数的范式。
**RAFT（Regression-Aware Fine-tuning）**：通过最小化平方误差损失直接优化模型输出分布，使推理时的期望得分最小化回归误差的微调方法。
**RAIL（Regression-Aware Inference）**：基于贝叶斯最优决策规则，在推理时对候选分数分布求期望以获得最优数值预测的解码策略。
**CoT-RAFT**：本文提出的联合训练目标，同时优化CoT生成的CE损失与分数预测的RAFT损失。
**CoT-RAIL**：条件化CoT的推理策略，先采样CoT再计算条件期望得分。
**Distribution Shift**：训练时使用的CoT分布与推理时模型自生成CoT分布之间的差异，是本文核心解决的问题。
**Pointwise LLM-as-a-Judge**：对单个样本独立输出分数的评估模式，区别于pairwise排名。
**Reward Model（RM）**：通常为配对训练、输出标量奖励的模型，用于RLHF等场景。

## 可复现要素
- **训练数据**：Feedback Collection（约100K样本），由GPT-4生成；论文声明训练数据公开。
- **代码/模型**：论文释放模型权重（Apache 2.0许可证），代码位于https://github.com/prometheus-eval/prometheus-eval（官方代码修改版）。
- **基座模型**：Mistral-7B-Instruct-v0.2、Llama-3.1-8B-Instruct。
- **训练超参**：LoRA rank=8，全线性层；学习率1.0e-5；cosine调度；2 epochs；warmup ratio=1.0；bf16精度；有效batch size=8；λ=1。
- **推理超参**：vLLM引擎；top-p=0.9；temperature=1.0；repetition penalty=1.03；max output tokens=1024；默认单CoT采样。
- **硬件**：单卡NVIDIA RTX A6000；全训练约100小时。
- **评估指标**：Pearson r、Spearman ρ、Kendall τ。
