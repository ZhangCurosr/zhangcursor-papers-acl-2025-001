---
title: "Direct-Prompt-Optimization-with-Continuous-Representations"
source: https://aclanthology.org/2025.acl-long.133.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:55:06"
field: "提示词优化与大语言模型应用"
keywords: ["prompt optimization", "Gumbel-Max", "gradient-based methods", "adversarial attack", "discrete optimization", "SGCG"]
innovations: ["提出SGCG框架将提示优化建模为概率分布的直接优化目标", "融合连续梯度更新与贪婪采样并利用滑动窗口保留历史信息", "发现恒等函数作为Straight-Through代理梯度优于Softmax"]
benchmarks: ["SST-2", "IMDB", "Amazon", "Yelp", "DBPedia", "AGNews"]
---

# 论文速读：Direct-Prompt-Optimization-with-Continuous-Representations

## 一句话总结
本文提出了 **SGCG（Soft Greedy Coordinate Gradient）** 框架，通过将提示词优化建模为概率分布优化问题，结合连续梯度更新与贪婪采样策略，在保持可微分优化的同时利用历史梯度信息，有效提升了离散提示词优化的性能与稳定性。

## 研究问题与动机
- 提示词优化是离散搜索空间中的难题，传统梯度方法无法直接应用。
- 现有硬表示方法（如 GCG、AutoPrompt）依赖贪婪搜索但丢失历史信息，软表示方法（如 GBDA）虽可利用梯度但非贪婪导致优化不稳定。
- 两类方法各有优劣且互补，缺乏一种能同时利用连续梯度信息和贪婪选择机制的统一框架。
- 连续表示在优化过程中需要"舍入"为离散提示词，造成灵活性和稳定性不足。

## 核心贡献（创新点）
- **提出 SGCG 框架**：将提示词优化转化为对单次采样的直接优化目标，通过 Gumbel-Max 技巧重参数化实现无参数分布采样。
- **融合贪婪策略与连续优化**：在概率分布参数更新后，采用类似 GCG 的贪婪采样机制，避免非贪婪方法导致的灾难性损失上升。
- **设计滑动窗口候选选择机制**：通过窗口缓存历史最优候选替代存储全部中间表示，提升效率并防止遗漏优质候选。
- **发现 Identity 函数作为代理梯度的优越性**：相比 Softmax，恒等函数作为 Straight-Through 代理梯度表现更优。
- **系统验证**：在 6 个数据集上的文本分类与文本攻击任务中，SGCG 优于 FluentPrompt、GCG、GS 等多个基线方法。

## 方法详解
**优化目标形式化：**
给定提示词序列 $x = (x_1, ..., x_n)$，每个 token 来自词汇表 $\mathcal{V}$，目标是 $\min f(x)$ 且 $x_i \in \mathcal{V}$。

**Gumbel-Max 重参数化：**
利用 $x = \arg\max[\log \pi - \log(-\log \epsilon)]$ 将离散采样转化为可优化形式，其中 $\epsilon \sim U[0,1]$。

**两步交替更新（Algorithm 1）：**
1. **更新分布参数**：$\theta^{(k+1)} \leftarrow \theta^{(k)} - \alpha \frac{\partial f}{\partial \theta^{(k)}}$，沿梯度方向调整 token 概率分布。
2. **采样更新 Gumbel 变量**：$\epsilon^{(k+1)} \leftarrow \arg\min_{\epsilon} f(\gamma(\log \pi - \log(-\log \epsilon)))$，通过贪婪采样找到使损失下降的 token。

**Straight-Through 代理梯度：**
前向传播使用 $\arg\max$ 进行离散决策，反向传播使用代理函数 $\hat{\gamma}$。研究发现恒等函数 $\hat{\gamma}(x) = x$ 比 Softmax 更优。

**滑动窗口机制：**
维护大小为 $k$（等于提示词长度）的队列 $Q$，仅记录最近 $k$ 次替换的 loss 值，当新 loss 小于窗口内最小值时才接受更新，避免全量候选集存储开销。

## 实验与结果
**数据集：** SST-2、IMDB、Amazon、Yelp（情感分析）；DBPedia、AGNews（主题分类），采用 16 样例 few-shot 设置。

**模型：** GPT-2-Large、OPT-350M、Vicuna-7B、LLaMA-2。

**分类任务结果（Table 1）：**
- GPT-2 Large：SGCG 在 IMDB（81.85%）、Amazon（86.12%）、Yelp（88.29%）均超越 GCG；在 DBPedia（70.79% vs 66.99%）和 AGNews（76.39% vs 75.87%）上也有提升。
- OPT-350M：SGCG 在 SST-2 达到 88.00%（GCG 为 87.02%），Amazon 达 83.99%。

**Token 数量实验（Table 4）：**
随着 token 数从 5→10→20 增加，SGCG 在多数数据集上精度稳步提升（SST-2: 83.76%→86.47%→88.00%），IMDB 呈现反常（可能过拟合）。

**文本攻击任务（Table 5）：**
- Vicuna Single 设置：SGCG loss=0.0220（成功率100%），远超 GCG（0.3096）。
- Vicuna Fluent 设置：SGCG loss=0.7179（成功率80.77%），优于 GCG（0.8966，76.92%）。
- LLaMA-2 上 SGCG 同样在 Single 和 Fluent 设置达到 100% 成功率。

**消融实验（Table 3）：**
贪婪机制贡献显著——SST-2 上 greedy 版本 88.00% vs nongreedy 75.29%，Amazon 上 83.99% vs 68.06%。

**最强结果：** SGCG 在 GPT-2 Large 的 Yelp 上达到 **88.29%** 准确率，较 GCG 提升 0.41%；在文本攻击的 Vicuna Single 设置下 loss 低至 **0.0220**（100% 成功）。

## 相关工作脉络
- **GCG（Zou et al., 2023）**：基于坐标贪心的离散提示优化，每次选择最优 token 替换；SGCG 在其基础上引入连续分布参数以保留历史信息。
- **FluentPrompt（Shi et al., 2022）**：投影梯度下降方法，将软提示投影到最近 token；SGCG 避免直接投影导致的近似误差。
- **GBDA（Guo et al., 2021）**：使用 Gumbel-Softmax 进行软优化但非贪婪；SGCG 通过贪婪采样克服其稳定性问题。
- **AutoPrompt（Shin et al., 2020）**：一阶 Taylor 展开近似 + token 替换搜索；SGCG 用连续参数化替代直接搜索。
- **GS / Gumbel-Softmax（Jang et al., 2016）**：将离散选择松弛为连续近似；SGCG 在此基础上引入贪婪约束确保单调下降。
- **RL-based / Evolutionary methods**：不依赖梯度信息，随机初始化下效率低；本文聚焦白盒梯度方法。

## 局限性与未来方向
- Gumbel 变量引入的随机性可能导致优化方差，对超参数（温度 $\tau$、学习率）较敏感。
- 当前仅在文本分类和攻击任务上验证，泛化到其它 NLP 任务（如生成、问答）尚待研究。
- 梯度方法需白盒访问 LLM，对封闭商业模型适用性受限（作者认为开源模型普及可缓解此问题）。
- 跨模型迁移性：在开源模型上优化的提示词能否迁移到闭源商业模型，仍是开放问题。
- 未来工作包括：探索提示表示在不同领域的可迁移性、缓解 Gumbel 变量引起的不稳定性。

## 研究启发与可借鉴点
- **两阶段交替更新策略**：先优化分布参数再贪婪采样，这种"连续优化+离散选择"的交替模式可有效平衡探索与利用，可迁移到其他离散优化问题（如代码生成、结构预测）。
- **Identity 作为 Straight-Through 代理梯度的发现**：表明在某些场景下恒等映射比 Softmax 更适合作为非可微函数的反向代理，值得在其他离散-连续混合优化任务中验证。
- **滑动窗口候选机制**：相比维护全量候选集，滑动窗口以恒定内存开销保留历史最优信息，可推广至内存敏感的优化场景。
- **IMDB 反常现象的分析方法**：简单模型在简单数据集上可能表现更好，提示词长度过大会导致过拟合；这种"数据集复杂度 vs 模型容量"的分析视角值得借鉴。
- **攻击任务的评估设计**：将文本攻击分解为 Single/Multiple/Fluent 三种设置，能更全面评估提示优化的鲁棒性，可作为安全研究的参考范式。

## 关键术语表
**SGCG（Soft Greedy Coordinate Gradient）**：本文提出的提示词优化框架，结合连续分布优化与贪婪采样策略。

**Gumbel-Max 技巧**：通过引入 Gumbel 噪声将分类分布采样转化为确定性 argmax 运算，实现可微分近似。

**Straight-Through Estimator**：前向使用非可微函数（如 argmax），反向用代理可微函数传递梯度的训练技术。

**Projected Gradient Descent**：在嵌入空间中优化软提示，每步将其投影回词汇表最近 token 的方法。

**FluentPrompt\***：去除 fluency loss 后的 FluentPrompt 基线版本，仅优化目标任务 loss。

**Few-shot Prompting**：使用少量（如 16 个）训练样例指导大语言模型完成下游任务的范式。

**Adversarial Text Attack**：通过优化提示词诱导语言模型生成期望输出（绕过其安全对齐）的攻击方式。

**Sliding Window Candidate Selection**：维护固定大小历史窗口记录最优候选，避免全量存储的贪心选择机制。

## 可复现要素
- **数据集**：SST-2、IMDB、Amazon、Yelp、DBPedia、AGNews（公开基准，论文未明确说明复用方式，通常按标准 few-shot split）。
- **代码/权重**：论文声明"代码将在会议前开源"（Reproducibility: We will release the code before the conference）。
- **关键超参**：学习率 $\alpha$、温度 $\tau$（初始 1，指数衰减至 0.1）、窗口大小 $k$（等于提示词长度）、weight decay $10^{-3}$、few-shot 样例数 16。
