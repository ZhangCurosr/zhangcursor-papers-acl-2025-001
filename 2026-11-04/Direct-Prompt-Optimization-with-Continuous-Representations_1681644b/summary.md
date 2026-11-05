---
title: "Direct-Prompt-Optimization-with-Continuous-Representations"
source: https://aclanthology.org/2025.acl-long.133.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:44:05"
field: "大语言模型提示优化与对抗攻击"
keywords: ["Prompt Optimization", "Gradient-based Attack", "Gumbel-Max", "Continuous Representation", "White-box Prompt Tuning", "LLM Alignment"]
innovations: ["提出SGCG框架，将连续梯度优化与贪婪接受结合，直接优化单个提示样本", "使用Straight-Through恒等函数作为代理梯度，替代传统Softmax以提升离散优化稳定性", "引入滑动窗口机制聚合历史梯度信息，兼顾内存效率与防止优化发散"]
benchmarks: ["SST-2", "IMDB", "Amazon", "Yelp", "DBPedia", "AGNews", "Vicuna/LLaMA-2 Text Attack"]
---

# 论文速读：Direct Prompt Optimization with Continuous Representations

## 一句话总结
本文提出 **Soft Greedy Coordinate Gradient (SGCG)** 框架，通过将提示优化问题建模为概率分布采样，并将贪婪选择策略嵌入连续梯度更新过程中，有效结合离散搜索的稳定性与连续表示的历史梯度利用能力。在文本分类与大模型对抗攻击任务上，SGCG 均显著优于现有基线方法。

## 研究问题与动机
1. **离散搜索空间巨大**：提示优化本质是离散组合优化，传统梯度下降无法直接应用，需依赖启发式或一阶泰勒近似。
2. **硬表示方法信息损失严重**：AutoPrompt、GCG、FluentPrompt 等方法在每步都将软表示强制取整为离散 token，丢失了历史梯度信息，且容易陷入局部最优或提前收敛 plateau。
3. **软表示方法优化不稳定**：GBDA 等基于 Gumbel-Softmax 的方法虽提供平滑梯度，但采用期望最小化目标（针对整个分布而非单一提示），且非贪婪更新极易引发损失 catastrophically 激增。
4. **缺乏针对性单一样本优化目标**：现有 RL/期望类方法优化的是分布期望，而实际部署更关注单个高质量提示；同时现有软方法未有效利用历史梯度进行稳健决策。

## 核心贡献（创新点）
1. **提出 SGCG 直接优化框架**：绕过两阶段（先优化分布再采样）流程，直接联合优化分布参数与单个提示样本，目标更聚焦。
2. **引入 Straight-Through 恒等代理梯度**：前向传播保持离散 `argmax`，反向传播使用恒等函数作为可微近似，替代传统 Softmax 代理，显著提升优化稳定性。
3. **贪婪接受 + 滑动窗口机制**：仅当更新后损失下降时才采纳新参数与 Gumbel 变量，并通过固定大小滑动窗口维护历史候选，避免全量存储的同时保留梯度历史信息。
4. **跨任务与跨模型验证**：在 6 个分类数据集与 3 种攻击场景下，于 GPT-2、OPT、Vicuna、LLaMA-2 等多个规模模型上均取得最优或极具竞争力的结果。

## 方法详解
1. **概率分布建模与 Gumbel-Max 重参数化**：将提示序列 $\boldsymbol{x}$ 建模为按参数 $\boldsymbol{\theta}$ 参数化的分布 $P(\boldsymbol{x};\boldsymbol{\theta})$ 的采样。通过 Gumbel-Max trick 将采样过程转化为可微形式：
   $\boldsymbol{x} = \arg\max [\log \pi - \log(-\log \epsilon)]$，其中 $\epsilon_i \sim \text{U}[0,1]$。
2. **两步交替更新（Algorithm 1）**：
   - **更新分布参数**：$\boldsymbol{\theta}^{(k+1)} \leftarrow \boldsymbol{\theta}^{(k)} - \alpha \frac{\partial f}{\partial \boldsymbol{\theta}^{(k)}}$，利用任务损失对概率参数的梯度进行常规下降。
   - **更新 Gumbel 变量**：通过重采样 $\tilde{\epsilon}_i$ 并结合平滑代理 $\gamma$（实际使用恒等函数）计算候选提示 $\tilde{\boldsymbol{x}}$。
3. **Straight-Through 恒等代理**：前向传播使用离散 $\arg\max$ 做出 token 选择，反向传播使用 $\hat{\gamma}(x)=x$ 传递梯度。实验表明该设计比 Softmax 代理更稳定且收敛更快。
4. **贪婪接受与滑动窗口**：每次更新后计算 $f(\tilde{\boldsymbol{x}})$ 并加入队列 $Q$，若队列大小超过窗口 $k$（设为提示长度 $n$）则丢弃最旧元素；仅当 $f(\tilde{\boldsymbol{x}}) < \min(Q)$ 时才正式接受新参数 $\boldsymbol{\theta}_i$ 与 $\epsilon_i$。该设计既防止非贪婪更新导致的损失爆炸，又避免了为大量候选保存中间表示的内存开销。

## 实验与结果
- **数据集与设置**：SST-2、IMDB、Amazon、Yelp（情感分类）；DBPedia、AGNews（主题分类）。均采用 Few-shot 设置（每类 16 个训练样本），每个实验运行 5 次取平均。
- **模型**：GPT-2 Large、OPT-350M、Vicuna-7B、LLaMA-2。
- **分类任务（Table 1）**：SGCG 在 GPT-2 Large 上于 Amazon (+1.81)、Yelp (+0.41)、DBPedia (+3.80)、AGNews (+0.52) 显著超越 GCG；OPT-350M 上 SST-2 达到 88.00（+0.98）。IMDB 在 OPT-350M 上出现轻微过拟合，属简单模型对短评结构的过度记忆现象。
- **贪婪消融（Table 3）**：移除贪婪约束的 SGCG nongreedy 性能大幅下滑（SST-2: 88.00→75.29，Amazon: 83.99→68.06），验证贪婪接受的必要性。
- **Token 长度分析（Table 4）**：准确率随 token 数增加（5→10→20）整体上升，但 IMDB 呈现反常下降，印证其简单结构只需较少上下文。
- **对抗攻击任务（Table 5）**：在 Vicuna 与 LLaMA-2 的 Single/Multiple/Fluent 三种攻击设置下，SGCG 均取得接近 0 的损失与 100%（或最高）的攻击成功率，明显优于 GCG 与 FluentPrompt*。
- **收敛曲线（Fig 1）**：SGCG 在 DBPedia 与 AGNews 上训练/评估损失下降更平稳，GCG 约 400 步后陷入 plateau，SGCG 则持续高效搜索。

## 相关工作脉络
1. **Hard 表示方法（AutoPrompt / GCG / FluentPrompt*）**：基于离散 token 替换或投影梯度下降，每步强制取整导致历史信息丢失；SGCG 通过连续参数化保留梯度记忆并加入滑动窗口，打破收敛 plateau。
2. **Soft 表示方法（GBDA / Gumbel-Softmax）**：对离散分布做连续松弛，但优化目标是分布期望且无贪婪约束，易发生损失突增；SGCG 改为直接优化单个样本，并以恒等 ST 代理+贪婪接受提升稳定性。
3. **Black-box / RL / 进化算法（RLPrompt / EvoPrompt / InstructZero）**：不依赖梯度，计算成本高、随机初始化效率低；SGCG 定位为白盒梯度方法，在显存与步数效率上更具优势。
4. **位置差异**：本文填补了“纯离散贪婪”与“纯连续期望优化”之间的空白，提供面向单一提示的直接优化范式，兼顾连续梯度的信息利用率与离散选择的决策安全性。

## 局限性与未来方向
- **局限性**：Gumbel 变量引入的随机性可能导致优化方差；对超参数（如温度衰减、窗口大小）较敏感；当前仅验证文本分类与攻击任务，跨领域适应性待确认；依赖白盒访问（虽对开源模型友好，但对闭源商业模型受限）。
- **未来方向**：扩展至更多 NLP/多模态任务；研究优化后提示向闭源商业模型的跨模型迁移能力；探索降低 Gumbel 随机性方差的技术；结合自动超参搜索提升工程易用性。

## 研究启发与可借鉴点
1. **恒等 Straight-Through 代理梯度**：在离散选择场景下，用 `identity` 替代 `softmax` 作为反向传播代理往往比理论上的“更平滑”函数更稳定，值得在 token/Embedding 混合优化任务中复用。
2. **滑动窗口替代候选集全量保存**：将历史梯度/候选信息压缩为定长队列并按时间淘汰，可在几乎不损失引导效果的前提下大幅降低显存占用，适用于长提示或批量化优化。
3. **连续优化 + 确定性贪婪接受**：在基于梯度的离散搜索中嵌入“仅当损失下降才更新”的硬性门槛，可有效抑制非凸优化中的 catastropic 波动，该策略可迁移至其他黑盒/白盒混合优化问题。
4. **Token 长度与数据复杂度的匹配分析**：系统性地扫描 5/10/20 token 揭示了不同数据集的最佳容量边界，为后续低资源场景下的提示压缩与部署提供实证参考。

## 关键术语表
**SGCG (Soft Greedy Coordinate Gradient)**：本文提出的提示优化框架，通过连续概率分布建模与贪婪接受机制联合优化单个提示序列。
**Gumbel-Max Trick**：将分类分布采样重参数化为可微形式的技术，常用于将离散选择融入梯度优化流程。
**Straight-Through Estimator (ST)**：前向使用离散函数、反向使用可微近似函数的梯度估计技巧；本文选用恒等函数作为代理。
**Sliding Window**：仅保留最近 $K$ 次优化结果的缓存机制，用于在控制显存的同时保留历史梯度指引。
**Hard / Soft Prompt**：分别指离散 token 序列提示与连续概率/嵌入空间提示，前者可解释性强但优化困难，后者平滑但需取整。
**Few-shot Prompting**：仅使用极少量标注样本（本文每类 16 个）驱动的提示优化设定，贴近实际低资源应用场景。

## 可复现要素
- **数据集**：SST-2、IMDB、Amazon、Yelp、DBPedia、AGNews（均为公开数据集，可从原论文链接或 HuggingFace 获取）
- **代码/权重**：论文明确声明 “We will release the code before the conference.”，项目主页指向 ACL Anthology 附录链接
- **关键超参**：学习率 $\alpha$、温度 $\tau$（初始 1，指数衰减至 0.1）、滑动窗口大小 $k$（等于提示长度 $n$）、Weight Decay $10^{-3}$、Few-shot 样本数 16/类、实验重复 5 次取平均
