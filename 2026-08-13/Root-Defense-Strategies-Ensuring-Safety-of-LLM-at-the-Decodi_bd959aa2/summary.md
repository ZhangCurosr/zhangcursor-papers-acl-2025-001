---
title: "Root-Defense-Strategies-Ensuring-Safety-of-LLM-at-the-Decodi"
source: https://aclanthology.org/2025.acl-long.97.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:20:04"
---

# 论文速读：Root-Defense-Strategies-Ensuring-Safety-of-LLM-at-the-Decodi

## 一句话总结
本文提出了解码级安全防御框架 RDS，通过轻量分类器在自回归采样过程中逐步评估并纠正候选 token 的危害性，并结合投机解码加速隐状态预测；该方法在不重新训练基础模型、不显著损害有用性的前提下，将有害查询合规率降至极低水平，同时将推理速度提升至基线方法的 2.12x~3.09x。

## 研究问题与动机
- **Prefill-level 防御易被绕过且过度防御**：现有输入侧防御（如安全 System Prompt、DRO）仅依赖用户输入进行单次判别，易被 GCG/Auto-DAN 等攻击绕过，且高误拒率严重损害模型有用性。
- **Output-level 防御缺乏过程视角**：SafeDecoding、Self-Examination 等“先生成后判决”方法仅做单点评估，无法感知 prefill 与已生成 token 的上下文关联，导致假阳性频发（如 Xstest 拒答率高达 64%~100%）。
- **解码级判别能力未被系统利用**：前期实验发现 LLM 无法在单步硬分类无害/有害 token，但能基于多步隐状态的分布差异逐步识别危害，这为“逐步纠正而非直接拒绝”的防御提供了理论依据。
- **安全与推理效率难以兼顾**：多数安全防御需额外前向计算或引入独立安全模型，显著拖慢生成速度；本文希望设计一种无需重训练主模型、且能加速的根因防御机制。

## 核心贡献（创新点）
1. **验证并量化 LLM 在解码阶段的危害判别能力**：通过 PCA 可视化与分类器训练揭示 LLM 能基于多步隐状态分布差异识别 token 危害。与既往仅关注输入表示或输出后验过滤的研究相比，本文首次将判别能力前移至自回归采样的每一步。
2. **提出 RDS 逐步骤安全生成框架**：在 top-k 采样阶段集成轻量分类器对候选 token 计算危害分数并重排序，优先选择低风险 token。与 SafeDecoding 等单次判决方法相比，该方法以“逐步纠正”替代“直接拒绝”，大幅降低良性查询误拒率。
3. **引入 EAGLE-inspired 投机解码头加速推理**：利用预测头直接从当前隐状态与候选 token 嵌入预测下一步隐状态，替代多层 Transformer 前向计算，实现 2.12x~3.09x 的加速。与纯加速类投机解码不同，本文将其与安全防御深度耦合，兼顾吞吐与根因安全。
4. **无需重训练主模型的即插即用范式**：仅外接可训练分类器与投机头，在五个不同安全对齐程度的开源模型上实现零参数修改的安全增强。与 DRO 等需优化 prompt embedding 或重新训练模型的方法相比，显著降低工程成本与灾难性遗忘风险。

## 方法详解
- **问题形式化**：设第 $t_i$ 步解码为 $x_i = [x_{i-1}; \max(\mathbb{C}_i)]$，其中 $\mathbb{C}_i = f(\mathbb{I}_i, x_{i-1})$ 为由分类器 $f$ 计算出的候选 token 安全得分集合，目标是在从第 1 步到第 $N$ 步的每一步采样均保证安全。
- **Step-by-step Safe Generation**：
  1. 对前一步隐状态 $\mathbf{h}_{i-1}$ 经 LM Head 得到 logits，取 top-k 获得候选集 $\mathbb{I}_i$ 与对应 logits $\mathbb{V}_i$。
  2. 将当前步拼接隐状态 $\mathbf{h}_i^k$ 输入分类器：先计算查询隐状态均值 $\mathbf{u} = \frac{1}{n}\sum_{q=1}^n \mathbf{h}^q$，再做 PCA 投影 $\mathbf{m}_k = \mathbf{V}^T(\mathbf{h}_i^k - \mathbf{u})$，最后经可训练参数得危害分数 $c_k = \mathbf{W}^T\mathbf{m}_k + \mathbf{b}$。
  3. 在 $\mathbb{C}_i$ 中选取 $c_k$ 最低（最安全）的 token 作为当前步输出 $x_i$。
- **Hidden State Prediction（投机解码）**：
  - 为加速隐状态计算，引入类 EAGLE 的 `EAGLE_Head`（全连接层 + 单层解码器）直接预测候选 token 隐状态：$\mathbf{h}_i^k = \text{EAGLE\_Head}(\mathbf{h}_{i-1}, \mathbf{e}_k)$，避免完整 Transformer 堆叠的前向传播。
  - 整体流程等价于一个 `Draft_Model`：$x_N = \text{Draft_Model}(\mathbf{h}_0)$，无需对主模型进行任何微调。
- **分类器训练**：
  - 训练集为自定义的 Custom 数据集（100 有害 + 100 良性查询），标签 $y_i=1$ 为有害、$y_i=0$ 为良性。
  - 损失函数采用二元交叉熵：$\mathcal{L} = -\frac{1}{n}\sum_{q=1}^n [y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)]$，通过梯度下降拟合 $\mathbf{V}, \mathbf{W}, \mathbf{b}$。

## 实验与结果
- **数据集与基线**：有害基准 HEx-PHI、AdvBench、MaliciousInstruct；良性基准 Held-out、Xstest；有用性评估 Just-Eval。基线包括 Safety Prompt、Self
