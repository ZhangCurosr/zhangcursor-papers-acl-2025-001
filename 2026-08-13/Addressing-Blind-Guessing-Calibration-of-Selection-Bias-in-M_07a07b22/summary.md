---
title: "Addressing-Blind-Guessing-Calibration-of-Selection-Bias-in-M"
source: https://aclanthology.org/2025.acl-long.162.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:50:15"
field: "多模态大模型评估与公平性"
keywords: ["selection bias", "video-language models", "multiple-choice question answering", "debiasing", "fairness evaluation", "post-processing calibration"]
innovations: ["首次系统分析VLM在视频MCQA中的选择偏差", "提出基于任务分解的后处理去偏方法BOLD", "去偏同时提升准确率与公平性指标"]
benchmarks: ["NExT-QA", "STAR", "Perception Test", "Video-MME"]
---

# 论文速读：Addressing-Blind-Guessing-Calibration-of-Selection-Bias-in-Multiple-Choice-Question-Answering-by-Video-Language-Models

## 一句话总结
本文首次系统研究了视频语言模型（VLMs）在多选题问答（MCQA）任务中的选择偏差（selection bias），提出了一种基于任务分解的后处理校准方法BOLD，通过移除视频、问题或答案组件来估计偏差向量，并从观测分布中减去先验偏差，从而在减少盲猜的同时提升模型的整体准确率和F1分数。

## 研究问题与动机
- **核心问题**：VLM在MCQA任务中普遍存在选择偏差，即模型倾向于基于答案位置（如选项顺序）而非真实内容理解来选择答案，这严重影响了评估的公平性和可靠性。
- **现有方法不足**：
  1. LLM领域的去偏方法（如PriDe的shuffle策略）未适配视频-文本模态；
  2. 图片-文本领域的工作（如Zhang et al., 2024）仅处理单模态偏差，无法覆盖视频任务中视觉、问题、答案三组件的交互偏差；
  3. 数据增强方法（如Wang et al., 2024a）需大规模扩充数据集，成本高昂；
  4. 基于置信度的校准（如Pezeshkpour & Hruschka, 2023）与偏差无稳定相关性，鲁棒性不足。
- **研究空白**：此前尚无针对视频MCQA中选择偏差的专门研究，VLM的复杂时空推理特性可能引入独特的偏差模式。

## 核心贡献（创新点）
1. **首次系统分析VLM的选择偏差**：通过11种数据集修改（如Empty Frames、All Identical Answers等），定位了偏差在视频、问题、答案三维度上的分布特征，揭示了VLM对答案位置的过度依赖。
2. **提出BOLD分解校准方法**：将偏差视为由视频、问题、答案三个投影平面构成的向量，通过移除关键组件构造"无定义任务"来估计先验偏差，再用概率减法去偏，无需重新训练模型。
3. **适配公平性偏差指标到视频MCQA**：引入F1_std、Recall_std、JS_std等标准差指标，量化模型对不同答案选项的不公平处理，弥补了原有评估范式的不足。
4. **去偏同时提升性能**：实验表明BOLD和Weighted_BOLD在降低选择偏差的同时，显著提高了Accuracy和F1_Mean，证明去偏能有效抑制"盲猜"行为。

## 方法详解
- **任务分解框架**：将MCQA任务T分解为video、question、answer options三个独立组件，移除任一组件后任务变为"无定义"（ill-defined），此时无偏模型应均匀随机选择选项。
- **三种攻击（解构）**：
  - $A_{v=0}$：将所有帧替换为空帧（移除视频信息）
  - $A_{q=0}$：将问题替换为空字符串（移除问题信息）
  - $A_{o=0}$：将所有答案选项替换为ID（移除答案内容）
- **先验偏差估计**（公式3）：在无定义任务中，观测分布$P_o$即为先验偏差$P_p$，因为去偏分布$P_d$应为均匀分布。
- **全局先验聚合**（公式2）：对$K$个样本应用三种攻击，累加各攻击下的先验分布，经softmax得到样本级先验$\tilde{P}_p(d_i|T)$，再平均得全局先验$\tilde{P}_p(d_i)$。
- **去偏公式**（公式4）：$P_d(d_i|T) = \text{softmax}(\log P_o(d_i|T) - \log \tilde{P}_p(d_i))$，从观测概率中减去先验偏差。
- **Weighted_BOLD扩展**（公式5）：引入权重$w_j$对三种攻击先验加权求和，通过5折交叉验证和COBYLA优化器求解最优权重，允许负权重以探索完整线性子空间。
- **评估指标**：Accuracy、F1_Mean衡量性能；Recall_std、F1_std、JS_std（Jensen-Shannon距离标准差）衡量公平性/偏差程度。

## 实验与结果
- **数据集**：NExT-QA（8564 QA对）、NExT-GQA、STAR（7098 QA对）、Perception Test（7656 QA对）、Video-MME（2700 QA对），覆盖因果、时序、描述、情境推理等多类视频理解任务。
- **模型**：Video-LLaMA、Video-LLaVA、SeViLA三个主流VLM架构。
- **关键结果**（k=0.5, 正权重配置）：
  - **Video-LLaVA在STAR上提升最大**：Accuracy从34.77%提升至37.53%（↑7.94%），F1_Mean从31.83%提升至36.18%（↑13.69%），Recall_std下降30.15%。
  - **Video-LLaMA在Perception Test上**：F1_Mean提升5.86%，Recall_std下降19.45%。
  - **SeViLA整体偏差最小**：因其训练目标接近均匀分布，去偏对其性能影响较小（NExT-QA上仅↑0.02%）。
- **核心结论**：去偏不仅降低了各选项间的不公平性（Recall_std、F1_std、JS_std全面下降），还提升了整体推理能力，证明模型之前确实依赖位置捷径而非内容理解。

## 相关工作脉络
1. **Zheng et al. (2024a)**：提出PriDe方法，通过shuffle分离LLM的先验偏差和无偏结果；本文与之区别在于不依赖shuffle，而是将偏差视为三维向量投影进行分解校准。
2. **Zhang et al. (2024)**：针对图片-文本MCQA的去偏工作，通过切断视觉输入识别单模态偏差；本文扩展到视频-文本任务，并将偏差分解为视频、问题、答案三个投影平面。
3. **Wang et al. (2024a) / Liu et al. (2023)**：通过重排选项和数据增强缓解偏差；本文将其改造为分析工具而非解决方案，通过组件移除定位偏差源头。
4. **Pezeshkpour & Hruschka (2023)**：基于模型不确定性重分配概率进行校准；本文指出该方法不够鲁棒，因为偏差与模型置信度无稳定相关性。
5. **Balepur et al. (2024a)**：研究LLM在无问题情况下的答案推断能力；本文沿用其Empty Questions修改用于偏差分析。

## 局限性与未来方向
- **分解不穷尽**：仅移除单一组件可能遗漏更优的偏差方向，未来可探索 latent space 中的更精细分解。
- **均匀分布假设的局限**：JS散度假设无先验信息时分布应均匀，但若测试数据本身答案分布不均，可能低估某些因素的重要性。
- **监督去偏的可能性**：本文追求无监督方案，但未来可尝试用真实标签配合COBYLA优化获得更优结果。
- **评估覆盖不足**：仅在原始数据集上验证了性能提升，未在11种修改场景下系统测试，未来需更细粒度的场景化评估。
- **模型泛化性**：主要针对三个VLM架构，需验证在其他架构（如Video-LLaMA2、VideoChat等）上的适用性。

## 研究启发与可借鉴点
1. **任务分解分析法可迁移**：通过构造"无定义任务"（移除关键组件）来暴露模型捷径行为的思路，可应用于其他多模态任务（如图像MCQA、多模态对话）的偏差诊断。
2. **概率减法去偏框架通用**：公式4所示的先验减法（log-space subtraction）简洁高效，可推广至文本、表格等其他MCQA场景的去偏。
3. **Fairness指标适配**：将公平性领域的标准差指标（Recall_std、JS_std）引入VLM评估，为多维度公平性量化提供了可复用范式。
4. **结合团队方向的机会**：若团队研究视频理解或多模态推理，可借鉴此方法诊断自身模型的偏差模式，或在训练阶段引入类似的组件扰动正则化。
5. **Weighted_BOLD的优化思路**：使用COBYLA进行无梯度权重优化，避免了对偏差向量的显式建模，适用于黑盒模型的去偏场景。

## 关键术语表
- **Selection Bias（选择偏差）**：模型基于答案位置、长度等非内容因素而非问题语义进行选择的系统性倾向。
- **Blind Guessing（盲猜）**：模型在缺乏有效信息时依赖位置偏好随机猜测的行为，是选择偏差的直接表现。
- **Ill-defined Task（无定义任务）**：因移除关键组件（视频/问题/答案）导致任务无法正确解答的构造场景，用于暴露偏差。
- **BOLD（Bias Optimisation Leveraging Decomposition）**：本文提出的基于任务分解的去偏校准方法。
- **Weighted_BOLD**：BOLD的扩展，引入可学习权重对三种攻击先验进行加权聚合。
- **Recall_std / F1_std**：跨选项的召回率/F1分数标准差，衡量模型对各选项的处理公平性。
- **JS_std**：预测分布与均匀分布之间Jensen-Shannon距离的标准差，量化偏差一致性。
- **COBYLA**：Constrained Optimization BY Linear Approximation，一种无梯度约束优化算法，用于Weighted_BOLD的权重搜索。

## 可复现要素
- **数据集**：NExT-QA、NExT-GQA、STAR、Perception Test、Video-MME均为公开数据集。
- **代码/权重**：论文未提供开源代码，但使用的模型（Video-LLaMA、Video-LLaVA、SeViLA）均为开源模型。
- **关键超参**：采样系数k=0.5（用于全局先验估计的样本比例）；权重约束0≤w_i≤1（正权重）或|w_i|≤1（允许负权重）；5折交叉验证；temperature=0.8（Video-LLaMA）、1（Video-LLaVA）；beam=2；最大生成尝试30次。
