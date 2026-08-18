---
title: "Position-aware-Automatic-Circuit-Discovery"
source: https://aclanthology.org/2025.acl-long.141.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:14:30"
field: "可解释AI（Mechanistic Interpretability）"
keywords: ["Mechanistic Interpretability", "Circuit Discovery", "Position-aware Analysis", "Edge Attribution Patching", "Dataset Schema", "Automated Pipeline"]
innovations: ["提出位置感知的边缘属性补丁（PEAP）方法，首次系统性地将位置区分引入自动电路发现，显著提升电路的faithfulness和精确度。", "引入数据集模式（Schema）概念及基于LLM的自动生成与saliency引导的增强管道，使位置感知电路发现能有效处理变长、非模板化的真实输入。"]
benchmarks: ["IOI (Indirect Object Identification)", "Greater-Than", "Winobias"]
---

# 论文速读：Position-aware-Automatic-Circuit-Discovery

## 一句话总结
本文提出了**位置感知的自动电路发现方法（PEAP）**，通过区分token位置并引入数据集模式（schema），实现了在可变长度输入上自动发现更精准、更精简的语言模型因果电路，其效果可与人工设计的电路媲美。

## 研究问题与动机
- **核心问题**：现有自动电路发现方法（如EAP）假设电路是**位置不变（position-invariant）**的，即认为模型组件（如注意力头）对所有输入token位置的贡献相同，无法捕捉跨位置交互或随位置变化的机制。
- **现有方法不足**：
  1. **抵消效应（Cancellation）**：将不同位置的边缘重要性分数直接求和，正负抵消可能导致真正重要的边缘被低估（低召回率）。
  2. **重要性高估（Overestimation）**：聚合后偏向那些在多个位置都有小幅影响、而非在少数位置有决定性影响的边缘（低精确率）。
  3. **无法处理变长输入**：现有位置感知方法多局限于模板化、严格对齐的数据集，难以推广到真实场景中长度各异的示例。

## 核心贡献（创新点）
1. **提出PEAP方法**：将边缘属性补丁（EAP）扩展至位置敏感版本，分别评估每个token位置上的跨位置边缘重要性，避免跨位置分数聚合带来的抵消和高估问题。*与Syed等（2023）工作的本质区别在于：首次系统性地为自动电路发现引入位置区分能力，并证明了其在提升电路faithfulness和减小电路规模上的优势。*
2. **引入数据集模式（Schema）概念**：定义一种将输入中语义相似的token span在不同示例间进行对齐的框架，使得位置感知电路发现能够处理变长输入。*与现有基于全对齐（full alignment）或部分对齐（partial alignment）的方法的本质区别在于：通过模式抽象，实现了更灵活、更通用的位置对齐，且该过程可由LLM自动化完成。*
3. **开发全自动LLM驱动的模式生成与应用管道**：利用大语言模型自动生成和匹配模式，并结合模型输入的saliency score增强生成过程，使得自动发现的电路在faithfulness上与人工设计的电路相当甚至更优。*与人工设计模式或纯LLM生成模式的本质区别在于：引入了模型计算重要性信息（mask）来引导模式生成，提升了模式的内在质量和外在效用。*

## 方法详解
1. **位置感知边缘属性补丁（PEAP）**：
   - **基础**：沿用EAP的间接效应（IE）线性近似公式 $g(e) = (z_u^* - z_u)^\top \nabla_v M(x)$，但将其应用于**跨位置注意力边**。
   - **核心扩展**：在Transformer中，位置$t$的注意力头$h_t^i$通过$q, k, v$向量与之前所有位置$t' \leq t$的头相连。PEAP分别计算对这些$q, k, v$边缘进行patching后对目标指标$M$的影响近似值（见公式4-6）。
   - **电路构建**：计算所有边缘（包括位置和跨位置）的归因分数后，采用改进的贪心算法（基于Hanna et al., 2024b）构建电路，确保从最终位置logits到嵌入节点的连通性。

2. **基于Schema的位置感知电路发现**：
   - **Schema定义**：Schema $S$ 将输入序列划分为$k$个有序的语义span（如“Subject”、“Verb”等）。
   - **抽象计算图**：为整个数据集构建一个长度为$k$的抽象计算图$G_S$，每个span对应一个抽象位置。
   - **边缘分数聚合**：通过映射函数$f_S^x$将具体示例$x$中的真实边缘（$e' \in G_x$）映射到抽象边缘（$e \in G_S$），然后在所有示例上平均聚合归因分数：$g_S(e) = \frac{1}{|D|} \sum_{x \in D} \sum_{e' \in f_S^x(e)} g_x(e')$。
   - **Faithfulness评估**：将抽象电路$C_S$映射回每个示例的具体电路$C_x$后进行消融评估。

3. **自动化Schema管道**：
   - **Schema应用**：使用LLM（如Claude 3.5 Sonnet）将schema应用到每个输入示例的token上。
   - **Schema生成**：从数据中采样子集，让LLM生成多个候选schema，再融合为一个统一schema。为提升质量，引入**Saliency-enhanced**方法：计算每个token位置的$saliency score$（输入$\times$梯度），生成mask后作为额外信息提供给LLM，指导其将重要token单独划分为span。

## 实验与结果
- **模型与数据集**：使用**GPT2-small**和**Llama-3-8B**。三个任务：**IOI**（间接宾语识别）、**Greater-Than**（大小比较）、**Winobias**（性别偏见核心指代）。每个任务抽取500个示例用于电路发现，500个用于faithfulness评估。
- **评估基线**：与位置无关的自动方法（Syed et al., 2023的EAP）、人工设计的电路/模式进行对比。
- **主要结果**：
  1. **位置感知优势**：在所有任务和模型上，PEAP发现的位置感知电路在**更小的电路规模**下达到了比位置无关电路**更高或相当的Hard Faithfulness**（见图6）。
  2. **Schema有效性**：对于变长输入，使用手动设计Schema的PEAP同样显著优于无位置信息的基线。
  3. **自动化管道性能**：
     - **LLM+Mask**生成的Schema在**外在质量（电路faithfulness）**上与人工Schema**相当甚至更优**（例如在Llama-3-8B的IOI任务中，LLM+Mask电路faithfulness最高）。
     - **内在质量**：LLM应用Schema的**有效性（Validity）**和**正确率（Correctness）**均很高（见表6，多数任务>95% validity）。
     - **Saliency Mask的增益**：在Greater-Than和Winobias任务中，加入saliency mask的LLM+Mask方法相比纯LLM生成能带来**显著的性能提升**。
  4. **最强结果**：在GPT2-small的Greater-Than任务上，**LLM+Mask Schema**发现的电路在极小规模下就达到了接近100%的Hard Faithfulness，与人工设计电路表现几乎一致。

## 相关工作脉络
1. **Edge Attribution Patching (EAP, Syed et al., 2023)**：本文基础方法，但EAP是位置无关的，本文将其扩展至位置感知。
2. **自动电路发现方法（如Conmy et al., 2023; Hanna et al., 2024b）**：此类方法通常忽略位置信息或仅适用于完全对齐的模板化数据，本文方法更通用且精度更高。
3. **手动电路发现（如Wang et al., 2023的IOI电路）**：本文证明了自动发现的位置感知电路可以逼近甚至超越专家手动设计的效果。
4. **位置感知的节点级定位方法（如Kramár et al., 2024的AtP\*）**：AtP*仅关注节点级定位且未在电路发现中验证，本文首次系统地将位置感知引入**边缘级**自动电路发现。
5. **稀疏特征电路（Sparse Feature Circuits, Marks et al., 2025）**：该方法虽考虑了位置，但仅在模板化数据上验证，且其“边”的定义与本文不同（本文是真实的计算图边），本文更关注边缘级因果机制。
6. **基于SAE/Transcoders的电路发现（Ge et al., 2024）**：与本文目标相关，但方法（PEAP vs. SAEs）、评估数据集（三个多样任务vs.少量示例）和模型规模（GPT2-small和Llama-3-8B）均有显著不同。

## 局限性与未来方向
- **Schema设计的普适原则缺失**：目前尚无明确的一般性原则来指导何种Schema能在不同任务中取得最佳的faithfulness与电路规模权衡。未来需要探索更通用的Schema设计理论。
- **Schema的顺序约束**：当前方法要求所有示例中span必须保持相同顺序，否则无法构建一致的抽象计算图。这限制了对结构更自由的数据的应用。
- **计算成本**：PEAP需要计算大量跨位置边缘的梯度，计算开销较大。虽然使用了近似方法，但在更大模型和更长序列上仍需优化。
- **LLM生成Schema的质量依赖**：尽管自动化管道效果良好，但其质量高度依赖于底层LLM（如Claude 3.5 Sonnet）的能力，且可能存在~10%的无效应用需要过滤。

## 研究启发与可借鉴点
1. **位置信息的重要性**：在 mechanistic interpretability 中，**严格区分token位置**是提升电路发现精确度和召回率的关键。团队在进行电路分析时，应优先考虑位置感知的评估方法，避免简单的跨位置分数聚合。
2. **Schema抽象框架**：**“通过语义span对齐实现位置感知”** 的思想非常有价值。可以借鉴此框架，将其他需要位置对齐的因果发现方法（如节点级的归因）扩展到变长输入场景。
3. **LLM辅助的自动化pipeline**：结合**LLM自动生成+模型内在信号（如saliency mask）引导**的策略，是平衡自动化程度与结果质量的有效范式。可在其他需要结构化先验的 interpretability 任务中尝试类似设计。
4. **评估指标的多样性**：同时使用**Soft Faithfulness**和**Hard Faithfulness**进行互补评估，后者更能反映模型最终行为的准确性。在评估电路时，建议引入行为层面的硬性指标。
5. **跨位置边缘的显式建模**：PEAP对注意力机制中**跨位置边（v, k, q edges）**的显式分解和近似计算，为理解Transformer内部的跨位置信息流动提供了精细的分析工具，值得在后续研究中深入挖掘这些边的语义。

## 关键术语表
- **Circuit（电路）**：模型计算图中执行特定任务的最小子图，由节点（组件）和边（连接）构成。
- **Edge Attribution Patching (EAP)（边缘属性补丁）**：通过线性近似计算直接边缘扰动对目标指标间接效应的自动电路发现方法。
- **Position-aware / Position-invariant（位置感知/位置不变）**：前者指区分不同token位置的分析方法；后者指将所有位置同等对待的假设。
- **Faithfulness（忠实度）**：衡量简化电路（如消融后）保留原模型行为程度的指标，分为Soft（概率分布相似度）和Hard（最终预测一致性）两种。
- **Dataset Schema（数据集模式）**：将数据集中各个示例的输入序列划分为一组有序、语义对齐的token span的框架。
- **Abstract Computation Graph（抽象计算图）**：基于Schema构建的、与具体示例长度无关的通用计算图表示。
- **Saliency Score / Mask（显著性分数/掩码）**：通过输入嵌入与目标指标梯度的乘积来量化每个token位置对模型输出的重要性。
- **Indirect Effect (IE)（间接效应）**：在因果中介分析中，通过干预某条边并观察目标指标变化来衡量该边的因果影响。

## 可复现要素
- **数据集**：IOI（Wang et al., 2023）、Greater-Than（Hanna et al., 2024a）、Winobias（Zhao et al., 2018）。论文未明确声明是否公开，但通常这些基准数据集可公开获取。
- **代码**：论文提及使用 **Transformerlens** 库实现，未明确声明自有代码开源状态。
- **模型**：GPT2-small, Llama-3-8B。
- **关键超参数**：每个任务使用500个示例进行电路发现，500个进行评估；Schema生成采样3组×5个示例；saliency mask阈值设为$1/n$（n为提示长度）。
