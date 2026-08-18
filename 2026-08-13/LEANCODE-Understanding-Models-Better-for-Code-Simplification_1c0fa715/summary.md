---
title: "LEANCODE-Understanding-Models-Better-for-Code-Simplification"
source: https://aclanthology.org/2025.acl-long.78.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:10:29"
field: "代码理解的模型压缩与简化"
keywords: ["code simplification", "attention mechanism", "pre-trained language models", "code search", "code summarization", "token importance"]
innovations: ["提出类别本地注意力平均替代全局平均自注意力，降低 token 重要性评估方差", "区分分类任务（CLS注意力）和序列生成任务（Encoder-Decoder注意力）设计差异化的token重要性度量机制"]
benchmarks: ["CodeSearchNet (Java)", "MRR", "BLEU-4"]
---

# 论文速读：LEANCODE-Understanding-Models-Better-for-Code-Simplification

## 一句话总结
论文提出 LEANCODE，一种基于上下文感知注意力分数的代码简化方法，通过利用 CLS 注意力（分类任务）和 Encoder-Decoder 注意力（序列生成任务）按语句类别计算 token 重要性，在代码搜索和代码摘要任务上显著优于 SOTA 方法 DIETCODE 和 SLIMCODE，同时大幅降低推理时间。

## 研究问题与动机
- **预训练模型计算开销大**：CodeBERT 等模型存在输入 token 长度限制（如 512），且更长输入导致显著的计算成本和时间开销。
- **现有方法 DIETCODE 的不足**：① 使用全局平均自注意力分数评估 token 重要性，忽略了同一 token 在不同上下文（语句类型）中的注意力方差很大；② 分类任务应关注 CLS token 的注意力而非普通自注意力；③ 序列生成任务忽略了 Encoder-Decoder 注意力，仅依赖编码器自注意力。
- **现有方法 SLIMCODE 的不足**：基于人工规则将 token 分为 8 个优先级，颗粒度过粗导致大量 token 优先级相同，且"人类认为重要"不等于"模型认为重要"。
- **核心假设**：模型自身的注意力知识比人工规则更适合作为代码简化的依据，且下游任务直接相关的注意力机制（CLS、Encoder-Decoder）比预训练阶段的自注意力更能反映 token 重要性。

## 核心贡献（创新点）
- **提出上下文感知的类别本地注意力平均（category-local attention average）**：按语句类别分组计算 token 的注意力平均分，替代 DIETCODE 的全局平均自注意力，降低了注意力分数的方差，使重要性评估更贴合具体上下文。
- **针对分类和序列生成任务分别设计注意力机制**：对代码搜索等分类任务，使用 CLS token 的自注意力分数作为 token 重要性指标；对代码摘要等序列生成任务，使用 Encoder-Decoder 注意力分数的最大值作为 token 重要性指标，与下游任务直接关联。
- **提出纯 token 级别的简化算法**：不同于 DIETCODE 两步法（先删语句再删 token），LEANCODE 直接迭代删除单个最低分 token，避免整句删除导致的重要 token 丢失。
- **系统性实验验证与跨模型迁移评估**：在 CodeBERT 和 CodeT5 上验证，代码搜索 MRR 提升最高达 60%，代码摘要 BLEU 提升最高达 29%；并将 CodeT5 生成的简化代码应用于 GPT-4o，验证跨模型迁移能力。

## 方法详解
**问题形式化**：给定代码片段 $d_j = \{t_1, \cdots, t_{n_j}\}$，token 重要性为 $w_i$，删除指示变量 $x_i \in \{0,1\}$，目标是最小化 $\sum w_i x_i$，满足删除 token 总数 $\sum x_i = \mathcal{X} = \text{SimplifiedRatio} \times n_j$。

**Token 重要性计算——类别本地注意力平均**：
$$\mu_t^c = \frac{\sum_{j=1}^{m} \sum_{t \in p_k, p_k \in d_j', L(p_k) \in c} s_t}{n_t^c}$$
其中 $p_k$ 是语句，$L(p_k)$ 是语句类别（共 21 类），$n_t^c$ 是 token $t$ 在类别 $c$ 语句中的出现次数，$s_t$ 为 CLS 注意力或 Encoder-Decoder 注意力分数。相比全局平均，类别本地平均将方差从 $0.55\times\sim5\times$ 降至 $0.1\times\sim156\times$，显著降低了噪声。

**分类任务（代码搜索）使用 CLS 注意力**：
$$s_i = \frac{q_{cls} \cdot k_i}{\sqrt{d}}$$
CLS token 的向量融合了代码与描述的配对信息，其注意力分数直接反映 token 对分类决策的贡献。

**序列生成任务（代码摘要）使用 Encoder-Decoder 注意力**：
$$s_i = \frac{q_t \cdot k_i}{\sqrt{d}}$$
对每个生成的 target token，取输入 token 在所有 decoder 步骤中的最大 Encoder-Decoder 注意力分数作为该 token 的重要性。

**简化算法（Algorithm 1）**：初始化简化数据集副本 → 对每个代码片段，按 $\text{SimplifiedRatio} \times n_j$ 确定删除数量 → 循环选择剩余 token 中分数最低者（记录 index:token）→ 直到删除数量达标 → 返回简化数据集。只删除 token 不删语句。

## 实验与结果
- **数据集**：CodeSearchNet（Java 子集），训练集 908,886（搜索）/ 164,923（摘要）条样本。
- **基线**：DIETCODE（全局自注意力 + 两步简化）和 SLIMCODE（人工规则 8 级优先级）。
- **模型**：CodeBERT、CodeT5（开源模型本地推理）、GPT-4o（API 调用跨模型迁移）。
- **代码搜索（MRR）**：
  - CodeBERT + 50% 删除率：LEANCODE MRR=0.688（降 5.23%），DIETCODE 降 40.9%，SLIMCODE 降 18.18%。LEANCODE 较 DIETCODE 提升最高 60.37%，较 SLIMCODE 提升 15.82%。
  - CodeT5 + 50%：LEANCODE MRR=0.706（降 5.48%），较 DIETCODE 提升 25.84%，较 SLIMCODE 提升 10.14%。
- **代码摘要（BLEU-4）**：
  - CodeBERT + 50%：LEANCODE BLEU=16.24（降 11.01%），DIETCODE 降 30.55%，SLIMCODE 降 29.29%。LEANCODE 较 DIETCODE 提升 29.36%，较 SLIMCODE 提升 27.04%。
  - CodeT5 + 50%：LEANCODE BLEU=18.46（降 10.17%）。
- **推理时间**：50% 删除率下，CodeT5 摘要推理时间降低 40.9%；CodeBERT 搜索推理时间降低 36.59%。
- **剪枝时间**：LEANCODE 约 3–48 分钟（取决于比例和任务），显著快于 DIETCODE（5h59m–9h24m），与 SLIMCODE（17–21 分钟）可比较。
- **跨模型迁移（GPT-4o）**：LEANCODE 在代码搜索 Precision 上略优，50% 删除率时 Precision=0.81（仅降 1.22%），总 token 数减少 23.68%。

## 相关工作脉络
- **DIETCODE (Zhang et al., 2022)**：基于全局平均自注意力分数的 token 重要性评估，两步剪枝策略（先删语句再删 token）。本文核心对比对象，本文指出其忽略了 CLS/Encoder-Decoder 注意力与下游任务的直接关联。
- **SLIMCODE (Wang et al., 2024)**：基于 8 级人工优先规则（方法签名最高、符号最低）的模型无关简化方法。本文认为人工规则与模型认知不一致，且颗粒度太粗。
- **Autofocus (Bui et al., 2019)**：通过 GGNN 注意力权重衡量语句相关性来定位关键代码。本文方法不同在于直接利用 Transformer 预训练模型的注意力机制。
- **SIVAND / P2IM**：基于 delta debugging 的程序简化方法，将代码分段后迭代缩小。本文方法是基于注意力权重的 token 级筛选，不依赖增量调试范式。

## 局限性与未来方向
- **编程语言限制**：仅在 Java 上验证，需扩展到 Python、Go、JavaScript 等其他语言。
- **外部有效性**：仅测试了 CodeBERT、CodeT5、GPT-4o 三个模型和代码搜索、摘要两个任务，需扩展到更多模型架构和代码相关任务（如缺陷检测、代码补全）。
- **内部有效性**：推理时间测量受硬件、操作系统等外部因素影响，虽然实验控制了环境但结论仍需谨慎推广。
- **未来方向**：多语言扩展、多任务泛化、更多模型架构验证、探索动态注意力分数的高效近似计算。

## 研究启发与可借鉴点
- **类别本地平均注意力**的思路可有效降低全局平均的方差，适用于其他基于注意力权重的模型解释或压缩任务，可迁移到文本理解、多模态模型等领域。
- **区分预训练注意力与下游任务注意力**：论文明确区分了 MLM/RTD 预训练阶段的自注意力与 fine-tuning 阶段 CLS/Encoder-Decoder 注意力的用途差异，这一分析框架对理解不同注意力机制的价值具有方法论意义。
- **纯 token 级简化策略**避免了语句级删除的信息损失，可推广到其他需要精细粒度的模型压缩场景。
- **跨模型迁移验证**（用 CodeT5 简化后用 GPT-4o 测试）提供了一个评估简化方法普适性的良好范式，值得在后续工作中借鉴。
- **消融替换实验**（用 DIETCODE 的删除算法替换 LEANCODE 的，反之亦然）清晰量化了 token 权重和删除算法各自的贡献，实验设计严谨，可复用于其他方法对比研究。

## 关键术语表
**CLS token**：分类任务中特殊标记 token，其最终层向量包含整个输入序列的聚合信息，用于下游分类预测。
**Category-local attention average**：按语句类别分组计算的 token 注意力平均分，用于替代全局平均以降低方差、提升上下文感知能力。
**Encoder-Decoder attention**：Transformer 解码器在生成 target token 时对编码器输出的注意力权重，反映输入 token 对生成任务的贡献。
**SimplifiedRatio**：代码简化率，表示被删除 token 数占原始 token 总数的百分比。
**MRR (Mean Reciprocal Rank)**：平均倒数排名，代码搜索任务的评价指标，衡量正确代码 snippet 在排序中的位置。
**BLEU-4**：基于 4-gram 重叠的自动评估指标，用于代码摘要任务衡量生成描述与参考描述的一致性。
**Bi-modal mappings**：双模态映射，指代码与文本描述之间的语义对应关系，是代码搜索和摘要任务的核心。

## 可复现要素
- **数据集**：CodeSearchNet（Java 子集），论文声明来源于公开数据集 CodeSearchNet Corpus (Husain et al., 2019)，公开可用。
- **代码/权重**：论文未明确声明代码开源；CodeBERT 和 CodeT5 为开源模型，GPT-4o 通过 API 访问。
- **关键超参**：SimplifiedRatio 取 10%–50% 步长；CodeBERT/CodeT5 使用默认超参数；Adam 优化器学习率 $1\times10^{-5}$（搜索）和 $5\times10^{-5}$（摘要）；注意力分数取自模型最后 Encoder/Decoder 层。
- **硬件**：2× Intel Xeon Gold 2.40GHz CPU + 2× NVIDIA A100 GPU。
