---
title: "BPP-Search-Enhancing-Tree-of-Thought-Reasoning-for-Mathemati"
source: https://aclanthology.org/2025.acl-long.40.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:40:56"
field: "大语言模型推理与运筹优化"
keywords: ["Tree of Thought", "Process Reward Model", "Mathematical Modeling", "Operations Research", "Beam Search", "Preference Model"]
innovations: ["提出BPP-Search算法，将PRM与成对偏好模型结合用于树搜索", "发布包含完整建模过程标注的StructuredOR数据集"]
benchmarks: ["StructuredOR", "NL4OPT", "Mamo-ComplexLP"]
---

# 论文速读：BPP-Search: Enhancing Tree of Thought Reasoning for Mathematical Modeling Problem Solving

## 一句话总结
本文针对运筹学数学建模任务，提出了包含完整过程标注的新数据集 StructuredOR，并设计了 BPP-Search 算法——一种将 Beam Search、过程奖励模型（PRM）与成对偏好算法结合的树形推理方法，以提升大语言模型在复杂建模任务中的准确率与搜索效率。

## 研究问题与动机
1. **现有数据集缺乏过程标注**：当前开源运筹学数据集（如 Mamo-ComplexLP、NL4OPT）多仅提供目标值，缺乏对变量、约束、集合等建模过程的细粒度标注，难以支撑基于强化学习或过程监督的方法。
2. **传统推理策略在弱策略模型下失效**：CoT 仅生成单一路径，Self-Consistency 缺乏验证器易导致误差传播，而 Tree-of-Thought（ToT）虽能生成多个候选解，但缺乏有效机制从叶节点中选出最优解。
3. **PRM 评分不够精细**：基于分类任务训练的 PRM 在推理时需输出连续分数，导致在区分结构相似但细节不同的候选节点时性能下降，单纯增加 Beam Search 宽度反而可能降低准确率。

## 核心贡献（创新点）
1. **发布 StructuredOR 数据集**：提供从自然语言问题到完整结构化建模过程（集合、参数、变量、目标、约束）的细粒度标注，支持 RL 与过程监督训练。
2. **提出 BPP-Search 算法**：将 Beam Search、PRM 与成对偏好模型（Preference Model）相结合，在搜索过程中利用 PRM 剪枝、在最终层利用偏好模型进行精细排序，兼顾效率与准确性。
3. **系统评估多种搜索策略**：对比 Greedy、Epsilon Greedy、Random Greedy 与 Beam Search 等多种启发式搜索与 PRM 结合的效果，验证 BPP-Search 的优越性。
4. **构建并训练专用奖励模型**：基于 Qwen2.5-Math-1.5B 微调得到 PRM 与 Preference Model，为树搜索提供可靠的节点评估与排序信号。

## 方法详解
**1. 结构化建模流程（Tree 结构）**
推理过程被划分为六层树结构：问题（Q）→ 集合与参数（SP）→ 变量（V）→ 目标与约束（OC），每层节点最多扩展 3 个子节点，以平衡搜索空间与计算成本。

**2. 过程奖励模型（PRM）训练**
使用 Qwen2.5-Math-1.5B 进行全参数监督微调，训练数据来源于正确/错误推理路径的分段标注。PRM 输出 logits 后通过 sigmoid 计算得分：
$$S_{PRM} = \frac{1}{1 + e^{-l_{prm}}}$$

**3. BPP-Search 搜索流程**
- **Beam Search 阶段**：在每层使用 PRM 对候选节点评分，保留 top-k 个节点（beam width 设为 2 或 3）。
- **成对偏好排序阶段**：在最后一层候选池中，使用微调的 Preference Model 对所有候选对 $(A, B)$ 计算偏好分数 $S_{PM}(A \succ B)$，并聚合得到每个候选的综合得分：
$$S_{PM}(A) = \frac{1}{n-1} \sum_{j \neq i} S(A \succ X_j)$$
最终选择综合得分最高的候选作为答案。

**4. Random Greedy 变体**
为缓解 PRM 评分不精确问题，引入 Random Greedy 策略：仅筛选得分与最高分差距小于阈值（threshold）的候选，再从中随机选取一个继续搜索，以探索多样性。

## 实验与结果
- **数据集**：StructuredOR（38 测试题）、Mamo-ComplexLP（211 题）、NL4OPT（289 题）。
- **基线**：CoT、Self-Consistency、ToT（Randomly-Chosen、Rethink、Fully-Traverse）。
- **主要结果**（固定策略模型 GPT-4o）：
  - 在 StructuredOR 上，BPP-Search (Width=2) 正确率 **93.3%**，推理步数仅 15 步，显著优于 ToT-Fully-Traverse（79.0%，39 步）与 Greedy Search（83.3%，9 步）。
  - 在 Mamo-ComplexLP 与 NL4OPT 上，BPP-Search 同样取得最高正确率（72.2% 与 80.4%），且推理步数低于 Fully-Traverse。
- **消融实验**：PRM 独立使用时 Beam Width 增大反而降低准确率；引入 Preference Model 后（BPP-Search）性能显著提升，验证了成对排序的有效性。

## 相关工作脉络
1. **Chain-of-Thought / Self-Consistency**：Wei 等（2022）与 Wang 等（2023）提出的单路径与多数投票策略，缺乏过程验证，易受弱模型限制。
2. **Tree-of-Thought**：Yao 等（2023）首次将树搜索引入 LLM 推理，但未解决叶节点选择问题。
3. **Process Reward Model**：Uesato 等（2022）与 Lightman 等（2023）证明过程监督可提升推理性能，但依赖人工标注或 MCTS，成本高。
4. **OR 领域 LLM 研究**：Mamo-ComplexLP（Huang 等，2024）、NL4OPT（Ramamonjison 等，2022）等数据集缺少过程标注，无法直接用于过程级奖励训练。
5. **偏好模型与排序**：本文引入成对偏好模型用于候选排序，区别于传统 ORM/PRM 的单点打分机制，提供更细粒度的比较信号。

## 局限性与未来方向
1. **计算成本与树深度的权衡**：当前实验受限于算力，未充分探索更深/更宽的树结构；扩展树深度可能进一步提升性能但会大幅增加 LLM 调用次数。
2. **LLM 处理大规模数值问题的能力有限**：当问题维度增加时（如三维参数数组），LLM 难以准确记忆和输出大量数值，制约了在复杂实例上的泛化。
3. **数据集规模较小**：StructuredOR 仅包含 124 个实例，虽经数据增强，但覆盖面仍有限，未来需扩展至更多工业场景。

## 研究启发与可借鉴点
1. **PRM + Beam Search 结合**：将过程奖励模型嵌入树搜索的剪枝阶段，可在保证覆盖率的同时显著减少无效展开，适用于其他复杂推理任务。
2. **成对偏好模型用于候选排序**：当单一评分不足以区分相似候选时，引入 pairwise preference 进行相对比较是一种有效且鲁棒的解决方案。
3. **结构化过程标注数据集构建范式**：通过 LLM 生成抽象模型、求解器验证、人工规则校验的流水线，可为其他垂直领域构建高质量过程标注数据集提供参考。
4. **Random Greedy 缓解评分不精确**：在奖励模型置信度不高时，引入随机性并保持多样性，可避免过早陷入局部最优。

## 关键术语表
- **BPP-Search**：本文提出的搜索算法，整合 Beam Search、Process Reward Model 与 Pairwise Preference 的树推理方法。
- **PRM（Process Reward Model）**：对推理过程中的每个中间步骤进行 correctness 评分的奖励模型。
- **ToT（Tree of Thought）**：将推理过程组织为树结构，每个节点代表一个中间状态，允许并行探索多条推理路径。
- **StructuredOR**：本文发布的新数据集，包含完整的运筹学数学建模过程标注（集合、参数、变量、目标、约束）。
- **Pairwise Preference Model**：通过成对比较学习偏好关系，用于在候选集中进行精细排序的奖励模型。
- **Random Greedy**：一种结合贪心选择与随机性的搜索策略，仅在高分候选中随机选取，以平衡利用与探索。

## 可复现要素
- **数据集**：StructuredOR 已在 Huggingface 和 GitHub 公开。
- **代码**：论文未明确提供代码链接，但提及 StructuredOR 在 GitHub 可获取。
- **基座模型**：PRM 与 Preference Model 基于 Qwen2.5-Math-1.5B 微调；策略模型使用 GPT-4o。
- **关键超参**：Beam Search width 设为 2 或 3；Random Greedy 的 threshold 未明确给出；PRM 与 Preference Model 均为全参数 SFT。
