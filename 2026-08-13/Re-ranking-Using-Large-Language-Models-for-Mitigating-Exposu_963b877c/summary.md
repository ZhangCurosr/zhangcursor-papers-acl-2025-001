---
title: "Re-ranking-Using-Large-Language-Models-for-Mitigating-Exposu"
source: https://aclanthology.org/2025.acl-long.44.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:18:56"
field: "推荐系统安全与内容治理"
keywords: ["内容安全", "大语言模型", "排序重排", "有害内容缓解", "零样本学习", "社媒推荐系统"]
innovations: ["基于LLM的pairwise偏好重排方法，零样本/少样本即可降低有害内容暴露", "提出PP-k和EWN两个新型评估指标，衡量排序质量与有害内容推迟效果", "利用RoBERTA嵌入+K-Means聚类选取代表性ICL示例，避免示例选择偏差"]
benchmarks: ["YouTube Harm Dataset", "Jigsaw Toxicity Dataset", "D-Lab Hate Speech Dataset"]
---

# 论文速读：Re-ranking-Using-Large-Language-Models-for-Mitigating-Exposu

## 一句话总结
本文提出一种基于大语言模型（LLM）的零样本/少样本排序重排方法，通过将推荐序列中的有害内容相对降序排列，在不依赖大量标注数据的前提下有效降低用户在社交媒体中暴露于有害内容的风险，且在多项指标上显著优于 Perspective API 和 OpenAI Moderation API 等商业分类器基线。

## 研究问题与动机
- 社交媒体推荐算法为最大化用户参与度，可能无意中引导用户接触有害内容（如进食障碍、自杀倾向、虚假信息、诈骗等），引发严重的心理健康和社会问题。
- 现有基于分类器的内容审核方法面临两大挑战：（1）需要大量人工标注数据进行训练，扩展性差；（2）有害内容类型随时间动态演化，分类器难以泛化到新形态的有害内容（concept drift）。
- 直接删除有害内容会影响透明度和表达自由，而通过相对降序（downranking）而非完全屏蔽的方式，能在最小化暴露的同时保留内容多样性。
- 现有工作多针对单一有害类型（如仇恨言论或虚假新闻），缺乏一个统一、系统、跨多种有害类型的通用缓解框架。

## 核心贡献（创新点）
- **提出 LLM 驱动的 pairwise 偏好重排方法**：将推荐序列中内容两两比较，由 LLM 判断哪一方更有害，据此计算分数并重新排序，与已有工作本质区别在于不依赖分类器的绝对有害/无害判决，而是利用 LLM 的上下文推理能力进行相对排序。
- **验证三种 LLM 提示策略的有效性与对比**：对比零样本（Zero-Shot）、零样本+提示工程（Zero-Shot + PE）和少样本 ICL（Few-Shot ICL）三种设置，发现即使在零样本场景下也显著优于商业分类器基线，而 Few-Shot ICL 效果最佳。
- **提出两个新型评估指标 PP-k 与 EWN**：Per-Pref-k（PPk）衡量用户需消费多少比例内容才能遇到第 k 个有害项；Exponentially Weighted Normalization（EWN）通过指数衰减权重对序列排序质量进行归一化评估（[0,1]，越高越好），弥补了传统指标对有害比例的依赖。
- **系统验证跨数据集、跨 LLM 的泛化能力**：在 YouTube 多类有害数据集、Jigsaw 毒性数据集和 D-Lab 仇恨言论数据集上，使用 GPT-3.5-Turbo、Mistral-7B-Instruct-v0.2 和 Llama2-13B 进行实验，证明方法的通用性和开源模型的可行性。
- **揭示 ICL 示例数量与性能的倒U关系**：发现增加 ICL 示例数量并不必然提升性能，N=4 时 EWN 最高，更多示例反而可能因过拟合或偏差引入导致性能下降。

## 方法详解
- **问题形式化**：给定内容序列 $X = \{x_i\}_{i=1}^{n}$，其中 $p$ 个为无害、$n-p$ 个为有害，利用二元函数 $\rho: X \to \{0,1\}$ 标记有害性，目标是用 LLM $\mathcal{L}$ 将 $X$ 变换为 $X^*$ 以最小化有害内容暴露。
- **Pairwise 偏好排序算法**（Algorithm 1）：枚举所有内容对 $(x_i, x_j)$，用含偏好约束 $\mathcal{C}$ 的提示查询 LLM 判断哪一方更有害；若 $x_i$ 更有害则 score[$x_i$] += 1，若 $x_j$ 更有害则 score[$x_j$] += 1，两者均无害则跳过；最终按分数升序排列得到重排序列 $X^*$。关键修改在于：当双方均为无害时不加分，避免非有害内容被错误降序。
- **三种偏好约束策略**：
  - **Zero-Shot**：仅要求 LLM 判断两内容中哪个更有害，不提供有害定义，依赖 LLM 预训练知识。
  - **Zero-Shot + Prompt Engineering**：在提示中显式定义六类有害内容（信息危害、仇恨与骚扰、成瘾、点击诱饵、性相关、身体危害）并举例，再进行判断。
  - **Few-Shot ICL**：提供代表性有害内容示例，利用 In-Context Learning 让 LLM 从示例中学习有害特征；示例通过 RoBERTA 嵌入 + K-Means 聚类选取各类簇中距质心最近的代表样本，以避免偏向某一类有害内容。
- **新评估指标**：
  - **PP-k**：$PP_k = \frac{\min\{m \mid \sum_{i=1}^{m} \rho(x_i) = k\}}{n}$，表示遇到第 k 个有害项前可消费的内容比例。
  - **EWN**：对排名分配指数衰减权重 $2^{-i}$，在最优（全无害在前）和最差（全有害在前）排序之间归一化：$\text{EWN} = \frac{\sum_{i=1}^{n}\{2^{-i}\cdot(1-\rho(x_i))\} - (2^{-p}-2^{-n})}{(1-2^{p-n})\cdot(1-2^{-p})}$，值域 [0,1]，1 表示最优排序。
  - **TP-k**（Top-Pref-k）：前 k 个内容中无害内容的比例。

## 实验与结果
- **数据集**：
  - **YouTube Harm Dataset**（主数据集）：19,422 条视频，9,832 条有害、2,679 条无害，覆盖 6 类有害；使用视频描述作为 LLM 输入。
  - **Jigsaw Toxicity Dataset**：1.8M+ 条评论，标注毒性。
  - **D-Lab Hate Speech Dataset**：50,070 条社交媒体帖子，标注仇恨言论。
  - 序列构造：从数据中均匀无放回采样生成长度 n=20 的序列，共 m=100 条序列；YouTube 数据集测试 10%~50% 有害比例，其余固定 30%。
- **基线**：OpenAI Moderation API（取最高类别得分）、Perspective API（使用毒性分数排序）。
- **LLM**：GPT-3.5-Turbo（主要）、Mistral-7B-Instruct-v0.2、Llama2-13B（均在 NVIDIA RTX A6000 本地运行开源模型）。
- **主要结果**（YouTube 数据集，30% 有害比例）：
  - **TP5**：Original 0.727 → OpenAI Moderation 0.812 → Perspective 0.780 → Zero-Shot 0.854 → Zero-Shot + PE 0.869 → **Few-Shot ICL 0.872**（相对 Original 提升 20.0%）。
  - **EWN**：Original 0.710 → OpenAI Moderation 0.760 → Perspective 0.738 → Zero-Shot 0.842 → Zero-Shot + PE 0.850 → **Few-Shot ICL 0.864**（相对 Original 提升 21.7%）。
  - 在 10%~50% 各有害比例下，所有 LLM 配置均全面优于两个商业基线；有害比例越高，LLM 方法优势越明显（50% 时 Zero-Shot + PE 相对 Original EWN 提升 51.3%）。
- **跨数据集**：D-Lab 数据集上 Zero-Shot 达到完美 TP10=1.000、EWN=0.99993；Jigsaw 数据集上 Perspective API 异常高（疑似测试集泄露），LLM 方法仍表现相当。
- **跨模型**：Mistral-7B EWN（Few-Shot ICL）达 0.824，落后 GPT-3.5 约 10%，但显著优于 Llama2-13B（EWN=0.684），证明轻量开源模型具备实用潜力。

## 相关工作脉络
- **Qin et al. (2024)**：提出 LLM 用于文本排序的 pairwise ranking 方法，本文借鉴其排序框架但修改了评分函数（避免无害内容间相互比较导致错误降序），并首次将其应用于有害内容缓解场景。
- **Perspective API / OpenAI Moderation API**：工业级分类器基线，依赖大量标注训练，本文证明 LLM 零样本重排在多项指标上全面超越，且无需任务特定训练。
- **Celis et al. (2019); Ovadya & Thorburn (2023)**：基于排序干预的内容治理方法，本文扩展了排序干预的思路，利用 LLM 推理而非预设规则实现动态降序。
- **Gupta et al. (2023)**：提出基于 BertScore 的 ICL 示例选择方法，本文借鉴其思想但改为 RoBERTA 嵌入 + K-Means 聚类选取代表样本，适配重排场景的多元有害分布。
- **Liu et al. (2024); Bonagiri et al. (2025)**：将 LLM 应用于虚假新闻/有害内容检测，属分类范式；本文聚焦于排序重排而非分类，关注相对有害程度而非绝对判定。
- **Jo & Wojcieszak (2025)**：提供 YouTube 多类有害数据集（MetaHarm），本文在此基础上构建序列并验证重排方法的有效性。

## 局限性与未来方向
- **成本与延迟**：LLM 调用在平台规模下耗时和基础设施成本较高，尽管开源轻量模型可缓解，但仍需进一步优化。
- **模态限制**：仅使用文本输入（视频标题、转录、描述），未利用视频帧等视觉信息，多模态扩展为未来方向。
- **LLM 自身鲁棒性**：LLM 在下游任务中可能存在鲁棒性问题（如对抗攻击、偏见），且 proprietary 模型的训练数据时效性限制其对新兴有害类型的识别。
- **过度审核风险**：LLM 可能反映训练数据中的偏见，导致少数观点或争议性内容被不当降序，需透明准则和人工审核机制保障言论自由。
- **未来方向**：扩展到多模态 LLM、探索其他内容排序优化目标（如提升 civic participation、mental well-being）。

## 研究启发与可借鉴点
- **Pairwise 重排替代分类**：将有害内容治理从"分类-删除"范式转向"相对排序-降序"范式，避免误杀同时保留内容多样性，可迁移至任何需权衡曝光与风险的排序场景。
- **ICL 示例选择的聚类策略**：用 RoBERTA 嵌入 + K-Means 选取代表性示例，避免 ICL 偏向某类有害内容，该方法可复用于其他 ICL 应用的示例筛选。
- **EWN 指标的归一化设计思路**：通过比较当前排序与理论最优/最差排序进行归一化，使得不同有害比例和序列长度的评估可比，该思路可用于其他排序质量评估任务。
- **开源模型的可行性验证**：Mistral-7B 在性能上接近 GPT-3.5 且成本低，为数据敏感场景（隐私保护、不依赖第三方 API）提供了实用部署路径。
- **创新评估指标的双重视角**：PP-k 关注"遇到有害内容的深度"，TP-k 关注"前排内容的无害比例"，EWN 提供全局排序质量，三者互补，为后续研究提供了完整的评估框架参考。

## 关键术语表
- **Pairwise Re-ranking**：两两比较内容并据此重新排序的方法，本文核心策略，利用 LLM 判断成对内容中哪个更有害。
- **In-Context Learning (ICL)**：通过在提示中提供示例让 LLM 在不更新参数的情况下学习任务特征，本文使用 K-Means 聚类选取代表性有害示例。
- **Concept Drift**：数据分布随时间变化导致模型性能下降的问题，传统分类器需重新标注训练，而本文 LLM 方法因其预训练知识无需持续重训即可泛化。
- **Per-Pref-k (PPk)**：新用户定义的评估指标，表示需消费序列中多少比例内容才能遇到第 k 个有害项，值越大说明有害内容被有效推迟。
- **Exponentially Weighted Normalization (EWN)**：新指标，对排名分配指数衰减权重并归一化到 [0,1]，衡量序列整体排序质量，可跨不同有害比例公平比较。
- **Top-Pref-k (TPk)**：前 k 个推荐内容中无害内容所占比例，反映用户初期接触到的内容质量。
- **Downranking**：将有害内容降至序列末尾而非完全删除的策略，平衡内容安全与表达自由。
- **Preference Constraints (C)**：提示中指定的有害内容定义或示例集合，决定 LLM 如何进行 pairwise 判断，有三种设置（Zero-Shot / PE / ICL）。

## 可复现要素
- **数据集**：YouTube Harm Dataset（公开，Jo & Wojcieszak 2025）、Jigsaw Toxicity Dataset（公开，Kaggle）、D-Lab Hate Speech Dataset（公开，HuggingFace）；代码和数据已开源。
- **代码**：https://github.com/rvoak/harm-ranking-llm/（论文附录 E 明确声明）
- **权重**：使用 GPT-3.5-Turbo（API）、Mistral-7B-Instruct-v0.2 和 Llama2-13B（本地运行）
- **关键超参**：序列长度 n=20，序列数 m=100，有害比例 10%~50%（YouTube）/ 30%（其余）；ICL 示例数 N ∈ {4, 8, 12, 16, 20}；K-Means 聚类数 K=N
- **硬件**：NVIDIA RTX A6000 GPU，256GB RAM
