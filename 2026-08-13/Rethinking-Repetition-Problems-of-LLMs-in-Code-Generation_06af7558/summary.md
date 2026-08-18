---
title: "Rethinking-Repetition-Problems-of-LLMs-in-Code-Generation"
source: https://aclanthology.org/2025.acl-long.48.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:06:46"
field: "代码生成与程序合成"
keywords: ["代码生成", "重复问题", "结构性重复", "解码方法", "语法约束", "大语言模型"]
innovations: ["首次形式化定义代码生成中的结构性重复问题并建立评估基准", "提出RPG基于语法的解码方法通过PDA和后缀数组检测并惩罚重复", "构建CodeRepetEval数据集覆盖三种场景评估重复缓解能力"]
benchmarks: ["HumanEval", "MBPP", "HumanEval-ET", "MBPP-ET", "CodeRepetEval"]
---

# 论文速读：Rethinking-Repetition-Problems-of-LLMs-in-Code-Generation

## 一句话总结
本文首次正式定义了代码生成中的**结构性重复（structural repetition）**问题，提出基于语法的解码方法 RPG（Repetition Penalization based on Grammar），通过下推自动机识别语法规则并指数衰减重复关键 token 的似然，有效缓解 LLM 代码生成中的重复问题。同时构建了覆盖三种场景的新评测数据集 CodeRepetEval。

## 研究问题与动机
1. **代码生成中重复问题的本质未被充分理解**：现有工作主要关注内容重复（content repetition，即完全相同的片段反复出现），但初步调查显示在 LLM 代码生成结果中，内容重复仅占小部分，更普遍且更具挑战的是结构性重复。
2. **结构性重复难以被现有方法检测和处理**：结构性重复表现为具有固定语法结构的代码片段反复出现，但每次重复的具体模式各不相同（如 `elif test: suite` 结构中条件语句不同），因此基于词频或 N-gram 的已有方法无法有效应对。
3. **结构性重复具有自我强化效应**：实验发现（Figure 2），随着重复次数增加，起始 token 的置信度以及整个重复片段的平均概率呈上升趋势，模型被困在由起始 token 锚定的结构循环中难以跳出。
4. **缺乏专门针对代码重复问题的评测数据集**：现有评测基准（HumanEval、MBPP）主要评估功能正确性，没有系统评估重复缓解能力。

## 核心贡献（创新点）
1. **首次形式化定义结构性重复（structural repetition）**：通过上下文无关文法将生成序列约简为语法规则序列，并以数学公式（Eq.1）刻画结构重复模式，揭示其在代码生成中的主导地位。*与已有工作的本质区别：此前工作仅关注字面内容重复，本文指出结构重复才是代码生成中更普遍的问题。*
2. **提出 RPG 解码框架**：利用下推自动机（PDA）在生成过程中实时约简语法规则，结合后缀数组与 LCP 数组检测重复，并通过动态权重函数指数衰减重复关键 token 的概率。*与已有工作的本质区别：RPG 是从语法视角解决重复，而非基于 token 频率uniformly 惩罚，避免了对代码中高频必要 token（如 `=`, `[`）的误伤。*
3. **构建 CodeRepetEval 数据集**：覆盖人工合成、代码生成基准、真实仓库三种场景共 2048 个测试样本，提供 EGP、TR-N、TR-S、CCP 等六项重复评估指标。*与已有工作的本质区别：这是首个专门针对代码生成重复问题的综合性评测基准。*
4. **实证表明 RPG 在重复缓解与代码质量两方面均显著提升**：在 CodeRepetEval 上全面优于最强基线，同时在 HumanEval 和 MBPP 上 Pass@1 相对提升最高达 11.3%。*与已有工作的本质区别：现有去重方法（如 Repetition Penalty/Dropout）虽能减少内容重复，但会显著损害代码生成性能；RPG 在去重的同时还能提升生成质量。*

## 方法详解
**RPG（Repetition Penalization based on Grammar）** 分为三个核心模块：

1. **约简到语法规则（Reduction to Grammar Rules）**：利用基于上下文无关文法（CFG）的下推自动机（PDA）将生成的 token 序列 $X$ 映射为语法规则序列 $\hat{R}$。对每个 token $x_t$，根据其时刻的状态 $q_t$ 和栈符号 $z_t$ 确定对应语法规则 $\hat{x}_t = [q_t, z_t]$（Eq.3），并合并相邻同类项得到最终约简序列 $\hat{R}_{1:t}$（Eq.4）。PDA 处理了 BPE 分词与文法终结符之间的三种映射关系（附录 E）。

2. **重复检测（Detection of Repetition）**：对约简序列 $\hat{R}_{1:t}$ 构建后缀数组（Suffix Array）和最长公共前缀数组（LCP Array），时间复杂度 $O(n \log n)$、空间复杂度 $O(n)$。通过 LCP 数组识别所有连续重复模式（Eq.5），条件为 $\text{LCP}[i] > 0$ 且前后后缀满足首尾相接。

3. **重复惩罚（Penalization of Repetition）**：定义动态权重函数 $\text{Pn}(x_t|x_{<t}) = \lambda^{\text{Count}(\text{Rep}(X_{1:t}))}$（Eq.6），其中 $\lambda \in (0,1)$ 为衰减因子，$\text{Count}$ 为该重复模式出现的次数。将原始 token 得分乘以该权重得到调整后得分 $s'(x_t) = s(x_t) \cdot \text{Pn}(x_t)$（Eq.7），最终取 $\arg\max$ 选择 token（Eq.8）。该机制使得随着重复次数增加，进一步重复的概率呈指数衰减，打破自我强化循环。

## 实验与结果
- **数据集**：自建 CodeRepetEval（三种场景共 2048 样本）+ 公开基准 HumanEval(164)、MBPP(974) 及扩展版 HumanEval-ET、MBPP-ET。
- **基线**：Greedy、Topk(k=5/10/30)、Temperature(t=0.1/0.2/0.8)、Topp(p=0.8/0.9/0.95)、Repetition Penalty、Repetition Dropout。
- **基础模型**：CodeLlama-7B（默认），另有 CodeGen、DeepSeek-Coder、Llama2、CodeLlama-34B 验证泛化性，以及 Go 语言验证跨语言适用性。
- **评估指标**：EGP（成功终止率↑）、TR-N（短语级结构重复↓）、TR-S（语句级结构重复↓）、CCP（可编译率↑）、Time↓、GenLen↓，以及 Pass@k。
- **主要结果**：
  - **CodeRepetEval Code Generation Benchmarks 场景**：RPG 在 EGP(0.912 vs 0.628)、CCP(0.805 vs 0.455)、TR-N(0.352 vs 0.415)、TR-S(0.391 vs 0.443) 上均大幅领先；Time 降至 13.68s（对比 Greedy 33.87s）。
  - **CodeRepetEval Real-world Repositories 场景**：RPG EGP=0.889、CCP=0.638，对比最优基线 Topk(k=30) 分别提升 +10.6% 和 +71.6%。
  - **HumanEval**：RPG Pass@1 = 0.325（vs Greedy 0.301），相对提升 **↑8.0%**；HumanEval-ET 提升 **↑11.3%**。
  - **MBPP**：RPG Pass@1 = 0.421（vs Greedy 0.396），相对提升 **↑6.4%**；MBPP-ET 提升 **↑10.3%**。
  - **跨模型**：在 CodeLlama、CodeGen、DeepSeek-Coder、Llama2 四种模型及 CodeLlama-34B 上均一致有效；代码训练模型对结构重复更敏感。
  - **跨语言**：在 Go 语言上 EGP 从 0.133 提升至 0.875，CCP 从 0.403 提升至 0.725。
  - **超参 λ=0.9** 为默认值，更小的 λ 抑制效果更强但需调优。

## 相关工作脉络
1. **Yin & Neubig (2018) TRANX / Sun et al. (2019, 2020) 语法驱动代码生成**：利用文法规则指导生成以提升语法正确性。*本文定位差异：这些方法生成的代码语法上正确，但结构性重复同样符合语法规则，因此无法解决重复问题；本文是从语法视角检测并惩罚重复，而非仅保证语法合法性。*
2. **Keskar et al. (2019) Repetition Penalty**：在解码阶段对已生成 token 进行 uniform 惩罚。*本文定位差异：RPG 识别的是特定语法模式的重复而非所有已出现 token，避免了惩罚代码中高频必要 token（如 `=`、`[`）。*
3. **Li et al. (2023a) Repetition Dropout**：训练阶段随机 mask 重复 n-gram 以降低注意力依赖。*本文定位差异：RPG 是解码阶段无侵入方法，无需重新训练模型；且 Repetition Dropout 仅针对内容重复，对结构性重复几乎无效。*
4. **Holtzman et al. (2020) Topp Sampling / Fan et al. (2018) Topk**：通用解码策略。*本文定位差异：实验表明低温度/低 k/低 p 反而加剧结构重复（模型过于自信）；RPG 与这些策略正交可叠加使用。*
5. **Chen et al. (2021) Codex / Rozière et al. (2023) CodeLlama / Guo et al. (2024) DeepSeek Coder**：主流代码 LLM。*本文定位差异：这些模型在代码生成能力上表现优异但仍受重复问题困扰；RPG 可作为即插即用的解码后处理方法增强任意代码模型。*
6. **Fu et al. (2021) 自强化重复理论分析**：揭示重复的自强化机制。*本文定位差异：本文在代码生成场景验证并利用了这一机制，通过语法驱动的指数衰减打破自强化循环。*

## 局限性与未来方向
1. **计算开销略高于纯采样**：RPG 需要额外构建后缀数组和 LCP 数组以检测重复，但作者认为相比 LLM 本身的计算开销可忽略不计（论文自述）。
2. **结构性重复的成因机制尚不明确**：作者承认目前尚未深入分析 LLM 为何容易产生结构重复，这留待未来工作。
3. **数据集规模有限**：CodeRepetEval 共 2048 样本，覆盖三个场景但样本量不大，可能对某些复杂重复模式覆盖不足。
4. **仅验证了 Python 和 Go 两种语言**：跨语言泛化性有待更多语言验证。
5. **λ 超参需人工设置**：当前默认 λ=0.9，更优值可能因场景而异。

## 研究启发与可借鉴点
1. **语法视角的重复检测思路可迁移**：利用 CFG/PDA 将 token 序列约简为语法结构序列，再用字符串算法（后缀数组/LCP）检测重复——这一框架可推广到其他结构化序列生成任务（如 JSON、Markdown、SQL 生成）。
2. **指数衰减式惩罚机制设计优雅且高效**：$\lambda^{\text{Count}}$ 的形式简单有效，可作为通用的"重复次数越多惩罚越强"的模块嵌入其他解码策略中。
3. **CodeRepetEval 的三种场景构建方法值得借鉴**：人工合成 + 从基准数据集中提取重复样本 + 从真实仓库选取触发重复的上下文，这种多层构建策略可复用于其他"负样本驱动"的评测集构建。
4. **可探索与现有方法的组合**：RPG 与 Temperature/Topk/Topp 正交，可进一步研究 RPG 与 Repetition Dropout 的联合使用效果，或在 SFT/RLHF 训练阶段引入语法重复感知损失。
5. **对团队方向的可能结合**：若团队关注代码生成可靠性/可执行性，RPG 的重复缓解可直接提升代码可编译率（CCP 提升显著），与测试生成、代码修复等方向形成互补。

## 关键术语表
**Structural Repetition（结构性重复）**：代码中反复出现的具有固定语法结构但具体内容各异的代码片段模式，由语法规则（如 `(elif test: suite)*`）决定，区别于字面内容重复。
**RPG（Repetition Penalization based on Grammar）**：本文提出的基于语法的重复惩罚解码方法，通过 PDA 约简语法规则、后缀数组检测重复、指数衰减惩罚重复 token。
**CodeRepetEval**：本文构建的专门评估代码生成重复缓解能力的数据集，含人工合成、代码生成基准、真实仓库三种场景共 2048 样本。
**EGP（End-of-sentence Generation Percentage）**：评估模型成功终止重复序列、生成 EOS token 的比例，越高越好。
**TR-N / TR-S**：分别衡量生成序列在短语级和语句级的结构重复程度（基于语法约简后的 n-gram 唯一性和语句唯一性），越低越好。
**CCP（Compiler Correctness Percentage）**：生成代码成功编译的比例，衡量代码语法/结构正确性。
**Pushdown Automaton（PDA，下推自动机）**：本文用于将 token 序列约简为语法规则序列的自动机模型，能处理嵌套和递归语法结构。
**Suffix Array & LCP Array**：高效的字符串匹配数据结构，本文用于在线检测语法约简序列中的连续重复模式。

## 可复现要素
- **数据集**：CodeRepetEval 论文未声明开源；HumanEval、MBPP、HumanEval-ET、MBPP-ET 均为公开基准。
- **代码/权重**：论文未声明代码或权重是否开源。
- **关键超参**：衰减因子 λ=0.9（默认）；最大生成长度 1024（真实仓库场景 4096）；基线默认温度 t=0.8；每个实验运行 5 次取平均；GPU 为 A6000 48GB；基础模型 CodeLlama-7B。
