---
title: "EscapeBench-Towards-Advancing-Creative-Intelligence-of-Langu"
source: https://aclanthology.org/2025.acl-long.39.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:51:04"
field: "Agent 评测与创造性推理"
keywords: ["creative intelligence", "language model agent", "benchmark", "room escape game", "tool use", "reasoning", "Foresight", "Reflection"]
innovations: ["提出首个基于密室逃脱游戏的创造性智力评测基准 EscapeBench", "设计双模块 EscapeAgent 框架（Foresight+Reflection）提升 Agent 创造性推理", "定义 Early Exit Progress 等细粒度创造性推理评估指标体系"]
benchmarks: ["EscapeBench"]
---

# 论文速读：EscapeBench: Towards Advancing Creative Intelligence of Language Model Agents

## 一句话总结
本文提出 EscapeBench，首个面向评估语言模型智能体**创造性智力**的评测基准（基于密室逃脱游戏环境），并在此基础上设计了 EscapeAgent 框架，通过 Foresight（创新工具假设生成）和 Reflection（隐式目标识别与任务列表维护）两个模块显著提升智能体的创造性推理能力。

## 研究问题与动机
1. **创造性智力被忽视**：现有 Agent 评测基准（如 WebArena、ScienceWorld、Minecraft 等）主要评估目标明确、路径清晰的目标导向任务，衡量的是分析性智力与实践性智力，而创造性智力——即在不熟悉环境中 adaptively reasoning、创新使用工具、发现隐式目标的能力——缺乏有效评估手段。
2. **训练偏差导致创造力缺失**：当前 LM Agent 的训练主要聚焦于记忆 tool-task 关联，而非探索工具 affordance 和适应非结构化场景，导致面对未见工具组合时易陷入常规思维模式。
3. **密室逃脱场景的天然适配性**：该类场景要求"跳出框框"思维，工具可能以非典型方式使用（如木棍用于撬开而非戳刺），目标路径不可预先规划、需 trial-and-error，且单局需要超过 100 步推理链（至少 40 个关键瓶颈步骤），恰好构成创造性推理的检验场。
4. **初步实验揭示明显差距**：即使是最强模型（GPT-4o、Llama-3.1-70B）在 BaseAgent 框架下完成最易场景也极度困难，平均需使用约 10 倍于最优行动链的步骤数，且远落后于人类平均表现（约 257 步 vs. 最优 107.83 步），暴露出现有方法在创造性推理上的根本性不足。

## 核心贡献（创新点）
1. **提出首个面向创造性智力的 Agent 评测基准 EscapeBench**：基于密室逃脱游戏设计，包含 36 个精心构建的游戏场景（三难度级别 × 三版本），其核心区分度在于要求非常规工具使用与隐式目标发现；相较于传统 benchmark（如 WebArena 评估网页导航、Minecraft 评估空间规划），本文聚焦于"创造性适应"而非"计划与纠错"。
2. **提出 EscapeAgent 框架，引入 Foresight + Reflection 双模块协同机制**：Foresight 模块使智能体在行动前显式生成并评估工具使用假设；Reflection 模块维护结构化任务列表以追踪未完成目标；两者结合相较独立模块具有互补性（消融实验 Table 7 显示 full EscapeAgent 在所有指标上均优于任一子模块）。
3. **定义并量化创造性推理的新评估维度**：除标准 Hints Used 和 Total Steps 外，引入 Early Exit Progress（首次求助前的关键步骤完成比例）、Tool Hints Used（收集工具的提示占比）、Key Step Hints Used（关键步骤的提示占比）等指标，提供超越传统完成率的新评测视角；特别指出"Input"和"Craft"动作是创造性推理的核心挑战点。
4. **系统性揭示当前 LM Agent 创造性能力的边界**：通过详尽的误差分析（Table 4）和跨模型对比，发现小模型倾向于无效动作，大模型倾向于"表面尝试后放弃"（superficial attempts），揭示创造性推理不足的结构性原因。

## 方法详解
**EscapeBench 环境设计**：
- **引擎三组件**：Scenes（场景节点构成图结构）、Items（场景中不可交互的对象，可被触发状态变化）、Tools（可收集并 apply/craft 的对象）。工具具有多状态（如钥匙可从"生锈"变为"光亮可用"）。
- **五类动作空间**：Move（场景间移动）、Click（点击查看）、Apply（工具→物品）、Input（字符串→物品）、Craft（工具→工具，合成新工具）。其中 Apply 和 Craft 最依赖创造性推理。
- **三难度设计**：Easy（环境描述可能暗示用途）、Normal（仅描述属性）、Difficult（无环境反馈），同一逻辑不同呈现粒度。

**BaseAgent（强基线）**：
- 工作在 Memory Length=10 的 working memory 之上，结合 Chain-of-Thought 推理决定下一步动作。
- 每 50 步无进展时触发 hint 介入（包含下一目标位置和具体动作）。

**EscapeAgent 增强框架**：
- **Reflection 模块（任务列表管理）**：在每次非 Move 动作后自动触发，维护结构化任务列表，每条记录包含任务名、目标物品、已尝试失败动作。支持三种操作：New（新增未解任务）、Update（记录新失败尝试）、Delete（目标达成后移除）。防止重复试错，增强行动目的性。
- **Foresight 模块（创新假设生成）**：在两种条件下激活：①新工具被收集；②新任务被识别。此时要求智能体基于工具描述和任务描述显式推理可能的 Apply/Craft 假设；若提出合理假设则进入 "Try Action" 状态依次尝试，否则保持 "Free Explore" 状态。
- 两模块协同：Reflection 提供任务上下文，Foresight 基于此生成假设，消融实验（Table 7）显示 Reflection 单独作用大于 Foresight 单独，但二者结合效果最优。

## 实验与结果
- **数据集**：EscapeBench 共 36 个游戏设置，三个难度级别（Easy/Normal/Difficult），每级别三个版本，总计场景图规模如图3所示（关键步骤数约 1200+ 步跨全部场景）。
- **评测模型**：涵盖闭源（GPT-4o、GPT-4o-mini、Claude-3.5-Sonnet、Gemini-1.5-pro）和开源（Llama-3.1-70B/8B、Qwen-2.5-72B/7B、DeepSeek-LLM-67B、Yi-1.5-34B、Phi-3-medium、Ministral-8B）共 12 个模型，<7B 参数模型因近乎随机行为被排除。
- **主要结果（BaseAgent，Table 3）**：
  - 最优模型 Claude-3.5-Sonnet：平均使用 8.97 条 hint、690.31 步，Early Exit Progress 为 28.95%。
  - 最优闭源 GPT-4o：10.30 hints、723.61 步；最优开源 Llama-3.1-70B：14.53 hints、982.42 步。
  - 相比之下，人类平均仅需 4.33 hints、257.83 步；oracle action chain 平均仅 107.83 步。
  - 各模型平均进度（无提示情况）约 15%。
- **主要结果（EscapeAgent，Table 5）**：
  - GPT-4o + EscapeAgent：hints 降至 5.03（↓51.3%）、steps 降至 452.75（↓37.5%）、Early Exit Progress 提升至 47.03%（↑22.28pp）。
  - Llama-3.1-70B + EscapeAgent：hints 降至 7.92（↓45.5%）、steps 降至 645.19（↓34.3%）。
  - 所有模型在 EscapeAgent 下均有提升，大模型获益更显著。
- **消融实验（Table 7）**：GPT-4o 上 Full EscapeAgent (5.03 hints) > Only Reflection (6.75) > Only Foresight (7.17) > BaseAgent (9.83)；Llama-3.1-70B 上同样趋势。
- **领域特化模型（Table 6）**：Qwen-2.5-Coder-7B 表现接近通用版，但 Qwen-2.5-Math-7B 性能急剧下降（45.4 hints），表明过度数学对齐可能损害创造性泛化。
- **核心结论**：当前最强 LM Agent 在创造性推理方面仍大幅落后于人类；EscapeAgent 框架可显著减少约 50% 的提示依赖和 40% 的步骤数，但距人类水平仍有较大差距。

## 相关工作脉络
1. **Agent 基准评测**：WebArena（Zhou et al., 2023）、ScienceWorld（Wang et al., 2022）、OSWorld（Xie et al., 2024）等侧重目标导向任务中的规划与纠错能力；本文的 EscapeBench 转向评估创造性推理，填补这一空白。
2. **工具学习与使用**：Toolformer（Schick et al., 2023）、ToolRL（Qian et al., 2025a）、OTC（Wang et al., 2025）主要关注工具调用学习与优化；本文强调工具的创新性再用途（unconventional tool use）而非记忆已有 tool-task 关联。
3. **创造力评估**：TTCT（Torrance Tests of Creative Thinking）、AUT（Alternative Uses Test）等心理测量方法曾用于评估 LLM 创造力（Guzik et al., 2023；Zhao et al., 2024）；本文将其从生成式创作任务迁移至 Agent 基于环境的 interactive reasoning 设定，更具生态效度。
4. **模拟环境中的 Agent**：TextWorld（Côté et al., 2019）、SwiftSage（Lin et al., 2024）、Voyager（Zhu et al., 2023）等；本文的密室逃脱环境相比 Minecraft/Roblox 等 sandbox 环境更聚焦于约束条件下的创新问题解决。
5. **反思与记忆机制**：Self-Refine（Madaan et al., 2024）、AgentPro（Zhang et al., 2024b）等；本文的 Reflection 模块与之不同，专注于维护任务追踪列表以支持隐式目标发现，而非一般性的自我修正。
6. **CoT 推理与规划**：CoT（Wei et al., 2022）、ToT（Yao et al., 2024）、Tree-of-Thoughts 等；本文在 CoT 基础上叠加了显式的工具使用假设生成（Foresight）和任务状态维护（Reflection），形成面向创造性推理的增强范式。

## 局限性与未来方向
1. **纯文本环境限制**：密室逃脱游戏天然包含视觉和听觉线索，但本文基准仅限文本输入，未能模拟多模态感知；未来可扩展至 vision-language 模型驱动的多模态版本。
2. **人工标注的扩展性瓶颈**：高质量场景需大量人工标注，尝试使用 GPT-4o 自动生成时发现其易遗漏关键物品且难以平衡难度；全文自动化生产仍不可行，仍需人工把关。
3. **核心模型能力仍是根本约束**：EscapeAgent 主要降低创造性推理的门槛，但小模型（<7B）即使在此框架下仍表现糟糕；提升模型内禀创造性推理能力仍是长期挑战。
4. **Step RL 尚未探索**：目前仅使用 prompt-based 方法，未引入强化学习；Future work 提出可探索 step-level reward 和 hierarchical RL 以提升超长跑链中的探索效率。
5. **过度特化损害泛化**：Qwen-2.5-Math-7B 的实验中观察到数学对齐过度反而削弱创造性能力，提示未来在模型训练时需平衡领域特化与通用创造性。

## 研究启发与可借鉴点
1. **Foresight 假设生成机制可迁移**：将"基于工具描述和任务上下文明式生成 Apply/Craft 假设再验证"的范式，可直接迁移至其他需要非常规工具使用的 benchmark（如 embodied task、robotics simulation），作为增强 Agent 探索能力的通用模块。
2. **Reflection 任务列表设计的通用价值**：结构化维护待解决任务及其失败历史的机制，适用于任何长链推理场景（代码生成、科学发现流程、多步文档处理），可防止 Agent 陷入重复试错的无效循环。
3. **创造性推理的细粒度评估指标体系**：Early Exit Progress、Key Step Hints 占比等指标的引入，为评测 Agent 的"主动发现能力"提供了可量化的新维度，可用于后续工作的基准对比设计。
4. **"Input" 和 "Craft" 动作的分析洞见**：本文发现这两类动作的 hint 使用率最高，揭示了当前模型在开放字符串空间和工具组合空间中的根本性困难；可启发后续工作针对这两个动作类型设计专项训练或评测。
5. **困难梯度设计的复用**：同一逻辑 × 不同描述粒度/反馈粒度的三难度×三版本设计，为构建可扩展的评测集提供了可复用的方法论模板，降低新 benchmark 的构建成本。

## 关键术语表
**EscapeBench**：本文提出的基于密室逃脱游戏环境的评测基准套件，专门用于评估 LM Agent 的创造性推理能力。
**Creative Intelligence（创造性智力）**：Sternberg Triarchic Theory 中的三大智力成分之一，指在 unfamiliar 和 unstructured 情境下创新性适应和思维发散的认知能力。
**Foresight Module**：EscapeAgent 中的创新假设生成模块，在新工具获取或新任务识别时显式推理工具的潜在应用或组合方式。
**Reflection Module**：EscapeAgent 中的任务追踪模块，维护结构化任务列表（含目标物品、已尝试失败动作），支持 New/Update/Delete 操作。
**Key Step（关键步骤）**：完成一局密室逃脱所必需的瓶颈动作序列，每个场景至少包含 40 个 key steps，是整个推理链中最具创造性的环节。
**Affordance（功能可供性）**：指对象基于其物理属性所允许的行为可能性（如硬木棍的"撬"功能），本文认为创造性工具使用本质上是发现并利用 affordance。
**Early Exit Progress**：评估指标，指 Agent 在首次需要 hint 之前已完成的关键步骤和收集工具的比例。
**Super-Long Reasoning Chain**：指单局游戏需要超过 100 步才能完成（至少 40 步为 bottleneck key steps），考验 Agent 的长期连贯推理能力。

## 可复现要素
- **数据集**：EscapeBench 共 36 个游戏设置，含 three difficulty levels × three versions per scenario；论文声明所有数据已公开发布（footnote 1 指向 release link）。
- **代码**：论文声明开源代码和 benchmark 实现（release link 同数据）。
- **权重**：评测使用标准 API（闭源模型）和 vLLM 框架（开源模型，2 × A100-80G GPU）；模型权重为公开预训练模型（GPT-4o、Llama-3.1、Qwen-2.5 等）。
- **关键超参**：Working memory length = 10；触发 hint 的无进展阈值 = 50 步；采样温度 T = 0，n = 1；Open-source 推理使用 vLLM。
- **人类基线**：招募参与者的平均游戏经验约 5.6 次离线密室逃脱，但自评技能水平差异较大（仅 20% 自认为 skilled）。
