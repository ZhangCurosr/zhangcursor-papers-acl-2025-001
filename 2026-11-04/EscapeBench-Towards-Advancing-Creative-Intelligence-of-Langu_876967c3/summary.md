---
title: "EscapeBench-Towards-Advancing-Creative-Intelligence-of-Langu"
source: https://aclanthology.org/2025.acl-long.39.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 07:32:42"
---

# 论文速读：EscapeBench-Towards-Advancing-Creative-Intelligence-of-Langu

## 一句话总结
本文提出EscapeBench——首个基于密室逃脱游戏的语言模型智能体创造力评测基准，并设计EscapeAgent框架（融合Foresight创新工具假设与Reflection隐式目标管理模块），显著提升智能体在不确定路径与超长推理链中的创造性问题解决效率。

## 研究问题与动机
- 现有Agent评测基准多聚焦目标明确的任务，主要考察分析型与实践型智力，缺乏对创造性智力的系统化评估。
- 当前大模型依赖参数化记忆中的工具-任务强关联，面对非结构化新场景时难以挖掘工具的可利用属性（affordance）并进行非常规复用。
- 密室逃脱场景天然要求“跳出框架”思维，涉及超长按钮步骤（>100步）、隐含目标路径与非常规工具/合成操作，是检验创造性的理想载体。
- 即便配备工作记忆与Chain-of-Thought的BaseAgent，在无提示下平均仅取得约15%的关键进展，提示依赖与步数消耗高达人类平均水平的10倍以上。

## 核心贡献（创新点）
- 提出EscapeBench基准：构建36个密室逃脱场景（含Easy/Normal/Difficult三档），首次将创造力评估聚焦于非常规工具使用、隐含目标发现与超长推理链。与现有基准的本质区别在于，其评测维度从“目标达成率”转向“无提示探索效率”与“工具新颖应用率”。
- 设计EscapeAgent框架：通过Reflection模块动态维护待办任务清单以识别隐式目标，通过Foresight模块在行动前提取工具应用假设并验证。与自回归反思类工作的本质区别在于，该框架将目标管理（Reflection）与假设生成（Foresight）解耦为上下游组件，而非对单次决策进行事后修正。
- 引入创造力细粒度指标：新增Early Exit Progress、Tool/Key Step Hints比例等维度。与单一成功率指标的本质区别在于，该指标体系能够分离“常规探索能力”与“瓶颈突破能力”，精准定位创造性推理断层。
- 揭示模型能力边界与领域副作用：证明<7B参数模型在该任务上接近随机行为，且过度垂直对齐（如Math型）会显著损害创造性泛化。与纯性能刷榜工作的本质区别在于，该研究明确了“框架增强”与“模型内在创造力”的互补边界，指出后续需从架构与对齐策略双端突破。

## 方法详解
- **游戏引擎核心组件**：由Scenes（场景图节点，通过门/隧道连接）、Items（场景中不可直接操作的客体，需Apply/Input/Click触发状态变化）与Tools（背包中可收集/应用/合成的客体）构成交互闭环。
- **动作空间定义**：Move（切换场景）、Click（观察/基础交互）、Apply（工具作用于物品）、Input（向物品输入字符串）、Craft（两工具合成新工具）。其中Apply与Craft被定义为最高创造力驱动动作。
- **BaseAgent基线**：采用Chain-of-Thought推理配合长度固定为10的工作记忆，每一步接收环境描述、可交互对象列表与背包工具，经显式推理后输出下一动作指令。
- **Reflection模块机制**：作为下游组件，在每次非Move动作后根据环境反馈更新任务清单。操作类型包括New（新增未解任务及target item）、Update（追加失败尝试与hindsight）、Delete（目标达成后移除）。该模块通过结构化记忆防止重复试错，并将隐式目标转化为显式待办队列。
- **Foresight模块机制**：作为上游组件，在新工具收集或新任务创建时激活。模块要求模型显式比对工具描述特征（如硬度、长度、材质）与任务/场景约束，生成Apply或Craft假设；若假设成立则进入“Try Action”状态并顺序执行验证，否则维持“Free Explore”状态由BaseAgent继续探索。
- **难度梯度构造**：同一逻辑谜题提供三个版本，差异仅在于环境描述粒度与交互反馈丰富度（Easy含解题提示，Normal仅描述物理属性，Difficult无任何环境反馈），核心解谜逻辑保持严格一致。

## 实验与结果
- **评测设置**：在36个场景上测试12个主流闭源/开源模型，采样温度T=0，n=1，工作记忆长度10；连续50步无进展则注入人工提示；报告Micro-average指标。
- **基线表现**：最优模型GPT-4o在BaseAgent下平均需10.30次提示、723.61步，Early Exit Progress仅24.75%；人类平均仅需4.33提示、257.83步，进展59.65%。Key Step提示占比高达24.27%，印证创造性瓶颈集中在关键谜题突破。
- **EscapeAgent最强结果**：GPT-4o配合EscapeAgent后，提示降至5.03（↓51%），总步数降至452.75（↓37%），Early Exit Progress提升至47.03%（↑22%）；Llama-3.1-70B提示降至7.92（↓54%），步数降至645.19（↓34%）。
- **扩展分析**：Reflection单独作用通常优于Foresight单独作用，双模块结合效果最佳；<7B模型（如Llama-3.2-1B/3B）关键动作提示率超95%，接近随机；Qwen2.5-Math-7B在数学对齐后创造性泛化显著下降，而Qwen2.5-Coder-7B保持接近通用模型水平。

## 相关工作脉络
- **LM创造力评估**：传统研究依赖Alternative Uses Test (AUT) 与Torrance Tests (TTCT) 等心理学量表评估文本生成的Novelty/Surprise；本文转向Agent交互场景中的工具创造性复用，填补行为与规划维度的评测空白。
- **文本冒险评测基准**：Zork、TextWorld、AI2Thor等侧重叙事理解、常识推理与结构化路径规划；本文引入密室逃脱的Open-ended puzzle solving，强调非预设路径探索与非常规工具应用。
- **沙盒环境评测**：Minecraft（Voyager/Plan4MC）、Roblox等侧重空间导航、协作与技能持续学习；本文聚焦单智能体在受限文本环境中的“工具affordance”挖掘与隐式目标推导，复杂度控制在可解释的长链推理层面。
- **Agent记忆与反思机制**：区别于Self-Refine、AgentPro等迭代自我修正框架，本文Reflection专注“任务清单动态状态管理”而非错误纠正；Foresight专注“行动前因果假设
