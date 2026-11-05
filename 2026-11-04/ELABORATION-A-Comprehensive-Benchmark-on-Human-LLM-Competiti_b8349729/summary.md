---
title: "ELABORATION-A-Comprehensive-Benchmark-on-Human-LLM-Competiti"
source: https://aclanthology.org/2025.acl-long.4.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:45:00"
field: "代码生成与人类反馈"
keywords: ["人机协作", "竞技编程", "大语言模型", "评测基准", "代码生成", "人工反馈", "Bug调试"]
innovations: ["提出首个覆盖编程全四阶段的人工反馈分类体系（理解→规划→生成→调试）", "构建ELABORATIONSET数据集（8,320道标注题目）与ELABORATION基准，支持LLM模拟器+真人双轨评估", "揭示人机能力互补性：人类擅长语义错误识别，LLM擅长语法修复，组合后Pass@1提升24%"]
benchmarks: ["ELABORATION", "ELABORATIONSET"]
---

# 论文速读：ELABORATION-A-Comprehensive-Benchmark-on-Human-LLM-Competitive Programming

## 一句话总结
本文提出了 **ELABORATION**，首个面向人机协作竞技编程的综合评测基准，包含覆盖完整编程流程的四阶段人工反馈分类体系（Problem Comprehension → Solution Planning → Code Generation → Debugging）以及专门为此构建的 **ELABORATIONSET** 数据集（8,320 道来自 Codeforces 和 AtCoder 的题目），并通过 LLM 模拟器和真实人类实验全面评估了人机协作的有效性与局限性。

## 研究问题与动机
- **LLM 单枪匹马能力有限**：现有研究表明，即使是最先进的 LLM（如 O1-Mini）在困难或未见过的竞技编程问题上表现不佳（未见难题为 30.8% Pass@1），且存在严重的训练数据污染问题（平均下降 9.2%）。
- **现有研究碎片化**：当前人机协作编程研究各自为政，仅聚焦特定反馈类型（如策略建议或错误识别），缺乏对全流程协作的系统性理解。
- **缺少统一评测基准**：现有基准（APPS、LiveCodeBench、OpenCoderInterpreter 等）要么仅评估自动编程，要么仅支持单阶段交互，无法全面衡量各阶段反馈的价值差异。
- **评估方式不可靠**：规则型模拟器缺乏现实性，而纯真人实验成本高昂、规模受限，需要一种兼顾效率与真实性的评估方案。

## 核心贡献（创新点）
- **提出首个覆盖全流程的人工反馈分类体系（Taxonomy）**：将竞技编程划分为 Problem Comprehension、Solution Planning、Code Generation、Debugging 四个阶段，为细粒度评估奠定基础；与已有工作相比，此前研究（如 OpenCoderInterpreter）仅关注调试阶段的对话式反馈，本文则实现了跨全阶段的结构化评测。
- **构建 ELABORATIONSET 数据集（8,320 道题目）**：从 Codeforces 和 AtCoder 收集题目，并逐题标注了题干澄清、算法知识摘要、标准答案；与既有数据集（如 APPS、TACO）相比，首次专门为"人机交互"场景设计了多阶段静态注解。
- **设计 ELABORATION 评测基准与协议**：通过对话式人机交互流程（每阶段最多 10 轮迭代），结合 LLM 模拟器（O1-Mini）与真实人类参与者进行评测；区别于 USCAOBench 等仅支持自动编程的基准，本文支持完整的多人机协作路径分析。
- **提供全面的实验分析与发现**：揭示了各阶段反馈的边际效益差异（编码阶段收益最大）、人类与 LLM 的互补性（人类擅长语义错误识别，LLM 擅长语法错误修复），以及成本效率权衡。

## 方法详解
**评估协议流程**（每道题最多 4 个阶段，每阶段最多 10 轮迭代）：

1. **Problem Comprehension（问题理解阶段）**：LLM 输出对题干的需求理解（Problem Requirements）和规格约束（Problem Specifications），由人工/模拟器判断是否满意，不满意则提供澄清反馈。
2. **Solution Planning（方案规划阶段）**：LLM 选择算法并生成伪代码，人工/模拟器评估算法正确性与伪代码质量，提供修正建议。
3. **Code Generation（代码生成阶段）**：LLM 生成可编译的 Python 代码，人工/模拟器从代码逻辑角度提供改进建议（不直接修改代码）。
4. **Code Debugging（调试阶段）**：LLM 根据编译器反馈迭代修复错误，人工/模拟器辅助定位语义/语法 bug。

**两种人工模拟器**：
- **Student Programmer（学生程序员）**：O1-Mini 仅凭内部知识提供反馈（中等水平）。
- **Teacher Programmer（教师程序员）**：O1-Mini 参考 ELABORATIONSET 全量标注信息提供专家级反馈（高水平）。

**评估指标**：Pass@k（k=1,3,5），重点报告 Pass@1；同时区分 **Contamination Evaluation**（含训练数据）和 **Contamination-free Evaluation**（仅含模型 cutoff 日期之后的新问题）。

**数据集构建**：三阶段流程——（1）HTML 爬虫从 Codeforces/AtCoder 抓取题目基本信息；（2）用 GPT-4o 为缺失测试用例的题目生成测试；（3）三阶段人工标注：题干澄清、33 种算法的知识摘要（定义+伪代码）、收集 5 份正确答案去重。

## 实验与结果
**实验设置**：13 个 LLM（O1-Mini、GPT-4o、GPT-4-Turbo、Claude-3.5、Gemini-1.5-pro、Deepseek-Coder-6.7B/33B、Qwen2.5-Coder-7B/14B/32B、CodeLlama-7B/13B/34B）、8,320 道题（Easy 3,642 / Middle 2,098 / Hard 2,580）。

**主要结果**：
- **LLM 自身能力**：O1-Mini 在无污染测试上 Pass@1 达 59.3%（最佳），但平均在困难未见过题上仅 3.4%；整体模型性能因数据污染平均下降 **9.2%**。
- **人机协作显著提升**：Teacher Programmer 反馈使所有模型平均 Pass@1 提升 **+10.1%**（无污染）/ **+13.7%**（有污染）；Student Programmer 反馈平均提升 **+3.3%**（无污染）。
- **分阶段效益**（Figure 3）：编码阶段（Code Generation）反馈收益最大，理解阶段最小；调试阶段单独使用收益有限，强调全流程指导的重要性。
- **真人实验**（300 道未见题，GPT-4-Turbo）：自动调试（Automatic Debug）错误识别精确率 23%、召回率 40%，Pass@1 仅提升 5%；人类调试精确率 **81%**、召回率 **71%**，Pass@1 提升 **24%**。
- **人机互补性**：LLM 生成的语义错误（350 个）远多于语法错误（33 个），人类在语义错误（引用、计算、控制流、逻辑方向错误）修复上优势显著，自动调试几乎无法解决此类问题。
- **成本效益**：专家反馈 token 开销大，按 Pass@1/#token 估算，规划阶段反馈性价比可能高于编码阶段。

## 相关工作脉络
- **APPS / CODE-CONTESTS / XCODEEVAL / TACO**：均为自动编程基准，仅评估 LLM 单轮代码生成能力，不涉及多阶段人机交互反馈。
- **LIVECODEBENCH**：关注无数据污染的自动编程评测，但未覆盖人机协作流程。
- **OpenCoderInterpreter（Zheng et al., 2024）**：唯一支持人机协作的现有基准，但仅限调试阶段的对话式反馈；本文相比之下覆盖了全部四阶段并提供了更细粒度的分期评估。
- **USCAOBench（Shi et al., 2024）**：评估 LLM 独立解决 Olympiad 编程问题的能力，不支持多轮人工反馈交互。
- **Codescope（Yan et al., 2023）/ KareCoder（Huang et al., 2024c）**：侧重编译器反馈驱动的自动编程，无人工介入环节。
- **InterCode（Yang et al., 2024）/ DebugBench（Tian et al., 2024）**：聚焦代码执行反馈下的自动调试，未涉及人类在理解/规划阶段的干预价值。

## 局限性与未来方向
- **Prompt 敏感性**：评测结果可能受 prompt 设计影响，虽经多轮验证但针对此任务的 prompt 优化仍是挑战。
- **泛化性受限**：仅聚焦竞技编程领域，结论推广至通用软件工程或其他编程任务需谨慎；作者承认可迁移至其他场景但尚未验证。
- **真实人类实验规模小**：仅 5 名研究生参与 300 道题的调试实验，统计效力有限。
- **成本问题**：专家级人工反馈 token 开销高，亟需更高效的人机协作集成方法。
- **未来方向**：探索低成本高效反馈机制、扩展至更广泛的编程任务领域、研究不同专业知识水平用户的协作模式。

## 研究启发与可借鉴点
- **四阶段反馈分类体系可迁移**：Problem Comprehension → Planning → Generation → Debugging 的分层框架可推广至代码补全、文档生成、SQL 生成等其他代码相关人机协作任务。
- **"LLM 模拟器 + 真实人类"的双轨评估范式**：先用强模型（O1-Mini）模拟不同能力水平用户实现大规模筛选，再用小规模真人实验验证，兼顾效率与可信度，适用于多数交互系统评测。
- **污染敏感评估的重要性**：通过时间切分确保测试集在模型 cutoff 之后，有效揭示数据污染问题（平均下降 9.2%），应成为代码 LLM 评测的标配实践。
- **人机能力互补分析框架**：本文对语法/语义错误的分类统计（Table 7）揭示了 LLM 和人类各自的识别盲区，为设计混合智能系统的错误处理分工提供了方法论参考。
- **成本-收益量化思路**：以 Pass@1/#token 衡量反馈性价比，为后续研究优化人机交互资源分配提供了可复用的评估维度。

## 关键术语表
- **ELABORATION**：本文提出的首个覆盖人机协作竞技编程全流程的综合评测基准，包含四阶段评估协议。
- **ELABORATIONSET**：专为该基准构建的数据集，包含 8,320 道来自 Codeforces/AtCoder 的题目及多阶段人工注解。
- **Human Feedback Taxonomy**：将人工反馈按编程四阶段（理解→规划→生成→调试）组织的结构化分类体系。
- **Pass@k**：生成 k 个候选解中至少有一个通过所有测试用例的概率，k=1 时等价于单次生成的通过率。
- **Contamination-free Evaluation**：排除模型训练数据中存在题目的评测方式，通过按发布时间切分实现。
- **Teacher/Student Programmer Simulator**：分别代表专家级和中级编程能力的 O1-Mini 用户模拟器，差异在于是否参考数据集标注信息。
- **Automatic Debug**：由 LLM 模拟器模拟的自动错误识别与修复流程，作为真人调试的对照基线。
- **Semantic/Syntactic Bug**：语义错误指逻辑/控制流层面的错误（LLM 主要短板），语法错误指编译/结构层面的错误（LLM 可自行修复）。

## 可复现要素
- **数据集**：ELABORATIONSET 已公开，包含 8,320 道题目及多阶段注解；代码已开源。
- **代码/权重**：代码和数据集已公开于 https://github.com/SCUNLP/ELABORATION；补充的错误标注数据集也将随主数据集一并公开。
- **关键超参**：nucleus sampling，temperature=0.7，top-p=0.95，每阶段最多 10 轮迭代；训练数据截止日期见 Table 9。
- **硬件**：7B 模型单卡 A100，13B 双卡 A100，34B 四卡 A100；使用 Xinference 框架。
