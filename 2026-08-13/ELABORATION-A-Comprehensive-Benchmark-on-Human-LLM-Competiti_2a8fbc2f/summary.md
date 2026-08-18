---
title: "ELABORATION-A-Comprehensive-Benchmark-on-Human-LLM-Competiti"
source: https://aclanthology.org/2025.acl-long.4.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:08:54"
field: "代码生成与人机协作"
keywords: ["Human-LLM Collaboration", "Competitive Programming", "Code Generation", "Benchmark", "Human Feedback", "LLM Evaluation"]
innovations: ["首个覆盖竞赛编程全流程四阶段的人类反馈分类体系与综合基准", "首个专为人类-LLM协作设计的竞赛编程数据集ELABORATIONSET（8,320题，含精细化注释）", "系统揭示编码阶段反馈收益最高但Token开销大、人类与LLM Bug识别能力互补的发现"]
benchmarks: ["ELABORATION", "ELABORATIONSET", "Codeforces", "AtCoder"]
---

# 论文速读：ELABORATION-A-Comprehensive-Benchmark-on-Human-LLM-Competiti

## 一句话总结
本文提出了 ELABORATION，首个面向人类-大语言模型协作竞赛编程的综合基准评测，包含覆盖完整编程流程的四阶段人类反馈分类体系、专用数据集 ELABORATIONSET（8,320 题），并通过 LLM 模拟者与真实人类实验系统评估了人机协作各阶段的效能与瓶颈。

## 研究问题与动机
- 现有竞赛编程基准（如 APPS、CODE-CONTESTS、OpenCoderInterpreter 等）主要面向**纯自动**代码生成，缺乏对人类-LLM 多轮协作过程的系统性评测。
- 已有的人机协作研究存在"碎片化"问题：有的仅关注解题策略建议（Mozannar et al., 2023; Wang et al., 2024），有的仅聚焦对话式错误识别（Zheng et al., 2024; Shi et al., 2024），**忽略了人类引导在问题理解、方案规划等环节的潜力**。
- 竞赛编程涉及四个关键阶段（问题理解 → 方案规划 → 代码生成 → 调试），但缺乏覆盖全流程的分阶段细粒度评估协议。
- LLM 单独求解竞赛编程的能力有限（尤其高难度/未见过的题目，性能平均下降 -9.2%），亟需通过人机协作探索有效补充路径。

## 核心贡献（创新点）
- **提出首个竞赛编程全流程人类反馈分类体系（Taxonomy）**，将编程过程划分为问题理解、方案规划、代码生成、调试四个阶段，支持分阶段细粒度评估；不同于此前工作仅关注单一阶段，本文实现了对整个竞赛编程流程的系统性覆盖。
- **构建 ELABORATIONSET 数据集**，包含来自 Codeforces 和 AtCoder 的 8,320 道题目，配备问题澄清注释、算法知识摘要、标准答案等，支持大规模模拟人机交互与低成本真实人类实验；区别于已有数据集，这是首个专为人类-LLM 协作设计的竞赛编程数据集。
- **设计 ELABORATION 评测基准**，采用对话式人机交互协议，结合 LLM 用户模拟器（Student/Teacher Programmer）与真实人类参与者进行双向验证；与现有基准相比，同时支持模拟与现实人参与的双重评估范式。
- **系统性揭示人机协作规律**：发现编码阶段反馈收益最高（Pass@1 提升 +7.0%~+11.5%）、人类与 LLM 在 Bug 识别上具有互补性（人类语义错误识别精度 81% vs 自动 23%），为后续研究提供可复现的实验结论。

## 方法详解
- **评估协议**：LLM 与人类（或模拟器）按四个阶段依次交互，每阶段人类提供文本反馈后 LLM 生成中间产物（问题理解摘要/算法选择+伪代码/完整代码/修正后代码），直至代码通过全部测试用例或达到最大迭代次数（10 轮）。
- **四阶段人类反馈分类体系**：
  - **问题理解（Problem Comprehension）**：澄清题目要求、边界条件、关键约束（如 O(n log n) 时间限制）。
  - **方案规划（Solution Planning）**：建议合适算法、提供理由及完整准确伪代码（如 Dijkstra 算法的适用场景与实现细节）。
  - **代码生成（Code Generation）**：优化数据结构（如用栈/优先队列）、细化算法实现。
  - **代码调试（Code Debugging）**：定位逻辑缺陷（如死循环），协助通过未见测试用例。
- **ELABORATIONSET 数据构建**：从 Codeforces 和 AtCoder 采集 8,320 题（2011.10–2024.11），难度按评分范围划分为 Easy/Middle/Hard；用 GPT-4o 补全缺失测试用例（每题 15 条），经人工校验；采用"LLM 自动初稿 + 人工复核"两阶段流程生成问题澄清和算法摘要（覆盖 33 种算法定义与伪代码）。
- **两类 LLM 用户模拟器**：Student Programmer（基于 O1-Mini 内部知识提供反馈，模拟中级水平）和 Teacher Programmer（借助 ELABORATIONSET 全部注释资源，模拟专家级水平）。
- **评估指标**：主要使用 Pass@k (k=1,3,5)，同时区分**含污染**（Contamination）与**无污染**（Contamination-free，仅评估模型截止时间后发布的问题）两类结果；分阶段通过比对地面真值实现细粒度自动评测。

## 实验与结果
- **数据集规模**（Table 1）：共 8,320 题（Easy 3,642 / Middle 2,098 / Hard 2,580），每道题平均约 14.4 个测试用例；真实人类交互子集 300 题（各难度 100 题），平均每轮人类反馈数随难度递增（Easy 3.4 轮 → Hard 6.9 轮）。
- **评测模型**：13 个 LLM（O1-Mini、GPT-4o、GPT-4-Turbo、Gemini-1.5-pro、Claude-3.5、CodeLlama-7B/13B/34B、Deepseek-Coder-6.7B/33B、Qwen2.5-Coder-7B/14B/32B）。
- **核心结果（Pass@1，无污染）**：
  - 最强模型 **O1-Mini** 在未见题上 Pass@1 达 **59.3%**；但所有模型在未见难题上的平均得分仅 **3.4%**，纯 LLM 不足以胜任竞赛编程。
  - 人机协作显著提升：Teacher Programmer 反馈下整体 Pass@1 平均提升 **+10.1%**（未见到 +8.1%），Student Programmer 平均提升 **+3.3%**（未见到 +3.0%）。
  - Qwen2.5-Coder-7B + Teacher 反馈：无论文 19.5% → 47.1%，提升近 28 个百分点。
- **分阶段分析**（Figure 3）：**编码阶段**人类反馈收益最大，**理解阶段**最小（LLM 本身已能较好理解题目）；但编码阶段伴随更高的 Token 开销，规划阶段性价比更高。
- **真实人类实验（Table 6）**：GPT-4-Turbo 在调试阶段，自动 Debug 错误识别精确率 23%、召回率 40%，Pass@1 仅提升 5%；真实人类精确率 **81%**、召回率 **71%**，Pass@1 提升 **24%**。
- **Bug 类型互补性**（Table 7）：LLM 主要产生语义错误（350 个 vs 语法错误 33 个），人类在引用错误、计算错误、不完整错误、逻辑方向错误上显著优于自动调试；两者结合形成强协同。
- **污染影响**：含污染评估 vs 无污染评估，所有模型平均 Pass@1 下降约 **9.3%**，表明相当比例的 LLM 性能来源于训练集记忆。

## 相关工作脉络
- **APPS / CODE-CONTESTS / XCODEEVAL / TACO / LIVECODEBENCH**：均为自动代码生成基准，不支持或仅部分支持人类交互评估；ELABORATION 首次覆盖全流程人机协作。
- **CODESCOPE / KareCoder / USCAOBENCH**：聚焦自动评测或单一反馈形式，缺乏跨阶段的人类指导分类；本文填补了全流程分阶段评估空白。
- **OpenCoderInterpreter（Zheng et al., 2024）**：同为 Human-LLM 方向，但仅聚焦对话式错误识别（调试阶段），未覆盖理解/规划阶段；ELABORATION 实现了四阶段全覆盖。
- **Mozannar et al. (2023) / Wang et al. (2024)**：仅关注策略建议（Solution Planning），忽略了其他阶段的潜在增益。
- **规则型用户模拟器**（早期工作）：缺乏对复杂编程反馈的建模能力；本文采用 LLM 驱动模拟器（O1-Mini），并辅以真实人类验证。
- **AlphaCode / CodeGen / StarCoder 等代码模型**：展示了 LLM 自动编程潜力但竞赛水平仍有限；本文通过系统性基准揭示了其局限与人机协作的补充价值。

## 局限性与未来方向
- **Prompt 敏感性**：结果受提示词设计影响，最优 prompt 优化仍是开放挑战。
- **泛化边界**：研究聚焦竞赛编程，尚未验证对人类软件开发生态或其他编程任务的适用性（作者承认潜力但暂不展开）。
- **Token 成本过高**：编码阶段反馈效率最高但 Token 开销大，当前性价比分析显示规划阶段更经济，需开发更高效的人机反馈集成方法。
- **真实人类实验规模有限**：仅 5 名 CS 研究生参与，且局限于调试阶段（300 题），难以充分反映更大规模人群的表现。
- **O1-Mini 模拟实验延迟**：因成本高未纳入部分模拟器对比实验。

## 研究启发与可借鉴点
- **全流程分阶段评估框架**可迁移至其他"人类-AI 协作"领域（如数学推理、代码审查、技术写作），四阶段 taxonomy 设计思路值得借鉴。
- **LLM 用户模拟器 + 真实人类的双重验证范式**：先通过 O1-Mini 进行大规模低成本筛选，再用真实人类做小规模精调验证，兼顾可扩展性与真实性。
- **数据构建的"两阶段人工校准"策略**（LLM 初稿→人工复核+分歧仲裁）在保证质量的同时控制成本，适用于大规模评测数据集建设。
- **人与 LLM 的 Bug 识别互补性发现**：人类擅长语义/逻辑错误、LLM 擅长语法错误，这一模式可指导后续混合智能调试系统的架构设计。
- **无污染评估的设计**：按发布时间划分训练/测试集，有效剔除记忆效应，值得在代码生成领域广泛推广。

## 关键术语表
- **ELABORATION**：本文提出的首个面向人类-LLM 协作竞赛编程的综合基准评测框架。
- **ELABORATIONSET**：专为人类-LLM 协作设计的竞赛编程数据集，包含 8,320 道带精细化注释的题目。
- **Human Feedback Taxonomy**：将人类指导划分为问题理解、方案规划、代码生成、调试四个阶段的结构化分类体系。
- **Contamination-free Evaluation**：无污染评估，仅测试 LLM 训练截止时间之后发布的题目，用于排除训练数据记忆的影响。
- **Student/Teacher Programmer Simulator**：基于 O1-Mini 的两类用户模拟器，分别以"内部知识"（中级）和"完整注释资源"（专家级）提供反馈。
- **Pass@k**：在 k 次生成尝试中至少有一次通过全部测试用例的概率，常用 k=1 作为主要指标。
- **Automatic Debug**：由 LLM 模拟器扮演的自动调试过程，替代真实人类进行 Bug 识别与修正建议。
- **Grammar/Semantic Bug**：语法错误（函数调用、变量声明等可编译检测的问题）vs 语义错误（逻辑缺陷、控制流错误等需深层理解的问题）。

## 可复现要素
- **数据集**：ELABORATIONSET（8,320 题），论文声明将在 https://github.com/SCUNLP/ELABORATION 开源。
- **代码**：评测实现代码已在 GitHub 公开。
- **权重**：未提供自有模型，评测使用现有商业/开源 LLM（API 调用或 Xinference 框架部署）。
- **关键超参**：Nucleus sampling, temperature=0.7, top-p=0.95, 每阶段最大迭代 10 轮, Pass@k 使用 macro average 计算。
- **硬件**：7B 模型单卡 A100，13B 双卡 A100，34B 四卡 A100。
- **真实人类实验**：5 名 CS 研究生，补偿 $150/人，受 IRB 批准，聚焦调试阶段 300 题。
