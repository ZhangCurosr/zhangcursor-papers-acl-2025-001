---
title: "CompileAgent-Automated-Real-World-Repo-Level-Compilation-wit"
source: https://aclanthology.org/2025.acl-long.103.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:41:47"
field: "软件工程系自动化与 LLM Agent 应用"
keywords: ["LLM Agent", "自动化编译", "仓库级代码构建", "Multi-Agent 系统", "软件工程自动化"]
innovations: ["提出首个仓库级编译 LLM Agent 框架 CompileAgent，集成五工具与流式策略", "构建 CompileAgentBench 基准（100 个 C/C++ 项目，14 个领域），验证成功率提升 10%-71%", "设计 Multi-Agent Discussion 机制，通过命令行重复项统计达成修复共识"]
benchmarks: ["CompileAgentBench", "Oss-Fuzz-Gen Baseline", "Readme-AI Baseline", "RAG Baseline", "Go Language Benchmark (20 projects)"]
---

# 论文速读：CompileAgent-Automated-Real-World-Repo-Level-Compilation-wit

## 一句话总结
论文提出 CompileAgent，首个面向代码仓库级别的 LLM-based Agent 自动化编译框架，通过集成 Shell、文件导航、指令提取、网站搜索与多智能体讨论五大工具及流式策略，显著提升 C/C++ 开源项目编译成功率并降低时间与经济成本。

## 研究问题与动机
1. **开源仓库编译复杂度极高**：真实软件工程中需处理环境适配、依赖管理、构建配置等复杂问题，人工调试耗时且易错。
2. **缺乏针对仓库级编译的自动化工具**：现有方法（如 Oss-Fuzz-Gen）仅依赖固定文件匹配预设命令，缺乏对动态环境和复杂错误处理的适应性。
3. **编译指令查找困难**：仓库文档分布杂乱（README、doc.html、install.txt 等），难以快速定位准确的编译命令。
4. **错误解决需要多轮交互**：依赖冲突、环境不匹配等问题需结合在线资源检索与团队协作推理，单一智能体难以高效应对。

## 核心贡献（创新点）
1. **提出首个仓库级编译 Agent 框架 CompileAgent**：将 LLM Agent 引入自动化编译领域，填补该方向研究空白。
2. **设计双模块五工具系统**：CompileNavigator 与 ErrorSolver 模块协同完成指令查找与错误修复，工具设计贴合真实开发者工作流。
3. **构建 CompileAgentBench 评测基准**：包含 100 个高质量 C/C++ 项目，覆盖 14 个领域，7:2:1 配比提供/不提供指南的项目，支持系统性评估。
4. **验证流式 Agent 策略优势**：对比 ReAct、Plan-and-Execute 等策略，证明流式策略在成功率和成本上的综合最优性。

## 方法详解
**整体架构**：采用 MasterAgent 作为主控节点，串联两个核心模块与五个工具，按顺序执行任务。

**CompileNavigator 模块（指令发现）**：
- **Shell 工具**：基于 Docker 容器（Ubuntu 22.04）隔离编译环境，通过 SSH 连接执行命令。
- **File Navigator 工具**：双 SearchAgent（SearchAgent I & II）协作讨论，基于仓库结构定位含编译指令的文件。
- **Instruction Extractor 工具**：SummarizeAgent 读取目标文件，提取编译指令；若含 URL 则抓取网页内容辅助解析。

**ErrorSolver 模块（错误修复）**：
- **Website Search 工具**：封装 Google Search，优先检索 GitHub、StackOverflow 等可靠来源。
- **Multi-Agent Discussion 工具**：三个 Agent 进行最多 3 轮多轮讨论，通过分割命令行并统计重复项判断共识，输出最终解决方案。

**流式 Agent 策略（Flow-based Strategy）**：
1. 下载仓库并挂载至容器。
2. 执行 `tree` 命令获取仓库结构。
3. 调用 File Navigator 定位指令文件。
4. 使用 Instruction Extractor 提取并执行编译命令。
5. 若成功则结束；若失败则先自主尝试修复，仍失败则激活 ErrorSolver 模块进行多轮讨论。

**关键设计逻辑**：将编译过程形式化为"查找→执行→修复"的流水线，不同任务匹配不同复杂度 Agent（导航用较小模型，错误修复用较大模型）。

## 实验与结果
**数据集**：CompileAgentBench，包含 100 个 C/C++ 项目（覆盖音频、压缩、密码学、数据库等 14 个领域），编译指南分布 7:2:1，人工编译耗时约 46 人时验证正确性。

**基线方法**：
- Oss-Fuzz-Gen（无 LLM 传统方法）
- Readme-AI（GPT-4o mini 生成文档后提取指令）
- RAG（检索增强生成，使用 text-embedding-3-large）
- 三种 Agent 策略：OpenAIFunc、Plan-and-Execute、ReAct

**主要结果**（Table 1）：
- **GPT-4o**：CompileAgent 达到 **89%** 成功率，较 Readme-AI（72%）提升 17%，较 RAG（67%）提升 22%，时间仅需 8.38 小时（节省 120.42/117.67 小时），成本仅 $16.53。
- **Claude-3-5-sonnet**：CompileAgent 达到 **96%** 成功率，为所有模型中最高，较 Readme-AI（79%）提升 17%，较 RAG（78%）提升 18%，时间 5.37 小时，成本 $22.02。
- **DeepSeek-v2.5**：CompileAgent 达到 **91%** 成功率，较 Readme-AI（71%）提升 20%，较 RAG（72%）提升 19%，时间 11.38 小时，成本 $3.31。
- **平均提升幅度**：相比各基线，CompileAgent 成功率提升 **10%~71%**。

**策略对比**（Table 2）：
- Flow-based 策略在 GPT-4o 上达到 89%，比 ReAct（72%）高 17%，比 Plan-and-Execute（40%）高 49%。
- Claude-3-5-sonnet 上 Flow-based 达 96%，比 ReAct（81%）高 15%。

**消融实验**（Table 3）：
- 移除 Multi-Agent Discussion 导致成功率下降 **18%**（89%→71%），时间增至 18.89 小时。
- File Navigator 双 Agent 效果优于单 Agent（83% vs 89%）与三 Agent（成本相当但时间更长）。
- 小模型替代实验（Tables 4-5）：CompileNavigator 可用较小模型（Qwen2.5-32B，成功率 82%，成本降低）；ErrorSolver 必须使用大模型（小模型导致成功率骤降至 68%）。

**Go 语言扩展实验**（Appendix D）：20 个 Go 项目上 CompileAgent 达 **90%** 成功率，单项目成本仅 $0.102，验证跨语言可扩展性。

## 相关工作脉络
1. **Oss-Fuzz-Gen**：早期仓库构建工具，基于文件匹配预设命令，缺乏动态适应与错误处理能力，本文在其基础上引入 LLM Agent。
2. **Readme-AI / RAG 技术**：知识库检索方案，但直接用于编译任务时因未考虑执行环境与错误修复环节而效果有限，本文通过工具集成弥补这一缺陷。
3. **Agent 策略研究（ReAct、Plan-and-Execute）**：通用推理策略，但编译任务具有明确时序性，本文设计专用流式策略更贴合实际工作流。
4. **Multi-Agent 讨论机制（ReConcile）**：借鉴其多轮协商思想，针对编译错误场景简化共识判定规则（命令行重复项统计），降低计算开销。
5. **SWE-Agent / OpenHands**：开源仓库级代码修改 Agent，侧重代码生成与修复；本文聚焦编译环节，二者互补可形成完整开发流水线。

## 局限性与未来方向
1. **LLM 理解误差**：Agent 可能误解读提示或指令，导致重复或错误操作，需通过微调模型提升指令理解能力。
2. **工具集较基础**：当前集成工具功能有限，未来可扩展更高级的编程与调试工具（如 Coverity Scan 等静态分析工具）。
3. **提示工程敏感**：系统性能高度依赖 Prompt 设计，需探索更自动化的提示优化方法。
4. **多架构支持待验证**：虽提出通过 QEMU 支持 ARM/MIPS 等架构，但尚未在实验中验证。
5. **失败案例类型集中**：主要失败源于复杂依赖链、工具链不匹配与配置复杂性，需针对性改进依赖解析与环境自动配置能力。

## 研究启发与可借鉴点
1. **双模块分工设计**：将"查找"与"修复"分离，匹配不同复杂度 Agent，为其他复杂任务（如部署、测试）提供可复用架构范式。
2. **流式策略优先于通用策略**：针对有明确步骤顺序的任务，专用流式策略比通用 ReAct/Plan-and-Execute 更有效，启发任务设计应贴合工作流。
3. **成本-效果权衡实验**：通过小模型替换实验揭示不同模块对模型能力的敏感度差异，为 Agent 系统的资源优化提供量化指导。
4. **共识判定简化设计**：Multi-Agent Discussion 中通过命令行文本重复项统计而非复杂推理达成共识，降低开销，值得在低成本场景中借鉴。
5. **跨语言扩展路径清晰**：通过 Docker 隔离环境+动态语言检测即可支持多语言，为多语言 Agent 系统构建提供方法论参考。

## 关键术语表
**CompileAgent**：首个面向仓库级编译的 LLM-based Agent 框架，集成五大工具与流式策略实现自动化编译。
**CompileAgentBench**：论文构建的评测基准，包含 100 个 C/C++ 项目，覆盖 14 个领域，编译指南分布 7:2:1。
**Flow-based Agent Strategy**：专为编译任务设计的流式策略，按固定顺序调用工具完成查找-执行-修复流程。
**Multi-Agent Discussion**：三个 Agent 进行多轮协商，通过命令行重复项统计达成共识的修复机制。
**CompileNavigator**：负责查找编译指令的模块，包含 Shell、File Navigator、Instruction Extractor 三个工具。
**ErrorSolver**：负责解决编译错误的模块，包含 Website Search 与 Multi-Agent Discussion 两个工具。
**ReConcile 思想**：借鉴的多轮协商机制，通过多 Agent 输入修正输出逐步逼近共识解。

## 可复现要素
**数据集**：CompileAgentBench，100 个 C/C++ 项目（含 Go 语言 20 个项目补充实验），论文声明代码与数据已开源（https://github.com/Ch3nYe/AutoCompiler）。
**代码**：已开源，包含完整 Agent 框架、工具实现与实验脚本。
**权重**：使用闭源模型 API（GPT-4o、Claude-3-5-sonnet、Gemini-1.5-flash）与开源模型（Qwen2.5-32B、Mixtral-8×7B、Llama3.1-70B、DeepSeek-v2.5）。
**关键超参**：Multi-Agent Discussion 轮数上限 R=3，Agent 数量=3，文件共识重复项阈值论文未明确给出（需查阅源码）。
**环境**：Docker 容器（Ubuntu 22.04），Google Search API（Website Search 工具），text-embedding-3-large（RAG 基线）。
