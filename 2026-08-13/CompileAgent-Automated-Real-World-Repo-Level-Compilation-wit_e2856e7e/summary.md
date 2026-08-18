---
title: "CompileAgent-Automated-Real-World-Repo-Level-Compilation-wit"
source: https://aclanthology.org/2025.acl-long.103.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:52:57"
field: "代码编译自动化"
keywords: ["仓库级编译", "LLM Agent", "代码编译自动化", "工具增强大模型", "软件工程自动化"]
innovations: ["首个基于LLM Agent的仓库级编译框架CompileAgent", "Flow-based Agent策略适配编译工作流", "Multi-Agent Discussion解决编译错误"]
benchmarks: ["CompileAgentBench"]
---

# 论文速读：CompileAgent-Automated-Real-World-Repo-Level-Compilation-with-Tool-Integrated-LLM-based-Agent-System

## 一句话总结
本文提出了CompileAgent，首个面向仓库级代码自动编译的LLM-based Agent框架，通过集成5个专用工具与基于流程的Agent策略，解决编译指令搜索与错误解决两大核心挑战。在自建基准CompileAgentBench上，该方法相比基线最高提升编译成功率71%，单个项目平均成本仅$0.22。

## 研究问题与动机
- **仓库级编译复杂性远超文件级**：开源仓库涉及多文件依赖、构建配置复杂、构建顺序约束（如cmake先于make），传统文件级编译工具无法直接适用。
- **现有自动化工具缺乏适应性**：如Oss-Fuzz-Gen依赖预定义文件匹配执行固定命令序列，在缺少指定文件（如bootstrap.sh/Makefile.am）时失效，且无法适应动态环境变化。
- **人工编译耗时且易错**：开发者需手动查找编译指南、处理依赖冲突、环境不匹配等问题，平均耗时且重复劳动频繁。
- **LLM Agent在其他软件工程任务中已有成功应用**：代码生成、bug修复、渗透测试等领域验证了Agent+工具调用的有效性，但尚无工作将Agent用于仓库级编译自动化。

## 核心贡献（创新点）
- **首次提出基于LLM Agent的仓库级自动编译框架CompileAgent**：与Oss-Fuzz-Gen等硬编码规则工具的本质区别在于，CompileAgent通过多轮交互与环境感知自主决策，而非依赖固定文件模式匹配。
- **设计了CompileNavigator与ErrorSolver两大模块及5个专用工具**：区别于通用Agent框架（如SWE-Agent），本文工具专为编译场景定制（如Shell容器隔离、Multi-Agent Discussion），解决编译特有的指令搜索与错误诊断问题。
- **构建首个仓库级编译基准CompileAgentBench**：包含100个真实C/C++仓库，覆盖14个领域，7:2:1的比例设计（仓库内有/外/无指南），填补该领域评测空白。
- **提出面向编译任务的Flow-based Agent策略**：不同于ReAct/Plan-and-Execute等通用策略，该策略严格模拟人工编译工作流（查找指南→执行→错误处理循环），在Claude-3-5-sonnet上达到96%成功率。
- **验证了小参数模型的适用边界**：发现CompileNavigator模块可用较小模型（Qwen2.5-32B）替代而不显著损失性能，但ErrorSolver模块必须使用大模型，为实际部署提供成本优化路径。

## 方法详解
**整体架构**：CompileAgent由MasterAgent驱动，调用两个核心模块（CompileNavigator、ErrorSolver），共集成5个工具，采用Flow-based策略编排。

**CompileNavigator模块**（解决编译指令搜索）：
- **Shell工具**：基于Ubuntu 22.04 Docker容器隔离执行，通过SSH接入终端，挂载仓库并执行命令，保障宿主机安全。
- **File Navigator工具**：包含SearchAgent I与SearchAgent II两个子Agent协作讨论，根据仓库结构定位可能包含编译指令的文件（如README、INSTALL等）。
- **Instruction Extractor工具**：由SummarizeAgent实现，读取目标文件内容，若含URL则爬取网页，最终总结提取编译指令。

**ErrorSolver模块**（解决编译错误）：
- **Website Search工具**：封装Google Search，优先检索GitHub、StackOverflow等可靠来源，聚合解决方案。
- **Multi-Agent Discussion工具**：3个Agent进行多轮讨论（最多R=3轮），每轮根据其他Agent输入修正分析；命令序列去重后若重复词超过阈值则视为达成共识。

**Flow-based Agent策略**：
1. MasterAgent用Shell下载仓库并挂载至容器
2. 用Shell执行"tree"获取仓库结构
3. 调用File Navigator定位编译指令文件
4. 调用Instruction Extractor提取指令并通过Shell执行
5. 若成功则完成；若失败，MasterAgent先尝试独立解决，否则激活ErrorSolver进行多轮讨论
6. 循环直到编译成功或达到最大轮次

## 实验与结果
**数据集**：CompileAgentBench，100个C/C++仓库，14个领域（Audio、Crypto、Database、HPC等），由3位有3-4年开发经验的研究人员手动编译验证。

**评估基线**：Oss-Fuzz-Gen（无LLM）、Readme-AI（GPT-4o mini生成文档）、RAG（text-embedding-3-large向量检索）、不同Agent策略（OpenAIFunc、Plan-and-Execute、ReAct）。

**主要结果**（Table 1）：
- **最优性能**：Claude-3-5-sonnet + CompileAgent达到96%编译成功率，比Oss-Fuzz-Gen(25%)、Readme-AI(79%)、RAG(78%)分别提升71%、17%、18%
- **时间成本**：平均每个项目耗时约5-11小时，相比Readme-AI节省47.64-121.96小时
- **费用成本**：单个项目平均成本$0.22，相比Readme-AI和RAG分别节省约$33
- **模型适配性**：7个LLM（32B-236B）均有效，闭源模型普遍优于开源模型；Mixtral-8×7B表现较差可能与架构有关

**策略对比**（Table 2）：Flow-based策略在Claude-3-5-sonnet上达96%，比ReAct(81%)、Plan-and-Execute(72%)、OpenAIFunc(80%)分别提升15%、24%、16%

**消融实验**（Table 3）：移除Multi-Agent Discussion导致成功率降至71%（-18%），验证其核心作用；三Agent版本效果与双Agent相当但耗时增加

**小模型替换实验**（Tables 4-5）：CompileNavigator用小模型可接受降级（87%→82%），ErrorSolver用小模型则严重下降（91%→68%）

## 相关工作脉络
- **Oss-Fuzz-Gen**：硬编码文件匹配规则，缺乏环境适应性；CompileAgent通过LLM自主搜索与错误解决克服此局限
- **SWE-Agent/OpenHands**：聚焦代码修复与软件开发全流程；本文聚焦编译这一特定且低资源的研究方向
- **AutoCodeRover**：针对程序改进的Agent；与本文ErrorSolver的Multi-Agent Discussion理念相关但应用目标不同
- **ChatDev/QueryAgent**：通用软件工程Agent；本文贡献在于工具设计针对编译场景（容器隔离、指令提取、多源搜索）
- **Readme-AI/RAG基线**：本文对比验证了Agent交互搜索优于静态文档生成或向量检索方式

## 局限性与未来方向
- **LLM理解偏差**：Agent可能误解释指令导致重复或错误操作，未来需探索微调提升指令理解能力
- **工具集相对基础**：当前仅封装5个基础工具，未集成高级编程/调试工具（如Coverity Scan），扩展空间大
- **Prompt设计敏感**：各Agent的Prompt质量显著影响系统性能，需要精细调优
- **多语言/多架构支持待验证**：虽提出可扩展至Java/Go/ARM等，但实验仅限C/C++和少量Go项目（20个）
- **失败案例分析**：复杂依赖链、工具链版本不匹配、配置复杂性仍是主要失败原因

## 研究启发与可借鉴点
- **工具设计应贴合领域工作流**：编译任务专用工具（如容器隔离Shell、Multi-Agent Discussion）比通用工具更有效，可迁移至其他工程自动化任务
- **大小模型分层使用策略**：简单任务用中小模型、复杂推理用大模型，平衡性能与成本，对Agent系统部署有参考价值
- **Flow-based策略的可复用性**：模拟人类专家工作流的Agent编排方式（查找→执行→诊断→修正）适用于其他分阶段任务
- **基准构建方法论**：CompileAgentBench的人工验证流程（多人交叉验证可编译性、指南分布设计）可为其他领域基准构建提供参考
- **失败模式分析框架**：将失败归因于依赖链、工具链、配置三类，为后续优化提供明确方向

## 关键术语表
**CompileAgentBench**：首个仓库级编译基准，包含100个C/C++仓库，人工验证可编译性，7:2:1的指南分布比例
**Flow-based Agent Strategy**：面向编译任务的Agent编排策略，严格遵循人工编译工作流顺序调用工具
**Multi-Agent Discussion**：3个Agent多轮协商机制，通过重复词阈值判断共识，用于编译错误解决方案生成
**CompileNavigator**：负责定位并提取编译指令的核心模块，集成Shell、File Navigator、Instruction Extractor三个工具
**ErrorSolver**：负责编译错误诊断与解决的模块，集成Website Search与Multi-Agent Discussion工具
**Oss-Fuzz-Gen**：基于文件模式匹配的自动化编译工具，依赖预定义文件（如bootstrap.sh）触发固定命令序列
**ReAct/Plan-and-Execute**：通用Agent推理-行动交替策略与规划-执行分离策略，作为本文对比基线

## 可复现要素
- **数据集**：CompileAgentBench公开于https://github.com/Ch3nYe/AutoCompiler（论文声明）
- **代码**：完整代码与数据已开源，链接同上
- **关键超参**：Multi-Agent Discussion最大轮数R=3；Embedding模型为text-embedding-3-large；Docker基于Ubuntu 22.04
- **基线实现**：Readme-AI使用GPT-4o mini；RAG使用OpenAI embedding模型
