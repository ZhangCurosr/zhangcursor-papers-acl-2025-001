---
title: "ANDROIDGEN-Building-an-Android-Language-Agent-under-Data-Sca"
source: https://aclanthology.org/2025.acl-long.138.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:49:22"
field: "移动端智能体与语言模型应用"
keywords: ["mobile agent", "Android", "language agent", "data scarcity", "in-context learning", "open-source LLM", "trajectory generation"]
innovations: ["提出四模块框架ExpSearch/ReflectPlan/AutoCheck/StepCritic增强data-scarce场景下的Android agent能力", "利用StepCritic细粒度评估构建无标注高质量轨迹数据并微调开源模型", "结合轨迹检索与反思规划实现easy-to-hard的自进化agent"]
benchmarks: ["AndroidWorld", "AitW (Android in the Wild)", "Popular Applications Benchmark"]
---

# 论文速读：ANDROIDGEN-Building-an-Android-Language-Agent-under-Data-Sca

## 一句话总结
ANDROIDGEN提出一个面向Android的语言agent框架，通过ExpSearch、ReflectPlan、AutoCheck和StepCritic四大模块在高质量标注数据稀缺的场景下增强LLM的agent能力；并利用该框架自生成轨迹数据微调开源模型（GLM-4-9B、Llama-3-70B），构建了无需人工标注的Android语言agent。

## 研究问题与动机
- **数据稀缺问题**：移动端agent缺乏高质量交互轨迹数据，人工标注成本高昂且耗时。
- **完成率不足**：即使GPT-4/Gemini等先进LLM在数字环境中完成率仍不理想，且缺少有效的自动化数据筛选策略。
- **三类数据收集困难**：场景多样性高（不同app/功能差异大）、复杂任务步骤多导致成本高、数据筛选需精确验证环境与操作一致性。
- **已有规划方法缺陷**：Chain-of-Thought、ReAct等方法缺乏对执行结果和环境状态的动态反馈机制，不适合多轮复杂交互场景。

## 核心贡献（创新点）
1. **提出ANDROIDGEN四模块框架**：ExpSearch（轨迹检索）、ReflectPlan（反思规划）、AutoCheck（操作预校验）、StepCritic（细粒度评估）协同增强agent推理与操作准确性；与直接prompting方法的本质区别在于引入了"自我积累-动态规划-主动校验"的闭环机制。
2. **引入StepCritic细粒度评估器**：将任务分解为子目标并提供逐步轨迹评估；与已有Pan等人、DigiRL的二元评估方式本质不同，StepCritic可定位每个子目标的完成步骤，最大化数据学习价值。
3. **无标注数据驱动的开源agent训练**：将ANDROIDGEN作为数据生成流水线，结合StepCritic标签构建1000+条高质量轨迹，微调开源模型；与M3A、DigiRL依赖RL或人工数据的方法本质区别在于完全无需人工标注。
4. **多维度实验验证**：在AndroidWorld、AitW、popular applications三个基准上验证有效性，揭示视觉/计数/跨应用/记忆四类错误分布，为后续改进指明方向。

## 方法详解
**整体架构**：ANDROIDGEN分三阶段运行——preliminary（初始化+检索示例）、task execution（规划+操作+校验）、update（轨迹评估+数据库更新）。

### 3.2.1 ExpSearch
- **轨迹收集**：agent自采样收集轨迹，利用StepCritic评估轨迹质量，保留所有轨迹及完成状态，构建轨迹数据库。
- **轨迹检索**：在相同上下文（如同一app内）中，使用**Contriever**对指令编码并与数据库嵌入计算相似度，选取top-1作为学习示例；每次任务完成后用StepCritic评估并更新数据库，实现easy-to-hard泛化。

### 3.2.2 ReflectPlan
- **Plan Initialization**：第一步前分析任务与环境，生成逐步计划。
- **Plan Reflection**：第二步起根据当前进度反思并更新计划；遇到失败状态或循环时重新规划，增强长期推理鲁棒性。

### 3.2.3 AutoCheck
- 生成操作后主动验证有效性，检测潜在问题后终止执行并向agent反馈。
- 验证规则见Table 1：如open_app检查app是否存在、Click检查id是否在屏幕上、Scroll/Swipe检查dir是否合法等。
- 采用"检查操作是否符合预期结果"策略（如元素id存在性、类型合规、滚动完成），避免self-checking导致的评估标准不一致问题。

### 3.2.4 StepCritic
- 基于GPT-4o构建，输入完整操作序列+设备最终状态，将任务分解为子目标并评估每步完成情况（-1表示未完成）。
- 细粒度评估最大化数据学习价值，为数据筛选和轨迹扩充提供信号。

### 数据算法（4.1节）
- 用GPT-4o生成约300条任务指令（无reward/golden label防泄漏）。
- **轨迹扩充**：对任务T的子目标序列$g_1,...,g_n$，找到第一个未完成子目标$g_k$，将$g_1+...+g_i$（i=1..k-1）与前面对应步骤拼接为新轨迹，扩充数据集至1000+条。
- **训练**：LoRA微调GLM-4-9B和Llama-3-70B，规划与执行步骤混合训练，epochs=3，max_lr=1e-4，seq_len=8192，batch_size=32。

## 实验与结果
**基准与指标**：AndroidWorld（任务成功率）、AitW（人工评估成功率）、popular applications（5任务/应用，人工评估）。

| 基准 | ANDROIDGEN (GPT-4o) | 最佳对比 | 提升 |
|---|---|---|---|
| AndroidWorld Avg. | **46.8%** | M3A (GPT-4o) 27.7% | +19.1pp |
| AitW General | **74.0%** (Llama-3-70B*) | DigiRL* 71.9% | +2.1pp |
| AitW Web Shopping | **79.2%** (Llama-3-70B*) | DigiRL* 67.2% | +12.0pp |
| Popular Apps SR | **65.0%** (GPT-4o) | AppAgent 57.5% | +7.5pp |

- **消融实验**（AndroidWorld, GPT-4o）：ReflectPlan使Avg SR从20.7%→32.4%（+56.5%整体，medium任务+149.2%）；AutoCheck +5.6%；ExpSearch +36.8%。
- **StepCritic评估精度**：子目标完成识别92.8%，步骤预测82.3%，整体87.9%，优于Captioner+Mixtral（82.4%）和Captioner+GPT-4（84.6%）。
- **效率对比**：生成1000条轨迹，ANDROIDGEN成本仅为人工标注的5%（质量过滤后），效率为人工的5.85倍。

## 相关工作脉络
1. **Mobile-Agent/AppAgent/SeeAct**：依赖闭源LLM（GPT-4V等），注重视觉感知；本文聚焦开源模型在无标注数据下的能力构建，使用XML而非截图降低token开销。
2. **M3A/DigiRL**：M3A基于Llama-3-70B/Gemini做agent基线；DigiRL采用Offline-to-On RL；本文用自生成轨迹SFT开源模型，数据成本更低且无需RL训练。
3. **Pan et al. (2024) 自主评估**：仅支持二元任务完成判断，缺乏细粒度轨迹分析；StepCritic通过子目标分解提供逐步评估。
4. **Chain-of-Thought/ReAct**：推理方法无法根据环境反馈动态调整计划；ReflectPlan引入执行结果驱动的规划更新机制。
5. **EPR/APE (In-Context Learning)**：依赖静态数据集或手动搜索；ExpSearch实现动态轨迹检索与自进化。

## 局限性与未来方向
- **视觉能力不足**：视觉相关错误占15%（如颜色识别），需集成vision model。
- **复杂多轮交互**：跨应用任务、计数场景（23%错误）仍困难，需引入大规模自适应推理搜索策略。
- **执行效率**：系统规模大，小模型作为executor表现良好但planner需更大模型；未来可探索轻量规划-执行分离架构。
- **安全性**：高风险操作（账户、支付）需开发classifier式auto-check模块，对用户授权进行验证。

## 研究启发与可借鉴点
1. **自生成数据pipeline**：将agent本身作为数据生产工具，配合细粒度评估器（StepCritic）构建训练数据，可迁移至web agent、desktop agent等场景。
2. **轨迹扩充策略**：截取已完成子目标的轨迹片段生成新样本（Algorithm 1），类似思路可用于网页导航、游戏agent的 Curriculum Learning。
3. **模块化设计范式**：检索增强（ExpSearch）+反思规划（ReflectPlan）+主动校验（AutoCheck）+细粒度评估（StepCritic）的四层架构可作为通用mobile agent设计模板。
4. **Python函数调用定义动作空间**：利用LLM对Python的熟悉度定义action space（含docstring），提升操作准确性，可推广至其他GUI agent。
5. **XML环境表示替代截图**：用结构化XML替代视觉输入，降低token消耗并提高元素解析精度，适合对视觉依赖低的纯文本操作场景。

## 关键术语表
**ExpSearch**：基于轨迹检索的上下文学习模块，通过Contriever检索top-1相似任务轨迹辅助当前任务执行。
**ReflectPlan**：在执行过程中根据环境状态动态评估任务进度并更新计划，解决多轮交互中的规划鲁棒性问题。
**AutoCheck**：在操作执行前对操作合法性进行预校验的模块，通过检查元素存在性、类型合规性等减少操作错误。
**StepCritic**：基于GPT-4o的细粒度轨迹评估器，将任务分解为子目标并定位每步完成情况。
**Contriever**：对比学习文本检索模型，用于轨迹语义相似度计算与检索。
**LoRA**：低秩适配器，用于高效微调大语言模型（论文中微调GLM-4-9B和Llama-3-70B）。
**AndroidWorld**：基于Android模拟器的dynamic benchmarking环境，提供细粒度任务完成评估。
**AitW (Android in the Wild)**：大规模真实场景Android设备控制数据集，含训练数据和离线评估指标。

## 可复现要素
- **代码**：已开源 https://github.com/THUDM/AndroidGen
- **模型**：GLM-4-9B*、Llama-3-70B* 已开源
- **数据集**：自构建1000+条轨迹（任务指令由GPT-4o生成，无人工标注）；AndroidWorld和AitW为公开基准
- **环境**：Android模拟器（使用XML格式环境表示）
- **关键超参**：LoRA fine-tuning，max_lr=1e-4，seq_len=8192，batch_size=32，epochs=3；Contriever用于轨迹检索；训练使用单节点8×A100-80GB
- **动作空间**：Python函数定义（open_app, do, quote, exit等，详见Appendix C）
