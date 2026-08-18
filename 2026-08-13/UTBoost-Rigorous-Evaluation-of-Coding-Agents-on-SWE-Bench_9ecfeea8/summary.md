---
title: "UTBoost-Rigorous-Evaluation-of-Coding-Agents-on-SWE-Bench"
source: https://aclanthology.org/2025.acl-long.189.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:24:00"
field: "代码生成评估与基准测试"
keywords: ["SWE-Bench", "代码生成评估", "测试用例生成", "内态射测试", "Coding Agent", "Benchmark Robustness"]
innovations: ["首个面向SWE-Bench的LLM自动化测试用例增强框架UTBoost", "首次将内态射测试应用于真实世界开源软件评测并构建测试预言机"]
benchmarks: ["SWE-Bench Lite", "SWE-Bench Verified"]
---

# 论文速读：UTBoost-Rigorous-Evaluation-of-Coding-Agents-on-SWE-Bench

## 一句话总结
本文提出UTBoost框架，利用LLM驱动的UTGenerator自动生成补充测试用例，并结合内态射测试构建测试预言机，对SWE-Bench进行更严谨的评估，发现原有测试不足导致大量错误补丁被误判为通过，并改进了SWE-Bench解析器缺陷，更新了排行榜。

## 研究问题与动机
1. SWE-Bench中手动编写的测试用例过于狭窄，无法全面评估代码生成代理（Coding Agent）产生的补丁是否真正解决了问题，部分错误补丁能在原测试中"作弊"通过。
2. 原始SWE-Bench解析器存在缺陷：无法正确处理跨越多行的测试日志，以及日志中的侧边消息，导致大量标注被错误解析。
3. 现有鲁棒性改进工作（如SWE-Bench+、人工校验）仅停留在揭示问题层面，缺乏自动化解决方案；即使93名专业工程师校验的SWE-Bench Verified仍存在26个测试不足实例未被发现。
4. 评测框架的可靠性直接影响排行榜排名，原有缺陷导致Amazon-Q-Developer-Agent等模型排名虚高。

## 核心贡献（创新点）
1. **首个面向SWE-Bench的自动化测试用例增强框架**：UTBoost通过内态射测试建立gold patch与generated patch之间的测试预言机，本质区别在于无需人工设计测试，而是自动发现测试覆盖缺口。
2. **三粒度LLM定位+依赖感知的测试生成器UTGenerator**：从文件级→函数/类级→行级逐步定位测试添加位置，结合代码库依赖信息生成完整测试用例；与EvalPlus等直接变异的方式本质不同，UTGenerator能理解真实项目结构和依赖关系。
3. **改进SWE-Bench解析器**：提出基于队列的多行日志解析方法，修正正则表达式解析缺陷；这是首次系统讨论并修复SWE-Bench评测管道中的解析错误问题。
4. **系统性重新评估SWE-Bench榜单**：综合运用增强测试与改进解析器，发现36个测试不足实例和大量标注错误，推动40.9%（Lite）和24.4%（Verified）的排名变化，重新校准了排行榜可信度。

## 方法详解
1. **内态射测试（Intramorphic Testing）测试预言机构建**：设P为应用gold patch的程序，P'为应用generated patch的程序，两者仅在组件C→C'处不同。测试预言机定义为 $P(T) = P'(T)$，即相同测试用例T下两者输出应一致；若不一致则标记为可疑。
2. **UTGenerator三阶段定位**：
   - **文件级**：将代码库组织为树状结构，LLM从issue描述和原始测试patch出发识别Top-N最需编辑的文件；
   - **函数/类级**：压缩代码仅保留类/函数头，LLM定位目标函数或类；
   - **行级**：提取目标代码片段，LLM确定具体行号范围。
3. **上下文窗口测试生成**：以定位行为中心扩展±x行（本文x=10），结合issue描述和原始测试patch，LLM生成补充分测试用例及其依赖；通过不同温度采样（0, 0.8, 0.9, 0.99）保证测试多样性。
4. **UTBoost两阶段评估流程**：
   - 阶段一：在原始测试集$T_{orig}$上验证intramorphic关系成立；
   - 阶段二：用生成的$T_{aug}$验证$P(T_{aug}) = P'(T_{aug})$，若不等则标记可疑并将$T_{aug}$加入测试套件。
5. **改进解析器设计**：使用deque队列追踪相邻日志行，通过正则`[a-zA-Z_]\w*\s\([w.]+\)`精确匹配测试名；当当前行解析结果不匹配时回溯队列查找正确测试名，解决多行日志拆分错误问题。

## 实验与结果
- **数据集**：SWE-Bench Lite（300实例）和SWE-Bench Verified（500实例），涵盖12个Python仓库。
- **测试不足发现**：23个Lite实例 + 26个Verified实例存在测试不充分问题，分布在全榜9/12个项目中，django和sympy占比最高（Lite 84.1%，Verified 82.6%）。
- **错误补丁识别**：Lite中28.4%（170/599）生成的通过补丁实际错误；Verified中15.7%（92/584）错误。
- **解析器修正**：影响54.7%（164/300）Lite实例和54.2%（271/500）Verified实例的标注。
- **排行榜更新**：结合两者共发现176个Lite错误补丁和169个Verified错误补丁；导致18次（40.9%）Lite排名变化和11次（24.4%）Verified排名变化；原Verified榜首Amazon-Q-Developer-Agent与devlo并列第一。
- **成本**：平均每实例生成成本$1.6，总评估耗时300小时（CloudLab服务器）。

## 相关工作脉络
1. **EvalPlus（Liu et al., 2024）**：通过类型感知变异（删除/重复列表元素）增强测试，但仅适用于简单函数级生成任务，无法处理SWE-Bench多文件/多依赖的复杂场景；UTBoost通过代码库感知定位解决此问题。
2. **SWE-Bench+（Aleithan et al., 2024）和Chen & Jiang（2024）**：仅通过人工检查少量Agent补丁揭示测试不足问题，无法规模化；UTBoost提供自动化可扩展方案。
3. **SWE-Bench Verified（OpenAI, 2024）**：93名工程师手工校验测试充分性和issue描述，但仍遗漏26个测试不足实例；表明人工校验对"测试是否充分"判断存在局限。
4. **Agentless（Xia et al., 2024）**：无工具交互的代码生成Agent架构，UTGenerator受其启发采用简化架构专注于定位+生成，但可适配为更复杂的测试Agent。
5. **Intramorphic Testing（Rigger & Su, 2022）**：原始白盒测试技术，本文首次将其应用于真实世界开源软件评测场景，扩展了技术应用边界。

## 局限性与未来方向
1. **覆盖率限制**：UTBoost只能对已有Agent成功解决的实例生成测试，目前仅覆盖Lite的74.6%（224/300）和Verified的81.6%实例，剩余未解决实例无法验证。
2. **LLM单一依赖**：仅使用GPT-4o，未来可集成多种LLM API提升测试多样性和降低API成本。
3. **架构简化**：采用简化版Agentless架构（无规划/无工具交互），未来可探索复杂Agent框架作为测试生成器以提升质量。
4. **手动审查依赖**：发现差异后仍需两名软件测试专家人工确认，难以完全自动化。
5. **跨语言扩展待验证**：作者指出框架可适配其他语言，但尚未在Python以外的项目中验证。

## 研究启发与可借鉴点
1. **内态射测试思想可迁移**：将"期望行为一致性"作为测试预言机，适用于任何有gold solution和generated solution对比的场景（如程序综合、代码翻译评估），无需人工定义期望输出。
2. **多粒度定位策略**：文件→函数→行的三阶段定位方法可在其他代码理解任务中复用（如漏洞检测、影响分析），平衡LLM上下文窗口限制与定位精度。
3. **温度采样多样性控制**：不同temperature下采样策略（确定性+多样性混合）生成多样化测试用例的思路，可迁移到测试生成、对抗样本生成等任务。
4. **评测管道修复价值**：SWE-Bench解析器改进揭示了benchmark实现细节对结果的影响，提醒研究者在使用公开benchmark时需审查评测代码，而非仅依赖论文数字。
5. **人机协作验证流程**：LLM自动生成+专家人工确认的模式兼顾效率与可靠性，可作为高质量测试生成的通用范式。

## 关键术语表
**UTBoost**：基于内态射测试的SWE-Bench测试用例增强框架，通过比对gold patch与generated patch在新测试上的表现来发现测试不足。
**UTGenerator**：LLM驱动的测试用例生成器，通过三粒度（文件/函数/行）定位和依赖感知生成补充测试用例。
**Intramorphic Testing（内态射测试）**：白盒测试技术，通过比较原始系统与修改系统在相同输入下的输出关系构建测试预言机。
**SWE-Bench Lite**：SWE-Bench的300实例子集，专注于功能性bug修复，保持跨11个仓库的多样性。
**SWE-Bench Verified**：OpenAI引入的500实例子集，由93名专业工程师手工校验测试充分性和issue描述质量。
**PASS_TO_PASS / FAIL_TO_PASS**：SWE-Bench两类单元测试：前者验证补丁不破坏原有正确功能，后者验证补丁修复了原本失败的测试。
**Test Oracle（测试预言机）**：判定系统行为是否正确的机制，本文通过intramorphic关系 $P(T)=P'(T)$ 自动构建。
**Context Window（上下文窗口）**：UTGenerator中提取定位行前后各x行作为LLM输入的范围，本文设为10行。

## 可复现要素
- **数据集**：SWE-Bench Lite（300实例）和SWE-Bench Verified（500实例），论文提供更新后的数据
- **代码/权重**：代码和数据已开源（论文标注了开源链接）
- **LLM**：GPT-4o（gpt-4o-2024-08-06）
- **关键超参**：文件级定位Top-3，上下文窗口10行，温度采样策略{0: 1个, 0.8: 20个, 0.9: 20个, 0.99: 20个}，定位阶段temperature=0.8
- **硬件**：CloudLab Ubuntu 22.04 LTS服务器
- **成本**：每实例平均API成本$1.6，总评估耗时300小时
