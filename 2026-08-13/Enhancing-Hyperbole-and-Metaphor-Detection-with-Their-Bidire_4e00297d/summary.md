---
title: "Enhancing-Hyperbole-and-Metaphor-Detection-with-Their-Bidire"
source: https://aclanthology.org/2025.acl-long.23.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:50:26"
field: "修辞语言理解与大模型推理"
keywords: ["夸张检测", "隐喻检测", "情感分析", "双向动态交互", "大语言模型提示", "修辞理解"]
innovations: ["提出情感引导的夸张与隐喻联合检测框架，首次将情感分析作为修辞检测的显式先验知识", "设计基于情感驱动的源域/目标域映射模块，增强跨域语义理解", "构建夸张与隐喻双向动态交互机制，实现任务间知识相互促进"]
benchmarks: ["HYPO", "HYPO-L", "LCC", "TroFi"]
---

# 论文速读：Enhancing Hyperbole and Metaphor Detection with Their Bidirectional Dynamic Interaction and Emotion Knowledge

## 一句话总结
论文提出 EmoBi 框架，通过情感分析引导 + 基于情感的源/目标域映射 + 夸张与隐喻的双向动态交互机制，显著提升夸张和隐喻的联合检测性能，在四个标准数据集上均达到 SOTA，Hyperbole F1 最高提升 28.1%，Metaphor F1 最高提升 23.1%。

## 研究问题与动机
- **情感因素被忽视**：现有方法（如 Badathala et al., 2023；Elzohbi & Zhao, 2023）主要依赖词汇和句法层面的表层特征，未深入挖掘修辞表达背后的情感线索，导致对"butcher's knife"等暗含残酷情绪的隐喻难以准确识别。
- **双向交互建模不足**：夸张（hyperbole）与隐喻（metaphor）虽表现形式不同，但都涉及语义偏离和跨域映射，现有方法要么独立建模，要么仅做隐式特征融合，缺乏显式的、动态的双向促进机制。
- **大模型直接提示效果有限**：标准 Prompt 和 CoT 提示在修辞检测任务上仍显不足，因为 LLM 本身对隐性修辞的理解能力有限，需要外部知识（如情感、域映射）进行引导。
- **缺乏可解释的推理链路**：现有方法输出标签缺乏中间推理过程，难以追踪模型是如何从情感推断出修辞结论的。

## 核心贡献（创新点）
- **情感引导框架**：首次系统性地引入情感分析作为修辞检测的先验知识，通过 LLM 提示挖掘句子深层情感，为后续域映射和交互提供情感线索。
- **基于情感的域映射模块**：利用情感分析结果，引导 LLM 从情感视角识别源域（source domain）和目标域（target domain），并通过情感连接桥接跨域映射，增强隐喻和夸张的语义理解。
- **双向动态交互机制**：设计夸张→隐喻、隐喻→夸张的双向推理流程，使夸张的强烈情感和程度变化丰富隐喻的概念映射，同时隐喻为夸张设定语义框架和情感基调，两者相互促进。
- **验证机制**：在推理链末端增加验证步骤，若检测到错误可重新评估和调整结果，提升检测的准确性和可靠性。
- **SOTA 性能与系统分析**：在四个数据集上全面超越基线，并通过多组实验揭示 LLM 规模、模型选择、交互机制等各因素对性能的影响规律。

## 方法详解
**整体架构**：EmoBi 由三个核心模块组成，串行+并行结合：

1. **情感分析模块**（Emotion Analysis）
   - 输入句子 $x$，通过 LLM 提示：`Prompt1: Please analyze the emotion of the following sentence.`
   - 输出情感分析结果 $x_e = LLM(x, Prompt1)$
   - 该步骤建立情感背景，为后续域映射和交互提供情感线索。

2. **基于情感的域映射模块**（Emotion-Based Domain Mapping）
   - 输入句子 $x$ 和 $x_e$，通过 LLM 提示：`Prompt2: Based on the above emotion analysis result, identify the source domain and target domain in the sentence, and analyze the emotion connection between the two domains.`
   - 输出域映射结果 $x_d = LLM(x, x_e, Prompt2)$，包含源域、目标域及两者间的情感关联说明。

3. **双向动态交互模块**（Bidirectional Dynamic Interaction）
   - **隐喻→夸张方向**：先用情感+域知识检测隐喻 $x_m = LLM(x, x_e, x_d)$，再结合 $x_m$ 检测夸张 $y_h = LLM(x, x_e, x_d, x_m, Prompt3)$。
   - **夸张→隐喻方向**：对称地，先用情感+域知识检测夸张 $x_h$，再结合 $x_h$ 检测隐喻 $y_m$。
   - 两种方向的推理结果共同决定最终的 $(y_h, y_m)$ 标签。

4. **验证机制**（Verification Mechanism）
   - 若检测到矛盾或不一致，触发重新评估和调整，确保最终输出的一致性。

**关键设计思想**：将多步 LLM 推理与结构化知识（情感、域映射）相结合，替代纯端到端黑盒预测，增强可解释性。

## 实验与结果
- **数据集**：HYPO（Troiano et al., 2018）、HYPO-L（Zhang & Wan, 2021）、LCC（Mohler et al., 2016）、TroFi（Birke & Sarkar, 2006）。
- **评估指标**：Precision（P）、Recall（R）、F1。
- **基线方法**：
  - MTL-F-BERT / MTL-F-ALBERT / MTL-F-RoBERTa（Badathala et al., 2023，当前 SOTA）
  - Standard Prompting
  - CoT-based Prompting

**主要结果**：
| 数据集 | 任务 | EmoBi F1 | 最佳基线 F1 | 提升幅度 |
|--------|------|----------|-------------|----------|
| HYPO | Hyperbole | 90.8 | 88.1 (MTL-F-RoBERTa) | +2.7% |
| HYPO | Metaphor | 84.5 | 78.7 | +5.8% |
| HYPO-L | Hyperbole | 79.3 | — | +6.5% |
| HYPO-L | Metaphor | 80.3 | 57.2 | **+23.1%** |
| LCC | Hyperbole | 84.9 | 77.5 | +7.4% |
| LCC | Metaphor | 91.3 | 83.6 | +7.7% |
| TroFi | Hyperbole | 84.2 | 56.1 | **+28.1%** |
| TroFi | Metaphor | 76.6 | — | +5.9% |

- **最强提升**：TroFi 上 Hyperbole F1 提升 28.1%；HYPO-L 上 Metaphor F1 提升 23.1%。
- **消融实验结论**：去掉情感分析模块性能下降最显著（HYPO-L Metaphor F1 降 5.7%），证明情感引导最关键；双向交互模块去除后也有明显下降（HYPO-L Metaphor F1 降 5.0%），验证了交互机制的必要性。
- **LLM 规模分析**：随着 Llama 模型增大（1B→70B），性能持续提升，且 EmoBi 的收益比标准 Prompt 更显著。
- **不同 LLM 对比**：GPT-4o 持续优于 Llama3-8b，但 EmoBi 在所有模型上均大幅超越对应基线。
- **交互机制对比**："Separate"（分别检测）< "Together"（同时检测）< "Ours"（双向动态交互），证明显式双向交互优于隐式多任务或独立检测。

## 相关工作脉络
- **Badathala et al. (2023)**：提出 MTL-F 多任务框架，同时检测夸张和隐喻，通过共享层实现任务间特征融合；本文定位差异：不仅依赖表层特征共享，更引入情感引导和显式双向动态交互，弥补其对深层语义和交互过程建模的不足。
- **Birke & Sarkar (2006)、Mohler et al. (2016)、Troiano et al. (2018)**：早期独立处理隐喻或夸张的基准数据集和工作；本文定位差异：不再孤立处理单一修辞类型，而是联合建模两者的交互关系。
- **Tian et al. (2024)、Zhang et al. (2024b)**：基于可解释词对/ grounding 的隐喻检测方法；本文定位差异：引入情感先验和 LLM 推理链，关注跨修辞类型的联合检测和语义深度挖掘。
- **Standard Prompting / CoT Prompting**：基础和大模型思维链提示方法；本文定位差异：在 CoT 基础上进一步结构化推理过程，引入情感知识和域映射作为中间知识，而非仅让 LLM 自由生成推理链。
- **Dankers et al. (2019)**： multitask 建模隐喻与情感关系；本文定位差异：将其思想扩展到夸张+隐喻的联合检测，并提出具体的双向动态交互机制和域映射策略。
- **Schneidermann et al. (2023)**：在预训练语言模型中探测夸张；本文定位差异：聚焦大模型时代下的多步推理框架设计，而非 probing 固定架构。

## 局限性与未来方向
- **误差传播问题**：多步推理链（情感→域映射→交互）中，前面步骤的错误会传递到后续步骤，影响最终结果。
- **情感分析精度有限**：当前情感分析模块依赖 LLM 输出，可能无法准确捕捉所有细微情感（如反讽中的表层正面、深层负面）。
- **未来方向**：提升情感分析的准确性和鲁棒性；探索误差容忍机制（如 confidence scoring、自我修正）；将框架扩展到其他修辞手法（如反讽、讽刺）的检测。

## 研究启发与可借鉴点
- **情感先验引导框架**：将情感分析作为中间知识引入其他修辞或隐义理解任务（如反讽检测、语用推理），具有高度可迁移性。
- **双向动态交互设计**：将"任务 A 辅助任务 B，任务 B 反过来增强任务 A"的思想应用到其他耦合任务对（如立场检测与情感分析联合建模）。
- **结构化 LLM 推理链**：用显式多步 Prompt 替代端到端黑盒预测，不仅提升性能，也增强了可解释性，适合需要审计的 NLP 任务。
- **域映射作为可解释工具**：源域/目标域的识别结果本身可作为人类可理解的中间解释，对下游应用（如智能聊天机器人、情感分析系统）有直接价值。
- **错误传播缓解策略**：本文自述的误差传播问题可作为独立研究方向，探索多步 LLM 推理中的置信度校准与自我纠错机制。

## 关键术语表
- **Hyperbole（夸张）**：一种修辞手法，通过明显夸大事实来表达情感或强调，如"等了一万年"。
- **Metaphor（隐喻）**：一种修辞手法，通过将某事物描述为另一事物来实现跨域概念映射，如"时间是小偷"。
- **Bidirectional Dynamic Interaction（双向动态交互）**：夸张与隐喻在推理过程中相互促进的机制，一方的信息增强另一方的判断。
- **Source Domain / Target Domain（源域/目标域）**：隐喻理论中，源域是被借用的概念领域，目标域是被描述的概念领域。
- **Emotion Guidance（情感引导）**：利用情感分析结果作为先验知识，指导后续域映射和修辞检测的框架设计思路。
- **Verification Mechanism（验证机制）**：检测框架中的纠错环节，在发现矛盾时重新评估推理结果以提升可靠性。
- **MTL-F（Multi-Task Learning with Fully Shared Layers）**：多任务学习基线方法，通过共享网络层同时处理夸张和隐喻检测。
- **CoT Prompting（Chain-of-Thought Prompting）**：引导大模型逐步推理的提示技术，本文对比的基线方法之一。

## 可复现要素
- **数据集**：HYPO、HYPO-L、LCC、TroFi 均为公开数据集，可从原文引用处获取。
- **代码/权重**：论文未提及开源代码和模型权重。
- **关键超参**：论文未详细说明 LLM 的具体型号、温度、最大 token 数等推理超参，仅提及使用了 Llama 系列不同规模和 GPT-4o。
- **LLM 依赖**：框架依赖 LLM 进行多步生成，具体实现需根据 LLM API 或本地部署配置调整。
