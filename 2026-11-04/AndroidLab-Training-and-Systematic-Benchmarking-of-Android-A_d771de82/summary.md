---
title: "AndroidLab-Training-and-Systematic-Benchmarking-of-Android-A"
source: https://aclanthology.org/2025.acl-long.107.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 07:30:10"
field: "多模态Agent与移动端交互"
keywords: ["Android Agent", "Mobile GUI Agent", "Vision-Language Model", "Benchmark", "Instruction Tuning", "SoM", "ReAct"]
innovations: ["提出统一XML/SoM双模式的Android agent操作框架，共享action space", "构建可复现的138任务benchmark与Android Instruct微调数据集", "证明开源小模型经微调后可接近GPT-4o等闭源模型的Android操作能力"]
benchmarks: ["ANDROIDLAB"]
---

# 论文速读：AndroidLab-Training-and-Systematic-Benchmarking-of-Android-A

## 一句话总结
本文提出了ANDROIDLAB，一个面向Android自主agent的统一训练与评测框架，包含标准化操作环境、可复现benchmark（138个任务、9个App）以及Android Instruct微调数据集，显著提升了开放模型在Android操作任务上的性能。

## 研究问题与动机
- **缺乏统一框架**：现有研究多依赖静态环境（如PixelHelp、MetaGUI）或仅支持closed-source模型，无法进行可复现的交互式评估与训练。
- **平台割裂**：LLM和VLM各自使用不同的交互方式（XML vs. 截图），缺乏统一action space导致模型能力难以公平比较。
- **不可复现性**：真实设备受网络、时间、账号等因素影响，难以保证实验可重复；AndroidWorld等虽有reward信号但未支持instruction-tuning。
- **开源模型能力不足**：开源LLM/VLM在Android操作上性能远低于GPT-4o等闭源模型，亟需有效的训练数据与评测标准。

## 核心贡献（创新点）
1. **提出ANDROIDLAB统一框架**：设计了XML mode与SoM mode两种操作模式，共享同一action space，使LLM和VLM可在相同维度上进行公平比较与训练。
2. **构建可复现benchmark**：138个任务覆盖9个常用App，采用预装历史数据的离线AVD，通过UI tree结构匹配精确评估子目标完成情况。
3. **开发Android Instruct数据集**：收集10,500条trace、94,300步操作，其中6,208步来自benchmark内App，用于fine-tuning开源模型。
4. **系统性评估与提升**：测试17个open/closed模型，证明fine-tuning可使LLM平均成功率从4.59%提升至21.50%，VLM从1.93%提升至13.28%。

## 方法详解
- **Action Space**：包含Tap、Swipe、Type、Long Press、Home、Back、Finish七个基本操作，统一适用于所有模式。
- **XML Mode**：专为纯文本模型设计，通过压缩XML展示屏幕信息，模型直接选择对应UI元素执行操作。
- **SoM Mode**：基于Set-of-Mark方法，为每个可点击/聚焦元素分配序列号，VLM通过序号选择目标元素，所选元素与XML模式一一对应。
- **ReAct框架**：让模型在输出action前进行逐步思维链推理（Thought），支持XML和SoM两种输入模式。
- **SeeAct框架**：两阶段交互——第一轮生成详细操作描述，第二轮输出function-call格式的实际动作，缓解多模态推理困难。
- **任务形式化**：每个任务定义为四元组T(E, I, F, M)，其中E为预配置AVD，I为自然语言指令，F为agent框架，M为backbone模型；分为Operation Task（输出状态序列）和Query Task（输出答案）。
- **评估指标**：Success Rate（SR）、Sub-Goal Success Rate（Sub-SR）、Reversed Redundancy Ratio（RRR）、Reasonable Operation Ratio（ROR），通过UI tree匹配判断子目标完成状态。

## 实验与结果
- **数据集**：ANDROIDLAB benchmark含138个任务（93 Operation + 45 Query），覆盖Bluecoins、Calendar、Cantook、Clock、Contacts、Maps.me、PiMusic、Settings、Zoom共9个App，全部离线可用。
- **基线模型**：闭源模型包括GPT-4o、GPT-4-1106-Preview、Gemini-1.5-Pro、Claude-3.5-Sonnet等；开源模型包括Llama-3.1-8B、GLM-4-9B、Qwen2-7B及其视觉版本。
- **最强结果**：XML模式下GPT-4-1106-Preview以31.16% SR领先；SoM模式下GPT-4o同样以31.16% SR并列第一，且RRR达107.45。
- **Fine-tuning效果**：LLM平均SR从4.59%提升至21.50%（+16.91%），VLM从1.93%提升至13.28%（+11.35%）；GLM4-9B-ft在XML mode达21.01% SR，Qwen2-VL-7B-ft在SoM mode达18.12% SR。
- **框架分析**：XML+ReAct对GPT-4o提升最显著（SR从25.36%→33.33%）；SeeAct在多模态下提升有限；微调后模型平均生成token仅4.96，远低于ReAct的67.89和SeeAct的129.12。

## 相关工作脉络
1. **PixelHelp / MetaGUI**：早期静态benchmark，依赖单步预测无需实际交互；ANDROIDLAB提供动态交互式环境。
2. **AITW / AndroidWorld**：大规模数据集但前者仅离线数据、后者不支持instruction-tuning；本文填补了可训练开源agent的benchmark空白。
3. **AndroidEnv / Android Arena**：前者为RL平台后者标准化AVD；本文进一步统一了LLM/VLM的action space并开源了训练数据。
4. **CogAgent / AppAgent**：依赖闭源模型设计交互逻辑；本文证明开源模型经微调后可接近闭源性能。
5. **Mind2Web / WebArena**：Web环境agent benchmark；本文将其思想迁移至移动端Android场景。
6. **ReAct / SeeAct**：原为Web agent设计；本文适配至Android环境并验证了XML+ReAct的有效性。

## 局限性与未来方向
- **任务扩展性有限**：所有138个任务为预定义hardcode，新增任务需人工逐个集成，难以大规模扩展。
- **固定等待时间**：操作后固定3秒等待，未考虑不同Android设备/型号/配置的响应速度差异。
- **缺乏跨平台能力**：仅支持Android，iOS等其他平台UI tree结构不统一，难以直接迁移。
- **XML质量不稳定**：部分App存在动态元素（如播放条）导致XML获取超时或内容不全。
- **潜在风险**：支付、发消息等敏感操作未在benchmark中测试，实际部署需增加用户显式确认机制。

## 研究启发与可借鉴点
1. **统一action space设计**：XML与SoM模式共享同一套操作语义，为多模态agent研究提供了"不同输入、相同输出"的可比评测范式。
2. **子目标分解评估**：将复杂任务拆解为多个sub-goal并通过UI tree精确匹配，提升了评估的细粒度与可解释性，可迁移至桌面/Web agent评测。
3. **离线AVD保证可复现性**：通过预装历史数据、固定时间与地理位置，消除外部依赖，此设计对移动端agentbenchmark建设具有示范价值。
4. **指令微调驱动开源模型追赶**：仅用6,208步高质量数据即让开源小模型接近闭源水平，证明了高质量domain data对agent训练的关键作用。
5. **自标注工具链设计**：基于Android Accessibility Service实现XML精确获取，并结合reward model辅助验证，为移动端数据收集提供了工程参考。

## 关键术语表
- **AndroidLab**：本文提出的统一Android agent训练与评测框架，包含操作环境、benchmark和训练数据集。
- **SoM (Set-of-Mark)**：在截图上为可交互元素添加标记序号的视觉标注方法，供VLM进行元素定位。
- **XML Mode**：通过压缩XML结构化表示屏幕信息的操作模式，适用于纯文本LLM。
- **ReAct**：将Reasoning与Acting结合的agent框架，允许模型在每一步输出思维链后再执行动作。
- **SeeAct**：两阶段交互框架，先描述操作意图再输出具体action，用于缓解多模态grounding难度。
- **RRR (Reversed Redundancy Ratio)**：衡量操作效率的指标，值为模型步长与人类基准步长之比的倒数，越高越高效。
- **ROR (Reasonable Operation Ratio)**：屏幕实际发生变化的操作占比，反映操作的合理有效性。
- **Android Instruct**：基于ANDROIDLAB收集的指令微调数据集，包含任务、屏幕状态、XML和操作轨迹。

## 可复现要素
- **数据集**：ANDROIDLAB benchmark和Android Instruct数据集已公开，见https://github.com/THUDM/Android-Lab。
- **代码/权重**：框架代码开源；fine-tuned模型权重需自行使用提供的数据训练。
- **关键超参**：batch size=32，max sequence length=4096，epochs=5，learning rate=1e-5；每任务最大25步，间隔3秒等待。
- **设备配置**：使用Android虚拟机（AVD），固定屏幕尺寸、Android版本和API level。
