---
title: "AndroidLab-Training-and-Systematic-Benchmarking-of-Android-A"
source: https://aclanthology.org/2025.acl-long.107.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:48:53"
field: "移动端自主智能体"
keywords: ["Android Agent", "Mobile AI", "Instruction Tuning", "Vision-Language Model", "Benchmark", "Reinforcement Learning"]
innovations: ["统一操作空间支持LLM/VLM双模态公平对比", "离线可复现基准测试设计消除外部依赖", "Android Instruct指令微调数据集弥合开源与闭源模型性能差距"]
benchmarks: ["ANDROIDLAB", "138 tasks across 9 apps"]
---

# 论文速读：AndroidLab-Training-and-Systematic-Benchmarking-of-Android-A

## 一句话总结
ANDROIDLAB是首个面向Android自主智能体的统一训练与系统评估框架，提供可复现的双模式操作环境（XML/SoM）、138任务的基准测试及Android Instruct微调数据集，使开源小模型微调后性能显著逼近闭源旗舰模型。

## 研究问题与动机
1. 现有Android智能体研究缺乏系统性的训练与评估统一框架，多数工作仅聚焦于闭源模型的prompt优化。
2. 已有基准测试（PixelHelp、MetaGUI、AITW等）多为静态截图预测，无法评估真实交互能力。
3. 开源模型在移动端操作任务上表现极弱（LLM平均成功率仅4.59%），缺乏有效的训练数据与评估标准。
4. 不同模态（纯文本LLM与多模态VLM）的智能体缺乏统一的操作空间定义，难以公平对比。

## 核心贡献（创新点）
1. 提出统一操作框架，支持XML/SoM双模式与ReAct/SeeAct双智能体架构，使LLM和VLM共享相同操作空间——本质区别在于首次实现"同动作空间、不同感知通道"的公平对比。
2. 构建可复现基准测试（138任务/9应用/离线预置数据/固定时间与地理位置）——区别于AndroidWorld等需要网络或存在非确定性的工作，本框架消除外部依赖。
3. 开发Android Instruct数据集（10,500轨迹/94,300步骤，其中6,208步骤用于微调）——首次提供专为移动端操作的指令微调数据，且明确解决隐私保护问题。
4. 设计任务级评估体系（Sub-SR/ROR/RRR）——突破以往仅用路径匹配的粗放评估，引入操作效率维度。

## 方法详解
**统一操作空间：** 定义4个基础操作（Tap、Swipe、Type、Long Press）、2个快捷键（Home、Back）及1个终止操作（Finish），LLM与VLM共享此动作空间。

**双模式设计：**
- XML模式：将屏幕UI转化为压缩XML文本，LLM直接选取对应元素执行操作。
- SoM模式：基于Set-of-Mark方法，为每个可交互元素分配序列号标注在截图中，VLM通过编号选择目标。

**智能体架构：**
- ReAct模式：模型先输出思维链（Thought），再执行操作。
- SeeAct模式：两阶段交互，第一轮生成详细描述，第二轮输出函数调用格式的操作。

**任务形式化：** 定义为4元组 T(E, I, F, M)，其中E为预配置AVD、I为自然语言指令、F为测试框架、M为骨干模型。

**评估指标：**
- Success Rate (SR)：任务整体完成比例。
- Sub-Goal Success Rate (Sub-SR)：子目标逐段完成比例。
- Reversed Redundancy Ratio (RRR)：操作路径长度与人类基准之比取倒数，越高越精简。
- Reasonable Operation Ratio (ROR)：导致屏幕变化的操作占比，衡量无效操作比例。

**数据构建流程：** 任务派生与扩展→LLM/VLM自探索（保留约500条轨迹训练奖励模型，剔除自探索数据）→人工标注（指令检查、熟悉界面、任务执行、交叉验证）。

## 实验与结果
**数据集：** 9个离线应用（Bluecoins、Calendar、Cantook、Clock、Contacts、Maps.me、PiMusic、Settings、Zoom），共138任务（93操作型+45查询型）。

**评估基线：** XML模式测试8个模型（GPT-4o、GPT-4-1106-Preview、Gemini系列、GLM4-PLUS、LLaMA3.1-8B、Qwen2-7B、GLM4-9B）；SoM模式测试9个模型（含Claude-3.5-Sonnet等）。

**最强结果：**
- XML模式：GPT-4-1106-Preview达到最高SR 31.16%、Sub-SR 38.21%。
- SoM模式：GPT-4o达到最高SR 31.16%、Sub-SR 35.02%；Claude-3.5-Sonnet在RRR达113.41，优于GPT-4o（87.32）。

**微调效果：**
- LLM平均SR从4.59%提升至21.50%（提升约17个百分点）。
- VLM平均SR从1.93%提升至13.28%（提升约11个百分点）。
- 微调后LLaMA3.1-8B-ft在XML模式SR达23.91%，Qwen2-VL-7B-ft在SoM模式SR达18.12%。

**框架分析：** ReAct仅在XML模式显著提升性能（GPT-4o从25.36%→33.33%）；SeeAct在VLM模式下效果不稳定。微调后模型平均生成token仅4.96，远低于ReAct的67.89和SeeAct的129.12。

## 相关工作脉络
1. PixelHelp (Li et al., 2020)、MetaGUI (Sun et al., 2022)：静态截图预测基准，无真实交互能力。
2. AndroidEnv (Toyama et al., 2021)：首个动态评估环境，但缺乏标准化与可复现设计。
3. AndroidArena (Xing et al., 2024)：支持跨应用评估并标准化AVD，但未提供指令微调数据。
4. AndroidWorld (Rawles et al., 2024)：提供116任务+奖励信号，但任务需网络连接且不支持instruction tuning。
5. AppAgent (Yang et al., 2023b)：提出XML操作空间与ReAct框架，但未构建系统化基准与开源训练数据。
6. CogAgent (Hong et al., 2023)：多模块协作的VLM智能体，在AITW上达SOTA但为闭源方案。

## 局限性与未来方向
1. 任务扩展性有限：所有138个任务均为硬编码，新增任务需手动集成，难以规模化。
2. 固定等待策略：操作后采用固定等待时间，无法适应不同Android设备的响应差异。
3. 单平台限制：框架仅支持Android，无法跨平台（如iOS）评估。
4. 潜在风险：虽规避支付/消息等敏感操作，但实际部署需增加用户显式授权机制。
5. 部分应用XML质量不佳：动态元素（如播放条）可能导致XML获取超时或内容缺失。

## 研究启发与可借鉴点
1. 双模式统一操作空间设计值得借鉴：同一动作空间适配不同模态模型，可用于跨模态公平对比实验。
2. 任务分解+UI树结构匹配的评估方法：将复杂任务拆分为子目标并精确判定完成状态，可迁移至其他交互型Agent任务评估。
3. 离线可复现基准设计：通过预置数据和固定设备状态消除网络/时间依赖，对移动端Agent研究具有通用参考价值。
4. 数据构建中"奖励模型辅助筛选"的策略：先用大模型自探索生成候选轨迹，再用小规模人工标注数据训练奖励模型自动筛选，可大幅降低数据构建成本。
5. 隐私保护与开源数据集构建的平衡：人工标注+敏感信息脱敏+本地存储的设计思路，可为移动端隐私敏感数据收集提供参考。

## 关键术语表
**SoM (Set-of-Mark)**：在多模态输入截图中为每个可交互元素添加序列号标注，引导VLM精准定位目标。
**XML Mode**：将Android屏幕UI树压缩为文本格式，供纯文本LLM解析并执行操作。
**ReAct**：让模型在输出操作前先进行思维链推理（Thought）的操作范式。
**ROR (Reasonable Operation Ratio)**：导致屏幕发生变化的操作占总操作的比率，衡量操作有效性。
**RRR (Reversed Redundancy Ratio)**：操作路径长度与人类基准之比的倒数，值越高表示操作越精简。
**Sub-Goal**：将完整任务拆解为多个阶段性子目标，用于细粒度评估。
**Android Instruct**：基于ANDROIDLAB构建的指令微调数据集，包含726条轨迹和6208步操作数据。
**AVD (Android Virtual Device)**：预配置好应用历史数据、固定时间与地理位置的虚拟设备镜像。

## 可复现要素
- 数据集：Android Instruct数据集已开源（部分受隐私保护未完全公开，约6,208步骤标注数据用于训练）
- 代码：https://github.com/THUDM/Android-Lab 已开源
- 关键超参：batch_size=32，max_seq_len=4096，epochs=5，learning_rate=1e-5
- 微调模型：LLaMA3.1-8B、GLM-4-9B、Qwen2-7B（LLM）；LLaMA3.2-11B-Vision、Qwen2-VL-7B、CogVLM2（VLM）
- 设备设置：最大执行步数25步，每次操作后等待3秒
- 任务分布：93个Operation Task + 45个Query Task
