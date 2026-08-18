---
title: "Can-Multimodal-Large-Language-Models-Understand-Spatial-Rela"
source: https://aclanthology.org/2025.acl-long.31.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:52:19"
field: "多模态空间理解"
keywords: ["空间关系推理", "多模态大语言模型", "基准评测", "视角代换", "SCS"]
innovations: ["提出无边界框、含视角代换且排除先验知识捷径的多选题空间关系基准SpatialMQA", "建立三轮质量控制标注流程以确保数据质量"]
benchmarks: ["SpatialMQA", "COCO2017"]
---

# 论文速读：Can-Multimodal-Large-Language-Models-Understand-Spatial-Rela

## 一句话总结
论文提出了 **SpatialMQA**，一个基于 COCO2017 的人工标注多模态空间关系推理基准，采用多选题格式、无需边界框、涵盖视角代换且无法仅凭先验知识作答；测试表明当前最优 MLLM（SpaceLLaVA-LoRA，48.14%）与人类水平（98.40%）差距巨大，凸显 MLLM 在空间关系理解上仍有显著不足。

## 研究问题与动机
- **现有基准依赖边界框（bboxes）**：SpatialVOC2K、Rel3D、SpatialSense+ 等均需 bboxes 标注主体/客体，但图像中某些对象（如"太阳"）无法用框框住，且 bboxes 可能使模型绕过真正的图像理解即可答题。
- **缺乏基于客观世界的视角代换**：既有无框基准（EmbSpatial、VSR、SpatialVLM）通常忽略第一人称/第三人称视角代换，VSR 中仅有 6% 的数据涉及视角代换，限制了模型在动态场景（如自动驾驶）中的评估。
- **部分题目可脱离图像仅凭先验知识作答**：如"书在公交车上方"可完全靠常识回答"No"，导致无法有效评测模型的图像理解能力。
- **空间关系标注未统一以客观世界为参照系**：部分基准（如 SpatialSense）使用主观坐标导致标注与客观世界认知存在偏差（例如"天空在森林后面"）。

## 核心贡献（创新点）
- **提出无边界框的多选题空间关系推理基准 SpatialMQA**：基于 COCO2017 构建 5,392 条高质量样本，覆盖 128 种主/客体类型，与依赖 bbox 的基准形成本质区别。
- **引入基于客观世界参照系的视角代换问题设计**：涵盖"图外视角"（Q1）、"图中第一人称视角"（Q2）和"图中第三人称视角"（Q3），解决了既有基准对视角代换覆盖不足的问题。
- **严格排除可脱离图像仅凭先验知识作答的题目**：通过两阶段人工校验确保所有样本必须依赖图像，使基准更纯粹地评测多模态理解能力。
- **系统评测主流开源/闭源 MLLM 并揭示多维度错误类型**：包括主体/客体识别错误（IRSO）、视角代换失败（FRS）、常识推理缺失（LCR）、字母数字空间识别错误（IILN），为后续研究提供诊断性分析。

## 方法详解
- **空间坐标系统（SCS）定义**：以重力向下、观察者为原点建立坐标系：X 轴从左（负）到右（正），Y 轴从后（负）到前（正），Z 轴从下（负）到上（正）。由此精确定义六种空间关系：
  - **left of / right of**：$x_s < x_o$ 为 left of，$x_s > x_o$ 为 right of。
  - **in front of / behind**：基于 Y 轴距离判断，满足 $y_s \cdot y_o > 0, |y_s| - |y_o| < 0$ 或 $y_s \cdot y_o < 0, y_s > 0 > y_o$ 时为 in front of；反之则为 behind。
  - **on/above / below**：$z_s > z_o$ 时为 on/above，$z_s < z_o$ 时为 below。
- **题型划分**：Q1（Out-of-image，图外视角）、Q2（In-image first-person，以客体为观察者）、Q3（In-image third-person，以图中第三生命体为观察者）。
- **标注流程**（三轮质量控制）：
  1. 第一轮：3 名 annotator 分别处理 10,000 张图，按指南生成问题、选项和答案。
  2. 第二轮：2 名 checker 同时校验，一人检查是否可脱离图像作答，一人检查主体/客体是否清晰可见；直至批次准确率 ≥ 90%。
  3. 第三轮：主作者抽检 20% 并反馈修正；直至批次准确率 ≥ 95%，最终获得 5,392 条样本。
- **数据划分**：Train 3,780 / Dev 536 / Test 1,076，比例 7:1:2。

## 实验与结果
- **数据集**：COCO2017，共 5,392 样本，空间关系分布：left of（27.52%）、right of（25.93%）、in front of（14.95%）、behind（14.00%）、on/above（9.33%）、below（8.27%）。
- **评估基线**：开源 MLLM（BLIP-vqa-base、BLIP2-opt-2.7B、InstructBLIP-3B、mPLUG-Owl-7B、IDEFICS-9B、LLaVA1.5-7B、SpaceLLaVA）+ 闭源 MLLM（GPT-4o、Gemini-1.5-flash）+ Random + Human（500 样本投票）。
- **主要结果**：
  - **最强模型**：SpaceLLaVA（LoRA 指令微调）准确率 **48.14%**，较 LLaVA1.5-7B（LoRA，46.85%）提升 1.29pp；相比之下 BLIP-vqa-base 仅 26.49%。
  - **闭源模型**：GPT-4o 0-shot 最佳（40.20%），但随 ICL 样本增加反而下降（3-shot 降至 37.80%）；Gemini-1.5-flash 随 ICL 增多逐步提升（3-shot 达 38.00%）。
  - **人类水平**：98.40%，与最优模型差距达 **50.26pp**。
  - **Text-only 人类回答**仅 24.40%，验证基准几乎不含可仅凭先验知识作答的题目。
- **分维度分析**（Table 5）：SpaceLLaVA 在 Q3（第三人称）得分 58.82% 最高，在 Az（Z 轴关系）仅 31.41%，最大性能差距达 24.59pp，说明不同空间维度和视角的推理能力不均衡。
- **错误类型**：FRS（视角代换失败）占比最高，其次为 IRSO、LCR、IILN。

## 相关工作脉络
- **SpatialVOC2K / Rel3D / SpatialSense+**：依赖 bbox 标注，无法处理不可见对象（如太阳），且部分标注未以客观世界为参照，本文在无框、客观参照系两方面与之区别。
- **SpatialSense (Yang et al., 2019)**：含 17,498 样本但为 True/False 题型，无视角代换，且存在"天空在森林后面"等与客观认知不符的标注；本文用 MQA 格式并引入客观参照。
- **VSR (Liu et al., 2023a)**：无 bbox 但仅 6% 覆盖第一人称视角，且部分题可脱离图像作答；本文全面覆盖视角代换并严格过滤此类题目。
- **SpatialVLM (Chen et al., 2024)**：无框但测试集仅 546 样本，未开源；本文提供 5,392 样本且完全开源。
- **EmbSpatial (Du et al., 2024)**：面向具身任务，仅 6 种空间关系且无视角代换；本文覆盖 6 种关系但扩展至多视角。
- **SpatialRGPT (Cheng et al., 2024)**：以客观世界为参照但仅含 OpenQA 题型；本文引入更严格的 MQA 格式以避免模型"蒙对"。

## 局限性与未来方向
- **规模受限**：人工标注保证质量但训练集仅 3,780 样本，不足以对 MLLM 进行充分微调；自动化工具因现实场景复杂性目前不适用。
- **仅覆盖 6 种基本空间关系**：未涉及更复杂的方位描述（如"对角线"、"介于...之间"），这些基础关系已对当前模型构成挑战，故优先保障深度而非广度。
- **第三人称视角样本极少**（仅 5.76%）：因需图中同时存在三个不同生命体，限制了该类问题的丰富性。
- **未来方向**：扩展更复杂的空间关系类型；提升模型在 Z 轴（on/above/below）和视角代换（Q2/Q3）上的均衡能力；探索更高效的自动化标注方法以扩大规模。

## 研究启发与可借鉴点
- **严格的质量控制流程**：三轮标注（annotator → checker → reviewer）+ 90%/95% 准确率阈值的设计值得借鉴，可迁移到其他基准构建任务。
- **多角度错误分类框架**：IRSO/FRS/LCR/IILN 四分类体系为诊断 MLLM 空间理解缺陷提供了系统性分析工具，可作为后续研究的评测维度。
- **"排除先验知识捷径"的基准设计理念**：通过双人校验确保题目必须依赖图像，这一思路可推广至其他多模态评测（如空间因果推理、场景理解）。
- **ICL 对齐性分析**：本文发现 GPT-4o 在 misaligned ICL 下性能下降而 Gemini 反而提升，揭示了不同模型对示例类型敏感度的差异，提示我们在设计评测时需谨慎选择 ICL 样本。
- **与团队方向的结合机会**：若团队关注空间推理或 MLLM 评测，可借鉴本文的 SCS 定义和视角代换机制，扩展至 3D 场景理解、机器人导航等下游任务。

## 关键术语表
- **SpatialMQA**：本文提出的多模态空间关系推理基准，基于 COCO2017，采用多选题格式，无边界框，含视角代换，共 5,392 条样本。
- **空间坐标系统（SCS）**：以观察者为原点、重力向下为 Z 轴负向建立的三维坐标系，用于统一定义六种空间关系。
- **视角代换（Perspective Substitution）**：将观察者视角从图外移至图中客体（第一人称）或第三方生命体（第三人称）的提问方式。
- **IRSO（Incorrect Recognition of Subject/Object）**：错误识别题目中主体或客体对象的模型错误类型。
- **FRS（Failure in Perspective Substitution）**：无法正确进行视角代换推理的模型错误类型，是本文中最常见的错误。
- **LoRA（Low-Rank Adaptation）**：低秩自适应参数高效微调方法，本文用于对 SpaceLLaVA 等开源模型进行指令微调。
- **ICL（In-Context Learning）**：上下文学习，通过在提示中提供少量示例引导模型输出，本文用于闭源模型的评测。
- **Out-of-image / In-image 视角**：两种问题视角分类，前者观察者在图外，后者观察者在图内（含第一/第三人称）。

## 可复现要素
- **数据集**：SpatialMQA 基于 COCO2017（CC-BY 4.0 许可），数据已开源，链接：https://huggingface.co/datasets/liuziyan/SpatialMQA。
- **代码**：论文声明 benchmark 和代码已开源（链接同上），标注工具细节见 Appendix F。
- **关键超参**（Open-source MLLMs，Table 12）：Epoch=10~30，Batch Size=8，LR=4e-5~6e-7（BLIP）或 2e-4（其他），Optimizer 主要为 AdamW，部分用 PagedAdamW_8bit；SpaceLLaVA：Ep=10，BS=8，LR=2e-4，Cosine LR schedule。
- **硬件**：双 NVIDIA A100-PCIE-40GB GPU。
- **评测设置**：开源模型均提供 Full/LoRA 两种微调设置；闭源模型测试 0-shot/1-shot/2-shot/3-shot。
