---
title: "Can-Multimodal-Large-Language-Models-Understand-Spatial-Rela"
source: https://aclanthology.org/2025.acl-long.31.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 07:31:09"
field: "多模态理解与推理"
keywords: ["多模态大语言模型", "空间关系推理", "基准构建", "视角转换", "视觉问答"]
innovations: ["提出无边界框、含视角转换、拒斥纯先验知识的多选题空间推理基准 SpatialMQA", "设计基于客观世界的六维空间关系形式化定义与三步人工质检流程", "系统性评测揭示当前 MLLM 在空间轴不平衡、视角换位、常识推理等多类错误上存在显著短板"]
benchmarks: ["SpatialMQA", "COCO2017"]
---

# 论文速读：Can Multimodal Large Language Models Understand Spatial Relations?

## 一句话总结
本文提出一个新的多模态空间关系推理基准 **SpatialMQA**（5,392 样本，基于 COCO2017，无边界框、含视角转换、拒绝纯先验知识可答），并通过系统性评测指出当前最强 MLLM（SpaceLLaVA-LoRA）仅达 **48.14%** 准确率，与人水平（98.40%）差距显著，揭示了 MLLM 在客观空间推理上的根本性不足。

## 研究问题与动机
1. **现有基准过度依赖边界框（bbox）**：如 SpatialVOC2K、Rel3D、SpatialSense+ 等需 bbox 标注主体/客体，但真实场景中主体/客体可能不可见（如“太阳”），bbox 方案天然受限。
2. **部分空间标注脱离客观世界**：例如 SpatialSense 将“天空位于森林上方”标注为“后方”，违背物理常识，导致评测与人类认知脱节。
3. **忽视视角转换（Perspective Substitution）**：已有无 bbox 基准（如 VSR）仅少量涉及第一人称视角（6%），无法评估模型从不同观察者位置理解空间关系的能力，而这在自动驾驶等动态场景中至关重要。
4. **存在可被纯先验知识回答的“捷径题”**：部分问题无需图像即可凭常识作答（如“书在公交车上方”），使模型能绕过视觉理解获得高分，削弱基准的区分力。

## 核心贡献（创新点）
1. **提出无 bbox 的多选题空间推理基准 SpatialMQA**：与 SpatialSense、VSR 等依赖 bbox 或二值判断的基准本质不同，本基准直接要求模型在多选项中选择正确空间关系，更贴近真实问答场景。
2. **引入基于客观世界的视角转换设计**：明确区分“图外视角”与“图内视角（第一人称/第三人称）”，60% 样本包含视角转换，优于 VSR（仅 6% 第一人称）和 EmbSpatial（无视角），使评测能真正检验模型的空间换位推理能力。
3. **设计严格的去先验知识过滤机制**：通过两步检查（是否可纯文字回答、主体/客体是否清晰可见）确保 97% 以上问题必须依赖图像，解决了 VSR、SpatialVLM 等基准中“常识可答”的污染问题。
4. **提供全面的基线评测与误差分析**：覆盖 7 个开源 + 2 个闭源 MLLM，并细化到 4 类错误（IRS O、FRS、LCR、IILN），为后续研究指出明确瓶颈方向。

## 方法详解
- **空间坐标系统（SCS）定义**：以重力向下为 Z 轴正向，观察者为原点，X 轴左负右正，Y 轴后负前正，Z 轴下负上正。主体 $(x_s,y_s,z_s)$ 与客体 $(x_o,y_o,z_o)$ 比较各轴坐标差判定关系。
- **六种空间关系逻辑**：
  - **left/right of**：$x_s < x_o$ 为左，$x_s > x_o$ 为右。
  - **in front/behind**：基于 Y 轴相对位置，需同时考虑同侧（$y_s·y_o>0$）与异侧（$y_s·y_o<0$）情况。
  - **on/above vs below**：$z_s > z_o$ 为 on/above，$z_s < z_o$ 为 below。
- **三步人工标注流程**：
  1. **初标**：3 名标注员按模板生成问题、选项并标注答案。
  2. **复核**：2 名检查员并行校验合理性，并专项排查“纯先验可答”与“主体/客体不可见”两类劣质样本，直至批次准确率达 90%。
  3. **终审**：主作者随机抽查 20% 样本，修正标准，循环至批次准确率达 95%。
- **问题类型划分**：
  - **Out-of-image（Q1）**：观察者在图外，模板如“Where is X located relative to Y?”。
  - **In-image 第一人称（Q2）**：以客体为观察者，“If you are [object] in the image, where is subject relative to you?”。
  - **In-image 第三人称（Q3）**：以图中其他生物为观察者，模板如“If you are the [living being]... from your perspective...”。

## 实验与结果
- **数据集**：SpatialMQA，5,392 样本（Train/Dev/Test = 7:1:2），128 类主体/客体类型，6 种空间关系，60% 含视角转换。
- **评测基线**：开源 MLLM（BLIP-vqa-base、BLIP2-opt-2.7B、InstructBLIP-3B、mPLUG-Owl-7B、IDEFICS-9B、LLaVA1.5-7B、SpaceLLaVA）；闭源 MLLM（Gemini-1.5-flash、GPT-4o）；Random 与 Human（500 样本投票）。
- **主要结果**：
  - 最佳开源模型 **SpaceLLaVA（LoRA）** 准确率达 **48.14%**，较未指令微调版本提升 16.82%。
  - 最佳闭源模型 **GPT-4o（0-shot）** 准确率为 **40.20%**，且随 ICL 样本增加反而下降。
  - 人类在相同 500 样本上准确率为 **98.40%**，纯文本（无图像）人类准确率仅 24.40%，与随机接近，验证基准无先验知识污染。
- **关键发现**：
  - 模型在不同空间维度（X/Y/Z 轴）表现极不均衡：SpaceLLaVA 在 X 轴（left/right）达 56%，但在 Y 轴（front/behind）仅 51.85%，Z 轴（on/above/below）仅 31.41%，相差 24.59%。
  - 视角难度：Q1（图外）> Q2（第一人称）> Q3（第三人称），所有模型在 Q3 上表现最差。
  - ICL 示例对齐性影响显著：GPT-4o 在 3-shot 对齐示例下比不对齐高 4.01%，而 Gemini 对不对齐示例更鲁棒。

## 相关工作脉络
1. **SpatialSense / SpatialSense+ / Rel3D**：依赖 bbox 的二值判断基准，主体/客体不可见时失效，且标注常违背客观物理关系（如天空标为“后方”）。
2. **VSR（Visual Spatial Reasoning）**：无 bbox 但视角覆盖极窄（仅 6% 第一人称），且存在可纯文本回答的常识题，评测有效性存疑。
3. **EmbSpatial / SpatialVLM**：无 bbox 但忽略视角转换，或测试集过小（SpatialVLM 仅 546 样本）且未公开。
4. **SpatialRGPT**：客观世界参考但未强制要求图像依赖，且同样无视角子集设计。
5. **CLEVR / NLVR2 / GQA**：合成或通用视觉问答基准，空间关系刻画简单或脱离真实场景分布，无法评估复杂多实体空间推理。

## 局限性与未来方向
1. **数据规模受限**：高质量人工标注导致训练集仅 3,780 样本，不足以对大型 MLLM 进行充分微调，未来需探索自动化弱监督标注方法。
2. **空间关系类型单一**：仅覆盖 6 种基本关系（左/右/前/后/上/下），未涉及更复杂的组合关系（如“左前方”“之间”“靠近”等）。
3. **第三人称视角样本极少**：因需图中存在三个独立生物，Q3 仅 185 条（5.76%），限制了该方向的充分评估。
4. **未来方向**：扩展关系类型、构建更大规模自动标注管道、设计针对视角转换的专用训练数据、研究空间坐标系统一表征学习。

## 研究启发与可借鉴点
1. **去先验知识污染的检验机制**：通过“纯文本回答准确率≈随机”验证基准质量，该方法可迁移至其他多模态评测构建中，避免模型走捷径。
2. **视角转换的结构化设计**：将观察者位置显式分为图外/图内第一人称/图内第三人称，并配套不同模板，为具身智能、自动驾驶等需要换位思考的任务提供评测范式。
3. **多维度误差分解**：按空间轴（X/Y/Z）、问题类型（Q1/Q2/Q3）交叉分析，能快速定位模型薄弱环节，指导针对性微调或架构改进。
4. **高质量人工三循环质检流程**：初标→双检（含专项排查）→作者终审，直至 95% 准确率，可作为小规模高质基准构建的标准操作流程。

## 关键术语表
- **SpatialMQA**：本文提出的多模态空间关系推理多选题基准，基于 COCO2017，5,392 样本，无边界框，含视角转换。
- **SCS（Spatial Coordinate System）**：以观察者为原点、重力向下为 Z 轴的三维笛卡尔坐标系，用于形式化定义六种空间关系。
- **Perspective Substitution（视角转换）**：问题中观察者位置的变化，包括图外视角、第一人称（客体为观察者）、第三人称（图中其他生物为观察者）。
- **IRSO（Incorrect Recognition of Subjects and Objects）**：模型误识别问题中主体或客体的错误类型。
- **FRS（Failure in Perspective Substitution）**：模型无法正确进行空间换位推理的错误类型，占比最高。
- **LCR（Lack of Commonsense Reasoning）**：模型违反物理常识（如将“椅子在桌子上”判为“下方”）的错误。
- **IILN（Incorrect Identification of Spatial Relations for Letters and Numbers）**：模型对字母/数字符号的空间位置判断错误。

## 可复现要素
- **数据集**：SpatialMQA（COCO2017 子集，5,392 样本）已公开于 Hugging Face：`https://huggingface.co/datasets/liuziyan/SpatialMQA`
- **代码**：论文声明代码可用，具体仓库链接见正文。
- **关键超参**：开源模型微调使用 LoRA（除 BLIP 全参数），Epoch=10~30，Batch Size=8，学习率 2e-4~6e-7，优化器 AdamW/PAW8，学习率调度 Cosine/ExpLR/Linear Decay（详见 Table 12）。
- **评测设置**：闭源模型采用 0/1/2/3-shot ICL，开源模型采用直接推理与指令微调两设置，均报告 P/R/F1/Acc，取三次运行平均。
