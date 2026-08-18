---
title: "Speaking-Beyond-Language-A-Large-Scale-Multimodal-Dataset-fo"
source: https://aclanthology.org/2025.acl-long.112.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:20:55"
field: "多模态对话生成"
keywords: ["非言语沟通", "多模态语言模型", "VQ-VAE", "对话生成", "3D人体建模", "大规模数据集"]
innovations: ["提出VENUS大规模多模态对话数据集，首次同时提供时间对齐文本、FLAME面部参数与SMPL-X全身参数", "提出MARS多模态语言模型，在同一Transformer中联合自回归预测文本、面部与肢体离散token", "设计面部/身体双分支VQ-VAE并引入运动速度损失，量化效果超越Ng et al. (2023)与Guo et al. (2024)"]
benchmarks: ["VMSE", "LVD", "Window Vertex L2", "Diversity", "Variance", "PPL", "BERT-score", "METEOR", "NLL"]
---

# 论文速读：Speaking-Beyond-Language-A-Large-Scale-Multimodal-Dataset-for-Learning-Nonverbal-Cues-from-Video-Grounded-Dialogues

## 一句话总结
本文提出了大规模多模态对话数据集 **VENUS**（含10分钟播客片段的时间对齐文本、3D面部表情与肢体参数），并基于此训练了多模态语言模型 **MARS**，首次实现了在同一自回归框架内联合生成文本与非言语线索（面部表情+上肢/手部动作）。

## 研究问题与动机
1. **现有LLM缺乏非言语建模能力**：大语言模型仍以纯文本交互为主，无法生成或理解面部表情、肢体语言等关键非言语信号。
2. **已有工作侧重面部而忽视全身**：如 Ng et al. (2022, 2023) 仅建模倾听者的面部运动，未覆盖全身肢体语言，限制了沉浸式对话的构建。
3. **大规模带非言语标注的对话数据集缺失**：既有视频对话数据集（YTD-18M、MultiDialog 等）要么规模不足、时长过短，要么缺少精细的3D非言语标注；而手势数据集（BEAT、EMAGE、TalkShow）又缺乏文本-视频对齐的对话上下文。
4. **跨模态token的统一预测难题**：多本量化码本（面部+身体）场景下的自回归生成方式尚未有成熟方案。

## 核心贡献（创新点）
1. **VENUS数据集**：首个大规模、真实对话场景下的多模态非言语数据集，包含89,459段对话、约1,114,328轮次、14,910小时，同时提供时间对齐的文本、FLAME面部参数（156维）和SMPL-X全身参数（179维）。
2. **MARS多模态语言模型**：首次在同一Transformer架构中联合预测文本、面部表情码和肢体码，实现了上下文感知的非言语生成。
3. **针对对话场景的VQ-VAE量化设计**：提出面部VQ-VAE与身体VQ-VAE双分支，引入运动速度损失（L_vel）以保留时序连续性，优于前作Ng et al. (2023)、Guo et al. (2024)。
4. **端到端数据流水线**：从YouTube播客采集→WhisperX语音转录→LightASD说话人检测→EMOCA-v2/OSX参数提取→安全过滤的全自动构建流程，具有强可扩展性。

## 方法详解
**数据流水线**：
- 通过YouTube API采集含"Podcast"关键词频道视频（2015–2023），每频道最多300条，经去重后共869个频道、27,128个视频。
- 每视频按10分钟分段（FPS=25），使用PyAnnote语音日志区分恰好两人对话的视频（F3），WhisperX生成时间对齐英文转录（F4/P2）。
- LightASD检测活跃说话人，YOLO提取人脸/身体ROI，MobileNet基于余弦相似度对齐多说话人帧（P4）。
- EMOCA-v2提取FLAME面部参数（50表达+3下颌，共53维/帧），OSX提取SMPL-X全身参数（27上肢+45×2手部，共117维/帧），经Savitzky–Golay平滑后得到序列 $\hat{M}^f$ 与 $\hat{M}^b$。
- WildGuard安全过滤：累计有害 utterance 超过3分钟则丢弃整段。

**VQ-VAE量化**：
- Encoder/Decoder为1D卷积网络，下采样率 $q=8$，码本 $\mathcal{Z} \in \mathbb{R}^{K \times C}$，K=512为最佳码本大小。
- 损失函数：
  $$\mathcal{L}_{face} = \mathcal{L}_{vq} + \lambda_{recon}^f \mathcal{L}_{recon}^f + \lambda_{vel}^f \mathcal{L}_{vel}^f$$
  其中 $\mathcal{L}_{recon}^f = \lambda_\psi L_1(\psi, \hat{\psi}) + \lambda_{jaw} L_1(\theta^{jaw}, \hat{\theta}^{jaw})$，速度损失 $v(p_l)=p_{l+1}-p_l$ 保持时序平滑；$\beta=0.02$，EMA更新码本。
- 最优配置：面部用L1损失、dim=8、size=512；身体用L1损失、dim=16、size=512。

**MARS模型**：
- 基于LLaMA 3.2 Instruct / Qwen 2.5 Instruct，输入序列按 $T=[x_1^w, x_1^f, x_1^b, x_2^w, \cdots]$ 交错排列。
- 自回归目标：
  $$p(T)=\prod_j p_\theta(x_j^w|T_{<j})\prod_k [p_\theta(x_k^f|T_{<k})\cdot p_\theta(x_k^b|T_{<k})]$$
- 将非言语token作为特殊token加入词表，SFT 50 epochs，batch size=8/GPU，max seq len=4096。

## 实验与结果
**数据集统计（VENUS）**：
- 89,459段对话，1,114,328轮次，7,118,654句，平均21轮/对话，平均170.8词/对话，非言语帧数均值547帧/utterance。
- T-SNE可视化显示面部与肢体token均自然聚类，覆盖中性/积极主导的情感分布。

**VQ-VAE量化评估**（Table 3）：
- 面部 VMSE=0.5106×10⁻¹、LVD=0.4020×10⁻³、w-VL2=0.2339×10⁻⁷、Diversity=7.8430、Variation=0.6236，全面超越Ng et al. (2023) 与 Guo et al. (2024)。
- 身体 VMSE=1.9946×10⁻¹、LVD=0.0962×10⁻⁴、Diversity=1.9998、Variation=0.1956，同样最优。

**MARS对话生成评估**（Table 4）：
- 以LLaMA 3B为例：PPL从5477.0降至926.9，BERT-score从0.818升至0.835，NLL-F从16.504降至8.057，NLL-B从17.574降至5.325。
- Qwen 3B上表现最佳：PPL=800.0，BERT-score=0.839，NLL-F=7.295，NLL-B=4.666。
- 定性结果显示MARS可生成与上下文一致的更丰富文本及对应面部/肢体动作，优于GT。

## 相关工作脉络
1. **多模态LLM**（LLaVA、Qwen-VL、VideoChat）：侧重视觉/视频理解，未涉及非言语生成；本文延伸至非言语连续参数生成。
2. **视频对话生成**（YTD-18M/Champagne、MultiDialog）：仅有文本或浅层视觉信号，无3D非言语参数；VENUS填补该空白。
3. **非言语识别**（Emotion-CLIP、Shafique et al. 2023）：仅做分类/识别，本文实现联合生成。
4. **人体动作合成**（Humantomato、MotionLLM、EMAGE）：聚焦文本→动作，缺乏对话上下文与倾听者双向互动建模；MARS支持对话驱动的联合生成。
5. **倾听者面部建模**（Ng et al. 2022, 2023）：仅面部，本文扩展至全身。
6. **3D手势数据集**（BEAT、TalkShow）：无文本-时间对齐对话；VENUS同时具备大规模对话与3D标注。

## 局限性与未来方向
1. **数据源单一**：主要来自播客频道，情感范围受限（缺少哭泣、愤怒等极端表情），话题以访谈/生活/回顾为主。
2. **伪标签误差**：非言语参数由EMOCA-v2/OSX自动提取，可能存在精度偏差，需人工校准。
3. **未使用全部数据**：受算力限制，仅用3,924视频/69,412 utterance训练，其余数据潜力未释放。
4. **评估指标不充分**：当前NLL/PPL等指标难以完全捕捉非言语交际的复杂语义，需开发更贴近人类感知的评价体系。
5. **未来方向**：引入语音语调等非言语模态、扩展更多话题领域、探索跨语言/跨文化非言语表达、优化隐私脱敏方案。

## 研究启发与可借鉴点
1. **双VQ-VAE分离量化设计**：面部与身体独立码本避免参数维度差异过大导致的表征拥挤，值得在多模态tokenization任务中借鉴。
2. **速度损失（L_vel）保留时序连贯性**：对连续动作参数（不仅是人脸，也包括肢体）加入帧差速度正则，可通用至任何时序动作生成任务。
3. **对话交互token交错策略**：$[x^w, x^f, x^b]$ 三元组交替排列，使LLM同时建模语义-表情-动作耦合关系，为多码本自回归提供简洁范式。
4. **全自动化伪标注流水线**：从视频采集到3D参数提取的端到端流程设计（安全过滤+说话人重对齐+批量resize+padding）具备高可复用性，可迁移至其他视频语料构建。
5. **与团队结合机会**：可将MARS的交错token机制迁移至具身对话、虚拟数字人、社交机器人情感反馈等方向；也可结合语音韵律（prosody）进一步扩展多模态非言语表征。

## 关键术语表
**VENUS**：Video with Nonverbal cues and Utterance Set，本文提出的大规模多模态对话数据集。
**MARS**：Multimodal lAnguage Model with nonveRbal-cueS，本文提出的联合生成文本与非言语线索的语言模型。
**VQ-VAE**：Vector-Quantized Variational Autoencoder，将连续特征离散化为码本索引的生成模型，用于非言语tokenization。
**FLAME**： facial Landmark Embedding Morphable model，本文用于提取面部表达式与下颌参数的3D人脸模型。
**SMPL-X**： Scalable body model with eXpressive fingers，本文用于提取全身及手部姿态的3D人体模型。
**EMOCA-v2**： Audio-driven 3D facial animation model，用于从视频重建细粒度面部表情参数。
**OSX**： One-stage 3D whole-body mesh recovery model，用于从视频重建全身姿态参数。
**NLL（Negative Log-Likelihood）**：负对数似然，本文用于评估非言语token生成质量的指标，越低越好。

## 可复现要素
- **数据集**：VENUS（JSON格式标注），已开源，含视频ID而非原始帧（隐私保护）；链接：https://github.com/winston1214/nonverbalconversation
- **代码**：已开源（同上仓库）
- **权重**：论文未明确说明是否公开预训练权重，仅给出源码与数据集访问方式
- **关键超参**：码本大小K=512，面部dim=8/身体dim=16，下采样率q=8，序列长度W=512，$\beta=0.02$，$\lambda_{recon}=1$，$\lambda_{vel}=0.5$，学习率1e-4，warmup 10%，训练100 epoch（VQ-VAE）/50 epoch（MARS SFT）
