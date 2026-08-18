---
title: "Pre-training-Distillation-for-Large-Language-Models-A-Design"
source: https://aclanthology.org/2025.acl-long.181.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:15:01"
field: "大语言模型预训练与模型压缩"
keywords: ["pre-training distillation", "knowledge distillation", "large language models", "design space exploration", "scaling law", "WSD scheduler"]
innovations: ["首次系统探索LLM预训练蒸馏设计空间，验证可行性并提出最优配置", "发现学生模型越大获益越多、教师模型越大未必更好的反直觉规律", "提出WSD-α+WSD-LR联合调度策略，实现+8.0%平均性能提升"]
benchmarks: ["HellaSwag", "WinoGrande", "PIQA", "MMLU", "C-Eval", "GSM8k"]
---

# 论文速读：Pre-training-Distillation-for-Large-Language-Models-A-Design-Space-Exploration

## 一句话总结
本文首次将知识蒸馏系统性地扩展到LLM预训练阶段（Pre-training Distillation, PD），并通过控制变量实验在logits处理、损失选择、扩展定律、离线/在线四个维度上探索设计空间，找到最优配置使1.9B模型相对纯LM损失预训练平均提升8.0%。

## 研究问题与动机
- 现有知识蒸馏主要应用于后训练阶段（post-training KD），让学生模型直接从教师生成的指令-响应对学习，而预训练阶段的知识蒸馏可行性尚未被系统验证。
- 教师模型生成的logits蕴含比hard label更丰富的知识分布信息，理论上可作为soft label平滑信号，加速学生模型预训练并提升性能上限。
- 预训练蒸馏涉及截断、归一化温度、损失函数组合、师生规模配比、离线/在线获取logits等多个设计维度，尚无系统探索。
- 现有LLM预训练蒸馏相关工作（如Gemma 2、AFM、LokiLM、Minitron）大多缺乏详细的蒸馏过程描述或仅聚焦剪枝+蒸馏联合优化。

## 核心贡献（创新点）
- **首次系统验证预训练蒸馏可行性**：使用GLM-4-9B蒸馏1.9B学生模型，在100B token上平均提升1.6%，证明该方向有效。
- **构建四维设计空间探索框架**：围绕logits处理、损失选择、扩展定律、离线/在线策略进行系统的控制变量实验，给出可复用的最佳实践。
- **发现"学生模型越大获益越多，教师模型越大未必更好"的反直觉规律**：指出师生容量差距（capacity gap）是关键约束，为后续研究提供指导。
- **提出WSD-α+WSD-LR联合调度策略**：在保持最大学习率阶段使用较高比例KD损失可显著提升性能，PD∗配置实现+8.0%平均提升。
- **揭示在线logits的实用价值与局限**：非收敛教师模型的在线logits仍可带来增益，建议在多规模模型预训练时优先采用"先训最大模型并存储在线logits"的策略。

## 方法详解
**目标函数**：
$$\theta_S^* = \arg\min_{\theta_S} \mathcal{L} = \arg\min_{\theta_S} [(1-\alpha)\mathcal{L}_{\text{lm}} + \alpha\mathcal{L}_{\text{kd}}]$$
其中$\mathcal{L}_{\text{lm}}$为标准负对数似然语言建模损失，$\mathcal{L}_{\text{kd}}$为蒸馏损失。

**Logits处理**：由于词汇表约150k导致完整logits存储开销巨大（58.6 PB / 100B tokens），采用两阶段top-p-k截断（先top-p=0.95再top-k）配合温度τ归一化：$F(z) = \text{softmax}(\text{Truncate}(z)/\tau)$，存储压缩至约15 TB。

**损失选择**：
- 蒸馏损失可选NLL（即负对数似然）、KLD（KL散度）或MSE；实验表明MSE会严重损害性能（-7.6%）。
- α（KD损失比例）静态最优值约0.9；动态调度中，线性递减优于线性递增。
- 最优组合：WSD-α（KD损失比例调度）配合WSD-LR（学习率调度），利用warmup-stable-decay阶段特性，在高学习率峰值期投入更多KD损失。

**扩展定律**：学生模型达教师模型规模10%以上时蒸馏开始显现增益，且增益随比例增大而上升；教师模型从9B增至32B并未带来更优结果。

**离线 vs 在线**：离线logits（来自预训练完成教师）效果稳定；在线logits（训练期间同步存储）效果略弱但对收敛不足的teacher仍有效，适合多尺度模型共享基础设施场景。

## 实验与结果
**实验设置**：教师模型GLM-4-9B（后文也用GLM-4-32B），学生模型330M~6.8B，预训练数据100B~500B token，评估含HellaSwag、WinoGrande、PIQA、MMLU、KBQA、C3、C-Eval、GSM8k。

**核心结果**：
- 初步实验（LLM-KD vs LLM-LM）：平均提升+1.6%，GSM8k提升最大（+24.6%）。
- **最优配置PD∗**（top-0.95-50截断 + τ=2.0 + KLD + WSD-α+WSD-LR，α_max=0.9）：
  - 1.9B学生：平均40.7（+8.0% vs LM baseline）；MMLU达31.8（+3.8绝对分）。
  - 3.8B学生：平均53.7（+1.7% vs LM baseline）。
- **在线logits**：LLM-Online-100B*在调优后仅+0.5%，低于离线配置，但仍证实可行性。
- **500B token实验**：1.9B与3.8B在全部检查点上均稳定优于LM baseline，增益随token数先增后收敛。
- **扩展定律**：学生模型越大获益越显著（6.8B提升幅度大于1.9B）；教师从9B→32B对330M/670M学生略有下降趋势。

## 相关工作脉络
- **后训练蒸馏（Alpaca、Vicuna、ORCA等）**：基于instruction-response对进行序列级蒸馏，与本文预训练阶段logits蒸馏形成阶段差异互补。
- **小模型预训练蒸馏（DistilBERT、TinyBERT、MiniLM）**：面向百万级参数BERT架构，配置不可直接迁移至十亿级LLM。
- **Gemma 2、AFM、LokiLM、Minitron**：均采用预训练蒸馏但细节描述有限，本文提供首个系统设计空间探索。
- **LLM剪枝+蒸馏联合（Muralidharan et al., 2024）**：聚焦剪枝为主、蒸馏为辅，本文独立聚焦蒸馏设计维度。
- **NormKD、WTTM、AdaKD（自适应温度）**：本文对比发现静态τ=0.5~2.0已足够，自适应温度未见显著额外收益。

## 局限性与未来方向
- 未探索不同设计维度之间的交互效应（组合实验），因计算成本过高。
- 训练数据规模仅达500B token，未触及万亿级token（主流先进LLM的训练规模）。
- 未探索学生模型规模超过教师模型的"弱到强泛化"（weak-to-strong generalization）场景。
- 性能增益的峰值比例（学生/教师规模比）未找到转折点，留作未来工作。

## 研究启发与可借鉴点
- **top-p-k截断+低τ归一化**的组合可作为标准化预处理流程，兼顾存储效率与蒸馏信号质量，适合大规模预训练蒸馏部署。
- **WSD调度器与KD损失比例联合调度**的策略具有通用性，可在其他需要动态平衡多损失的训练场景中借鉴。
- **师生容量匹配原则**：学生模型规模建议不低于教师的10%，否则蒸馏收益有限，可为模型选型提供指导。
- **在线logits存储策略**：为多尺度模型家族预训练提供了"一次训练、多模型蒸馏"的经济路径。
- **MSE损失在LLM预训练蒸馏中不适用**：这一发现挑战了图像分类领域的经验，提醒NLP社区需重新审视蒸馏损失设计。

## 关键术语表
**Pre-training Distillation (PD)**：将知识蒸馏从后训练阶段拓展至预训练阶段，让学生模型在预训练语料上直接从教师logits学习。
**top-p-k Truncation**：两阶段logits截断策略，先按累积概率top-p筛选再按数量top-k截断，用于大幅压缩logits存储。
**WSD (Warmup-Stable-Decay) Scheduler**：一种非线性调度策略，包含线性warmup、稳态和余弦衰减三个阶段，本文用于联合调度学习率与KD损失比例。
**Capacity Gap**：师生模型之间的容量差距，差距过大会阻碍蒸馏效果，是解释"大教师不一定更好"现象的核心概念。
**Offline vs Online Logits**：离线指从已完成的教师模型前向传播获取logits；在线指在教师预训练过程中同步存储logits。
**Label Smoothing Analogy**：教师logits软分布可作为hard label的平滑版本，减少过拟合并提供更丰富的监督信号。

## 可复现要素
- **数据集**：预训练数据为公开语料（100B/500B token），评估数据集包括HellaSwag、WinoGrande、PIQA、MMLU、KBQA、C3、C-Eval、GSM8k；论文未提及预训练语料的具体开源地址。
- **代码/权重**：论文声明"we do not publish additional artifacts"，代码与模型权重未公开。
- **关键超参**：top-p=0.95、top-k=50、τ=2.0（最优配置）；α_max=0.9；WSD warmup=10%、decay=1%；batch size=2048；max sequence length=4096；学习率6×10⁻⁴（cosine）；Optimizer=Adam。
