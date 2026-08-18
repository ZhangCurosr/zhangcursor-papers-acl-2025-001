---
title: "The-Hidden-Attention-of-Mamba-Models"
source: https://aclanthology.org/2025.acl-long.76.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:22:56"
field: "模型可解释性"
keywords: ["Mamba", "State Space Model", "XAI", "Attention Mechanism", "Interpretability", "S6"]
innovations: ["揭示S6层隐式因果注意力机制并将其重构为数据控制线性算子", "开发首个面向Mamba的Attention Rollout与Attribution可解释工具", "证明S6层表达能力强于因果线性注意力且不低于Softmax注意力"]
benchmarks: ["ImageNet-1K", "ImageNet-Segmentation", "IMDb", "ARC-Easy"]
---

# 论文速读：The-Hidden-Attention-of-Mamba-Models

## 一句话总结
本文揭示了Mamba模型（S6层）中隐藏的隐式注意力机制，将其重构为数据控制型因果线性自注意力模型，并据此开发了首个面向Mamba的可解释性工具（Attention Rollout与Attribution方法），在视觉与NLP任务上取得了与Transformer可解释性方法相当的效果。

## 研究问题与动机
1. **机制黑盒问题**：Mamba模型在语言建模、视觉、视频等多领域表现优异，但其token间信息流动力学与学习机制仍属未解之谜，不清楚其本质更贴近RNN、CNN还是注意力机制。
2. **可解释性缺失**：当前缺乏面向Mamba的跨模型互操作分析方法，阻碍了调试、公平性评估以及在医疗等需可解释性的社会敏感领域的应用。
3. **理解鸿沟**：传统观点将SSM视为卷积或循环层，本文试图建立第三视角——注意力视角，以便用成熟的Transformer可解释技术来分析Mamba。

## 核心贡献（创新点）
1. **揭示Mamba的隐式注意力本质**：将S6层重排为数据控制型线性算子，证明其等价于隐式因果自注意力，每个通道产生N个内部注意力矩阵，总量比Transformer多约三个数量级。
2. **开发Mamba专属可解释工具**：提出Mamba版Attention Rollout（类别无关）与Mamba-Attribution（类别相关），首次将成熟的Transformer XAI技术迁移至Mamba架构。
3. **理论表达能力分析**：证明单通道S6层可表达单个注意力头的全部函数，同时存在单个注意力头无法表达的函数（如count-in-row问题），说明S6的表达能力强于因果线性注意力且不低于Softmax注意力。
4. **经验可比性验证**：在Vision与NLP任务上，Mamba注意力机制的可解释性指标与Transformer相当，部分指标（如正扰动AUAC、分割任务mIoU）甚至更优。

## 方法详解
1. **隐式注意力矩阵推导**：将S6层的递归公式展开，得到输出表达式 $y_t = C_t \sum_{j=1}^{t} \left(\prod_{k=j+1}^{t} \bar{A}_k\right) \bar{B}_j x_j$，写成矩阵形式 $y = \tilde{\alpha} x$，其中 $\tilde{\alpha}$ 为下三角矩阵，元素 $\tilde{\alpha}_{i,j} = C_i \left(\prod_{k=j+1}^{i} \bar{A}_k\right) \bar{B}_j$ 捕捉了 $x_j$ 对 $y_i$ 的影响。
2. **注意力形式的重构**：通过将Softplus近似为ReLU，定义 $\tilde{Q}_i = S_C(\hat{x}_i)$、$\tilde{K}_j = \text{R}(S_\Delta(\hat{x}_j)) S_B(\hat{x}_j)$、$\tilde{H}_{i,j} = \exp\left(\sum_{k=j+1}^{i} S_\Delta(\hat{x}_k) A\right)$，将隐式注意力矩阵近似为 $\tilde{\alpha}_{i,j} \approx \tilde{Q}_i \tilde{H}_{i,j} \tilde{K}_j$，这与Transformer的 $QK^T$ 形式形成类比，其中 $\tilde{H}$ 编码了连续历史上下文。
3. **Attention Rollout方法**：对每个通道 $d$ 和层 $\lambda$ 提取隐式注意力矩阵 $\tilde{\alpha}^{\lambda,d}$，通过期望聚合得到全局注意力矩阵 $\tilde{\alpha}^{\lambda} = \mathbb{I} + \mathbb{E}_{d}(\tilde{\alpha}^{\lambda,d})$，再沿层维度连乘得到最终相关性图 $\rho = \prod_{\lambda} \tilde{\alpha}^{\lambda}$，提取CLS token行作为最终热图。
4. **Mamba-Attribution方法**：将Transformer-Attribution方法适配到Mamba，用门控机制与S6混合器的输出梯度 $\nabla \hat{y}'^{\lambda,d}$ 替代原始注意力梯度，用隐式注意力矩阵 $\tilde{\alpha}^{\lambda,d}$ 替代LRP相关性分数，通过Hadamard积融合：$\tilde{\beta}^{\lambda} = \mathbb{I} + \left(\mathbb{E}_d(\nabla \hat{y}'^{\lambda,d}) \odot \mathbb{E}_d(\tilde{\alpha}^{\lambda,d})\right)^+$。

## 实验与结果
- **视觉任务**：使用ImageNet-1K上的ViM（Tiny）与DeiT（Tiny）对比，评估Raw-Attention、Attn-Rollout、Attribution三种方法。正扰动测试中Mamba Raw-Attention AUC（17.27）优于ViT（20.69）；分割任务中Mamba Attn-Rollout mIoU达51.51%，超过ViT的47.85%。
- **NLP任务**：在IMDb情感分类上使用Mamba-130M与BERT-large，在激活与剪枝任务中Mamba-Attr表现与Transformer-Attr相当或更优；在ARC-Easy零样本推理上，Mamba 2.7B的正扰动AUAC达0.918，与Pythia 2.8B的0.920基本持平。
- **最强结果**：Mamba-Attribution在ImageNet分割任务上取得Pixel acc 74.72%、mAP 81.70%、mIoU 54.24%，综合性能优于原始LRP方法及多数Baseline。

## 相关工作脉络
1. **S4/DSS/S5等早期SSM**：具有固定混合元素，与Mamba的数据依赖非对角混合器形成对比，本文证明S6是首个具备数据控制非对角混合能力的SSM。
2. **GSS/Hyena**：具有固定混合元素加对角数据控制机制，区别于S6的非对角数据控制。
3. **Poli et al. (2023) Hyena层次**：提出数据控制线性算子是Transformer表达力的关键，本文将此概念具体化到S6层的数据控制非对角混合器。
4. **Zimerman & Wolf (2024)**：从长卷积视角分析Transformer，本文从相反方向将Mamba重构为注意力形式，两者形成对称理解。
5. **Attention-Rollout (Abnar & Zuidema, 2020) 与 Transformer-Attribution (Chefer et al., 2021b)**：经典Transformer可解释性方法，本文首次将其适配到Mamba架构。
6. **Mamba-LRP (Jafari et al., 2024)**：基于LRP的Mamba可解释方法，本文的Attribution方法在多项指标上超越该方法。

## 局限性与未来方向
1. **计算复杂度**：生成注意力图需要构造每通道 $L \times L$ 矩阵，复杂度为 $O(\Lambda D L^2 N)$（可优化至 $O(\Lambda D L N)$），在大模型与长序列上开销较大。
2. **大模型验证缺失**：实验主要在ViM-Tiny、Mamba-130M/2.7B等中等规模模型上进行，未在LLaMA-405B或GPT-4量级模型上验证。
3. **负扰动表现较弱**：Mamba在负扰动测试中AUC低于Transformer，可能对遮蔽patch更敏感，需进一步研究模糊替换等替代方案。
4. **未来方向**：探索利用Mamba内在线性注意力结构的更高效XAI方法，避免显式计算完整注意力矩阵。

## 研究启发与可借鉴点
1. **视角转换的启发**：将S6层从递归/卷积视角转换为注意力视角，为理解新型序列模型提供了通用方法论——寻找隐式等价形式以复用成熟分析工具。
2. **可解释性迁移策略**：将Transformer的Attention Rollout和Attribution适配到Mamba时，关键在于用门控梯度替代注意力梯度、用隐式注意力矩阵替代LRP分数，这一适配思路可推广到其他架构。
3. **表达能力理论分析**：通过构造count-in-row等特定任务进行理论证明，展示了形式化分析方法在模型对比中的价值。
4. **XAI辅助模型改进**：结合AMPLIFY框架，利用Mamba的XAI结果进行prompt工程，提升了少样本ICL性能，开辟了XAI反哺模型优化的新路径。
5. **注意力热图质量验证**：通过正/负扰动、分割任务等多维度指标综合评估可解释性，避免了单一指标的局限性。

## 关键术语表
**S6 / Selective SSM**：Mamba的核心组件，一种输入依赖的时间可变状态空间层，通过门控机制实现选择性信息传递。
**隐式注意力矩阵（Hidden Attention Matrix）**：$\tilde{\alpha}$，由S6层的递归结构推导出的下三角矩阵，元素表示输入token对输出token的隐式影响权重。
**Attention Rollout**：通过逐层聚合注意力矩阵来分析token间信息流的可视化方法，本文首次适配到Mamba。
**Mamba-Attribution**：结合梯度与隐式注意力矩阵的类别相关可解释方法，是Transformer-Attribution在Mamba上的适配版本。
**数据控制非对角混合器（Data-controlled non-diagonal mixer）**：S6层的核心特征，混合操作由输入数据动态控制且元素间存在非对角交互。
**ICL（In-Context Learning）**：上下文学习，模型根据输入示例自适应调整行为的能力，S6被证明具备与Transformer相当的ICL能力。
**CLS Token**：分类令牌， Vision Mamba模型序列末尾附加的特殊token，用于聚合全局信息进行分类。
**正向/负向扰动测试**：评估可解释性质量的两种范式，正向测试按相关性从高到低遮蔽输入，负向测试从低到高遮蔽。

## 可复现要素
- **数据集**：ImageNet-1K（视觉分类）、ImageNet-Segmentation（分割评估）、IMDb（情感分类）、ARC-Easy（推理）、The Pile（预训练）。部分数据集公开，ImageNet-Segmentation可公开获取。
- **代码**：论文声明代码已公开（链接见摘要），部分实验代码作为补充材料提供，包含用户友好接口和notebook演示。
- **模型**：ViM-Tiny、Mamba-130M/2.7B/1.3B、Pythia-1.4B/2.8B、BERT-large。
- **超参数**：论文未详细列出训练超参数，主要关注可解释性方法设计而非模型训练细节。
