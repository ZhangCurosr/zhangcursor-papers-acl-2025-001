---
title: "Modeling-Uncertainty-in-Composed-Image-Retrieval-via-Probabi"
source: https://aclanthology.org/2025.acl-long.61.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:12:05"
field: "多模态检索"
keywords: ["Composed Image Retrieval", "Probabilistic Embedding", "Uncertainty Quantification", "Multi-modal Retrieval", "Gaussian Distribution"]
innovations: ["提出COPE概率嵌入框架，将查询与候选表示为高斯分布以建模不确定性", "设计不确定性感知距离度量与分层学习目标，同时建模实例级与特征级不确定性"]
benchmarks: ["Fashion-IQ", "CIRR"]
---

# 论文速读：Modeling-Uncertainty-in-Composed-Image-Retrieval-via-Probabi

## 一句话总结
论文提出了COPE（Composed Probabilistic Embedding）框架，将CIR查询和候选图像表示为潜在空间中的高斯分布而非固定点，通过设计概率距离度量与分层学习目标，显式建模实例级与特征级不确定性，在Fashion-IQ和CIRR数据集上均取得SOTA检索性能。

## 研究问题与动机
1. 现有CIR方法基于确定性点嵌入的度量学习，无法刻画用户意图的模糊性（如"make it more natural"有多重解读）。
2. 输入数据质量参差不齐：参考图像可能存在模糊/遮挡/低分辨率，修改文本可能存在语法错误或信息不足。
3. 目标图像属性具有多义性（如袖长、风格的个体理解差异），现有方法对所有样本采用统一训练策略，忽视样本可靠性差异，导致训练不稳定和过拟合风险。
4. 现有方法缺乏对嵌入特征维度间语义歧义的建模能力。

## 核心贡献（创新点）
1. 提出COPE概率嵌入框架，将CIR的查询与候选嵌入为高斯分布，方差反映输入不确定性，无需额外标注即可量化不确定性。
2. 设计不确定性感知距离度量 $d(z_1,z_2)=\|\mu_1-\mu_2\|^2+\|\sigma_1-\sigma_2\|^2+2D\bar{\sigma}_1\bar{\sigma}_2$，第一项为Wasserstein-2距离，第二项惩罚高不确定性匹配，使模型优先选择可靠实例并降低模糊维度权重。
3. 提出分层学习目标，联合实例级不确定性对比损失与特征级邻域偏差损失，分别建模数据质量不确定性与语义歧义不确定性。
4. 实验在Fashion-IQ与CIRR双数据集上均达到SOTA，并验证了更强的训练稳定性与跨骨干兼容性。

## 方法详解
**问题建模：** 每个嵌入建模为 $z \sim \mathcal{N}(\mu, \sigma^2 I)$，假设维度独立，仅对角协方差，$\mu, \sigma^2 \in \mathbb{R}^D$ 均为D维向量。

**编码器设计：** 使用CLIP-ViT-L/14，视觉与文本分支各设均值头 $f_V(\cdot), f_T(\cdot)$ 与方差头 $g_V(\cdot), g_T(\cdot)$；参考图与候选图共享视觉分支。查询嵌入通过高斯加法组合：$z_q = z_r + z_t \sim \mathcal{N}(\mu_r+\mu_t, (\sigma_r^2+\sigma_t^2)I)$。

**多粒度特征提取：** 将CLIP视觉Transformer下层特征与最后一层hidden state连接，通过文本embedding调制的Cross Attention门控实现：$h'_l = \text{LN}(h_l + \text{XA}(h_l, \mu_t))$，再对多层输出取平均。

**不确定性量化头：** 输入encoder最后一层hidden，经MLP→局部注意力→残差连接→Generalized Pooling Operator (GPO)得到 $\sigma$。

**距离度量：** $d(z_1, z_2) = \|\mu_1 - \mu_2\|^2_2 + \|\sigma_1 - \sigma_2\|^2_2 + 2D\bar{\sigma}_1\bar{\sigma}_2$，满足非负性、对称性与可判别性（自身最近）。

**实例级损失：** 借鉴SigLIP的Sigmoid对比损失 $\mathcal{L}_C$，其中距离 $d$ 使用上述概率距离，使模型对高置信正样本赋予更高权重。

**特征级损失：** 邻域偏差损失 $\mathcal{L}_{ND} = \sum_{x \in \{r,t,c\}} \|\sigma_x - \frac{1}{K}\text{std}(\mu_{N_x})\|^2_2$，强制嵌入方差与K-NN邻域特征标准差成正比，捕捉特征维度的语义歧义。

**总损失：** $\mathcal{L} = \mathcal{L}_C + \lambda \mathcal{L}_{ND}$，$\lambda=0.02$。

## 实验与结果
**数据集：** Fashion-IQ（18K训练三元组，dress/shirt/top&tee三类，R@10/R@50评估）；CIRR（36.5K三元组，Recall@K与Recall_subset@K评估）。

**主要结果：**
- **Fashion-IQ Avg：** COPE R@10=44.50，R@50=68.60，均值56.55，超越SADN（55.63）约+0.92。
- **CIRR R@10：** COPE=49.18，超越SADN（44.27）约+4.91；Recall_subset@2=88.65，略低于SADN（89.33）但整体SOTA。
- **消融：** Hard Contrastive仅得50.81（Fashion-IQ Avg），COPE全量得56.55；实例级+特征级联合损失优于单独使用任一；$\lambda=0.02$、$K=10$最优。
- **不确定性有效性：** 查询不确定性越高，召回率越低；模型正确识别出颜色/袖子长度等维度的语义歧义。
- **计算效率：** 借助Milvus混合向量搜索，1M向量库平均检索耗时13.6ms，具备部署可行性。
- **骨干兼容性：** 换用SigLIP后COPE提升更大（Avg 58.27 vs 56.55）。

## 相关工作脉络
1. **TIRG/CIRPLANT/CoSMo/ARTEMIS：** 早期监督CIR方法，依赖点嵌入三元组对比学习，未建模输入不确定性。
2. **CLIP4CIR/BLIP4CIR：** 基于预训练CLIP/BLIP的零样本或微调方法，将视觉特征转为伪文本token，缺乏显式三元组不确定性感知。
3. **CompoDiff/DWC/SSN/SADN：** 最新SOTA基线，CompoDiff引入扩散模型辅助，SADN使用语义蒸馏；本文从概率表示角度切入，不依赖额外数据源。
4. **PFE/DUL（人脸识别）：** 将概率嵌入用于人脸，仅建模实例级不确定性，本文扩展至CIR并同时建模特征级不确定性。
5. **PCME/Chun(2024)：** 跨模态检索的概率嵌入工作，闭式距离度量有相似之处，但未针对CIR的多粒度歧义设计特征级邻域偏差损失。
6. **Xu et al.(2024) Set of Diverse Queries：** 引入不确定性正则化但属于训练正则手段，本文直接从表示层建模概率分布。

## 局限性与未来方向
1. 对严格top-1排序（Recall@10/Recall_subset@K）提升相对有限，存在召回覆盖率与排名尖锐度的trade-off。
2. 多向量距离度量带来额外计算开销，需依赖Milvus等混合检索框架支撑高效近似最近邻搜索。
3. 当前假设每张图片-query对的不确定性是静态的，未考虑上下文或任务自适应的动态不确定性调整。
4. 未来方向：引入动态/上下文相关的不确定性建模；结合fine-grained contrastive alignment或dynamic re-ranking提升top-1性能。

## 研究启发与可借鉴点
1. **不确定性表示范式可迁移：** 将嵌入从点估计升级为高斯分布的思路，可直接迁移至开放域多模态检索、医学影像诊断等需要置信度估测的场景。
2. **特征级邻域偏差损失设计巧妙：** 通过K-NN邻域特征标准差 supervise 方差向量的思路，为其他含多义语义的检索任务提供了可复用的特征歧义建模方案。
3. **闭式概率距离度量具备通用性：** Wasserstein-2+不确定性惩罚项的距离公式，可推广至图文对齐、跨模态聚类等其他度量学习任务。
4. **实验设计值得借鉴：** 按不确定性分箱分析召回率变化、多维度消融（参考图/文本/候选图分别加特征损失）的颗粒度很足，可作为论文实验规范参考。
5. **与SigLIP等现代骨干兼容性强：** 证明不确定性模块与特征提取器解耦，可灵活替换，有利于后续研究快速接入最新backbone。

## 关键术语表
**Composed Image Retrieval (CIR)：** 结合参考图像与文本修改指令进行图像检索的多模态任务。
**COPE（Composed Probabilistic Embedding）：** 本文提出的将CIR查询与候选表示为高斯分布的概率嵌入框架。
**Aleatoric Uncertainty：** 由数据本身噪声、模糊性或语义歧义引起的不确定性，不可通过增加训练数据消除。
**Instance-Wise Uncertainty：** 表征整体样本质量的不确定性，通过方差向量的L2范数衡量。
**Feature-Wise Uncertainty：** 表征不同特征维度语义歧义程度的不确定性，通过方差向量的逐维分布刻画。
**Neighborhood Deviation Loss ($\mathcal{L}_{ND}$)：** 约束嵌入方差与K-NN邻域特征标准差成正比的损失函数。
**Generalized Pooling Operator (GPO)：** 一种聚合不确定性信息并保持概率可解释性的池化算子。
**Recall_subset@K：** CIRR数据集提出的严格评估指标，要求在6张相似图中召回正确目标。

## 可复现要素
- **数据集：** Fashion-IQ（公开）；CIRR（公开，源自NLVR2）。
- **代码：** 已开源，https://github.com/tanghme0w/ACL25-CoPE。
- **权重：** 论文未提及预训练权重下载。
- **关键超参：** batch_size=128，learning_rate=2×10⁻⁶，optimizer=AdamW（β₁=0.9，β₂=0.999，ε=1e⁻⁷），EMA rate=0.99，λ=0.02，K=10，数据增强：cutout/HSV/rotation/scaling/Gaussian noise（各p=0.2）。
- **硬件：** 单卡A100-80G。
- **骨干：** CLIP-ViT-L/14。
