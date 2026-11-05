---
title: "Cross-model-Transferability-among-Large-Language-Models-on-t"
source: https://aclanthology.org/2025.acl-long.185.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:42:29"
field: "大语言模型可解释性与表示工程"
keywords: ["steering vectors", "cross-model transferability", "representation engineering", "linear transformation", "LLM interpretability", "weak-to-strong transfer", "Platonic representations"]
innovations: ["提出L-Cross Modulation线性变换方法实现跨模型SV迁移", "发现不同概念共享同一线性变换矩阵的泛化现象", "揭示小模型SV可有效调控大模型行为的弱到强迁移能力"]
benchmarks: ["CAA数据集", "RepE数据集", "HarmBench"]
---

# 论文速读：Cross-model-Transferability-among-Large-Language-Models-on-the-Platonic-Representations-of-Concepts

## 一句话总结
论文提出L-Cross Modulation方法，通过简单的线性变换将不同LLM中同一概念的steering vector（SV）相互迁移，实验证明跨模型概念表示具有强线性和通用结构，且小模型提取的SV可有效控制大模型行为。

## 研究问题与动机
- 现有SV研究聚焦于单模型内概念表示的发现与调控，但不同LLM训练数据、目标、架构各异，其内部概念表示空间是否存在可迁移的"柏拉图式"统一结构尚不清楚。
- 现有方法缺乏跨模型概念表示对齐机制，无法利用一个模型中学到的概念知识调控另一个模型的行为。
- 基础模型时代探索通用任务范式具有重要意义，理解跨模型表示共性能推动更适应性的语言模型发展。
- 如何以轻量级方式实现跨模型安全调控（如用小型模型提取的SV调控大模型）尚未被充分研究。

## 核心贡献（创新点）
- **首次系统研究LLM概念表示的跨模型迁移性**：提出跨模型视角分析SV的普适性，而非局限于单模型内部表征。与已有工作聚焦单模型Self Modulation的本质区别在于探索了不同训练数据/架构下的表示共性。
- **提出L-Cross Modulation线性变换方法**：通过最小二乘法学习变换矩阵T，将源模型SV映射到目标模型表示空间。与直连添加SV（w/o T）或随机矩阵的核心区别在于提供了语义对齐的桥梁。
- **发现跨概念泛化的线性变换**：证明不同概念共享同一线性变换矩阵，揭示了LLM概念表示空间的统一底层结构。区别于按概念单独训练变换的方法。
- **揭示弱到强迁移能力**：小模型（Qwen2-0.5B）提取的SV可有效调控大模型（Qwen2-7B），为高效安全调控提供新途径。与同规模模型间迁移的关键区别在于突破了模型大小限制。

## 方法详解
- **L-Cross Modulation核心框架**：给定源LLM $m_s$ 的概念SV $\bar{\lambda}_W^{m_s}$ 和目标LLM $m_t$，通过学习变换矩阵T实现跨模型迁移：$\bar{\lambda}_W^{m_t} = \bar{\lambda}_W^{m_s} \mathbf{T}_{\mathcal{D}}$，再将缩放后的向量 $\beta \bar{\lambda}_W^{m_s} \mathbf{T}_{\mathcal{D}}$ 叠加到目标模型隐藏状态。
- **变换矩阵T的优化**：采用普通最小二乘法（OLS），基于共享语料库$\mathcal{D}$对齐两个模型的表示空间：$\mathbf{T}_{\mathcal{D}} = \underset{\mathbf{T}'}{\mathrm{argmin}} \|\lambda_{\mathcal{D}}^{m_t} - \lambda_{\mathcal{D}}^{m_s} \mathbf{T}'\|$，闭式解为 $\mathbf{T}_{\mathcal{D}} = (\lambda_{\mathcal{D}}^{m_s\top} \lambda_{\mathcal{D}}^{m_s})^{\dagger} \lambda_{\mathcal{D}}^{m_s\top} \lambda_{\mathcal{D}}^{m_t}$。
- **语料选择策略**：可使用概念相关语料$Y_W$优化T以实现针对性对齐，也可使用概念无关语料学习通用变换，后者能验证跨概念泛化能力。
- **与Self Modulation对比**：Self Modulation直接使用目标模型自身提取的SV进行调控，而L-Cross Modulation通过线性变换桥接不同模型的表示空间。

## 实验与结果
- **数据集**：11个基准概念，7个来自CAA数据集（AIC, CORR, HALLU, MR, SI, SYC, REF），4个来自RepE数据集（HARM, FAIR, HAPPY, FEAR）。
- **评估模型**：Llama2-7B、Qwen2-7B、Llama3.1-8B（Chat版本）。
- **评估指标**：输出概率（CAA）、GPT-4o mini评分0-10（CAA）、HarmBench分类器百分比（HARM）、性别词频率（FAIR）。
- **RQ1结果**：L-Cross Modulation在216个案例中仅4个失败（失败率约1.9%）。以HARM为例，Llama2无调制时0%有害输出，使用Qwen2的SV后提升至96%。31/42案例优于Self Modulation。
- **RQ2结果**：使用概念无关语料学习的T在216个案例中仅17个失败（约7.9%失败率），证明T的强泛化能力。不同概念的T矩阵相似度高（SSIM 0.87-0.95 vs 随机0.05-0.13）。
- **RQ3结果**：Qwen2-0.5B（弱模型）的HARM SV经变换后调控Qwen2-7B（强模型）产生86%有害输出，比Self Modulation的54%提升32个百分点。
- **最强结果**：Llama3.1作为目标模型时，HAPPY概念达到9.34分（Self Modulation为8.74），FEAR达到9.28分（Self为9.28持平）。

## 相关工作脉络
- **CAA (Rimsky et al., 2024) 和 RepE (Zou et al., 2023a)**：单模型内SV提取与调控的基础方法，本文在此基础上拓展到跨模型场景。
- **线性表示假说 (Park et al., 2024; Jiang et al., 2024)**：证明单模型内概念表示呈线性结构，本文进一步验证跨模型的线性可迁移性。
- **柏拉图表示假说 (Huh et al., 2024)**：理论支撑，认为不同网络收敛到共享的现实统计模型，本文通过实验提供实证支持。
- **Zou et al. (2023c) 和 Huang et al. (2024)**：研究越狱提示的通用性和线性迁移，但聚焦安全攻击场景；本文聚焦概念表示的普适性，范围更广（11个概念）。
- **Prompt Tuning迁移 (Zhang et al., 2024; Su et al., 2022)**：研究软提示的跨模型迁移，与本文的表示空间线性对齐形成对比——前者操作输入空间，后者操作隐层表示空间。

## 局限性与未来方向
- **概念范围有限**：仅覆盖11个基准概念（有用性、诚实性、无害性、情感类），未探索更高层次抽象概念。
- **评估指标分散**：依赖不同数据集的异构评估方法，缺乏统一的标准化评测框架。
- **超参数手动调优**：缩放因子β需人工调整，自动化最优β选择仍是开放问题。
- **语料依赖性**：变换矩阵T的质量受训练语料影响，语料选择策略未充分探讨。

## 研究启发与可借鉴点
- **线性变换的对齐思路可迁移**：对于任何需要在不同表示空间间建立映射的场景（如多模型集成、跨架构知识蒸馏），最小二乘对齐提供简洁有效的解决方案。
- **弱到强迁移的实用价值**：小模型提取的SV可有效调控大模型，为资源受限环境下的安全控制提供低成本方案，可结合本团队在模型压缩/效率方面的研究方向。
- **跨概念泛化验证方法**：通过概念无关语料验证变换通用性的实验设计值得借鉴，可用于评估其他表示学习方法的泛化能力。
- **t-SNE可视化辅助解释**：论文用t-SNE展示不同概念表示的线性关系（翻转、缩放、旋转），为表征对齐提供直观解释，可在类似研究中复用。

## 关键术语表
**Steering Vector (SV)**：从LLM对比文本对的表示差异中提取的概念方向向量，用于调控模型输出倾向。
**L-Cross Modulation**：本文提出的线性变换方法，通过学习变换矩阵T将源模型的SV对齐到目标模型的表示空间。
**Self Modulation**：直接使用目标模型自身提取的SV进行行为调控的方法，作为本文的对照基线。
**Platonic Representation Hypothesis**：柏拉图表示假说，认为不同训练目标和数据的神经网络在表示空间中收敛到共享的现实统计模型。
**CAA (Contrastive Activation Addition)**：通过对比激活相加提取SV的方法，使用二元选择文本对。
**RepE (Representation Engineering)**：通过主成分分析从对比表示中提取SV的框架。
** Ordinary Least Squares (OLS)**：普通最小二乘法，用于求解变换矩阵T的最优化方法。
**Weak-to-Strong Transfer**：从小模型向大模型的SV迁移能力，突破模型规模限制。

## 可复现要素
- **数据集**：CAA和RepE开源数据集（MIT许可证），包含概念相关对比文本对。
- **代码**：使用CAA和RepE的开源代码库进行SV提取。
- **模型**：Llama2-7B-Chat、Qwen2-7B-Instruct、Llama3.1-8B-Instruct、Qwen2-0.5B-Instruct（均为开源模型）。
- **关键超参**：β值（CAA所有概念设为1，RepE概念依附录表格手动选取）；SV提取层（Llama2: 13层，Qwen2: 18层，Llama3.1: 13层）。
- **实验环境**：单张A6000 GPU。
- **评估工具**：GPT-4o mini用于文本评分，HarmBench Llama-2-13b-cls分类器用于有害性评估。
