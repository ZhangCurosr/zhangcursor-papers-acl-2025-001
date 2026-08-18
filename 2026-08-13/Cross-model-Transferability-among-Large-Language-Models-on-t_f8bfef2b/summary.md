---
title: "Cross-model-Transferability-among-Large-Language-Models-on-t"
source: https://aclanthology.org/2025.acl-long.185.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:54:03"
field: "大语言模型可解释性与表示工程"
keywords: ["Steering Vectors", "跨模型迁移", "LLM可解释性", "表示工程", "弱到强迁移", "线性表示假设"]
innovations: ["提出L-Cross Modulation线性变换实现跨模型概念向量迁移", "发现不同概念共享同一线性变换矩阵T并具有强泛化能力", "证实小模型SV经变换后可有效调控大模型行为"]
benchmarks: ["CAA 7概念评测", "RepE 4概念评测", "Llama2-7B", "Qwen2-7B", "Llama3.1-8B"]
---

# 论文速读：Cross-model Transferability among Large Language Models on the Platonic Representations of Concepts

## 一句话总结
本文首次系统研究了不同大语言模型（LLM）之间概念表示（Steering Vectors, SVs）的可迁移性，提出一种线性变换方法 L-Cross Modulation，通过普通最小二乘法学习跨模型表示空间的映射矩阵 T，实现从源模型提取的概念向量在目标模型中直接生效，并验证了该映射具有跨概念通用性与"弱模型→强模型"的迁移能力。

## 研究问题与动机
1. **核心问题**：训练目标、数据、架构和规模各不相同的多个 LLM，其内部对同一"柏拉图式"普适概念（如有害性、幸福感、公平性）的表示是否共享底层结构，能否跨模型直接迁移？
2. **现有方法不足**：既往工作（CAA、RepE）几乎全部聚焦于单个 LLM 内部的 Self Modulation，缺乏对跨模型概念表示对齐与迁移的实证研究；安全控制依赖微调或系统提示，成本高且难以复用。
3. **理论支撑缺失**：尽管 Huh et al. (2024) 提出 "Platonic Representation Hypothesis"（不同网络收敛于共享统计现实），但鲜有工作从 Steering Vectors 角度验证不同 LLM 间表示线性可对齐。
4. **应用痛点**：若能在小模型上低成本提取 SV 并通过简单变换控制大模型行为，将显著降低安全对齐与红队测试的计算开销。

## 核心贡献（创新点）
1. **开创性研究 LLM 间概念表示的跨模型可迁移性**：与以往仅关注单模型内 SV 提取/应用的路线不同，首次构建"从源模型到目标模型"的概念向量迁移范式并提供大规模实证。
2. **提出 L-Cross Modulation 线性变换框架**：基于 CAA/RepE 提取的对比文本对表示差构建最小二乘回归，学习一个闭式解的变换矩阵 T，实现源模型 SV 到目标模型表征空间的高效对齐；与 Zou et al. (2023c) 的 jailbreak 特征迁移不同，本文聚焦普适概念而非单一攻击特征。
3. **发现跨概念泛化的线性变换**：实验证明，由某一概念语料 $Y_{W_1}$ 优化得到的 T 可直接用于不同概念 $W_2$ 的 SV 迁移（216 例中仅 17 例失效），且不同概念的 T 矩阵在 SSIM、特征值差异、Frobenius 范数上高度相似，揭示共享底层表征结构。
4. **揭示弱模型到强模型的可迁移性**：Qwen2-0.5B 自调制难以生成高比例有害内容，但其有害 SV 经 T 变换后对 Qwen2-7B 可达 88% 有害输出，较 0.5B 自调制提升 32%，为低成本安全对齐提供新思路。
5. **实证不同架构/数据训练的 LLM 间表示内在线性关系**：t-SNE 可视化与消融实验共同表明，跨模型表示差集可通过旋转、缩放、翻转等线性操作相互近似，挑战"架构与训练数据差异导致表示不可通约"的直觉。

## 方法详解
1. **Steering Vector（SV）提取背景**：给定概念 W 的对比文本对 $Y_W=\{(Y(0), Y(1))\}$，在选定层计算末 token 表示 $\lambda_0, \lambda_1$，得到差值集合 $\{\lambda_\delta = \lambda_1-\lambda_0\}$。CAA 取均值作为 $\bar{\lambda}_W$，RepE 取第一主成分。缩放后得到 $\beta\bar{\lambda}_W$ 加入隐藏状态完成 Self Modulation。
2. **L-Cross Modulation 核心公式**：给定源模型 $m_s$、目标模型 $m_t$ 与共词库 $\mathcal{D}$（可概念相关 $Y_W$ 或无关），学习变换矩阵：
   $$\mathbf{T}_{\mathcal{D}} = \arg\min_{\mathbf{T}'} \|\lambda_{\mathcal{D}}^{m_t} - \lambda_{\mathcal{D}}^{m_s} \mathbf{T}'\|$$
   闭式解为 $\mathbf{T}_{\mathcal{D}} = (\lambda_{\mathcal{D}}^{m_s\top}\lambda_{\mathcal{D}}^{m_s})^{\dagger}\lambda_{\mathcal{D}}^{m_s\top}\lambda_{\mathcal{D}}^{m_t}$。
3. **跨模型 SV 变换**：源模型 SV $\bar{\lambda}_W^{m_s}$ 经 T 映射得到目标模型近似的概念方向 $\bar{\lambda}_W^{m_t} \approx \bar{\lambda}_W^{m_s}\mathbf{T}_{\mathcal{D}}$，再以缩放因子 β 叠加进 $m_t$ 隐状态完成调制。
4. **两个关键设计选择**：(a) 使用线性变换而非复杂非线性映射，以避免引入阻碍迁移的归纳偏置并保留概念间几何关系；(b) 尝试概念无关语料 $\mathcal{D}$ 学习 T，验证跨概念泛化能力。
5. **超参数 β**：与现有工作一致，采用手动调参（CAA 所有概念统一 β=1，RepE 四个概念按表 5/6 手动设定），未解决自动搜索最优 β 的开放问题。

## 实验与结果
1. **数据集与评测基准**：11 个基准概念——CAA 的 7 个（AIC、CORR、HALLU、MR、SI、SYC、REF）与 RepE 的 4 个（HARM、FAIR、HAPPY、FEAR）；测试集分别为各概念 50 题。
2. **模型与基线**：三组开源 Chat 模型 Llama2-7B、Qwen2-7B、Llama3.1-8B；基线为 No Modulation，参考上限为 Self Modulation。
3. **RQ1 有效性（Table 1）**：L-Cross Modulation 在 216 个实验中仅 4 例未能有效调制；代表性案例：Qwen2 SV 使 Llama2 有害输出从 0.0% 跃升至 96.0%；CAA 七个概念中 L-Cross 在 31/42 情况下优于 Self Modulation，RepE 四个概念中最佳占 7/12。
4. **消融（Table 2）**：无 T 的 Cross Modulation 及随机 T 变体在 35 个实验中 23/35 劣于基线，证明线性变换 T 不可或缺。
5. **RQ2 跨概念泛化（Table 3 & 4）**：用概念无关语料学习的 T 在 216 例中仅 17 例失效；不同概念所得 T 矩阵间 SSIM 达 0.87–0.95、ME 低至 0.07–1.76、Frobenius 范数差 27–573，远优于随机矩阵（SSIM 0.05–0.13、ME 51–55、范数差 3800+）。
6. **RQ3 弱→强迁移（Figure 4）**：Qwen2-0.5B 自调制有害输出最高 54.0%，其有害 SV 经 T 变换后对 Qwen2-7B 可达 88.0%，提升 32%，与同规模跨模型结果相当。
7. **模型间相似度规律**：同期发布或架构相近的模型（Qwen2↔Llama3.1、Llama2↔Llama3.1）迁移效果更优。

## 相关工作脉络
1. **CAA / RepE（Rimsky et al. 2024; Zou et al. 2023a）**：单模型 SV 提取与调制的基础工作；本文在其之上引入跨模型视角，而非重新发明提取流程。
2. **线性表示假设（Park et al. 2024; Jiang et al. 2024）**：论证单 LLM 内概念方向线性可分；本文进一步将其推广到跨模型场景，揭示不同模型共享"柏拉图式"线性结构。
3. **Zou et al. (2023c) 通用越狱攻击**：发现 jailbreak 特征的线性可迁移性，归因于"普适有害特征"假设；本文与之不同，聚焦 11 类普适概念而非单一攻击向量，并强调通用性而非安全性漏洞利用。
4. **软提示迁移（Zhang et al. 2024; Su et al. 2022）**：研究 prompt tuning 跨任务/模型迁移；本文研究对象是模型隐状态的语义方向，两者在表征粒度与物理意义上截然不同。
5. **Platonic Representation Hypothesis（Huh et al. 2024）**：从理论上预言不同网络收敛于共享现实模型；本文为该假设在 LLM SV 层面的首个系统性实验证据。
6. **安全对齐与红队（Feng et al. 2024; Mazeika et al. 2024）**：依赖微调或额外分类器；本文展示利用小模型低成本提取 SV 即可调控大模型行为的替代路径。

## 局限性与未来方向
1. **概念覆盖范围有限**：仅覆盖 11 个代表性概念，更多高层抽象概念（如战略意图、元认知）有待探索，需耗费大量人工标注成本。
2. **评估指标缺乏统一框架**：CAA 与 RepE 指标体系差异大，混合评测难以横向比较；需建立标准化、跨概念的统一评测基准。
3. **超参数 β 仍需手动调优**：自动选择最优 β 是领域开放问题，本文未解决，限制了工程落地。
4. **未探索非线性变换**：仅验证线性映射的有效性，对于更大架构/训练差异的模型对是否仍需非线性对齐尚待研究。
5. **潜在滥用风险**：方法可被用于跨模型越狱攻击，需在部署时配套安全审查机制。

## 研究启发与可借鉴点
1. **"表示对齐矩阵"范式可迁移**：用配对共词库做最小二乘回归学习跨模型变换矩阵 T 的思路，可推广至多模态模型（图文对齐）、多语言模型（跨语言概念迁移）等场景。
2. **弱→强迁移降低安全评测成本**：利用小模型提取 SV 再变换调控大模型，可大幅缩减红队测试和安全性对齐的算力开销，适合团队在资源受限环境下开展安全评测。
3. **跨概念泛化 T 的实验设计**：用概念无关语料学习 T 以验证泛化性的实验设置，可作为检验其他表征可迁移性的标准协议。
4. **与团队方向的结合机会**：若团队关注 LLM 安全对齐或模型压缩，可将 L-Cross Modulation 作为低成本"概念注入"工具，或将 T 矩阵分析用于诊断不同版本模型的语义漂移。
5. **自动 β 搜索的改进空间**：结合梯度信息或贝叶斯优化自动选取 β，是可直接复用的工程改进点。

## 关键术语表
**Steering Vector (SV)**：从 LLM 隐藏层表示中提取的、指向特定概念方向的单位向量，叠加进隐状态可引导模型生成偏向该概念的文本。
**L-Cross Modulation**：本文提出的线性变换方法，通过最小二乘学习变换矩阵 T，将源模型的概念 SV 映射到目标模型的表征空间从而实现跨模型调制。
**Self Modulation**：利用目标模型自身提取的 SV 进行调制，代表单模型内的性能上限参考。
**Platonic Representation Hypothesis**：Huh et al. (2024) 提出的假说，认为不同训练目标与数据的神经网络在表示空间中收敛于对同一现实结构的共享统计模型。
**CAA (Contrastive Activation Addition)**：Rimsky et al. (2024) 提出的 SV 提取方法，通过对比文本对的二元选择计算概念方向。
**RepE (Representation Engineering)**：Zou et al. (2023a) 提出的自顶向下表示工程框架，使用 PCA 等方法提取 SV。
**弱→强迁移 (Weak-to-Strong Transferability)**：从小参数模型（如 0.5B）提取的 SV 经线性变换后能有效调控大参数模型（如 7B）行为的现像。

## 可复现要素
- **数据集**：CAA 与 RepE 开源数据集（MIT 协议），11 个概念的对比文本对与测试题均已公开。
- **代码**：实验基于 CAA 与 RepE 官方开源代码库实现，链接见论文脚注 7、8。
- **模型**：Llama2-7B-Chat、Qwen2-7B-Instruct、Llama3.1-8B-Instruct、Qwen2-0.5B-Instruct，均为公开权重。
- **关键超参数**：SV 提取层（Llama2: 13，Qwen2: 18，Llama3.1: 13）；CAA 统一 β=1；RepE 各概念-模型组合 β 详见 Appendix 表 5/6。
- **硬件**：单卡 A6000 GPU。
- **未提及**：变换矩阵 T 的存储与分发格式、跨设备批量推理的脚本。
