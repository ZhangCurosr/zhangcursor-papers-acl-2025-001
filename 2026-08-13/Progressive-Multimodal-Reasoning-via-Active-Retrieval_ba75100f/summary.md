---
title: "Progressive-Multimodal-Reasoning-via-Active-Retrieval"
source: https://aclanthology.org/2025.acl-long.180.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:15:05"
field: "多模态大模型推理"
keywords: ["多模态推理", "主动检索", "蒙特卡洛树搜索", "过程奖励模型", "MCTS", "RAG"]
innovations: ["在MCTS扩展阶段引入主动检索替代自采样，动态获取每步推理所需知识", "构建混合多模态检索语料库并结合知识概念过滤提升检索质量", "两阶段课程学习对齐PRM（DPO预对齐+逐点微调），实现自动化逐步推理验证"]
benchmarks: ["MATHVISTA", "WE-MATH", "GAOKAO-MM"]
---

# 论文速读：Progressive-Multimodal-Reasoning-via-Active-Retrieval

## 一句话总结
本文提出 AR-MCTS 框架，通过主动检索（Active Retrieval）与蒙特卡洛树搜索（MCTS）结合，为多模态大语言模型（MLLMs）自动获取高质量逐步推理标注，并据此对齐过程奖励模型（PRM），实现可验证的多步多模态推理。

## 研究问题与动机
- **MLLMs 多步推理能力不足**：复杂推理任务需要多步推导，现有 MLLMs 常因模态间不对齐导致内部知识不足以支撑推理路径扩展。
- **现有 MCTS 方法仅优化模拟阶段**：多数方法依赖 beam search 进行扩展，仅在文本推理中有效，未考虑多模态场景下每步所需的差异化知识支持。
- **PRM 标注依赖人工，难以扩展**：过程奖励模型虽能提供细粒度验证，但需要大量人工标注的逐步推理轨迹，缺乏可扩展性。
- **小模型推理潜力未被充分挖掘**：轻量级 MLLMs 在多步推理任务中表现较差，需要外部知识补充以提升可靠性。

## 核心贡献（创新点）
- **理论建模揭示 MCTS 在多模态推理中的局限性**：形式化推导表明扩展阶段决定了推理质量上限，现有方法忽略多模态场景中每步需动态获取差异化知识的本质需求。
- **首次将检索机制引入多模态推理的 MCTS 扩展阶段**：用主动检索替代 beam search，根据当前推理状态动态获取解题洞察，提升采样多样性与准确性。
- **提出 AR-MCTS 通用框架**：结合 MCTS 与主动检索自动获取逐步推理标注，并通过课程学习对齐 PRM，实现过程级自动推理验证，无需人工标注。
- **构建高质量混合多模态检索语料库**：整合数学特定知识（GSM8K、MATH、MATHVISTA 等）与通用推理知识（Wikipedia、COIG），覆盖 20+ 数学子领域。

## 方法详解
- **混合多模态检索语料库构建**：数学特定源 $D_M$ 包含 22K 纯文本 QA 对（GSM8K、MATH）和 12.5K 多模态样本（MATHVISTA、MathVerse、MathVision、WE-MATH）；通用源 $D_G$ 包含中英文 Wikipedia 和 COIG，合并为 $D_H = D_M \cup D_G$。
- **统一多模态检索模块**：文本检索使用 Contriever（余弦相似度）；跨模态检索使用 CLIP（ViT-L/14），对图文对编码为 $E_x(x,t) = (E_I(x) + E_T(t))/2$（纯文本时退化为 $E_T(t)$），通过 FAISS 索引检索 Top-K。
- **知识概念过滤**：对检索结果额外计算与知识概念标签 $L_{kc}$ 的相似度，双阈值过滤：$\text{Sim}(r, Q^m) \geq T_r$ 且 $\text{Sim}(r, L_{kc}) \geq T_{kc}$，确保检索知识与问题在概念层面一致。
- **AR-MCTS 推理标注（四步）**：
  - **选择**：从根节点出发，按 UCB 公式 $\mathrm{UCB}(i) = w_i + C \cdot \sqrt{2 \ln(N_i/n_i)}$ 递归选择子节点。
  - **主动检索扩展**：在每个状态 $s_i$，将当前多模态查询与历史推理步骤拼接后，从 $D_{ins}$ 动态检索 $r_i$，替换上一步的 $r_{i-1}$，再采样 $k$ 条推理路径：$p_\theta(y|x) = \prod_{j=1}^{k} p_\theta(y_i^j | Q_i^m, r_i)$。
  - **模拟**：对每个展开节点做一步 rollout，价值函数 $V(s_i) = \frac{1}{k}\sum_{j=1}^{k} \mathbb{I}(y_j = \hat{y}_i)$，答案为正确则 $V=1$，否则 $V=0$。
  - **回溯传播**：沿路径更新访问次数 $N(s,a) \leftarrow N(s,a)+1$ 和 Q 值 $Q(s,a) \leftarrow Q(s,a) + \frac{1}{N(s,a)}(V(s) - Q(s,a))$。
- **课程过程奖励建模（两阶段）**：
  - **逐步 DPO 预对齐**：从 AR-MCTS 采样中筛选 $v_j > 0.8$ 为正样本 $y^+$、$v_j = 0$ 为负样本 $y^-$，构造偏好对 $D^{step}$，使用 DPO 损失 $\mathcal{L}_{SDPO} = -\mathbb{E}[\log\sigma(\beta\log\frac{\pi_\theta(y^+|Q^m)}{\pi_\theta(y^-|Q^m)}) - \beta\log\frac{\pi_{ref}(y^+|Q^m)}{\pi_{ref}(y^-|Q^m)})]$ 对齐偏好。
  - **逐点微调（PFT）**：对预对齐 PRM 施加交叉熵损失 $\mathcal{L}_{PFT} = \sum_i [\hat{y}_i \log r_i + (1-\hat{y}_i)\log(1-r_i)]$，其中 $\hat{y}_i$ 为 golden label，$r_i$ 为 PRM sigmoid 输出。
- **推理阶段**：使用 PRM 对每步评分，保留最高分节点，设置第 4 轮早停以降低计算复杂度。

## 实验与结果
- **数据集**：MATHVISTA（testmini，含几何、代数、统计等6类）、WE-MATH（8类指标）、GAOKAO-MM（中文高考多模态基准，含8个学科）。
- **基线方法**：Zero-shot、Self-Consistency、Self-Correction、ORM（结果奖励模型）。
- **骨干模型**：GPT-4o、GPT-4V（闭源）；LLaVA-OneVision-72B、InternVL2-8B、Qwen2-VL-7B、Llama3-LlaVA-NeXT-8B（开源）。
- **主要结果（Qwen2-VL-7B）**：MATHVISTA ALL 64.1%（vs Zero-shot 58.8%，+5.3%）；WE-MATH S3 40.6%（vs ORM 34.6%，+6.0%）；GAOKAO-MM ALL 37.4%。
- **最强结果（GPT-4o）**：MATHVISTA ALL 62.6%（+3.6% vs Zero-shot）；WE-MATH S3 56.4%（+6.1% vs ORM）。
- **消融实验**：移除 PRM 导致 MATHVISTA -3.1%、WE-MATH S3 -2.9%；移除主动检索导致 -2.2%/-1.9%；移除知识概念过滤导致 -1.3%/-1.1%。
- **采样分析**：AR-MCTS 随采样数增加准确率单调上升，PRM（soft labels）优于硬标签训练。

## 相关工作脉络
- **Chain-of-Thought / Tree-of-Thought**：引导模型分解问题，但未解决多模态推理中知识缺失问题；本文通过主动检索补充每一步所需的外部知识。
- **Self-Consistency / Self-Correction**：依赖模型自采样或自反馈，后者在小模型上稳定性差（Qwen2-VL-7B 下降超 8%）；本文用 PRM 验证路径质量，避免自纠错的不稳定性。
- **ORM vs PRM**：ORM 仅验证最终结果；本文证明 PRM 在复杂多步推理（WE-MATH S3）上显著优于 ORM（GPT-4o: 56.4% vs 50.3%）。
- **MCTS for LLM reasoning**：已有工作（如 AlphaMath、Math-Shepherd）在纯文本推理中使用 MCTS；本文首次将其扩展到多模态，并解决模态不对齐导致的扩展瓶颈。
- **Retrieval-Augmented Generation**：RAG 通常一次性检索后生成；本文在 MCTS 每步动态检索，实现"边推理边检索"的渐进式知识补充。

## 局限性与未来方向
- **计算开销**：MCTS 标注需要较多推理计算，虽远低于人工标注，但相比零样本推理仍有额外开销；高效推理引擎（如 vLLM）可缓解。
- **PRM 未与 MLLM 基础训练联合优化**：当前 PRM 独立于 MLLM 训练，理想情况应融合到模型基础训练中以增强图文交互理解。
- **检索与推理的深度集成仍有空间**：目前检索主要基于相似度，未来可探索基于模型反馈的动态知识缺失补偿机制。

## 研究启发与可借鉴点
- **主动检索替代自采样策略**：在 MCTS/Tree Search 类框架中，用外部知识检索替代模型内部 beam search，可显著提升多模态推理的采样多样性与准确性。
- **知识概念过滤机制**：在检索后增加语义一致性过滤（双阈值），可有效降低噪声知识的干扰，值得在 RAG 系统中推广。
- **课程学习对齐 PRM**：先 DPO 预对齐偏好，再交叉熵微调评分能力，这种"先判别后打分"的两阶段策略具有通用性，可迁移到其他奖励模型训练场景。
- **梯度归因与 PRM 验证结合**：论文中通过 rollout 价值函数评估路径质量，可将此思路与梯度归因方法结合，用于更细粒度的推理步骤重要性分析。

## 关键术语表
- **AR-MCTS**：Active Retrieval Monte Carlo Tree Search，本文提出的融合主动检索与 MCTS 的多模态推理框架。
- **Process Reward Model (PRM)**：过程奖励模型，对推理每一步给出细粒度验证信号，区别于仅验证最终结果的 ORM。
- **Active Retrieval**：主动检索，在推理的每个 MCTS 扩展步动态检索相关知识，替代传统的静态检索或自采样。
- **Hybrid-Modal Retrieval Corpus**：混合多模态检索语料库，同时包含纯文本和图文对的高质量推理知识库。
- **Knowledge Concept Filtering**：知识概念过滤，通过多模态嵌入计算检索结果与问题知识概念的相似度，双阈值筛选确保概念一致性。
- **Curriculum PRM Alignment**：课程式 PRM 对齐，先通过 DPO 学习步骤偏好判别，再通过交叉熵微调学习步骤评分。
- **Step-wise DPO**：逐步直接偏好优化，在 MCTS 采样过程中构造正负偏好对，直接优化推理模型的偏好分布。
- **Soft vs Hard Labels**：软标签使用 PRM 连续输出概率，硬标签使用二值 0/1，软标签在推理验证中表现更优。

## 可复现要素
- **数据集**：MATHVISTA、WE-MATH、GAOKAO-MM 均有公开版本；检索语料库由 GSM8K、MATH、MathVerse、MathVision、Wikipedia、COIG 等开源数据构建。
- **代码/权重**：论文未明确声明开源；PRM 基于 Qwen2-7B 微调，检索使用 Contriever 和 CLIP-ViT-L/14 公开模型。
- **关键超参**：DPO 学习率 5e-7、β=0.3、warmup=0.1、batch size=64、2 epochs；PFT 学习率 7e-6、batch size=128、warmup=20 steps、3 epochs；最大上下文长度 DPO 4096 tokens、PFT 8192 tokens；推理早停设为第 4 轮；检索阈值 $T_r$ 和 $T_{kc}$ 见附录。
