---
title: "Fine-Tuning-on-Diverse-Reasoning-Chains-Drives-Within-Infere"
source: https://aclanthology.org/2025.acl-long.191.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:52:07"
field: "大语言模型推理增强"
keywords: ["Chain of Thought", "Diverse CoT", "self‑correction", "instruction tuning", "reasoning refinement", "large language models"]
innovations: ["提出DCoT训练范式，使模型在单次推理中生成序列化的多样化推理链以实现内部修正", "通过定量与人工分析证实修正增益源于逻辑改进而非随机扰动", "证明DCoT可与self‑consistency等解码扩展兼容，且在单次修正(k=2)时性价比最高"]
benchmarks: ["GSM8K", "ARC", "HotpotQA", "ConditionalQA", "Big Bench Hard", "CSQA", "AQuA", "SVAMP"]
---

# 论文速读：Fine-Tuning-on-Diverse-Reasoning-Chains-Drives-Within-Inference-CoT-Refinement

## 一句话总结
本文提出**Diverse Chain of Thought (DCoT)** 微调方法，使大语言模型能在单次推理步骤中连续生成多个相互参考的推理链，从而实现无需外部反馈的**推理链内部修正**；该方法在1.3B至70B多系列模型及多种推理任务上均稳定提升性能，尤其对结果状态空间大的任务（如数值计算）增益显著。

## 研究问题与动机
1. **现有CoT方法局限**：已有工作（如self‑consistency、meta‑reasoning）通常独立生成多条CoT，再通过投票、集成等后处理策略聚合，模型在推理时无法访问已生成的推理链，限制了**链间参照与迭代修正**的能力。
2. **缺乏内部自我修正机制**：多数自纠错研究依赖两步系统（生成答案→识别错误）或外部反馈，而大模型在无需显式错误定位的情况下能否自发改进推理链仍不明确。
3. **小模型CoT能力不足**：小型模型在未见任务上生成正确CoT的能力较弱，且独立采样多条CoT的成本较高，需要一种更高效的训练与推理范式。
4. **训练数据利用方式单一**：现有指令微调数据集通常将`(question, CoT)`作为独立样本，未利用同一问题下多条正确推理链之间的**多样性互补**信息。

## 核心贡献（创新点）
1. **首次实现单次推理内的链式修正**：提出DCoT训练范式，将同一问题的多条正确CoT拼接为单一序列进行指令微调，使模型学会在生成后续CoT时参考先前推理，实现**within‑inference refinement**。
2. **证明修正增益源于逻辑改进而非随机扰动**：通过定量分析与人工评估（LLaMA‑2 7B）表明，第二条CoT能修正第一条的错误推理步骤或得出正确结论，而非简单重复或随机生成。
3. **方法兼容现有CoT增强技术**：DCoT微调后的模型可与self‑consistency等解码扩展无缝结合，且DCoT+greedy在部分模型上即超过CoT+self‑consistency，说明其本身已具备更强的推理质量。
4. **揭示任务特性与增益的关系**：系统验证DCoT在**输出空间较大**的任务（数值生成、跨度提取、多选题）上提升明显，而在二元选择等小输出空间任务上收益有限甚至持平，为后续方法设计提供选用依据。

## 方法详解
### 1. DCoT指令模板
- **输入格式**：`[Question] Question [Options] Options [Number of answers] k`  
  （`[Options]`仅多项选择题包含；`k`为要求生成的CoT数量）
- **输出格式**：`[Answer 1] CoT₁ [Answer 2] … [Answer k] CoT_k [Final answer] answer`  
  （`[Final answer]`作为收敛指令，使模型综合所有推理链输出最终答案）

### 2. 训练数据构建
- **CoT触发器**：沿用Ott et al. (2023)的多种提示后缀（如“Let’s think step by step”），每个问题随机采样4个触发器，由GPT‑3.5‑turbo（零样本，温度0.7）生成推理链。
- **数据过滤**：仅保留导向**正确答案**的推理链（依据数据集标签）；每个问题最多保留4条CoT以适应上下文窗口。
- **样本组织**：DCoT将同一问题的多条CoT拼接为一个训练样本`(q, cot₁, cot₂, …)`；对比基线CoT则保持传统`(q, cot)`独立样本形式，两者使用完全相同的推理链集合。

### 3. 模型训练
- **基础模型**：Phi‑1.5 (1.3B)、Phi‑2 (2.7B)、LLaMA‑2 7B、13B、70B，以及LLaMA‑2 13B Chat（用于探讨已指令微调模型的适配性）。
- **微调方式**：LoRA（`r=64`, `alpha=16`, `dropout=0.1`），学习率2e‑4，3个epoch，按dev集平均性能选择最佳checkpoint。
- **训练并行**：9个数据集（ARC、BGQA、CoinFlip、ConditionalQA、GSM8K、HotpotQA、LLC、Quartz、StrategyQA）联合训练，提升泛化性。

### 4. 推理控制
- 推理时可通过提示中的`k`参数灵活控制生成CoT数量（`k ∈ {1,2,3,4}`）；`k=2`即为一次修正，兼顾性能与计算开销。

## 实验与结果
### 评测设置
- **评测数据集**：包括数值（GSM8K）、跨度提取（ConditionalQA、HotpotQA）、多选题（ARC、BGQA、Quartz）、二元（StrategyQA）、符号（LLC）五类任务。
- ** unseen任务**：CSQA、AQuA、SVAMP、Object Count，检验通用性。
- **鲁棒性检查**：Big Bench Hard，验证方法在CoT可能有害的小模型困难任务上不产生退化。
- **评估指标**：分类任务用macro‑F1，跨度提取用SQuAD指标。

### 主要结果（Table 1）
| 模型 | 基线(CoT) | DCoT | 提升幅度 |
|------|-----------|------|----------|
| Phi‑1.5 (1.3B) | 47.2 | 49.39 | **+2.19** |
| Phi‑2 (2.7B) | 60.85 | 62.6 | **+1.75** |
| LLaMA‑2 7B | 58.97 | 60.8 | **+1.83** |
| LLaMA‑2 13B | 64.39 | 66.18 | **+1.79** |
| LLaMA‑2 70B* | 66.96 | 68.63 | **+1.67** |

- **最强结果**：LLaMA‑2 13B + DCoT+self‑consistency (SC) 在GSM8K上达到**54.51**，较CoT+SC（50.27）提升**4.24**个百分点；DCoT+SC在多项任务上均超越CoT+SC。
- **任务分布规律**：数值与跨度提取任务增益最大（如Phi‑2在GSM8K上从56.71升至60.73），多选题次之，二元与符号任务提升有限。
- **k值分析**（Table 2）：**k=2**（一次修正）在所有模型上平均性能最优且稳定；k≥3后收益饱和甚至略降，表明单次修正已足够。

### 关键验证实验
1. **DCoT@1 ≈ CoT**：单链生成性能与基线持平，证明DCoT训练不会破坏原有CoT能力，可安全替换或混合现有指令微调数据。
2. **Unseen任务无退化**（Table 3）：在CSQA等未见任务上DCoT保持或小幅提升性能，未见明显下降。
3. **Big Bench Hard鲁棒性**（Table 4）：在CoT可能有害的小模型困难任务上，DCoT性能与CoT相当，部分模型随k增加仍有小幅提升。
4. **人工分析**（Section 4.6‑4.7）：LLaMA‑2 7B的DCoT@2修正案例中，第二条CoT约50%改变了推理模式并得出正确答案，约10%修正了第一链的逻辑结论，证实了**内部修正机制**的存在；DCoT@3的答案模式显示模型并非盲目追求差异，而是“若第一链已正确则维持”。

## 相关工作脉络
1. **Self‑consistency (Wang et al., 2023)**：通过多次独立采样CoT并投票聚合；DCoT与之本质区别在于**推理链在单次生成中相互可见**，支持链间修正而非事后投票。
2. **Meta‑reasoning / 元提示聚合 (Yoran et al., 2023)**：设计元提示比较各CoT优劣；同样属于独立采样后的后处理，无法利用生成过程中的逐步信息。
3. **CoT指令微调 (Ho et al., 2023; Huang et al., 2023; Kim et al., 2023)**：将大模型CoT蒸馏至小模型，但每条数据仅含单个CoT，未利用多样性；本文将其扩展为多链序列格式。
4. **Self‑correction / 自我修正 (Madaan et al., 2023; Shinn et al., 2024; Estornell et al., 2025)**：通常依赖两步流程或外部反馈；本文方法无需显式错误检测，直接在生成序列中完成修正。
5. **Code prompting / Graph of thoughts (Puerto et al., 2024; Besta et al., 2024)**：探索结构化推理表示；DCoT可视为自然语言序列版的一种轻量实现，未来可与这些结构化合流。

## 局限性与未来方向
1. **上下文窗口限制**：当前最多生成4条CoT，更长的推理链可能受限于模型序列长度。
2. **多样性不保证**：训练目标并非强制链间差异，而是允许修正；实际推理中后续链可能与前一条高度相似（尤其当第一链已正确时）。
3. **单一供应商数据源**：CoT仅由GPT‑3.5‑turbo生成，混合多供应商数据可能导致性能下降（见附录C）。
4. **70B模型实验规模有限**：受算力所限，70B仅用900个问题训练、每数据集100题测试，且未调参，结果仅为趋势性验证。
5. **未来方向**：① 探索将DCoT扩展至代码提示、图思维等结构化推理形式；② 研究如何在无外部反馈条件下进一步提升k的扩展性（目前k>2收益递减）；③ 验证在更大规模商业模型（如GPT‑4o）上的零样本提示可行性。

## 研究启发与可借鉴点
1. **训练数据重组策略**：将同一问题的多条正确推理链拼接为序列样本，是一种低成本、高收益的数据增强手段，可迁移至任何包含CoT的指令微调数据集。
2. **单次推理内修正机制**：证明了模型能在生成过程中逐步参照前序输出进行自我修正，该范式可推广至其他序列生成任务（如代码生成、数学证明）。
3. **k=2的性价比**：一次修正即可捕获大部分收益，为实际应用提供了明确的超参建议，避免盲目增加生成数量带来的token浪费。
4. **与解码策略正交**：DCoT可与self‑consistency、反事实采样等解码扩展结合，且DCoT自身greedy解码即接近甚至超越CoT+SC，说明其提升了**单链质量**而非仅依赖集成。
5. **任务类型选择**：后续研究可优先将DCoT应用于结果状态空间大、需要多步推导的任务（如数值推理、开放域问答），而在二元判定类任务上需谨慎评估。

## 关键术语表
- **Chain of Thought (CoT)**：要求模型在输出最终答案前显式生成中间推理步骤的提示技术。
- **Diverse Chain of Thought (DCoT)**：本文提出的训练方法，使模型学会在单次推理中顺序生成多条相互参考的推理链。
- **Within‑inference refinement**：在同一个生成交叉过程中，后续推理链参照先前链进行修正，无需多次独立采样或外部反馈。
- **Self‑consistency**：多次独立采样CoT后通过投票聚合选出最一致答案的解码策略。
- **Result state space**：任务可能输出的答案集合大小；空间越大，DCoT的修正收益越明显。
- **LoRA**：低秩自适应微调技术，通过冻结预训练权重并注入低秩矩阵实现高效参数更新。

## 可复现要素
- **数据集**：训练数据基于ARC、BGQA、CoinFlip、ConditionalQA、GSM8K、HotpotQA、LLC、Quartz、StrategyQA；代码与数据**已公开**（GitHub链接见论文）。
- **代码/权重**：训练代码、指令模板、触发器列表均在附录及开源仓库中提供；模型权重需读者自行在Hugging Face获取基座模型后微调。
- **关键超参**：LoRA `r=64, alpha=16, dropout=0.1`，学习率2e‑4，batch size 4，最大序列长度4096，3个epoch；最佳checkpoint按dev集平均性能选取。
- **推理温度**：生成CoT时使用温度0.7（见附录J）；评测时主要报告greedy解码结果，并与SC（4次采样）结合。
