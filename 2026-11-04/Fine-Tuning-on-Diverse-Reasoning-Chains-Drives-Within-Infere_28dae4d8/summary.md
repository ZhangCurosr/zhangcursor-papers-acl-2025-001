---
title: "Fine-Tuning-on-Diverse-Reasoning-Chains-Drives-Within-Infere"
source: https://aclanthology.org/2025.acl-long.191.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 17:02:06"
field: "大语言模型推理能力增强"
keywords: ["Chain of Thought", "self-correction", "instruction tuning", "reasoning", "LLM fine-tuning", "diverse reasoning chains", "within-inference refinement"]
innovations: ["提出DCoT训练范式，在单次推理步骤中顺序生成多条CoT并实现自我修正", "首次实证无外部反馈的within-inference自我修正可带来稳定性能增益", "仅通过重新组织训练数据格式（独立CoT对→串联多CoT序列）即可显著提升小到大模型推理性能"]
benchmarks: ["GSM8K", "HotpotQA", "ConditionalQA", "ARC", "Big Bench Hard", "CSQA", "AQuA", "SVAMP"]
---

# 论文速读：Fine-Tuning on Diverse Reasoning Chains Drives Within-Inference CoT Refinement in LLMs

## 一句话总结
本文提出 Diverse Chain of Thought (DCoT) 训练方法，通过instruction-tuning让LLM在**单个推理步骤**中顺序生成多条多样化推理链，使模型能够基于前一条推理链进行自我修正；实验表明DCoT在1.3B至70B多种模型和任务上均显著提升性能，且在数值类大输出空间任务上增益最为突出，且仅需一次修正（k=2）即可获得最佳收益。

## 研究问题与动机
1. **现有CoT增强方法的局限**：Self-consistency等主流方法通过独立采样多条CoT再投票聚合，但各链间相互独立，模型无法在推理过程中参考已有推理路径进行迭代优化。
2. **within-inference自我修正能力缺失**：已有self-correction工作多依赖两阶段外部反馈机制（先生成答案再识别错误），而非在单次推理中实现自然的链式修正。
3. **如何高效利用CoT多样性**：现有instruction-tuning数据集虽含多条CoT样本，但均以独立(q, CoT)对形式使用，未将其串联为单次推理的多链生成任务。
4. **小型模型CoT能力不足**：小规模LLM（<7B）在未见任务上生成正确CoT的能力较差，需要更有效的训练范式来增强其推理能力。

## 核心贡献（创新点）
1. **提出DCoT单次推理多链生成方法**：设计训练数据格式为`(question, [CoT₁, CoT₂, ..., CoTₖ])`，使模型学会在单个推理步骤中顺序生成多条推理链并逐步修正。
2. **首次实证within-inference自我修正带来的性能增益**：通过定量与人工评估证明，DCoT的增益源于模型"在无前向反馈的情况下参考前一条CoT修正自身推理"，而非随机扰动或外部引导。
3. **系统验证跨模型族与规模的有效性**：在Phi(1.3B/2.7B)、LLaMA-2(7B/13B/70B)五款模型上全面测试，DCoT在所有模型上平均优于CoT基线，且增益在数值/span提取等大输出空间任务上尤为显著。
4. **证明DCoT的可扩展兼容性与安全性**：DCoT可与self-consistency等既有CoT增强技术叠加使用；DCoT@1性能与CoT基线相当，可安全替换现有instruction-tuning数据集中的CoT数据。

## 方法详解
**DCoT指令模板设计：**
- 输入提示：`[Question] Question [Options] Options [Number of answers] k`，其中k为请求生成的CoT数量。
- 输出格式：`[Answer 1] CoT₁ [Answer 2] ... [Answer k] $\text{CoT}_k$ [Final answer] answer`，通过`[Final answer]`指令作为收敛机制。
- CoT（基线）使用完全相同的训练数据，但每个样本独立格式为`(question, CoT)`，仅在训练时利用多样性，推理时仍生成单条CoT。

**训练数据生成流程：**
1. 使用GPT-3.5-turbo（zero-shot）+ 多种CoT trigger（如"Let's think step by step"等）为每个问题生成最多4条CoT。
2. 仅保留最终答案正确的推理链作为训练数据（通过数据集标签验证）。
3. 将同一问题的多条CoT拼接为顺序序列构成DCoT样本。

**关键原理：**
- 训练时所有CoT均为正确推理链，但多样性使模型学到"从不同角度推理同一步骤"的能力。
- 推理时模型基于前一条CoT的上下文进行修正，实现without external feedback的within-inference refinement。
- 训练数据量与CoT基线完全相同，唯一区别在于response格式的组织方式。

## 实验与结果
**实验设置：**
- **模型**：Phi 1.5 (1.3B), Phi 2 (2.7B), LLaMA-2 7B, LLaMA-2 13B, LLaMA-2 70B（后者因成本限制仅小规模训练与测试）
- **训练数据集**（9个）：ARC, BGQA, CoinFlip, ConditionalQA, GSM8K, HotpotQA, LLC, Quartz, StrategyQA
- **未见任务**：CSQA, AQuA, SVAMP, Object Count, Big Bench Hard
- **评估指标**：Macro F1（分类）、SQuAD metric（span extraction）
- **超参数**：LoRA(r=64, α=16, dropout=0.1), lr=2e-4, max_seq_length=4096, 训练3 epochs

**主要结果（Table 1）：**
- **Phi 2 (2.7B)**：DCoT平均62.60 vs CoT 60.85（+1.75），GSM8K上56.71→60.73（+4.02）
- **LLaMA-2 7B**：DCoT平均60.80 vs CoT 58.97（+1.83），GSM8K上28.51→29.57
- **LLaMA-2 13B**：DCoT平均66.18 vs CoT 64.39（+1.79），GSM8K上42.53→44.28（+1.75）
- **LLaMA-2 70B**：DCoT平均68.63 vs CoT 66.96（+1.67），GSM8K上56→66（+10.0）

**最强结果：** LLaMA-2 70B在GSM8K上DCoT达到66.0（vs CoT的56.0），绝对提升达10.0点；DCoT整体平均最优为LLaMA-2 13B的66.18。

**关键发现：**
- **k=2即最优**（Table 2）：所有模型上DCoT@2平均性能最佳，增加至k=3/4无进一步提升（数学任务GSM8K例外，k=3略有增益）。
- **DCoT@1 ≈ CoT基线**：单链生成性能与标准CoT相当，证明DCoT训练不损害基础CoT能力。
- **DCoT + SC仍有效**（Table 1最后两行）：与self-consistency结合后，DCoT+SC在所有模型上均超过CoT+SC。
- **未见任务无退化**（Table 3）：DCoT在CSQA等未见任务上无显著性能下降。
- **Big Bench Hard鲁棒性**（Table 4）：在CoT可能对小型模型有害的任务上，DCoT仍保持与CoT相当或更优的性能。
- **任务类型差异**：数值（Numeric）和Span Extraction任务提升最大，Binary/Symbolic任务提升较小或波动。

## 相关工作脉络
1. **Self-Consistency (Wang et al., 2023)**：独立采样多条CoT并通过多数投票聚合答案；本文DCoT的关键区别在于CoT在**单次推理**中顺序生成且后链可参考前链，而非独立生成后投票。
2. **Meta-Reasoning over Multiple CoTs (Yoran et al., 2023)**：构建元提示聚合多条推理路径；同样基于并行独立生成，而DCoT通过序列生成实现内在的链间依赖与修正。
3. **CoT Distillation (Hsieh et al., 2023; Ho et al., 2023)**：将大模型CoT蒸馏到小模型；Ho等人提到"每个问题采样多条CoT进行数据增强"但未在推理时利用多样性，本文首次将多链多样性引入推理阶段。
4. **Self-Correction / Self-Refine (Madaan et al., 2023; Huang et al., 2023)**：依赖两阶段机制（生成+错误识别），通常需要外部反馈或显式反思步骤；DCoT无需额外模块即可在单次推理中实现自我修正。
5. **CoT Instruction Tuning (Chung et al., 2024; Kim et al., 2023)**：通过指令微调赋予小模型CoT能力；本文在其基础上进一步利用同问题多CoT的串联训练方式提升推理质量。
6. **Self-Contrast (Zhang et al., 2024)**：显式对比不同CoT并反思最终答案；与DCoT不同的是其为推理时prompt工程方法，而DCoT是训练阶段的方法且无需额外prompt设计。

## 局限性与未来方向
1. **上下文窗口限制**：当前最多生成k=4条CoT，受限于模型上下文长度，无法支持更深层次的迭代修正。
2. **多样性不保证**：训练多链多样性不代表推理时模型一定会生成真正不同的推理链（尤其在首链已正确时，后续链可能重复）。
3. **70B模型实验不充分**：因计算成本限制，70B模型仅用900个问题训练、每个数据集仅测试100题，无法进行全面超参调优。
4. **未探索多LLM供应商协同**：训练数据仅来自单一商业LLM提供商（GPT-3.5），混合多供应商生成CoT可能导致性能下降（初步实验发现）。
5. **未覆盖LLaMA 3/3.1**：因模型尺寸跨越大（8B/70B），无法在同一模型族内做规模对比；LLaMA 3.2发布晚于实验完成时间。
6. **未来方向**：扩展到code prompting、Graph of Thoughts等推理范式；提升k的可扩展性以实现无需外部反馈的更深修正。

## 研究启发与可借鉴点
1. **训练数据重组即可产生显著增益**：相同CoT数据仅通过改变response格式（从独立对到串联序列）即可大幅提升推理性能，提示我们重新审视现有instruction-tuning数据的组织方式——大量已有数据集包含同一问题的多CoT样本，可直接转化为DCoT训练数据。
2. **"最小修正"范式的高效性**：k=2（仅一次修正）即达到最优，提示在self-correction/多轮推理设计中，不必盲目追求多轮修正，一次精准的within-inference修正可能比多次独立采样更具性价比。
3. **可安全叠加到现有CoT增强框架**：DCoT与self-consistency的组合仍有效，表明本文方法可作为通用基础设施层嵌入现有CoT pipeline，未来可与reward modeling、process reward等结合探索。
4. **小模型CoT能力增强的有效路径**：DCoT在1.3B~2.7B小模型上同样有效，为低资源场景下的推理增强提供了低成本方案（仅增加约0.00007美元/次调用的token成本）。
5. **手动分析揭示的行为模式**：本文对k=2和k=3的详细人工分析（包括"为何第三链不继续改进"的归因）为后续研究如何设计有效的多链修正协议提供了可借鉴的分析框架。

## 关键术语表
**Chain of Thought (CoT)**：通过让模型生成中间推理步骤来回答问题的提示/训练方法，已被证明可显著提升LLM推理能力。
**Diverse Chain of Thought (DCoT)**：本文提出的训练方法，通过将同一问题的多条CoT串联为单个训练样本，使模型学会在单次推理中顺序生成并修正多条推理链。
**Self-Consistency (SC)**：Wang et al. (2023)提出的CoT增强方法，通过多次独立采样CoT并按多数投票选择最终答案。
**Within-inference Refinement**：在单次推理步骤中，利用前一条推理链的上下文信息对后续推理链进行修正，无需外部反馈或多轮独立推理。
**Result State Space**：任务可能的输出空间大小；本文发现DCoT在大输出空间任务（如数值生成）上增益更显著。
**Instruction Tuning**：通过指令格式的微调数据训练模型，使其遵循自然语言指令完成指定任务的技术。
**LoRA (Low-Rank Adaptation)**：一种参数高效微调方法，通过低秩矩阵分解更新预训练模型参数，大幅减少训练成本和显存占用。
**Big Bench Hard**：Suzgun et al. (2023)提出的难例评测集，包含对小模型而言CoT可能无效甚至有害的复杂推理任务。

## 可复现要素
- **数据集**：9个训练数据集（ARC, BGQA, CoinFlip, ConditionalQA, GSM8K, HotpotQA, LLC, Quartz, StrategyQA）均为公开数据集；4个未见评测数据集（CSQA, AQuA, SVAMP, Object Count/BBH）亦为公开。**代码和数据已公开**（论文声明："Our code and data are publicly available"，GitHub链接见于作者主页ukp.tu-darmstadt.de）。
- **代码/权重**：论文未提供具体仓库链接，但声明代码和数据已公开；模型基于Phi-1.5/Phi-2/LLaMA-2开源模型族，权重可用。
- **关键超参数**：LoRA r=64, α=16, dropout=0.1；learning_rate=2e-4；batch_size=4；max_seq_length=4096；warmup_ratio=0.03；训练epochs=3；seed={0, 42, 2024}；best checkpoint按dev集平均F1选择。
- **推理配置**：Greedy decoding为主；DCoT的k值通过dev集选定（各模型-数据集最优k见Table 16，通常在1~4之间）。
- **训练框架**：PEFT库+LoRA；vLLM推理引擎；GPT-3.5-turbo通过Azure OpenAI服务生成CoT训练数据（temperature=0.7）。
- **GPU环境**：Nvidia A100集群；模型以fp8加载训练、fp16推理；70B模型使用8-bit量化训练、4-bit量化推理。

---
