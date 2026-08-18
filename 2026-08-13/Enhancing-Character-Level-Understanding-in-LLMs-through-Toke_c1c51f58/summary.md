---
title: "Enhancing-Character-Level-Understanding-in-LLMs-through-Toke"
source: https://aclanthology.org/2025.acl-long.194.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:49:58"
field: "大语言模型字符级能力增强"
keywords: ["字符级理解", "中文拼写纠错", "Token化", "大语言模型", "位置预测", "反向字符预测", "TIPA", "MTIPA"]
innovations: ["提出TIPA方法，利用tokenizer词表构造反向字符预测任务增强模型token内部字符位置感知，无需修改架构", "提出MTIPA扩展至句子级多token序列，进一步提升位置预测精度", "重新定义CSC任务为带位置预测的版本，设计PPA等细粒度评估指标"]
benchmarks: ["CSCD-NS", "LEMON", "TyDi QA", "IFEVAL", "GSM8K", "MMLU", "HumanEval", "AExams", "KoBEST"]
---

# 论文速读：Enhancing-Character-Level-Understanding-in-LLMs-through-Toke

## 一句话总结
本文提出了 TIPA（Token Internal Position Awareness）和 MTIPA（Multi-Token Internal Position Awareness）方法，通过将 Token 内部字符结构转化为反向字符预测任务进行训练，显著提升大语言模型在字符级别位置感知和中文拼写纠错（CSC）任务上的性能，且不改变模型架构或分词器。

## 研究问题与动机
- **BPE 分词隐藏字符内部结构**：主流 LLM 使用的 BPE（Byte-Pair Encoding）将文本切分为子词单元，导致模型无法精准感知 token 内部的字符位置与组成结构。
- **GPT-4o 等强模型仍存在字符级缺陷**：例如模型在统计 "strawberry" 中字母 'r' 数量、或定位中文句子中特定汉字位置时频繁出错（如将"阁下"的"阁"定位错误）。
- **现有字符级模型需要架构改动**：ByT5 等 byte-level 模型虽能实现字符级精度，但需改变模型架构，难以低成本适配已有子词 LLM。
- **CSC 任务中位置精确定位至关重要**：传统 CSC 仅输出纠错结果，但引入位置预测可同时验证模型定位能力并减少输出 token 数（见表 1，Position Task 仅需 1 个 token 而非多个词元），提升效率。

## 核心贡献（创新点）
1. **提出 TIPA 方法**：利用 tokenizer 词表中的 token，构造反向字符预测任务（reverse character prediction），使模型学习 token 内部的字符组成与位置信息；与已有工作本质区别在于无需修改分词器或架构，仅需在训练数据中注入字符结构信息。
2. **提出 MTIPA 扩展**：将 TIPA 的任务形式扩展至完整句子级（multi-token sequences），通过随机采样训练数据并构建反向字符映射，进一步提升模型在多 token 上下文中的位置感知能力。
3. **重新定义 CSC 任务为带位置预测的版本**：要求模型输出错误字符的位置、错误字符及修正结果，并设计新的评估指标（PPA、SA、SAIP 等），使 CSC 任务更具细粒度可评估性；与传统 CSC 方法的本质区别在于同时要求位置精确定位。
4. **展示 TIPA 在通用模型上的迁移有效性**：通过 Llama-3.1-8B 全参数 SFT 实验证明，TIPA 不仅提升中文 CSC，还在多语言场景（TyDi QA）、指令遵循（IFEVAL）和字符级计数任务上取得提升，且不影响标准 benchmark 性能。

## 方法详解
**TIPA（Token Internal Position Awareness）：**
- 设 $T$ 为 tokenizer，$V = \{t_1, t_2, \ldots, t_m\}$ 为其词表。对每个可完全用 UTF-8 表示的 token $t$，将其分解为字符序列 $C_t = [c_1, c_2, \ldots, c_n]$。
- 构造**反向位置映射** $D_t = \{(i, c_i) \mid i = n, n-1, \ldots, 1\}$，即从末尾到开头逐字符输出其位置与字符，以 JSON 格式呈现（如 "girl" → `{"4": "l", "3": "r", "2": "i", "1": "g"}`）。
- 采用**反向排序**而非正向的核心原因：模型首先输出的数字即为 token 长度 $n$，从而同时将长度信息、分词信息和位置信息整合到一个任务中，避免正向排序下模型可能通过位置序列间接推导长度带来的歧义。
- TIPA 数据集构建为 $\mathcal{D}_{TIPA} = \{(t, D_t) \mid t \in V(\text{UTF-8})\}$，可通过对训练数据中 token 去重后生成（论文中精简后得到 24,994 个 unique tokens）。

**MTIPA（Multi-Token Internal Position Awareness）：**
- 从目标任务训练集中随机采样一定比例（论文取 $r=10\%$）的句子，对每个句子 $s$ 进行与 TIPA 相同的反向字符分解，得到 $D_s = \{(i, c_i) \mid i = n, n-1, \ldots, 1\}$。
- MTIPA 仅在需要精确定位的任务（Experiment 1：带位置预测的 CSC）中使用；传统 CSC 任务（Experiment 2）仅使用 TIPA，避免长序列训练对 LoRA 微调效率的负面影响。

**全参数 SFT 扩展：**
- 从 Llama-3.1 tokenizer 提取全部词表，构建 TIPA 数据集，与 tulu-3-sft-mixture 混合后进行 Llama-3.1-8B 全参数 SFT，得到 Llama-3.1-Tulu-TIPA-8B，无需额外字符级任务数据集。

**LoRA 训练配置**：rank=16，alpha=16，dropout=0.05，优化器 AdamW，学习率 $1\times10^{-4}$，batch size=16，在单卡 NVIDIA A800 80GB 上训练。

## 实验与结果
**数据集：**
- 训练：CSCD-NS（30,000 条）+ Wang271K（271,329 条），TIPA 精简后 24,994 unique tokens
- 验证：CSCD-NS Dev（5,000 条）
- 测试：CSCD-NS Test（5,000 条）+ LEMON（多领域，7个子域）

**Experiment 1（带位置预测的 CSC，CSCD-NS test）：**
- 基线 Pure-SFT-7B：PPA=79.45%，SA=69.58%，CF1=53.17%
- **TIPA-7B**：PPA=**84.72%**（↑5.27），SA=**70.70%**（↑1.12），CF1=**54.90%**（↑1.73）
- **MTIPA-7B（r=10%）**：PPA=**87.52%**（↑8.07），SA=**72.40%**（↑2.82），CP=**63.25%**（↑6.64），CF1=**58.81%**（↑5.64）——最强结果
- GPT-4o 的 PPA 仅 11.14%（因其字符级分词优势而相对较高，但仍有较大提升空间）

**Experiment 2（传统 CSC，CSCD-NS test + LEMON）：**
- TIPA(←)（反向）始终优于 TIPA(→)（正向），且在大模型（7B）上差距更显著
- TIPA-7B 在 LEMON 各子域上均取得更高 CF1，AVG 从 53.28% 提升至 55.51%

**Experiment 3（通用模型，Llama-3.1-8B 全参数 SFT）：**
- IFEVAL strict：64.88 → 67.84（↑2.96）
- AExams：38.92 → 40.22（↑1.30）
- TyDi QA 多语言平均 F1：47.85 → 52.81（↑4.96），其中芬兰语↑11.25、印尼语↑9.88、韩语↑10.21
- 字符级任务：Occurrence↑3.95%，Length↑7.11%，Distinct 从 9.29% 提升至 18.74%（近乎翻倍）

## 相关工作脉络
1. **BPE 与字符级理解问题**：Sennrich (2015) 的 BPE、Wang et al. (2020) 的 BPE 变体改进计算效率但隐藏字符结构；Xu & Ma (2024) 和 Shin & Kaneko (2024) 指出 LLM 在字符级任务中存在系统性缺陷，本文在此基础上提出低成本训练方案而非架构改动。
2. **ByT5 与 byte-level 模型**：Xue et al. (2022) 的 ByT5 直接处理原始字节实现字符级精度，但需改变模型架构；本文方法在保持原有 subword 架构不变的前提下实现字符级增强。
3. **CANINE 混合方法**：Clark et al. (2022) 的 CANINE 在字符和 word 层面同时编码，但多字符 token 的位置建模效率不足；本文通过训练数据注入字符位置信息，无需额外架构设计。
4. **中文拼写纠错（CSC）**：Liu et al. (2024) 的 ReLM 将 CSC 重构为句子改写；Jiang et al. (2024) 证明仅用正确数据训练的模型也可超越使用混淆集的模型；Li et al. (2024) 的 C-LLM 使用字符级分词增强 CSC；本文与之区别在于不依赖字符级分词，而是通过反向预测任务强化位置感知。
5. **字符位置敏感性问题**：Itzhak & Levy (2022) 发现模型隐式编码了正写信息但显式拼写训练未提升性能；Berglund et al. (2023) 和 Thawani et al. (2023) 研究了 "reversed curve phenomenon"；本文通过反向排序的训练设计针对性解决位置理解问题。

## 局限性与未来方向
- **OOV（未登录词）的位置预测能力待验证**：tokenizer 词表中不存在的 token 是否仍能从字符层面正确预测位置，尚需进一步研究。
- **全参数 SFT 中 TIPA 数据比例过高可能引入偏差**：混合过多 TIPA 数据可能导致模型偏向生成较短文本序列，需设置最小 token 约束，更根本的解决方案需从 pretraining 或 RL 阶段入手。
- **MTIPA 的训练效率问题**：MTIPA 数据集较长，训练时间显著增加（如 MTIPA-7B 需 99 GPU hours vs Pure-SFT-7B 的 52.6 hours），采样率需权衡。
- **当前仅验证了 LoRA 微调和全参数 SFT 两种范式**，预训练阶段的 TIPA 集成效果尚未探索。

## 研究启发与可借鉴点
1. **"反向排序"设计精妙**：将 token 长度信息通过首个输出位置隐式编码，避免正向排序中长度可被间接推断的歧义问题，这一设计可迁移至其他需要同时学习长度与结构的任务（如字数统计、文本分割）。
2. **无需修改架构的低成本增强策略**：仅通过构造额外的训练数据（利用 tokenizer 自身词表）即可增强字符级理解，为其他需要细粒度感知的任务（如代码生成中的字符匹配、公式解析）提供了可复用的训练范式。
3. **位置预测任务设计的启发性**：将 CSC 重新定义为同时输出位置+错误字符+修正字符的三元组形式，既能评估定位能力又压缩了输出 token 数，这种"多功能任务设计"值得在文本纠错、事实核查等场景借鉴。
4. **TIPA 在多语言和非拉丁语系上的增益**：芬兰语、印尼语、韩语的提升尤为显著，说明字符级理解增强的方法具有跨语言的泛化潜力，可探索在其他 agglutinative 语言（如土耳其语、芬兰语）上的应用。
5. **Pruned vs Unpruned TIPA 的策略权衡**：论文展示了从大规模数据中提取 unique token 进行精简（24,994 tokens 覆盖几乎所有中文字符）的有效性，这一"词汇覆盖优先"的构造策略对资源受限场景有参考价值。

## 关键术语表
- **TIPA（Token Internal Position Awareness）**：一种通过反向字符预测任务训练 LLM 学习 token 内部字符组成与位置信息的方法。
- **MTIPA（Multi-Token Internal Position Awareness）**：TIPA 的扩展版本，将反向字符预测任务从单个 token 推广至完整句子级多 token 序列。
- **BPE（Byte-Pair Encoding）**：一种将文本切分为子词单元的常见分词方法，提升计算效率但会掩盖 token 内部的字符结构。
- **PPA（Position Prediction Accuracy）**：衡量模型预测的错误字符位置准确率的指标，仅评估位置感知能力不考虑实际纠错。
- **SA（Sentence-Level Accuracy）**：句子级准确率，要求模型在所有预测的位置和修正字符上完全正确。
- **SAIP（Sentence-Level Accuracy Ignoring Position）**：忽略位置错误的句子级准确率，仅评估修正字符是否正确。
- **CF1（Character-Level F1 Score）**：字符级别的精确率与召回率的调和平均，衡量错误字符检测与纠正的综合性能。
- **CSC（Chinese Spelling Correction）**：中文拼写纠错任务，识别并纠正中文文本中的错别字。

## 可复现要素
- **数据集**：CSCD-NS（公开）、Wang271K（公开）、LEMON（公开）；TIPA 数据集由 tokenizer 词表构造，未单独开源声明
- **代码/权重**：论文未明确声明代码仓库，提及相关工具为 Hugging Face Transformers 和 LLaMA-Factory（Zheng et al., 2024）；模型基于 Qwen2.5-7B 和 Llama-3.1-8B 微调
- **关键超参**：LoRA rank=16，alpha=16，dropout=0.05，学习率 $1\times10^{-4}$，batch size=16，epoch=10（Exp1）/ epoch=3（Exp2最优），MTIPA 采样率 r=10%；全参数 SFT 使用学习率 $5\times10^{-6}$，batch size=32（含梯度累积），epoch=2，4×NVIDIA H100 80GB
- **硬件**：LoRA 实验使用单卡 NVIDIA A800 80GB；全参数 SFT 使用 4×NVIDIA H100 80GB
