---
title: "LANGSAMP-Language-Script-Aware-Multilingual-Pretraining"
source: https://aclanthology.org/2025.acl-long.88.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 17:02:48"
field: "多语言表示学习"
keywords: ["多语言预训练", "语言嵌入", "跨语言迁移", "语言中立性", "零样本迁移", "MLM"]
innovations: ["将语言+文字嵌入注入Transformer输出端以解耦语言特异性信息，提升表征中立性", "证明预训练辅助嵌入可用作跨语言源语言选择的副产品", "在500+语言规模上验证后嵌入式设计对零样本跨语言迁移的稳定提升"]
benchmarks: ["Bible Sentence Retrieval", "Tatoeba Sentence Retrieval", "Taxi1500", "SIB-200", "WikiANN NER", "Universal Dependencies POS"]
---

# 论文速读：LANGSAMP-Language-Script-Aware-Multilingual-Pretraining

## 一句话总结
本文提出 LANGSAMP，通过在 Transformer 输出后、语言建模头前注入可学习的语言嵌入和文字嵌入，将语言/文字特异性信息从上下文 token 表示中解耦，从而提升表征的语言中立性，在覆盖 500+ 语言的零样本跨语言迁移任务上稳定超越基线。

## 研究问题与动机
- 近期 mPLMs（如 XLM-R、mBERT）已摒弃早期 XLM 中的语言嵌入，导致 token 表示被迫编码大量语言/文字特异性信息，损害跨语言表征的中立性（language neutrality）。
- 现有多数改进工作聚焦于预训练后的对齐（post-alignment），从架构角度在预训练阶段增强语言中立性的探索极少。
- 语言嵌入的移除虽使模型成为无需语言 ID 输入的通用编码器，但也削弱了低资源语言（tail languages）和非拉丁文脚本语言在跨语言迁移中的表现。
- 需要一种既保持通用编码器兼容性，又能从表征中剥离语言/文字特异性负担的预训练机制。

## 核心贡献（创新点）
1. **将语言+文字嵌入加到 Transformer 输出之后、MLM 头之前**：与早期 XLM 将嵌入加到输入端不同，本文方案使预训练后的骨干网络（token embeddings + Transformer）无需任何语言/文字 ID 即可作为通用编码器使用，与主流 mPLM 完全兼容。
2. **从架构层面提升零样本跨语言迁移性能**：持续预训练 XLM-R 于 Glot500-c（500+ 语言），在句检索、文本分类、序列标注等 6 类下游任务上 consistently 优于基线，尤其对低资源和非拉丁文语言提升显著。
3. **揭示语言/文字嵌入的可迁移副产品价值**：嵌入向量捕获了语言类型学与文字系统结构特征，其余弦相似度可用于选择更优的跨语言迁移源语言（donor language），减少英语中心偏差。

## 方法详解
- **嵌入维度**：语言嵌入 $E^{Lang} \in \mathbb{R}^{L \times D}$（$L=610$，$D=768$），文字嵌入 $E^{Script} \in \mathbb{R}^{S \times D}$（$S=30$），随机初始化。
- **注入位置**：对输入句子 $X=[x_1,\cdots,x_n]$，经 Transformer 得到最后一层上下文表示 $H=[h_1,\cdots,h_n]$，然后在 MLM 预测前叠加：$o_i = h_i + E_l^{Lang} + E_s^{Script}$，其中 $l$ 为句子语言 ID，$s$ 为文字 ID。
- **损失函数**：标准 MLM 损失 $\mathcal{L}_{MLM} = -\sum_{i \in \mathcal{M}} \log P_{MLM}(x_i | o_i)$，仅在使用语言/文字嵌入的预训练阶段计算。
- **微调阶段**：骨干网络（$E^{token}$ + Transformer blocks）完全移除语言/文字 ID，以标准方式接任务分类头微调，参数无增量。
- **数据采样**：按多项分布采样（温度 0.3）提升低/中资源语言训练实例数。

## 实验与结果
- **数据集**：Glot500-c（500+ 语言，30 种文字），6 个下游任务：SR-B（Bible 句检索）、SR-T（Tatoeba）、Taxi1500、SIB200（文本分类）、NER（WikiANN）、POS（Universal Dependencies）。
- **评估协议**：英语中心零样本跨语言迁移——在英文训练集上微调，评估所有目标语言。
- **主要结果**：LANGSAMP 在所有分组（head/tail 语言、Latn/non-Latn 文字）上均优于或持平基线；SR-B 低资源语言提升最大（tail 语言 +7%，head 语言 +1%）；非拉丁文语言整体有稳定增益。
- **消融实验**（5% 数据）：仅有 $E^{Lang}$ 或仅有 $E^{Script}$ 均有效，两者结合效果最佳；序列级任务（检索、分类）提升最大，token 级任务（NER、POS）提升较小。
- **语言中立性验证**：10 种类型学差异大的语言两两余弦相似度全部提升；最显著的为 arb_Arab（阿拉伯语）与其他语言的相似度提升（eng+4.7%，rus+4.1%）。
- **源语言选择案例研究**：使用语言嵌入相似度选择最优 donor 语言替代英语，在分类任务上（尤其非拉丁文语言）取得明显提升，如 yue 用 zho 作源语言在 NER 上从 25.7 提升至 73.5。

## 相关工作脉络
- **XLM（Conneau & Lample, 2019）**：最早引入语言嵌入到 token 输入端的 mPLM，但需语言 ID 参与下游，本文方案在结构上与之形成对比（嵌入位置不同，下游无需 ID）。
- **XLM-R / mBERT**：放弃语言嵌入以提升通用编码器能力，本文认为这是以牺牲语言中立性为代价。
- **后对齐方法（Cao et al., 2020; Pan et al., 2021; Xhelili et al., 2024）**：通过对比学习或 transliteration 对齐多语言表征，属于 post-hoc 策略；本文从预训练架构层面原生增强中立性。
- **语言相似度与源语言选择（Lin et al., 2019; Lauscher et al., 2020）**：已有工作利用语言相似度选 donor，本文进一步证明预训练过程中的语言嵌入本身即能学习到有意义的类型学表征。
- **Glot500-m（ImaniGooghari et al., 2023）**：基线模型复现，本文在其上叠加 LANGSAMP 模块验证增益。

## 局限性与未来方向
- 消融实验受算力限制仅用 5% Glot500-c 数据，未能在全量数据上逐项验证语言/文字嵌入的独立贡献。
- 未探索将语言/文字嵌入添加到 Transformer 之前的架构可能性（虽与"通用编码器"目标不符，但值得研究）。
- 模型仅覆盖 610 种语言，对 Glot500-c 之外的低资源语言无法直接利用语言嵌入选择 donor。
- 语言中立性改善与下游性能提升之间的因果关系尚未严格证明（现有文献也表明对齐改善不一定转化为更好迁移效果）。
- 未来方向：在中间层引入辅助嵌入与对比学习目标、扩展至 decoder-only/enc-decoder 模型的受控多语言生成。

## 研究启发与可借鉴点
- **"后置嵌入"设计思想**：将辅助嵌入注入 Transformer 输出端而非输入端，既可在预训练阶段卸载语言特异性负担，又保证下游无需额外输入——这一权衡思路可迁移到其他需要"增强中立性但不破坏通用性"的场景。
- **消融实验的小数据策略**：受限于算力时用 5% 数据做控制变量消融是务实做法，但需在论文中明确说明局限性；可借鉴其在严格公平比较下的结论有效性论证方式。
- **语言嵌入的副产品价值挖掘**：将预训练辅助参数（语言/文字嵌入）用于下游任务（源语言选择）的思路，对"高效利用预训练中间产物"有启发；可考虑将此类嵌入用于语言聚类、代码混合检测等任务。
- **分层中性化探索**：本文聚焦最后一层输出，而 prior work 表明中间层通常更具语言中立性；可将 LANGSAMP 思想扩展至中间层的显式中性化引导。

## 关键术语表
**Language Neutrality（语言中立性）**：不同语言的语义等价句子的表征共享统一子空间，是跨语言迁移有效的核心前提。
**Zero-shot Crosslingual Transfer（零样本跨语言迁移）**：在单一源语言（通常为英语）上微调模型后，直接应用于其他目标语言测试，无需目标语言训练数据。
**Head vs. Tail Languages（头部/尾部语言）**：Glot500-c 中被 XLM-R 覆盖的语言为 head languages（高资源），其余为 tail languages（低资源）。
**MLM（Masked Language Modeling）**：掩码语言建模，随机掩码输入 token 并要求模型预测原始 token 的自监督预训练目标。
**Donor Language（源语言/捐献语言）**：在零样本跨语言迁移中，作为微调数据源的语言；本文探索用语言嵌入相似度自动选择最优 donor 替代英语。
**Sentence Retrieval（句检索）**：给定查询句，从候选集中检索其语义等价翻译句，使用余弦相似度排序，无需微调。

## 可复现要素
- **数据集**：Glot500-c（公开可用，来自 Glot500 项目）；下游评测数据集（SR-B/Bible、SR-T/Tatoeba、Taxi1500、SIB200、WikiANN/NER、UD/POS）均为公开数据集。
- **代码/模型**：代码和模型已开源至 https://github.com/cisnlp/LangSAMP。
- **关键超参**：语言嵌入 $\mathbb{R}^{610 \times 768}$，文字嵌入 $\mathbb{R}^{30 \times 768}$；AdamW（$\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-6}$），初始学习率 5e-5，有效 batch size=1024（8 卡梯度累积 × per-GPU batch 32），序列长度 512，温度 0.3 采样，最多 150K 步，每 5K 步保存 checkpoint。
