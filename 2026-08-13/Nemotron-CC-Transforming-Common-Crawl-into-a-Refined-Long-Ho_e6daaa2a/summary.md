---
title: "Nemotron-CC-Transforming-Common-Crawl-into-a-Refined-Long-Ho"
source: https://aclanthology.org/2025.acl-long.123.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:12:52"
---

# 论文速读：Nemotron-CC: Transforming Common Crawl into a Refined Long-Horizon Pretraining Dataset

## 一句话总结
本文提出了一套面向长窗口预训练（long-horizon）的英文 Common Crawl 数据处理流程，通过多分类器集成打分、差异化合成数据生成（改写低质文本/蒸馏高质文本）以及弱化启发式过滤规则，构建了 6.3T token 的 Nemotron-CC 数据集；该数据集在唯一真实 token 数量上是 DCLM 的 4 倍，并支撑 8B 模型在 15T token 规模下全面超越 Llama 3.1 8B。

## 研究问题与动机
1. **现有主流数据集在长窗口训练中数据多样性严重不足**：FineWeb-Edu 与 DCLM 通过激进过滤丢弃约 90% 的原始数据，且剩余数据中近 80% 为模糊重复，导致 DCLM 唯一真实 token 仅约 1T，难以支撑 10T 以上的大规模训练。
2. **重复样本导致长跨度训练的边际收益递减**：Muennighoff et al. (2024) 指出训练超过 4 个 epoch 后收益骤降，若直接用低唯一性数据集进行 15T token 训练，模型将陷入反复学习相同内容的低效循环。
3. **传统启发式过滤会误伤高质数据**：基于规则（如长度、perplexity、正则等）的后处理管道虽能剔除明显噪声，但会额外移除约 18.1% 已被模型确认为高质量的关键 token。
4. **单一质量分类器视角有限**：FineWeb-Edu（侧重教育价值）与 DCLM（侧重信息密度）的分类标准差异显著，单独使用均只能召回约 10% 的高质文档，难以最大化长周期训练所需的数据产量。

## 核心贡献（创新点）
1. **提出“分类器集成+合成改写+弱化启发式过滤”的长周期数据集构建新范式**。与 FineWeb-Edu/DCLM 依赖单一强过滤管道不同，本文以学习型分类器为核心，通过多视角投票扩大高质召回，并用合成数据弥补去重后的唯一 token 缺口。
2. **设计基于下游任务校准的 5 级质量分桶体系**。现有工作多直接使用分类器原始分数切分数据，本文通过 50B token 小模型继续预训练（annealing）验证各分数桶的真实下游增益，将 20 个细粒度桶映射为 5 个与模型实际表现对齐的质量等级。
3. **差异化合成数据生成策略（低质重写降噪 + 高质多模态变体提取）**。与 TINYStories 或 T5-Mix 依赖强模型从零生成虚构内容不同，本文仅将中等规模模型（12B）作为文本风格转换器，用 Wikipedia 风格重写低质文本来压制噪声，并用 QA/Distill/知识列表等 prompt 从高质量文本中提取新鲜唯一 token。
4. **实证证明启发式过滤在长窗口场景下的负面作用并给出替代方案**。本文首次系统量化了传统规则过滤对高质 token 的损失（-18.1%），提出“仅对低质分桶应用过滤、对高质分桶直接放行”的策略，在 MMLU 不降反升（+2.0）的同时大幅提升了有效数据产量。

## 方法详解
1. **HTML 提取与语言/去重管线**：对比 Trafilatura 与 Justext，选择 Justext（在 FineWeb-Edu 标准下高质 token 产量高 28.6%）。使用 pycld
