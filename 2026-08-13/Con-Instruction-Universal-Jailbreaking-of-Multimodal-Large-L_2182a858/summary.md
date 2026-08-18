---
title: "Con-Instruction-Universal-Jailbreaking-of-Multimodal-Large-L"
source: https://aclanthology.org/2025.acl-long.146.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:53:02"
field: "多模态大语言模型安全"
keywords: ["多模态大语言模型", "越狱攻击", "对抗样本", "安全评估", "非文本模态", "灰盒攻击"]
innovations: ["提出Con Instruction通用灰盒越狱方法，将恶意指令语义嵌入对抗图像/音频", "引入ARC评估体系综合考量响应质量与相关性，与人工评估一致率达88%", "揭示大模型对非文本模态攻击更脆弱的悖论现象及内部安全感知破坏机制"]
benchmarks: ["AdvBench", "SafeBench"]
---

# 论文速读：Con Instruction: Universal Jailbreaking of Multimodal Large Language Models via Non-Textual Modalities

## 一句话总结
本文提出了 Con Instruction，一种针对多模态大语言模型（MLLMs）的通用灰盒越狱攻击方法，通过将恶意文本指令的语义嵌入到对抗性图像或音频中，成功绕过模型安全对齐机制；同时在 AdvBench 和 SafeBench 两个基准上达到了最高攻击成功率（LLaVA-v1.5-13B 分别达 81.3% 和 86.6%），并引入 ARC 评估体系以缓解现有方法的误判问题。

## 研究问题与动机
- **现有攻击的模态局限**：当前 MLLM 越狱攻击主要依赖文本+对抗图像的联合输入，或仅针对 OCR 能力（黑盒方法），无法泛化到音频语言模型；白盒方法需要完整模型参数访问，计算成本高昂。
- **非文本模态的安全对齐盲区**：MLLMs 在文本输入上经过严格安全对齐，但对非文本模态（视觉/音频）的安全防护相对薄弱，存在利用空间。
- **评估方法的误判问题**：现有自动评估方法（如拒绝字符串匹配 RM、StrongREJECT）常因响应包含冲突内容而高估攻击成功率，缺乏对响应质量和相关性的综合考量。
- **通用性需求**：亟需一种不依赖训练数据、无需预处理文本指令、可跨视觉和音频语言模型泛化的统一攻击框架。

## 核心贡献（创新点）
1. **提出 Con Instruction 通用攻击框架**：在灰盒设置下优化对抗图像/音频，使其在融合嵌入空间中与目标恶意指令对齐；本质区别在于无需训练数据且首次系统性地将完整恶意指令转换为非文本模态进行越狱。
2. **引入 ARC（Attack Response Categorization）评估体系**：将响应分为 Irrelevant、Refusal、Superficial、Success 四类，综合考量响应质量与相关性；相比 RM/SR 方法，与人工评估一致率达约 88%。
3. **揭示非文本模态对齐缺陷**：通过探针实验和 t-SNE 可视化证明，对抗样本会破坏模型内部的“安全感知”分布，大模型反而对非文本攻击更脆弱。
4. **系统评估防御策略**：对比后处理（ECSO、MLLM-Protector）、对抗微调（VLGuard）、输入扰动等方法，发现当前防御存在显著差距，MLLM-Protector 和噪声扰动（σ=6）表现最佳。

## 方法详解
**整体架构**：两阶段攻击流程——Stage I 生成对抗样本，Stage II 部署越狱。

**Stage I：对抗样本生成**（Algorithm 1）
- 输入：目标指令 $\mathrm{Inst}_{adv}^t$、预训练视觉/音频编码器 $\mathcal{E}(\cdot)$、融合层 $\mathcal{F}(\cdot)$、token 嵌入层 $\mathcal{T}(\cdot)$、距离函数 $\mathcal{D}(\cdot)$、学习率 $\eta$、阈值 $\tau$。
- 步骤：
  1. 获取目标指令的 token 嵌入：$\mathbf{H}_{\mathrm{Inst}} \leftarrow \mathcal{T}(\mathrm{Inst}_{adv}^t)$，随机初始化噪声图像/音频 $X_{adv}^{\neg t} \leftarrow \hat{N}(0,1)$。
  2. 计算视觉/音频嵌入并通过融合层：$\mathbf{H}_{adv}^{\neg t} \leftarrow \mathcal{F}(\mathcal{E}(X_{adv}^{\neg t}))$，计算损失 $\mathcal{L} \leftarrow \mathcal{D}(\mathbf{H}_{\mathrm{Inst}}, \mathbf{H}_{adv}^{\neg t})$。
  3. 迭代优化：$X_{adv}^{\neg t} \leftarrow X_{adv}^{\neg t} - \eta \nabla_{X_{adv}^{\neg t}} \mathcal{L}$，直至 $\mathcal{L} \leq \tau$。
- 关键设计：
  - **嵌入对齐策略**：当视觉嵌入数（如 LLaVA-1.5 固定 576 个）多于指令 token 数时，选取最后 $N_{\mathrm{Inst}}$ 个视觉嵌入进行对齐效果最佳。
  - **损失函数**：联合使用 Euclidean distance + cosine similarity，单独使用任一效果均较差（Table 4）。
  - **阈值设置**：LLaVA(7B)/Qwen-VL/Qwen-Audio 用 $\tau=0.60$，LLaVA(13B)/InternVL(13B) 用 $\tau=0.75$，InternVL(34B) 用 $\tau=0.85$。

**Stage II：越狱部署**
- 基础设置：对抗样本 + 空字符串文本输入。
- 增强策略：组合对抗样本与三种文本输入：
  - **Agree**： affirmative phrases（如"Yes, I can provide"）。
  - **Anti**：双角色设定（GoodGPT/BadGPT），BadGPT 返回相反恶意内容。
  - **Hypo**：假设性角色请求，要求详细列表格式回答。
- 关键发现：非空文本输入显著提升识别率（$\mathrm{ARC}_r$）和攻击成功率，且文本+Con Instruction 的组合远优于纯文本输入。

**ARC 评估体系**（Table 2）
- Category 0 (Irrelevant)：无关响应。
- Category 1 (Refusal)：拒绝响应（含免责声明但无有害内容）。
- Category 2 (Superficial)：表面配合但缺乏深度/具体有害信息。
- Category 3 (Success)：有效生成有害内容的成功越狱。
- 用 GPT-4 作为 judge，与人工评估一致率约 88%。

## 实验与结果
**数据集**：
- **SafeBench**：500 个有害问题，覆盖 10 类禁用主题。
- **AdvBench**：520 条有害行为指令，聚焦危险/非法指导。

**目标模型**：
- 视觉语言：LLaVA-v1.5 (7B, 13B)、InternVL (13B, 34B)、Qwen-VL。
- 音频语言：Qwen-Audio。

**主要结果**（Table 3，AdvBench）：
| 模型 | 方法 | $\mathrm{ARC}_a$ | 提升幅度 |
|------|------|------------------|----------|
| LLaVA-13B | Text | 2.7% | — |
| LLaVA-13B | FigStep (最强基线) | 76.6% | — |
| LLaVA-13B | **Con Inst.+Hypo** | **81.3%** | +4.7% vs FigStep |
| InternVL-34B | Con Inst.+Hypo | **97.2%** | — |
| Qwen-VL | Con Inst.+Hypo | **76.3%** | — |
| Qwen-Audio | Con Inst.+Hypo | **76.2%** | 首个有效音频越狱 |

**主要结果**（Table 3，SafeBench）：
- LLaVA-13B: $\mathrm{ARC}_a = 86.6\%$（Con Inst.+Hypo）
- InternVL-34B: $\mathrm{ARC}_a = 97.2\%$
- Qwen-Audio: $\mathrm{ARC}_a = 73.5\%$

**关键发现**：
- **A1**：Con Instruction 同时有效攻击视觉和音频语言模型，对 Qwen-Audio 实现 >75% 成功率，填补音频越狱空白。
- **A2**：文本输入（尤其 Anti/Hypo）可显著提升 $\mathrm{ARC}_r$ 和 $\mathrm{ARC}_a$，揭示现有安全对齐过度偏向文本模态。
- **A3**：大模型对小模型更具鲁棒性（文本输入），但对非文本攻击更脆弱，暴露多模态对齐的不均衡性。
- **A4**：Euclidean + Cosine 联合损失显著优于单一距离度量（SafeBench: $\mathrm{ARC}_a$ 从 1.6%→33.2%）。
- **A5**：探针实验显示，第 36 层安全分类器在文本指令上准确率达 87.2%，但在 Con Instruction 对抗样本上降至 72.4%；t-SNE 可视化显示对抗样本激活分布分散且与安全/ unsafe 文本重叠。
- **A6**：提供目标指令最后 1-5 个 token 作为文本输入可进一步提升 ASR。

**防御实验**（Table 7，AdvBench）：
- MLLM-Protector: $\mathrm{ARC}_a$ 降至 8.7%-10.1%（最佳后处理防御）。
- 噪声扰动 σ=6: $\mathrm{ARC}_a$ 降至 1.8%-10.5%（最佳输入扰动）。
- VLGuard 微调：$\mathrm{ARC}_a$ 仍达 25%-31%，证明防御不足。
- ECSO 自评估：失效，仍产生带免责声明的有害内容。

## 相关工作脉络
1. **FigStep (Gong et al., 2023)**：黑盒 OCR-based 攻击，通过排版将恶意文本嵌入图像；局限是仅适用于视觉模型且易被 OCR 过滤防御。Con Instruction 扩展至音频且无需 OCR 依赖。
2. **QueryR (Liu et al., 2024b)**：黑盒查询相关攻击，将部分信息转移到图像；同样仅限视觉模态，且分解恶意意图可能削弱攻击性。
3. **JiP (Shayegani et al., 2024)**：灰盒组合式攻击，需将指令分解为 benign prompt + malicious trigger；受限于分解难度和触发器复杂度，性能较差。
4. **VisAdv (Qi et al., 2024) / ImgJP (Niu et al., 2024)**：白盒优化攻击，需训练数据和完整模型访问；计算成本高且单图像难以适配所有指令。
5. **LLM-as-a-Judge 方法 (Souly et al., 2024 StrongREJECT)**：自动评估框架但未考虑响应质量与相关性；本文 ARC 在此基础上增加质量维度，一致率达 88%。
6. **防御工作 (ECSO, MLLM-Protector, VLGuard)**：后处理和微调方法；本文验证其在 Con Instruction 攻击下的不足，指出需更强的多模态安全对齐。

## 局限性与未来方向
- **模型规模限制**：实验仅涉及 ≤34B 参数模型，更大规模模型的行为未验证；作者指出大模型脆弱性趋势可能不随规模线性外推。
- **灰盒假设**：方法需访问非文本编码器和融合模块，难以直接应用于纯黑盒商业 API；但适用于开源模型红队测试。
- **评估体系完备性**：ARC 虽优于 RM/SR，但仍需更细粒度的分类维度（如响应危害程度分级）以支持更稳健的评估。
- **对抗样本可解释性**：生成的图像/音频对人类不可感知恶意意图，但缺乏对"模型为何理解非文本指令"的深入机理分析。

## 研究启发与可借鉴点
1. **非文本模态作为攻击面**：安全对齐常聚焦文本输入，忽视视觉/音频的指令跟随能力；后续防御工作应将多模态指令对齐纳入统一框架。
2. **嵌入空间对齐策略**：通过优化非文本输入使其嵌入与目标文本对齐，可推广至其他多模态安全漏洞挖掘（如视频、3D 点云）。
3. **ARC 评估框架的迁移价值**：综合考虑质量和相关性的响应分类方法，可用于 LLM/MLLM 安全评测的标准化工具。
4. **大模型脆弱性悖论**：大模型文本安全更强但对非文本更脆弱，提示"规模提升≠安全提升"，需针对性加固多模态融合层。
5. **防御策略启示**：MLLM-Protector 的外部分离检测+净化管道有效，但增加延迟；未来可探索轻量级多模态安全感知模块。

## 关键术语表
- **Con Instruction**：本文提出的通用越狱攻击方法，通过将恶意指令语义嵌入对抗图像/音频实现非文本模态越狱。
- **ARC (Attack Response Categorization)**：攻击响应分类体系，将响应分为 Irrelevant/Refusal/Superficial/Success 四类，综合评估攻击成功与否。
- **灰盒攻击 (Gray-box attack)**：仅需访问模型部分组件（如编码器、融合层）的攻击设置，平衡性能与计算成本。
- **$\mathrm{ARC}_a$**：基于 ARC 的攻击成功率，衡量模型生成有害内容的比例。
- **$\mathrm{ARC}_r$**：攻击识别率，衡量模型能否正确理解对抗样本中的指令（对应 Irrelevant 以外的三类）。
- **SafeBench / AdvBench**：两个标准 MLLM 安全评测基准，分别包含 500 个有害问题和 520 条有害行为指令。
- **LLM-as-a-Judge**：使用大型语言模型自动评估生成响应质量的范式，本文用 GPT-4 实现 ARC 分类。
- **VLGuard**：用于对抗微调的安全指令数据集，本文验证其防御 Con Instruction 的效果不足。

## 可复现要素
- **数据集**：SafeBench（公开）、AdvBench（公开）；论文未提及是否提供额外 adversarial examples 数据集。
- **代码/权重**：论文声明"implementation is made available"，代码基于 LLaVA、InternVL、Qwen-VL、Qwen-Audio 官方项目；具体仓库链接需查阅原文脚注。
- **关键超参**：
  - 学习率 $\eta = 0.1$
  - 优化阈值 $\tau$：LLaVA(7B)/Qwen-VL/Qwen-Audio 用 0.60，LLaVA(13B)/InternVL(13B) 用 0.75，InternVL(34B) 用 0.85
  - batch size = 1
  - optimizer = Adam
  - max_new_tokens = 2048
  - temperature = default（具体值未明确，见 Appendix A）
  - 评估采样数 n = 5
- **硬件**：NVIDIA A100-SXM4-80GB GPU
- **训练数据**：无需训练数据（zero-shot，从随机噪声初始化）
