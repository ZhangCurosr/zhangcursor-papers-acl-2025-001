---
title: "Con-Instruction-Universal-Jailbreaking-of-Multimodal-Large-L"
source: https://aclanthology.org/2025.acl-long.146.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:41:56"
field: "多模态大模型安全与越狱攻击"
keywords: ["multimodal large language models", "jailbreaking attack", "adversarial example", "gray-box attack", "non-textual modality", "Attack Response Categorization"]
innovations: ["提出 Con Instruction，通过非文本模态嵌入对齐实现通用越狱攻击，无需训练数据且支持音频模态", "引入 ARC 评估框架与 LLM 裁判，综合考量响应相关性与质量，提升自动评估可靠性", "揭示多模态安全对齐的模态偏差，证明较大模型对非文本指令注入更脆弱"]
benchmarks: ["AdvBench", "SafeBench"]
---

# 论文速读：Con Instruction: Universal Jailbreaking of Multimodal Large Language Models via Non-Textual Modalities

## 一句话总结
本文提出 Con Instruction，一种通用的灰盒越狱攻击方法，通过优化对抗图像/音频使其在融合嵌入空间中与恶意文本指令对齐，从而绕过 MLLM 安全防护；同时引入考虑响应质量与相关性的 ARC 评估框架，并在 AdvBench 和 SafeBench 上对多种视觉/听觉语言模型验证了其有效性。

## 研究问题与动机
- **现有攻击的模态局限**：多数 MLLM 越狱攻击依赖“文本+对抗图像”组合，无法直接泛化到音频语言模型；纯黑盒方法多利用 OCR 能力，难以适配非文本模态。
- **灰盒设置的效率与泛化需求**：白盒方法需要全模型参数梯度且依赖训练数据，计算成本高；现有灰盒方法要么针对特定视觉-语言任务、要么需人工构造触发器，通用性不足。
- **非文本模态的隐蔽性潜力**：MLLM 具备理解非文本指令的能力，但安全防护通常偏向文本输入，对图像/音频模态的指令注入风险识别不足。
- **评估方法的失真问题**：现有自动评估（拒绝词匹配、StrongREJECT）常高估攻击成功率，未充分考虑响应是否与恶意指令相关及内容质量。

## 核心贡献（创新点）
- **提出 Con Instruction 通用攻击框架**：通过灰盒访问非文本编码器与融合模块，将恶意文本语义嵌入对抗图像/音频中；区别于白盒方法无需训练数据，区别于黑盒方法不依赖 OCR 且支持音频模态。
- **设计基于嵌入对齐的两阶段优化过程**：在融合空间中迭代更新对抗样本的嵌入，使其与目标指令的 token 嵌入尽可能接近；以往工作多仅使用余弦相似度，本文实证表明欧氏距离与余弦相似度组合更有效。
- **引入 Attack Response Categorization (ARC) 评估体系**：从响应质量与相关性出发定义 Irrelevant/Refusal/Superficial/Success 四类标签，并开发 LLM 裁判（与人工标注一致率约 88%），弥补了传统拒绝词匹配的误判缺陷。
- **系统验证跨模型与跨模态的通用性**：在 LLaVA-v1.5、InternVL、Qwen-VL（视觉）和 Qwen-Audio（音频）上均取得高攻击成功率，并揭示结合简单文本提示（如 Hypo）可显著提升攻击效果。
- **全面评测多种防御策略的有效性差距**：对比后验检测（ECSO、MLLM-Protector）、对抗微调（VLGuard）与输入扰动，发现现有防御存在显著不足，为后续安全研究提供基准。

## 方法详解
- **灰盒设置假设**：攻击者可获得 MLLM 的非文本编码器 $\mathcal{E}(\cdot)$（视觉或音频）、token 嵌入层 $\mathcal{T}(\cdot)$ 及多模态融合模块 $\mathcal{F}(\cdot)$，无需访问生成层全参数，降低计算开销。
- **第一阶段：对抗样本生成**
  1. 获取目标恶意指令的 token 嵌入 $\mathbf{H}_{\mathrm{Inst}} = \mathcal{T}(\mathrm{Inst}_{adv}^t)$。
  2. 从随机噪声初始化对抗图像/音频 $X_{adv}^{\lnot t}$，经编码器与融合层得到视觉嵌入 $\mathbf{H}_{adv}^{\lnot t} = \mathcal{F}(\mathcal{E}(X_{adv}^{\lnot t}))$。
  3. 迭代优化：以距离函数 $\mathcal{D}$（欧氏距离 + 余弦相似度）最小化为目标，通过梯度下降更新 $X_{adv}^{\lnot t}$，直至损失低于阈值 $\tau$。
  4. **关键设计**：当视觉 token 数多于文本 token 数时，选取融合后视觉嵌入的**最后 $N_{\mathrm{Inst}}$ 个**进行对齐，经验证效果最佳。
- **第二阶段：越狱部署**
  1. 将生成的对抗样本与不同文本输入组合（空串、Agree、Anti、Hypo 等），形成完整输入送入 MLLM。
  2. 文本输入不提供显式恶意意图，但通过引导句式（如肯定提示、双角色设定、假设性请求）增强模型对对抗样本的响应倾向。
- **ARC 评估流程**：对每个攻击响应，由 LLM 裁判按 0–3 级分类（Irrelevant/Refusal/Superficial/Success），并结合人工标注校准；攻击成功率 $\mathrm{ASR}$ 定义为多条采样响应中至少一条达到 Success 的比例。

## 实验与结果
- **数据集**：AdvBench（520 条有害行为指令）、SafeBench（500 条涵盖 10 类禁忌话题的有害问题）。
- **受害模型**：视觉语言模型 LLaVA-v1.5 (7B/13B)、InternVL-Chat (13B/34B)、Qwen-VL-Chat；音频语言模型 Qwen-Audio-Chat。
- **评估基线**：Text（纯文本指令）、VisAdv、ImgJP、JiP+Img、JiP+Text、FigStep、QueryR；评估指标包括 RM、SR 及本文提出的 $\mathrm{ARC_a}$（攻击成功）与 $\mathrm{ARC_r}$（指令被识别）。
- **主要结果**：
  - 在 LLaVA-v1.5 (13B) 上，Con Instruction + Hypo 取得 **AdvBench $\mathrm{ARC_a}$ 81.3%**、**SafeBench $\mathrm{ARC_a}$ 86.6%**，显著高于既有方法（如 FigStep 在 AdvBench 上最高约 76.6%）。
  - 在 Qwen-Audio 上，Con Instruction + Hypo 达到 **AdvBench $\mathrm{ARC_a}$ 76.2%**、**SafeBench $\mathrm{ARC_a}$ 73.5%**，为当时首个有效攻击音频语言模型的方法之一。
  - 结合文本提示（Agree/Anti/Hypo）可普遍提升 $\mathrm{ARC_r}$ 与 $\mathrm{ARC_a}$，其中 Hypo 效果最佳。
  - 损失函数消融显示：联合使用欧氏距离与余弦相似度使 SafeBench $\mathrm{ARC_a}$ 从 1.6%–2.6% 跃升至 **33.2%**（LLaVA 7B）。
- **防御实验**：MLLM-Protector 与高斯噪声扰动（$\sigma=6$）表现最佳，但 Con Instruction + Hypo 仍能在多数防御下保持超过 25% 的成功率；VLGuard 微调防御仅将成功率降至约 25%，未完全消除威胁。

## 相关工作脉络
- **FigStep (Gong et al., 2023)**：黑盒方法，利用排版视觉提示绕过 OCR；本文方法无需文本重排且支持音频，且不被 OCR 过滤类防御轻易检测。
- **VisAdv / ImgJP (Qi et al., 2024; Niu et al., 2024)**：白盒优化方法，需少量有害语料且试图用单张图像适配所有指令，通用性弱；本文无需训练数据且针对每条指令独立生成对抗样本。
- **JiP (Shayegani et al., 2024)**：灰盒分解攻击，将恶意指令拆分为良性文本与恶意触发图像；本文直接在全嵌入空间对齐语义，避免分解导致的意图失真。
- **QueryR (Liu et al., 2024b)**：黑盒查询相关方法，依赖文本‑图像信息转移，无法推广至音频；本文灰盒设置直接优化非文本模态嵌入，跨模态通用。
- **VLGuard (Zong et al., 2024)**：安全微调防御数据；本文实验表明其对 Con Instruction 类非文本指令注入防御不足，提示现有安全微调可能未充分覆盖多模态融合漏洞。

## 局限性与未来方向
- **模型规模局限**：实验最大仅覆盖 34B 模型，更大规模或新架构模型的脆弱性未验证；作者指出“更大模型更脆弱”的观察未必随规模线性成立。
- **灰盒假设的限制**：商业部署的 MLLM 通常仅提供黑盒接口，直接应用受限；但开源模型生态广泛，仍适用于红队测试与安全评分。
- **评估分类的粒度**：ARC 虽改进现有评估，但仍属粗粒度四维分类，未来可探索多维度细粒度 taxonomy（如意图强度、危害等级、响应完整性等）。
- **防御手段的可持续性**：当前最优防御（MLLM-Protector、强噪声扰动）均以增加延迟或损害正常生成为代价，亟需轻量、无损的安全对齐机制。

## 研究启发与可借鉴点
- **非文本模态的语义嵌入对齐可作为通用攻击范式**：将文本指令映射至视觉/音频嵌入空间的思想可迁移至其他多模态大模型（如视频语言模型、触觉‑语言交互系统）的安全评估。
- **联合距离函数（欧氏 + 余弦）提升嵌入优化稳定性**：该策略不仅适用于图像/音频对抗生成，也可推广至任何基于对比学习的多模态表示攻击或鲁棒性测试。
- **ARC 评估框架可直接复用或扩展**：其“相关性‑质量”二维分类逻辑与 LLM 裁判实现，可替代或补充现有 RM/SR 指标，用于任何大模型安全评测流水线。
- **文本提示增强攻击的协同效应值得深挖**：简单中性文本（如 Hypo）与非文本对抗样本的组合可大幅突破安全对齐边界，提示未来防御需联合考量多模态上下文与引导句式。
- **内部激活分布分析揭示安全机制失效机理**：通过探针 classifier 与 t‑SNE 可视化发现，对抗样本使安全/不安全激活混叠，为可解释安全审计提供新思路。

## 关键术语表
- **Con Instruction**：本文提出的通用越狱攻击方法，通过非文本模态（图像/音频）传递恶意指令语义。
- **灰盒攻击**：攻击者仅访问模型部分组件（如编码器、融合层），无需全参数梯度，计算成本低于白盒攻击。
- **Attack Response Categorization (ARC)**：四类响应分类体系（Irrelevant/Refusal/Superficial/Success），综合考量相关性与内容质量。
- **$\mathrm{ARC_a}$**：攻击成功率，指响应落入 Success 类别的比例。
- **$\mathrm{ARC_r}$**：指令识别率，指模型能感知对抗样本中包含指令（即响应属于后三类）的比例。
- **StrongREJECT (SR)**：基于 LLM 的自动评估方法，通过比较原指令与反事实安全指令的得分差判断拒绝强度。
- **VLGuard**：用于安全微调的视觉‑语言指令数据集，可增强模型对视觉对抗输入的鲁棒性。
- **MLLM‑Protector**：后验防御模块，含外部危害检测器与内容解毒器，可拦截或改写不安全响应。

## 可复现要素
- **数据集**：AdvBench、SafeBench（均为公开基准）。
- **代码/权重**：论文声明实现已公开（链接见摘要脚注 1）；基于 LLaVA、InternVL、Qwen‑VL、Qwen‑Audio 官方项目构建。
- **关键超参**：学习率 $\eta=0.1$，Adam 优化器，batch size=1；阈值 $\tau$ 依模型调整（LLaVA 7B/Qwen‑VL/Qwen‑Audio 设为 0.60，LLaVA 13B/InternVL 13B 设为 0.75，InternVL 34B 设为 0.85）；max\_new\_tokens=2048，temperature 使用默认设置。
- **硬件环境**：Linux AWS 实例，NVIDIA A100‑SXM4‑80GB GPU。
- **防御实验细节**：VLGuard 微调使用 LoRA（lr=2e‑4，3 epochs，global batch size=128）；噪声扰动添加零均值高斯噪声，标准差 $\sigma$ 取 1/4/6。
