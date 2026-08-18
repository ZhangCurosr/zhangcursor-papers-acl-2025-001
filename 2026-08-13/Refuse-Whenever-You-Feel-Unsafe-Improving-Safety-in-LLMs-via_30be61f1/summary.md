---
title: "Refuse-Whenever-You-Feel-Unsafe-Improving-Safety-in-LLMs-via"
source: https://aclanthology.org/2025.acl-long.158.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:07:16"
field: "LLM安全对齐"
keywords: ["LLM安全", "jailbreak防御", "安全微调", "拒绝位置偏差", "RLHF", "安全对齐"]
innovations: ["揭示安全微调数据中拒绝token位置偏差问题", "提出DeRTa解耦拒绝训练框架，含有害前缀+RTO双组件"]
benchmarks: ["HarmBench", "Do-Not-Answer", "XSAFETY", "XStest"]
---

# 论文速读：Refuse Whenever You Feel Unsafe: Improving Safety in LLMs via Decoupled Refusal Training

## 一句话总结
这篇论文揭示了现有LLM安全微调数据中存在"拒绝位置偏差"（refusal position bias）——模型仅在响应开头拒绝不安全内容，一旦生成就无法停止，导致CodeAttack等攻击轻易突破防线。为此提出**解耦拒绝训练（DeRTa）**，通过在训练时嵌入有害响应前缀并引入强化转移优化（RTO），使模型能够在响应的任意位置学会拒绝生成。

---

## 研究问题与动机
- **拒绝位置偏差是核心症结**：现有安全微调数据（如BeaverTails）中，拒绝词（"sorry"、"I cannot"等）几乎全部出现在响应前5个token内。表1显示，LLaMA3-8B/70B-Instruct在200个攻击样本中，第5个token之后才拒绝的仅有2次（0.5%）。
- **缺乏拒绝决策所需信息**：模型被迫仅凭query在响应起始处做出拒绝判断，缺少对有害内容的完整感知，在CodeAttack等"先完成有害代码再问出问题"的陷阱式攻击中尤为脆弱。
- **缺乏响应中途转向的能力**：位置偏差使模型依赖位置特定特征，一旦已开始生成有害内容，就无法在中途切换至拒绝，安全防线形同虚设。
- **与已有工作的关键分歧**：Qi et al. (2025)、Xu et al. (2024b) 也观察到类似问题并提出有害前缀的数据增强方案，但本文证明仅靠前缀不足以抵御CodeAttack等复杂攻击（Figure 3, Table 3）。

---

## 核心贡献（创新点）
1. **首次系统性揭示并量化"拒绝位置偏差"**：区别于Qi et al. (2025) 从生成分布角度切入，本文从拒绝token的位置分布出发，用表格与图4提供了完整的实证证据。
2. **MLE with Harmful Response Prefix**：在安全响应的开头随机嵌入长度为k（k∈[0, |r̂|]）的有害响应片段作为前缀，使模型在训练中反复经历"从有害上下文切换到安全拒绝"的过程，从而学会在任何起始位置做出拒绝决策。
3. **强化转移优化（RTO）**：在前缀方法基础上，引入辅助损失——对有害响应的**每一个token位置t**都训练模型输出"sorry"，使模型在整个有害序列长度内被训练 |r̂| 次而非仅1次，显著提升中途拦截能力。
4. **新构建CompletingAttack白盒攻击**：去除所有格式化token（如[INST]），将query转为声明式文本完成格式，以检验模型在完全开放生成情境下的安全性。
5. **全面实验验证方法普适性**：在LLaMA3（8B/70B）和Mistral（7B/8×7B MoE）四个规模模型上，ASR从最高79.1%降至最低2.2%，且GSM8K/MMLU/AlpacaEval三项能力指标未受损。

---

## 方法详解

### 整体框架 DeRTa
核心思想：**用构造的三元组 (q, r̂, r) 替代传统二元组 (q, r)**，其中q为有害query、r̂为有害响应、r为安全响应，使模型"见识"有害内容的全貌后仍能安全拒绝。

### 损失函数（公式2）
$$
\mathcal{L}(\theta) = \underbrace{-\mathbb{E}_{(q,r,\hat{r})\sim\widehat{\mathcal{D}}}\log P_\theta(r \mid q, \hat{r}_{<k})}_{\text{MLE with Harmful Prefix}} + \underbrace{-\mathbb{E}_{(q,\hat{r})\sim\widehat{\mathcal{D}}}\sum_{t=1}^{|\hat{r}|}\log P_\theta(\text{"sorry"} \mid q, \hat{r}_{<t})}_{\text{RTO}}
$$

其中 $\hat{r}_{<k}$ 表示取有害响应的前k个token（k为随机采样值），**有害token不接收梯度回传**（防止模型学会生成有害内容）。

### 两组件的作用机制
- **MLE with Harmful Prefix**：提供额外的"有害上下文"作为输入，帮助模型在更丰富的信息下做出拒绝决策；随机长度k确保模型学会在响应的不同位置做出拒绝，而非仅依赖固定起始位置。
- **RTO**：对有害响应的每个位置t独立计算拒绝概率，迫使模型在每一时刻都保持拒绝意识——相当于将整个有害序列变成"触发训练"，训练次数从1次提升至 |r̂| 次。

---

## 实验与结果

### 实验设置
- **数据集**：安全性数据来自BeaverTails（3000条），辅助生成safety response用GPT-3.5-turbo、harmful response用恶意微调的LLaMA3-8B-Instruct；帮助性数据来自Evol-Instruct（60K条）。
- **评估集**：Do-Not-Answer（100条）+ HarmBench（100条）= 200条有害query。
- **攻击类型**：6种——CodeAttack、PAIR、JailbreakChat、SelfCipher（黑盒），CompletingAttack、AutoDAN（白盒）。
- **基线**：Vanilla safety training、Goal-Priority、SoFA、RecAug、DPO。
- **模型**：LLaMA3（8B/70B）、Mistral（7B/8×7B-MoE），部分实验用LoRA（rank=96, alpha=16）。
- **超参**：batch size=128，epochs=2，学习率2e-5（全参）或1e-4（LoRA）。

### 主要结果

| 模型 | Vanilla ASR | Ours ASR | 提升幅度 | GSM8K | AlpacaEval |
|---|---|---|---|---|---|
| Mistral-MoE (8×7B) | 79.1% | 8.7% | ↓70.4pp | 55.0→55.8 | 92.0→91.7 |
| LLaMA3-70B | 70.6% | 8.8% | ↓61.8pp | 78.6→77.6 | 97.0→96.3 |
| LLaMA3-70B-Instruct | — | 2.2% | 综合最优 | 89.3 | 97.8 |

- **相对于基线的优势**：在CompletingAttack上ASR仅4.0%（最佳基线RecAug为25.0%）；在CodeAttack上ASR仅21.5%（最佳基线SoFA为73.0%）。
- **消融实验（Table 3）**：仅加Harmful Prefix平均ASR=55.8%→30.5%；仅加RTO平均ASR=17.6%→8.5%；两者结合平均ASR=11.8%→3.0%（黑盒）/3.0%（白盒）。
- **模型尺寸普适性（Figure 6）**：从Mistral-7B到LLaMA3-70B，各尺寸均获得显著安全提升。

### 鲁棒性分析
- **多语言**：在英/法/西/德/中五种语言上（XSAFETY 1000条），性能稳定。
- **解码策略**：在不同采样温度/核采样下表现稳定。
- **Fully Safe ASR（Figure 9）**：绝大多数拒绝发生在有害内容生成之前，而非"先生成有害步骤再拒绝"。
- **自适应攻击**：即使攻击者预设"Sorry, I cannot..."前缀诱导模型，ASR仍维持在极低水平。
- **过度敏感缓解**：加入200条XStest边界数据后，过度敏感率从64.0%降至18.0%，ASR仅增加4.4pp。

---

## 相关工作脉络
1. **Qi et al. (2025)**：同样观察到安全对齐存在"偷懒"问题（仅适应前几个token），并提出从有害响应前缀切换到安全拒绝的数据增强策略——本文证明该策略单独使用仍不足以抵御CodeAttack，需配合RTO。
2. **Xu et al. (2024b)**：从生成分布角度分析安全对齐捷径问题，提出合成偏好数据——本文从拒绝位置偏差角度切入，方法论完全不同。
3. **RecAug (Qi et al., 2025) / Goal-Priority (Zhang et al., 2024b) / SoFA (Lu et al., 2024)**：均为无需额外推理开销的数据/策略调整方案，与本文可比；本文在六种攻击上全面超越。
4. **DPO (Rafailov et al., 2024)**：使用安全/有害响应对进行偏好优化——DPO在SelfCipher上有效，但在CodeAttack上无效，说明偏好排序本身不足以解决位置偏差问题；本文方法在所有任务上全面优于DPO。
5. **CodeAttack (Ren et al., 2024)**：将恶意问题伪装为代码补全任务——本文以此作为关键测试场景，证明仅靠前缀方法无法防御此类OOD攻击。
6. **HarmBench (Mazeika et al., 2024) / Do-Not-Answer (Wang et al., 2024c)**：标准安全评测基准，本文在其上建立统一评估协议。

---

## 局限性与未来方向
- **攻击覆盖不全**：仅评测六种代表性攻击，实际jailbreak方法繁多，全面评测成本过高。
- **单轮对话局限**：仅验证了单轮场景，多轮对话的延续性拒绝能力尚未验证（论文认为可自然扩展但未验证）。
- **过度敏感风险**：DeRTa加剧了过度敏感问题（ASR从8.8%升至13.2%），需额外引入XStest边界数据缓解。
- **有害响应获取依赖**：harmful response需通过恶意微调的模型+GPT-3.5后处理生成，存在数据质量敏感性。
- **多语言数据缺失**：训练数据仅含英文，虽多语言测试稳定但训练语言单一可能影响非英语场景的最优表现。

---

## 研究启发与可借鉴点
1. **拒绝位置偏差的测量框架可迁移**：表1的"拒绝token位置分布"统计方法可作为安全微调数据质量评估的标准诊断工具，适用于任何安全对齐研究。
2. **有害前缀+多位置训练的思路可复用到其他领域**：在幻觉检测、知识边界判断等场景中，均可借鉴"让模型先见识有害/错误内容，再在任意位置学会拒绝"的训练范式。
3. **RTO的token-wise辅助损失设计值得复用**：对序列中每个位置独立施加同一监督信号，是提升时序决策能力的高效且低成本手段，可迁移至代码生成、医学建议等高风险序列任务。
4. **CompletingAttack的新攻击范式**：去除所有格式化token的"纯声明式完成"攻击，对通用安全评测协议设计具有参考价值，可作为新的标准攻击基线。
5. **过度敏感的低成本缓解策略**：仅需添加200条XStest边界样本即可将过度敏感率降低46pp，这一低成本策略对后续安全对齐研究极具参考价值。

---

## 关键术语表
- **DeRTa（Decoupled Refusal Training）**：解耦拒绝训练，本文提出的安全微调新方法，通过有害前缀+RTO双组件实现任意位置拒绝能力。
- **Refusal Position Bias（拒绝位置偏差）**：安全微调数据中拒绝词集中出现在响应前几个token的统计偏差，导致模型无法在响应中途拒绝。
- **MLE with Harmful Prefix**：在安全响应开头随机嵌入有害响应片段作为前缀的辅助训练策略。
- **RTO（Reinforced Transition Optimization，强化转移优化）**：对有害响应每个token位置独立施加拒绝学习信号的辅助损失。
- **CompletingAttack**：本文提出的新白盒攻击，去除格式化token后将query转为声明式文本完成格式。
- **ASR（Attack Success Rate）**：攻击成功率，即模型生成有害响应的比例，越低越安全。
- **BeaverTails**：大规模安全微调数据集，本文作为有害指令来源。
- **XStest**：过度敏感测试数据集（Röttger et al., 2024），用于评估模型对边界问题的拒绝敏感度。

---

## 可复现要素
- **数据集**：BeaverTails（公开）、Evol-Instruct（公开）、Do-Not-Answer（公开）、HarmBench（公开）、XSAFETY（公开）、XStest（公开）；安全微调三元组数据由作者自行构建。
- **代码**：论文GitHub项目链接在 Appendix D 提及，代码已开源。
- **权重**：未明确声明公开；基于LLaMA3和Mistral官方模型进行微调。
- **关键超参**：batch size=128，epochs=2，learning rate=2e-5（全参）/1e-4（LoRA），LoRA rank=96、alpha=16，max length=512~1024，dropout 0~95%（按"Sorry" token而定）。
- **硬件**：8×A800 80GB GPU；LLaMA3-70B训练约100 GPU小时。

---
