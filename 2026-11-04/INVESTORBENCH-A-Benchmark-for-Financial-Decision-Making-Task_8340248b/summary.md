---
title: "INVESTORBENCH-A-Benchmark-for-Financial-Decision-Making-Task"
source: https://aclanthology.org/2025.acl-long.126.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 07:34:57"
field: "金融AI与大语言模型应用"
keywords: ["金融 agent", "LLM benchmark", "序列决策", "分层记忆", "股票交易", "加密货币"]
innovations: ["首个跨股票/加密货币/ETF的LLM agent金融决策统一评测基准", "基于FINMEM分层记忆架构的跨资产泛化设计", "开源多源多模态金融决策数据集与评估平台"]
benchmarks: ["INVESTORBENCH"]
---

# 论文速读：INVESTORBENCH-A-Benchmark-for-Financial-Decision-Making-Task

## 一句话总结
本文提出 **INVESTORBENCH**，首个专门针对 LLM-based agent 在多样化金融决策任务（股票交易、加密货币交易、ETF投资）上进行统一评估的开源基准测试平台，解决了现有金融 agent 框架局限于单一任务、缺乏标准化评测环境的痛点。

## 研究问题与动机
1. **缺乏综合性评估基准**：现有金融 LLM agent 框架（如 FINMEM、FINAGENT、CRYPTOTRADE、FINROBOT、FINCON 等）各自聚焦特定任务，无法跨任务类型进行统一比较。
2. **评估数据集不开放**：已有研究多依赖专有金融数据，导致性能难以复现、可比性受限。
3. **任务类型覆盖不足**：单一股票交易 vs. 加密货币 vs. ETF 的决策逻辑差异显著（如加密货币对新闻情绪更敏感，ETF 强调长期被动策略），现有框架缺乏跨资产类型的通用性。
4. **推理与决策能力评估缺失**：金融场景要求 agent 在多模态信息（数值、文本、图表）和时间敏感性信号中做出高质量序列决策，当前缺乏系统性评测手段。

## 核心贡献（创新点）
1. **首个跨资产类型的金融 agent 评测基准**：涵盖股票、加密货币、ETF 三类任务，提供统一的评估框架，区别于 FINMEM（仅股票）和 CRYPTOTRADE（仅加密货币）的单一任务定位。
2. **分层记忆架构的跨任务泛化**：基于 FINMEM 的工作记忆+分层长期记忆设计，引入不同衰减率（浅层14天、中层90天、深层365天）处理不同时间敏感度的金融信号，使其可适配多类资产而非仅单只股票。
3. **开源多源市场环境与数据集**：整合 Yahoo Finance（OHLCV）、SEC EDGAR（10-K/10-Q）、Refinitiv/加密货币新闻等多模态数据，提供完全开放的评测平台，填补金融 agent 基准领域空白。
4. **13 个主流 LLM 的全面评测**：系统对比闭源模型（GPT-4/o1-preview）、开源通用模型（Llama、Qwen、DeepSeek、Yi）和领域微调模型（Palmyra-Fin），揭示模型规模与领域适配性的关系。

## 方法详解
1. **Agent 架构五模块**：
   - **Brain/Backbone**：LLM 作为推理核心，生成预测分析和交易决策。
   - **Perception**：将多模态市场数据（数值 OHLCV、文本新闻/财报、情感标签）转换为结构化输入。
   - **Profile**：以自然语言定义 agent 角色（资深投资者）和动态风险偏好（根据历史动量自适应调整）。
   - **Memory**：包含工作记忆（即时+延伸反思）和分层长期记忆（浅/中/深三层，按衰减率管理信息生命周期）。
   - **Action**：输出 Buy/Sell/Hold 决策，分 warm-up（校准阶段，利用未来价格差）和 evaluation（严格历史数据）两阶段。
2. **分层记忆检索机制**：
   - 每个记忆事件得分 $\gamma_l^E = S_{Recency}^E + S_{Relevancy}^E + S_{Importance}^E$。
   - 近期得分 $S_{Recency_l}^E = e^{-\delta^E/Q_l}$，其中 $Q_{shallow}=14$、$Q_{intermediate}=90$、$Q_{deep}=365$ 天。
   - 重要性得分 $S_{Importance_l}^E = v_l^E \cdot \alpha_l^{\delta^E}$，深层信息衰减更慢（$\alpha_{deep}=0.988 > \alpha_{shallow}=0.9$）。
3. **POMDP 建模**：将金融决策建模为无限horizon POMDP，状态空间 $\mathcal{X} \times \mathcal{Y}$（可观测+不可观测），动作空间 $\{Buy, Sell, Hold\}$，奖励函数为每日 PnL，优化目标为 $\max_{\pi \in \Pi} \mathbb{E}\left[\sum_{t \in \mathbb{T}} \alpha^t R_t^\pi\right]$。
4. **评估指标**：累计收益率（CR）、夏普比率（SR，主要指标）、年化波动率（AV）、最大回撤率（MDD）。

## 实验与结果
- **数据集与环境**：
  - 股票：7只（MSFT、JNJ、UVV、HON、TSLA、AAPL、NIO），warm-up 2020-07-01 至 2020-09-30，test 2020-10-01 至 2021-05-06。
  - 加密货币：BTC、ETH，warm-up 2023-02-11 至 2023-04-04，test 2023-04-05 至 2023-11-05。
  - ETF：NIFTY 数据集，warm-up 2019-07-29 至 2019-12-30，test 2020-01-02 至 2020-09-21。
- **评测模型**：13 个（GPT-4/o1-preview/GPT-4o、Qwen2.5-7B/32B/72B、Llama3.1-8B/70B、DeepSeek-V2-Lite/67B、Yi-1.5-9B/34B、Palmyra-Fin-70B）。
- **硬件**：vllm 部署，<10B 用 2×RTX A6000，10B-65B 用 4×A6000，>65B 用 8×A100。
- **主要结果**：
  - **股票交易**：专有模型平均表现最优；Llama-3.1-70B-Instruct 在 MSFT 上 CR=17.396%、SR=0.921；Yi-1.5-34B-Chat 在 TSLA 上 CR=22.093%；GPT-4o 在 NIO 上 CR=75.952%、SR=0.887。
  - **加密货币**：GPT-o1-preview 在 BTC 上 CR=34.060%、SR=1.613；小型开源模型普遍弱于 Buy & Hold 基线。
  - **ETF**：Palmyra-Fin-70B 表现最强（CR=24.759%、SR=1.152、MDD=8.203%），远超专有模型平均。
  - **趋势**：模型参数增大提升决策质量；专有模型在混合/高波动市场更稳定；领域微调模型（Palmyra-Fin）在 ETF 表现突出但在加密货币表现差。

## 相关工作脉络
1. **FINMEM (Yu et al., 2024a)**：单股票投资决策 agent，提出分层记忆架构；本文在此基础上扩展到股票、加密货币、ETF 三类任务，验证记忆模块跨资产泛化能力。
2. **FINAGENT (Zhang et al., 2024a)**：多模态金融 agent，但仅使用纯相似度记忆检索；本文引入带衰减率的分层长期记忆，增强信息时效性管理。
3. **CRYPTOTRADE (Li et al., 2024)**：专注加密货币零样本交易；本文将其纳入统一框架，实现跨资产类型对比。
4. **FINROBOT (Yang et al., 2024)**：市场分析及 agent 平台，侧重工具调用；本文聚焦决策能力评测而非工具使用。
5. **FINCON (Yu et al., 2024b)**：多 agent 投资组合管理，但仅支持3只股票的紧凑组合；本文附录展示可扩展至多资产组合，但仍以单资产为主。
6. **金融 LLM 基准**：FLUE、Pixiu、FinBen 主要评估 NLP 任务（情感分析、NER、问答），不涉及 agent 决策能力；本文填补金融 agent 序列决策评测空白。

## 局限性与未来方向
1. **仅限单资产任务**：未充分探索多资产投资组合管理（附录仅展示3只股票案例），缺乏真正的大规模资产配置评测。
2. **数据版权限制**：部分新闻数据受版权保护，可能影响数据集规模和质量。
3. **模态单一**：当前仅使用数值（OHLCV）和文本（新闻/财报），未包含音频（财报电话会议）和图表（K线图）等多模态信息。
4. **时间跨度有限**：数据集集中在 2020-2023 年，未覆盖更长周期或极端市场事件（附录 E 虽有短期极端市场案例，但数据有限）。

## 研究启发与可借鉴点
1. **分层记忆架构迁移**：将 FINMEM 的三层记忆（浅/中/深）与不同衰减率机制应用于其他时序决策领域（如量化风控、供应链调度），实现信息生命周期管理。
2. **Profile 模块的动态风险偏好设计**：以自然语言描述 agent 角色并根据市场动量自适应调整风险倾向，可迁移至其他需要"人设"的 agent 应用（如客服、理财顾问）。
3. **Warm-up 与 Evaluation 两阶段分离**：warm-up 阶段允许访问未来价格用于校准策略，evaluation 阶段严格限制历史数据，该设计可借鉴于需要在线学习的 agent 评测。
4. **多基准任务的统一平台设计**：INVESTORBENCH 提供两种使用模式（集成自研 LLM 或直接复用环境），为构建跨任务 benchmark 提供架构参考。

## 关键术语表
**POMDP（部分可观察马尔可夫决策过程）**：金融决策的形式化建模框架，状态包含可观测与市场不可观测两部分，agent 基于部分信息做出序列决策。
**分层长期记忆（Layered Long-Term Memory）**：借鉴人类认知系统，将记忆分为浅（14天）、中（90天）、深（365天）三层，按不同衰减率存储和检索金融信号。
**工作记忆（Working Memory）**：包含观察、摘要和即时/延伸反思机制，整合当前市场指标与长期记忆事件生成决策。
**累计收益率（CR）**：投资期间每日对数收益的累加，衡量总回报幅度。
**夏普比率（SR）**：超额收益与波动率的比值，衡量风险调整后收益，为核心评估指标。
**暖化阶段（Warm-up Stage）**：agent 学习期，可访问未来价格差用于策略校准，侧重趋势理解。
**评估阶段（Evaluation Stage）**：正式测试期，仅使用历史数据，要求 agent 独立输出 Buy/Sell/Hold 及推理。
**专有模型（Proprietary Models）**：指 GPT-4 等闭源商业模型，文中表现出比开源模型更强的复杂市场决策能力。

## 可复现要素
- **数据集**：开源，来自 Yahoo Finance、SEC EDGAR、CoinMarketCap、Refinitiv Real-Time News、NIFTY 等公开来源，MIT 协议发布。
- **代码**：已开源，GitHub Repo: https://github.com/felis33/INVESTOR-BENCH
- **权重**：评测使用各模型官方 API 或开源权重，未微调。
- **关键超参**：temperature=0.6；记忆三层时间常数 Q_shallow=14天、Q_intermediate=90天、Q_deep=365天；重要性衰减基 α_shallow=0.9、α_intermediate=0.967、α_deep=0.988；top-K 记忆检索未明确，需查代码。
