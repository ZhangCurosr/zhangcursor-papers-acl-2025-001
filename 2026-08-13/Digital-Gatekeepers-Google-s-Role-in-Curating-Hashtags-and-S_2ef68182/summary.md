---
title: "Digital-Gatekeepers-Google-s-Role-in-Curating-Hashtags-and-S"
source: https://aclanthology.org/2025.acl-long.119.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:55:03"
field: "计算社会科学 / 搜索引擎公平性"
keywords: ["search engine bias", "SERP", "Reddit", "Twitter/X", "digital gatekeeping", "rank turbulence divergence"]
innovations: ["将 SERP 偏见分析从页面级扩展到 subreddit/hashtag 社区层级，揭示宏观叙事塑造机制", "跨平台对比显示搜索引擎对 Reddit 与 Twitter 的活跃度-可见性关联强度存在显著差异", "结合 RTD、毒性评估与语义聚类构建多维度 SERP 偏见量化框架"]
benchmarks: ["Rank Turbulence Divergence (RTD)", "Toxic-BERT/Detoxify 毒性评估", "MPNet-Base-V2 + UMAP 语义聚类"]
---

# 论文速读：Digital Gatekeepers: Google's Role in Curating Hashtags and Subreddits

## 一句话总结
该论文通过对比 Google 搜索结果页（SERP）与 Reddit/Twitter 未经采样的完整社区数据，揭示搜索引擎在 subreddit 和 hashtag 层级存在系统性偏见：偏向高参与度、游戏/娱乐类内容，而系统性抑制色情、阴谋论、广告和加密货币相关内容，从而主动塑造公共话语的信息生态。

## 研究问题与动机
- **核心问题**：在社交媒体 API 访问受限的"后 API 时代"，搜索引擎（以 Google 为例）如何通过算法策展充当数字守门人，影响 subreddit 和 hashtag 的可见性与舆论框架。
- **现有方法不足 1**：已有 SERP 偏见研究多停留在单篇帖子/页面的粒度（如 Poudel & Weninger, 2024），缺乏对社区/话题层级整体叙事被塑造机制的系统考察。
- **现有方法不足 2**：Post-API 时代研究者难以获取 Twitter/Reddit 完整数据，SERP 常被当作代理数据，却未充分量化其引入的采样偏差与算法过滤效应。
- **动机与应用价值**：搜索排名偏差可改变公众对信息的接触范围，已知研究表明可能显著影响消费者选择甚至选举倾向（SEME 效应），因此亟需理解搜索引擎在公共话语中的把关作用。

## 核心贡献（创新点）
1. **社区层级的 SERP 偏见量化**：将偏见分析从"单篇帖子"提升到" subreddit / hashtag"层级，首次系统刻画搜索引擎对整个社区可见性的塑造作用。与先前工作本质区别在于，本文揭示了偏见不仅存在于帖子层面，更在更高层级上放大了某些叙事、压制了另一些叙事。
2. **跨平台对比揭示算法差异化**：同时考察 Reddit 和 Twitter/X，发现搜索引擎在这两个平台上关联用户参与度与可见性的强度不同（Reddit R²=0.423 高于 Twitter R²=0.214），说明算法并非统一标准，而是按平台特性差异化策展。
3. **毒性分析与分类映射的联合评估**：结合 Toxic-BERT 毒性评估、GPT-4 标签分类（9 类）与 RTD（Rank Turbulence Divergence）语义聚类，构建多维度画像，全面刻画被压制/推广的内容特征。
4. **方法论上的 SERP 抽样鲁棒性验证**：通过 5 折交叉验证（5 组 80% 关键词样本）与 SERP 重复查询（3 次），证明结论在 SERP 非确定性下仍然稳健。
5. **公开可复现的跨平台偏见证据集**：提供 SERP 与原始平台数据的配对数据集、可视化示例（如图 4–8）与代表性 subreddit/hashtag 列表，为后续研究提供基准参照。

## 方法详解
- **数据来源**
  - Reddit：使用 Pushshift 获取 2023 年 1 月全量数据，共 36,090,931 篇帖子、253,577,506 条评论，覆盖 336,949 个 subreddit。
  - Twitter/X：使用学术 API 获取 2022 年 9 月 20–21 日 24 小时数据，共 374,937,971 条推文。
  - SERP：针对 1,000 个分层采样的关键词（基于文档频率，覆盖常见/中等/罕见词），使用 `site:reddit.com {keyword}` 与 `site:twitter.com {keyword}` 查询 Google，时间窗口与源数据对齐；采用 SERP-API（多全球代理）并每个查询重复 3 次合并结果。
- **活动相关性分析**
  - Reddit 活动度量：子版块在采样时段内的发帖数；Twitter 活动度量：hashtag 出现频次。
  - 使用 hexbin 散点图展示 SERP 频次与非采样频次的相关性，报告 R² 与显著性。
- **样本表征与分类**
  - Reddit subreddit 按可见性划分为 public / restricted / forbidden / private；使用 GPT-4 按 9 类主题（games、politics、celebrities、sex、entertainment、advertisement、finance 等）对 hashtag 分类。
  - 比较"出现在 SERP"与"不出现在 SERP"两类群体的比例差异（图 4、5）。
- **毒性分析**
  - 使用 Toxic-BERT（基于 Wikipedia 评论训练的 detoxify 模型）评估 Reddit 帖子标题与推文毒性概率（toxicity、obscenity、insults）。
  - "In SERP"组：随机抽取 5,000 篇来自 top-1,000 搜索结果中的帖子/推文；"Not In SERP"组：随机抽取 5,000 篇来自未被 SERP 收录 subreddit/hashtag 的帖子/推文。
- **抑制与推广分析**
  - 采用 Rank Turbulence Divergence (RTD, Dodds et al., 2023) 衡量 SERP 分布与非采样分布的差异：
    - 元素级公式：`|1/r_ξ,1^α − 1/r_ξ,2^α|^(1/(α+1))`，其中 α=1/3。
    - 归一化聚合后给出全局 RTD 值（Reddit 0.64、Twitter 0.73），分数越高表示分布差异越大。
  - 绘制"有偏抑制/推广程度"随活动频次的函数（图 7），识别极端被压制与被推广的案例。
  - 语义聚类：对 subreddit 内容用 MPNet-Base-V2 做嵌入平均，并用 UMAP 降维可视化（图 8），识别明显缺席 SERP 的类目（成人内容、加密货币、政治等）与突出类目（科技、音乐、漫画、游戏、健康等）。

## 实验与结果
- **数据集规模**：非采样数据中 Reddit 含 336,949 个 subreddit，Twitter 含 3,014,574 个 hashtag；SERP 召回 Reddit 35,094 个 subreddit、Twitter 21,920 个 hashtag（表 1）。SERP 结果总计 Reddit 1,296,958 篇帖子、Twitter 80,651 条推文。
- **活动相关性**
  - Reddit subreddit 活动与 SERP 出现的关联性更强（R²=0.423，p<0.001）。
  - Twitter hashtag 活动与 SERP 出现的关联性较弱（R²=0.214，p<0.001）；低活跃度 hashtag 仍可因历史热度进入 SERP。
- **RTD 分布差异**
  - Reddit SERP 与非采样数据的 RTD=0.64；Twitter SERP 与非采样数据的 RTD=0.73，均显示显著分布偏移（表 2）。
- **可见性类别差异**
  - Reddit：SERP 显著偏好 public subreddit，压制 restricted/forbidden/private subreddit（图 4）。
  - Twitter：Games 与 Finance 类别 hashtag 被过度代表；Advertisement、Politics、Entertainment 被代表不足（图 5）。
- **毒性差异**
  - Reddit：不在 SERP 的 subreddit 整体毒性更高，表明 SERP 主动过滤高风险内容（图 6）。
  - Twitter：In SERP 与 Not In SERP 的 hashtag 毒性差异较小，可能与采样时段集中在娱乐/金融/游戏等争议性较低议题有关。
- **被压制/推广的典型项**
  - Reddit 被严重压制的 subreddit 包括：r/AskReddit、r/AutoNewspaper、r/dirtykikpals 及各成人/色情子类；被推广的主要为 StableDiffusion、Pathfinder2e、ffxiv、OnePiece、Cooking、DestinyTheGame、StardewValley、leagueoflegends、BaldursGate3、Genshin_Impact 等游戏/创作/生活类社区。
  - Twitter 被压制的 hashtag 包括：#voguegala2022xmileapo、#nft、#nsfwtwt、#putinisdead、#trumpmeltdown、#cryptocurrency 等；被推广的主要为 #unga77、#firstdayoffall、#cybersecurity、#linux、#peaceday、#worldalzheimersday 等事件/公益/科技类 hashtag。
- **最强结果与提升幅度**：跨平台 RTD 最高达 0.73（Twitter），显示搜索引擎与原始平台分布差异显著；Reddit 活动-可见性相关系数 R²=0.423，显著高于 Twitter 的 0.214，反映搜索引擎在 Reddit 上对活跃度的敏感度更高。

## 相关工作脉络
- **Poudel & Weninger (2024)**：研究单篇帖子/页面在 SERP 中的 promotion/suppression 动态；本文将其扩展至 subreddit/hashtag 社区层级，揭示更宏观的叙事框架效应。
- **Freelon (2018)**：提出"后 API 时代"概念，指出社交媒体数据获取受限对计算社会科学的影响；本文以 SERP 为替代数据源，并量化其引入的偏差。
- **Epstein & Robertson (2015)**：发现搜索排序偏差可导致选举偏好偏移超过 20%；本文在此基础上将焦点从个体决策延伸至公共信息生态的结构化偏见。
- **Dodds et al. (2023)**：提出 Rank Turbulence Divergence (RTD) 度量；本文将其引入搜索引擎与社会媒体数据分布的比较，提供可复用的跨分布差异指标。
- **Morstatter et al. (2013)**：分析 Twitter 采样偏差（garden-hose vs fire-hose）；本文进一步揭示即便拥有近全量数据，仍可能因依赖 SERP 产生新的系统性偏差。
- **Hanu & Unitary team (2020) / Detoxify**：提供毒性分类模型；本文借此量化 SERP 与原始数据在有害内容过滤上的差异，支撑"安全-表达"权衡讨论。

## 局限性与未来方向
- **SERP API 非确定性**：每次查询结果存在随机波动，尽管采用多代理与重复查询缓解，仍仅为搜索引擎输出的近似样本。
- **跨平台数据覆盖不一致**：Reddit 数据为整月（1 月），Twitter 数据仅为 24 小时，可能影响两平台结论的可比性与泛化性。
- **仅聚焦英语 hashtag**：为保持一致性过滤了非英语内容，可能遗漏其他语言社区的关键偏差模式。
- **因果机制不明确**：SERP 抑制行为是算法排序、商业化合作、内容政策共同作用的结果，本文未拆解各因素的独立贡献。
- **未来方向**：扩展至其他搜索引擎（Bing、DuckDuckGo）与社交平台（TikTok、Telegram、微博）；开展纵向研究追踪算法更新与政策变化对偏见的长期影响；探索 SERP 偏差对公众信任、信息极化与民主进程的实质后果。

## 研究启发与可借鉴点
- **分层粒度跃迁**：从页面级分析升级至社区/话题级分析，能捕捉到单篇研究无法显现的系统性叙事塑造效应，值得在其他平台审计场景复用。
- **RTD 指标迁移**：Rank Turbulence Divergence 可用于任意"观测样本 vs 总体分布"的对比任务，适合搜索引擎审计、推荐系统公平性评估等场景。
- **毒性-可见性联合分析**：将 toxicity 评分与 SERP 出现概率结合，为"安全过滤是否过度"提供定量证据，可迁移至平台内容治理评估。
- **五折交叉验证 + 多重复查询**：针对非确定性 API 的稳健性验证策略，可作为 SERP 研究的标准实验规范。
- **公开对照基准价值**：本文提供的代表性 subreddit/hashtag 名单与可视化结果，可作为后续研究对比的基线参照，减少重复数据采集成本。

## 关键术语表
- **Digital Gatekeepers**：控制信息可发现性与可见性的平台/算法机制（如搜索引擎），决定哪些内容能被用户看到。
- **SERP (Search Engine Result Pages)**：搜索引擎返回的结果页面，代表算法对用户查询的策展与排序结果。
- **Post-API 时代**：社交媒体平台收紧数据接口访问后，研究者被迫转向替代数据源（如 SERP）的研究生态阶段。
- **Rank Turbulence Divergence (RTD)**：衡量两个有序分布之间差异的指标，对 Zipfian/长尾分布具有鲁棒性。
- **Toxic-BERT (Detoxify)**：基于预训练语言模型的在线毒性检测工具，输出 toxicity、obscenity、insults 等标签概率。
- **Search Engine Manipulation Effect (SEME)**：搜索排序偏差对公众意见与决策（如选举）产生实质性影响的理论与实证现象。
- **MPNet-Base-V2**：用于文本语义嵌入的预训练模型，本文用于 subreddit 内容表示与 UMAP 降维聚类。
- **U-MAP**：Uniform Manifold Approximation and Projection，非线性降维方法，用于可视化高维语义空间。

## 可复现要素
- **数据集**：Reddit 数据来自 Pushshift（约 2023 年 1 月全量），Twitter/X 数据来自学术 API（2022-09-20 至 2022-09-21 24 小时）；论文未声明公开具体数据集压缩包，但提供表格与图示的统计结果。
- **代码/权重**：论文未声明开源代码仓库；使用了开源工具 Lucene StandardAnalyzer、GPT-4 API、Toxic-BERT/Detoxify、MPNet-Base-V2、UMAP。
- **关键超参**：关键词采样 1,000 个、分层均匀覆盖；SERP 查询每个关键词重复 3 次；毒性分析采样各 5,000 条帖子/推文；RTD 中 α=1/3；关键字过滤条件为非字母字符剔除、长度≥3 字符、出现频次≥100 次。
