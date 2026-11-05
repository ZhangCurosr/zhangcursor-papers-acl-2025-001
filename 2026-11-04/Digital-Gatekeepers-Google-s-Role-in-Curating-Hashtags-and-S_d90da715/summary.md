---
title: "Digital-Gatekeepers-Google-s-Role-in-Curating-Hashtags-and-S"
source: https://aclanthology.org/2025.acl-long.119.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:43:48"
field: "计算社会科学与搜索系统公平性"
keywords: ["Search Engine Manipulation", "Social Media Data Collection", "Algorithmic Curation", "Rank Turbulence Divergence", "Post-API Era", "Content Moderation Bias"]
innovations: ["在社区/话题粒度量化Google SERP对Reddit subreddit与Twitter hashtag的系统性抑制与推广偏差", "将RTD引入SERP社会数据代表性评估并提出可复现的分层采样+多代理查询审计框架", "结合毒性分析、主题分类与语义嵌入聚类三重视角揭示被过滤内容的共同属性"]
benchmarks: ["Pushshift Reddit (2023-01)", "Twitter/X Academic API Full 24h Snapshot (2022-09-20~21)", "SERP-API with Multi-Proxy Sampling"]
---

# 论文速读：Digital-Gatekeepers-Google-s-Role-in-Curating-Hashtags-and-S

## 一句话总结
本研究在"后API时代"背景下，通过对比Reddit和Twitter/X的完整数据与Google SERP结果，揭示了搜索引擎作为数字守门人，如何系统性抑制成人内容、阴谋论、广告及加密货币相关内容，同时优先展示高参与度、游戏/娱乐类话题，从而塑造网络公共话语。

## 研究问题与动机
- 社交媒体平台（如Reddit、Twitter/X）已大幅限制API访问，进入"Post-API era"（Freelon, 2018），传统数据集采方式受阻，研究者被迫依赖Search Engine Results Pages（SERPs）作为替代数据源。
- 已有研究表明，SERP本身存在算法偏见（如Poudel & Weninger, 2024的页面级分析），但缺乏从"社区/话题层级"（subreddit/hashtag）进行的系统性调查，不清楚搜索引擎如何过滤整个社区或话题。
- 搜索排序偏差可显著影响公众认知与行为（如Epstein & Robertson, 2015发现可改变超20%犹豫选民的投票倾向），亟待量化这种"搜索操纵效应"在reddit/tweet话题层面的表现。
- 现有社会计算研究长期面临数据不可得困境（涉及话语分析、计算社会科学、计算语言学、人类行为等多领域），亟需新的可复现、量化的方法框架来评估SERP的代表性与偏差。

## 核心贡献（创新点）
- 首次在社区与话题粒度（而非单篇帖子粒度）上系统度量Google SERP对Reddit subreddit与Twitter/X hashtag可见性的系统性偏差，填补了页面级偏见的社区级研究空白。
- 引入Rank Turbulence Divergence（RTD）作为衡量SERP样本与非采样社交数据分布差异的核心指标（Reddit RTD=0.64、Twitter/X RTD=0.73），并配套可视化"偏向度 vs. 活动量"关系曲线。
- 结合毒性分析（Toxic-BERT）与主题分类（GPT-4）双重探针，揭示SERP更倾向于呈现低毒性、偏游戏/娱乐/金融的议题，同时抑制成人内容、政治、加密货币相关话题。
- 针对后API时代的社会计算研究，提出并验证一套以SERP为替代数据源、以分层关键词采样+多代理查询去偏为手段的可复现评估框架。
- 通过语义嵌入（MPNet-Base-V2）+ UMAP降维的聚类可视化，将抽象的"偏差"转化为可直观理解的社区地理图谱（色情/政治/加密货币社区远离SERP聚集区）。

## 方法详解
- **数据采集**：Reddit数据来自Pushshift（2023年1月，覆盖336,949个distinct subreddits、约3609万帖子、2.54亿评论）；Twitter/X数据来自Academic API全量24小时抓取（2022/9/20-21，约3.75亿条tweets、含约80%转发/引用/回复）。
- **关键词采样**：基于文档频率的**分层采样**（stratified sampling）从去噪（标点过滤、停用词去除、词长≥3、频次≥100）后的token中抽取1000个关键词，以保证Zipfian分布下常见/中等/低频词均被充分代表。
- **SERP查询**：使用`site:reddit.com {keyword}`与`site:twitter.com {keyword}`格式查询Google，附加时间窗口约束以匹配上述非采样时间段；调用SERP-API并通过多个全球代理缓解地域偏差；每个关键词重复3次取并集以弱化SERP非确定性。
- **活动度度量**： subreddit活跃以投稿数衡量，hashtag活跃以词频衡量；用hexbin图展示SERP出现频次与原始频次的相关性（Twitter X: $R^2=0.214$；Reddit: $R^2=0.423$，均$p<0.001$）。
- **RTD度量**：采用Dodds et al. (2023)的Rank Turbulence Divergence，公式（E.1/E.2）对两个排序序列（非采样排名 vs. SERP排名）的元素级 rank 偏差做加权求和，$\alpha=1/3$。得分越高表示排名分布差异越大。
- **毒性分析**：使用Toxic-BERT预测每条帖子的toxicity/obscenity/insult概率，随机抽取各5000条"In SERP"与"Not In SERP"样本进行对比；呈现均值与95%置信区间。
- **hashtag分类**：用GPT-4按九分类体系（games, politics, celebrities, sex, entertainment, advertisement, finance, other, unknown）对top 1000英文hashtag打标签，prompt模板见附录A.2.1。
- **语义聚类**：对含≥10条帖子的subreddit，用MPNet-Base-V2编码5条随机post的平均嵌入，再经UMAP降至二维可视化分布。
- **稳健性检验**：5-fold交叉验证（每次取80%关键词子集），结论保持一致。

## 实验与结果
- **数据规模**：非采样数据中Reddit有336,949个subreddit、Twitter/X有3,014,574个hashtag；时间匹配的SERP样本分别仅包含35,094个subreddit与21,920个hashtag。
- **RTD结果**（表2）：Reddit RTD = 0.64、Twitter/X RTD = 0.73，说明SERP排名分布与平台原始分布存在显著偏离。
- **活动度相关性**：hexbin显示相关性中等偏强，Reddit相关性（$R^2=0.423$）高于Twitter/X（$R^2=0.214$），说明Google对Reddit的排序更贴合热度，但对Twitter/X还受其他因素（如事件时效）更强影响。
- **社区类型偏差**：
  - Reddit：SERP显著偏好public subreddit，抑制restricted/forbidden/private subreddit；
  - Twitter/X：Games、Finance类别在SERP中过度代表（over-represented），而Advertisement、Politics、Entertainment被低估（under-represented）。
- **毒性差异**：Reddit层面，"Not In SERP"的subreddit平均毒性显著高于"In SERP"；Twitter/X层面两者差异较小（可能与采样时段集中在非争议性话题有关）。
- **典型被抑制/被推广话题**：
  - Reddit被抑制：r/AskReddit、r/AutoNewspaper、r/dirtykikpals等；
  - Reddit被推广：/r/StableDiffusion、/r/Pathfinder2e、/r/ffxiv、/r/OnePiece等游戏/娱乐向社区；
  - Twitter/X被抑制：#voguegala2022xmileapo、#nft、#nsfwtwt、#putinisdead、#trumpmeltdown、#eth、#cryptocurrency等；
  - Twitter/X被推广：#unga77、#cybersecurity、#linux、#firstdayoffall、#abbottelementary等。
- **语义嵌入聚类**（图8）：成人内容、加密货币、政治讨论相关subreddit普遍不在SERP红点范围内；技术/音乐/漫画/游戏/健康类则密集呈现。

## 相关工作脉络
- **Poudel & Weninger (2024, Web Conf)**：本文与其一脉相承，但前者聚焦单篇post/page级排名偏见，本文扩展至subreddit/hashtag社区与话题级系统性偏差，视角更宏观。
- **Freelon (2018, Political Communication)**：提出"Post-API era"概念，指出社会计算研究因平台数据政策收紧而面临数据荒，本文直接响应此号召并以SERP作为替代数据源的实证评估。
- **Epstein & Robertson (2015, PNAS)**：提出"Search Engine Manipulation Effect"并量化其对选民决策的影响（>20%偏向），为本文论证"SERP偏见可实质性影响公共舆论"提供因果链条支撑。
- **Dodds et al. (2023, EPJ Data Science)**：引入Rank Turbulence Divergence（RTD）以应对Zipfian重尾分布下的排名比较难题，本文将其移植到SERP–social data的分布对齐评估中。
- **Pfeffer et al. (2023, ICWSM)**：提供Twitter/X完整24小时基准数据集（Just another day on Twitter），本文沿用该采样思想但将其用于"SERP覆盖度评估"的参照系构建。
- **Gillespie (2020, Big Data & Society)**：论述平台内容审查与AI治理的规模化困境，本文以实证方式佐证"算法策展不仅放大某些叙事，也在系统性边缘化另一些叙事"。

## 局限性与未来方向
- SERP-API的非确定性导致单次查询结果不完全可复现；虽通过多代理、三重复查询和5-fold交叉验证缓解，但本质上仍是样本次数与代理分布受限的近似。
- Reddit数据为整月（2023年1月）、Twitter/X数据仅为单天（2022年9月），时间跨度不一致影响两类平台的跨数据集对比与泛化结论。
- 仅聚焦英语hashtag，排除了大量非英语话题，可能遗漏语言维度上的偏差结构。
- 未深入拆解Google排名与过滤策略背后的具体信号（如广告收入、品牌安全协议、法律合规要求），难以区分"算法相关性"与"人工/政策干预"的贡献份额。
- 未来方向：（1）扩展到其他搜索引擎（Bing、Baidu等）与其他平台（TikTok、YouTube、Telegram等）；（2）纵向追踪算法更新与治理政策调整带来的动态演变；（3）结合用户侧实验评估SERP偏差对认知/态度的真实影响强度。

## 研究启发与可借鉴点
- **分层关键词采样策略**可迁移至任何需要"代表性语料"的跨平台对比研究，尤其适合Word-frequency重尾分布场景。
- **RTD度量**为"两个排名/频次分布"的比较提供了统一、可解释的距离指标，适用于SERP审计、推荐系统公平性评估、话题热度漂移检测等任务。
- **双探针评估框架**（毒性分析+主题分类）可在不依赖人工标注的前提下快速刻画"被过滤内容"的属性画像，适合扩展到其他敏感属性（如民族、性别、地域）的偏差审计。
- **SERP作为替代数据源**的方法路径可直接复用于其他数据获取受限的平台（如LinkedIn、Facebook公开内容），配合API降级、代理切换与重复查询策略提升覆盖度。
- 可结合本团队方向（如多模态偏差审计、跨语言SERP公平性、推荐系统可解释性）进一步引入因果推断或干预实验设计，识别Google排名信号中的关键驱动因子。

## 关键术语表
- **Digital Gatekeepers**：控制信息可见性的关键中介（此处指搜索引擎），通过算法排序与审核决定用户能看到哪些内容。
- **Post-API era**：社交媒体平台收紧甚至关闭公开API的时代背景，研究者被迫寻找替代数据源（如SERP）以维持计算社会科学研究的连续性。
- **Search Engine Result Pages (SERPs)**：搜索引擎返回的排名结果页面集合，其排序本身就是一种隐式的内容策展行为。
- **Rank Turbulence Divergence (RTD)**：基于Zipfian分布特性设计的排序差异度量，通过元素级rank偏差的幂律加权求和反映两个分布的结构性偏离程度。
- **Stratified keyword sampling**：按词频分层抽取关键词的方法，确保高/中/低频词均被覆盖，克服纯随机采样在重尾分布下的代表性偏差。
- **Toxic-BERT**：基于BERT架构训练的在线毒性检测模型，可输出toxicity/obscenity/insult等维度的概率，用于评估社区/话题的有害内容水平。
- **MPNet-Base-V2**：Microsoft出品的预训练句子表示模型，用于将subreddit内容编码为低维向量，配合UMAP实现语义空间可视化。
- **SERP-API**：第三方聚合搜索引擎结果的API服务，提供多代理查询、分页抓取等功能，用于自动化采集Google搜索结果。

## 可复现要素
- **Reddit数据**：Pushshift系统收集（2023年1月），非开源数据集，但作者声明可通过Pushshift公开渠道回溯。
- **Twitter/X数据**：Academic API获取的24小时完整快照（2022/9/20-21），由Pfeffer et al. (2023)数据集衍生，非本论文新开源。
- **代码/模型权重**：论文未提供官方代码仓库链接；Toxic-BERT、MPNet-Base-V2、UMAP为开源组件；GPT-4分类为商业API调用。
- **关键超参**：分层采样抽取1000个关键词；token过滤阈值（词长≥3、频次≥100）；SERP查询重复3次；RTD参数$\alpha=1/3$；交叉验证5-fold、每次取80%关键词子集。
