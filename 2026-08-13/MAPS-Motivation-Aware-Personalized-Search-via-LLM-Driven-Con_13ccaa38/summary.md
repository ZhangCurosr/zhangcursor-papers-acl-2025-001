---
title: "MAPS-Motivation-Aware-Personalized-Search-via-LLM-Driven-Con"
source: https://aclanthology.org/2025.acl-long.152.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:11:13"
field: "个性化检索与排序"
keywords: ["个性化搜索", "动机感知", "LLM对齐", "MoAE", "咨询对话", "对比学习", "电商推荐"]
innovations: ["首次将用户咨询历史显式建模为搜索动机源，提出双路对齐框架桥接自然语言咨询与商品ID语义", "设计MoAE混合注意力专家池化网络，自适应聚焦关键token语义，替代传统平均/CLS池化"]
benchmarks: ["Commercial电商数据集", "Amazon Reviews合成数据集", "HR@k, NDCG@k, MRR@k"]
---

# 论文速读：MAPS-Motivation-Aware-Personalized-Search-via-LLM-Driven-Con

## 一句话总结
本文针对电商个性化搜索中用户查询难以完整表达真实动机的问题，首次将"搜索动机"建模为可学习信号，提出MAPS框架：利用LLM对齐咨询文本与查询、通过MoAE注意力专家网络提取关键语义，并设计通用对齐（对比学习桥接ID-文本鸿沟）与个性化对齐（双向注意力融合咨询/历史序列），在检索与排序任务上均显著优于现有方法。

## 研究问题与动机
1. **查询≠真实动机**：现有个性化搜索方法假设用户查询已充分表达需求，但实践中用户常基于模糊动机发起搜索（如图1示例），需要多次搜索/比较才能找到合适商品。
2. **咨询蕴含动机信号**：电商平台引入AI咨询服务后，用户在搜索前往往先进行咨询（图2显示有相当比例搜索会话伴随相关咨询），咨询对话隐含了即将搜索的深层动机。
3. **三大对齐挑战**：
   - 咨询文本长且复杂，与简洁关键词式查询存在语义错位；
   - 自然语言表达的动机与产品类别属性之间存在领域鸿沟；
   - 历史咨询序列中混杂不相关内容，需噪声过滤与动机筛选。

## 核心贡献（创新点）
1. **首次显式建模"搜索动机"**：将用户咨询历史视为动机源，明确其在个性化搜索中的增强价值，区别于仅依赖查询-商品匹配的传统范式。
2. **LLM驱动的MoAE语义对齐**：冻结预训练LLM生成token嵌入，通过参数化注意力、自注意力、搜索中心交叉注意力三类专家组成的混合池化网络，自适应聚焦关键语义，弥补传统平均池化缺乏世界知识与重点聚焦能力的缺陷。
3. **双路对齐机制**：通用对齐（映射表+双向对比损失）桥接token-商品ID语义空间；个性化对齐（双向Transformer编码器）从咨询序列与查询历史中抽取动机感知嵌入并与用户偏好融合，区别于CHIQ等仅做query rewriting的工作。
4. **工业级验证**：在含AI咨询服务的真实电商数据集与GPT-4o合成Amazon数据集上，检索与排序任务均实现约15%~35%的绝对提升，并系统性验证了LLM规模、序列长度、激活函数等设计选择。

## 方法详解
**整体框架**（图3）包含三个模块：(1) ID-Text表示融合（LLM+MoAE）；(2) 映射式通用对齐；(3) 序列式个性化对齐。

**(1) ID-Text Representation Fusion with LLM**
- 文本表示：将查询/咨询/商品标题等文本输入冻结LLM获取token嵌入（不直接做平均池化），经FFN映射到统一维度$d_t$后，送入MoAE池化网络。
- MoAE包含三种专家：
  - Parameterized Attention Pooling：可学习查询向量$\mathbf{q}$作为注意力Query；
  - Self-Attention Pooling：token间自注意力加权；
  - Search-Centered Cross-Attention Pooling：以当前查询文本嵌入$\mathbf{e}_s^{\mathrm{text}}$作为Query，使其他文本聚焦于查询语义。
- 门控网络计算$3N_E$维得分，选取Top-K专家输出$\mathbf{e}^{\mathrm{text}} = \sum_{j=1}^K gate_j \mathbf{e}_j^{\mathrm{pool}}$。
- 类别ID表示：通过查找表$\mathbf{e}_{g_{id}}^{\mathrm{ID}}=\mathrm{lookup}_g(id)$获得，拼接所有类别特征得$\mathbf{e}^{\mathrm{ID}}$。
- 最终表示：$\mathbf{e}_u, \mathbf{e}_v, \mathbf{e}_s, \mathbf{e}_c$均由concat(IFN映射到$d_{\mathrm{uni}}$，act激活函数，默认tanh。

**(2) Mapping-Based General Alignment**
- 对每个商品$v$，聚合全场景相关文本（查询、咨询、标题、描述、广告等）构建全集$\mathcal{A}_v$。
- 按搜索场景词频阈值$t$过滤低频噪声：$\mathcal{A}_v^S = \{w \in \mathcal{A}_v \mid \mathrm{freq}^S(w) > t\}$，建立token→商品的映射$M$。
- 双向对比损失$\mathcal{L}_{\mathrm{GA}}$：
$$
\mathcal{L}_{\mathrm{GA}} = -\lambda_1 \sum_{(t,v)} \log \frac{\exp(\mathrm{sim}(\mathbf{e}_t,\mathbf{e}_v)/\tau_1)}{\sum_{t^-}\exp(\mathrm{sim}(\mathbf{e}_{t^-},\mathbf{e}_v)/\tau_1)} - \lambda_2 \sum_{(t,v)} \log \frac{\exp(\mathrm{sim}(\mathbf{e}_t,\mathbf{e}_v)/\tau_2)}{\sum_{v^-}\exp(\mathrm{sim}(\mathbf{e}_t,\mathbf{e}_{v^-})/\tau_2)}
$$
使正确token-item对在共享空间中相近。

**(3) Sequence-Based Personalized Alignment**
- 动机感知查询嵌入：以当前查询$\mathbf{e}_{s_{N+1}}$为anchor，分别与咨询历史$[\mathbf{e}_{c_1},\dots,\mathbf{e}_{c_M}]$、查询历史$[\mathbf{e}_{s_1},\dots,\mathbf{e}_{s_M}]$输入独立Transformer编码器，取首向量得$\mathbf{e}_{s_{N+1}}^{\mathcal{C}}$与$\mathbf{e}_{s_{N+1}}^{S}$，加权融合：
$$
\mathbf{e}_{s_{N+1}}' = \alpha_1 \mathbf{e}_{s_{N+1}}^{\mathcal{C}} + \alpha_2 \mathbf{e}_{s_{N+1}}^{S} + \alpha_3 \mathbf{e}_{s_{N+1}}
$$
- 结合商品历史与用户偏好：将$\mathbf{e}_{s_{N+1}}'$与候选商品嵌入集合$\mathbf{E}_{\mathrm{items}}$输入Final Transformer编码器，与用户嵌入$\mathbf{e}_u$逐元素相加得$\mathbf{e}_{s_{N+1}}''$。
- 排序概率：$p(v|s_{N+1},H,u) = \mathrm{sim}(\mathbf{e}_{s_{N+1}}'', \mathbf{e}_v)$。
- 个性化对齐损失$\mathcal{L}_{\mathrm{PA}}$采用InfoNCE形式（负采样自当前batch内99个随机负样本），总损失：
$$
\mathcal{L}_{\mathrm{overall}} = \mathcal{L}_{\mathrm{PA}} + \lambda_3 \mathcal{L}_{\mathrm{GA}} + \lambda_4 \|\Theta\|_2
$$

## 实验与结果
**数据集**：
- Commercial：某电商AI咨询服务真实交互数据（31天，过滤<5次交互用户/商品，前29天训练，后2天验证/测试），2096用户、2691商品、18774次搜索+40567次咨询交互，稀疏度99.56%。
- Amazon：Amazon Reviews数据集（PersonalWAB处理版）+ GPT-4o生成咨询文本，967用户、35772商品、7263搜索+35772咨询交互，稀疏度99.98%。

**基线**：
- 排序：ZAM, HEM, AEM, QEM, TEM, CoPPS；多场景融合：SESRec, UnifiedSSR, UniSAR。
- 检索：BM25, BGE-M3, CHIQ。

**主要结果**：
- **排序（Commercial）**：MAPS的HR@10=0.7071、NDCG@10=0.4359，较最强基线CoPPS（HR@10=0.5637, NDCG@10=0.3445）分别提升约25%与26%；Amazon上相对CoPPS提升约20%~35%。
- **检索（Commercial）**：MAPS MRR@10=0.3805，较最强基线CHIQ（0.3192）提升19%；超密集检索BGE-M3（0.2976）达28%提升。
- **多场景对比（Commercial）**：相对UniSAR（HR@10=0.5838, N@10=0.3577）提升约21%与22%。
- **消融**：去掉通用对齐（w/o general align）下降最显著（HR@10从0.7071降至0.6198），说明token-item语义对齐至关重要；MoAE vs 平均池化带来约1.7% HR@10提升。
- **扩展性**：LLM越大越好（Qwen2.5-0.5B→7B，N@20从0.3394升至0.3780）；Transformer层数增加持续增益；序列长度过长（40 vs 30）反而略降（含噪声）。
- **配置分析**：映射阈值$t=3$最优；激活函数PReLU略优于默认tanh（Amazon N@20=0.5097 vs 0.4995）。

## 相关工作脉络
1. **个性化搜索基线（ZAM/HEM/AEM/QEM/TEM/CoPPS）**：从早期零注意力模型演进至Transformer编码器（TEM）与对比学习（CoPPS），但均以"查询-商品历史"为核心，未建模咨询动因。
2. **对话检索（CHIQ）**：利用LLM世界知识做query rewriting，侧重上下文历史辅助，但未将咨询对话显式建模为动机源并与商品ID对齐。
3. **多场景融合（SESRec/UnifiedSSR/UniSAR）**：联合搜索与推荐行为，利用cross-attention或双分支建模行为转换，仍聚焦ID序列交互，缺少咨询文本语义挖掘。
4. **ID-文本对齐（BGE-M3/LEARN）**：引入预训练语言模型做embedding，但仅用于检索端文本表征，未设计对比损失桥接token与商品ID，也未考虑MoAE自适应加权。
5. **序列推荐与搜索统一（SAR系列工作）**：强调search-recommendation行为迁移，但未覆盖"咨询-搜索"这一新兴交互链路及其动机传播机制。
6. **MAPS定位**：首次将"咨询动机"作为可学习信号，通过LLM+MoAE+双路对齐打通"自然语言咨询→关键词查询→商品ID"的全链路语义对齐，填补了动机感知个性化搜索的空白。

## 局限性与未来方向
1. **实时性与可扩展性**：未充分讨论工业部署下的推理延迟与计算开销，LLM冻结+MoAE虽降低训练成本，但线上检索仍需高效近似。
2. **动态偏好建模不足**：未显式建模用户随时间演进的长期偏好变化（引用Shen et al., 2023），动机可能随会话推进漂移。
3. **领域知识缺失**：通用LLM嵌入缺乏垂直领域专业知识（如家电参数、化妆品成分），在专业品类中可能受限。
4. **咨询相关性判定依赖启发式规则**：Appendix A采用Lenient/Moderate/Strict三档字符串匹配规则定义"相关咨询"，可能遗漏语义等价但字面不同的动机。
5. **未来方向**：优化实时适应能力、引入外部领域知识图谱、探索动态动机追踪、结合强化学习在线调优。

## 研究启发与可借鉴点
1. **MoAE自适应语义聚焦**：将LLM token嵌入与可学习门控专家池化结合，替代简单平均/CLS池化，可在任意文本对齐任务（如对话摘要、意图识别）中复用，尤其适合长文本中关键信息稀疏的场景。
2. **通用对齐+个性化对齐解耦设计**：先用对比学习在全局空间对齐"词-商品"语义（世界知识），再用序列编码器做用户特定动机提取（个性化），两者分离便于分阶段训练与增量更新，值得迁移至多模态召回/排序 pipeline。
3. **噪声过滤阈值$t$的敏感性分析**：映射表构建中频率阈值过小引入跨域噪声、过大限制覆盖范围，本文给出可视化曲线（Fig.4），提示下游工作需结合业务词频分布精细调参。
4. **GPT-4o合成咨询数据的可行性**：在Amazon等公开数据缺乏咨询链路时，用LLM基于用户画像与行为生成仿真咨询，可为动机建模研究提供低成本数据扩充路径（需注意LLM合成偏差）。
5. **扩展至多轮咨询建模**：当前仅用第一轮咨询历史，可探索多轮对话中动机演化（如用户先问"续航久吗"→再问"充电快吗"→最终搜某型号），结合对话状态追踪(DST)与动机累积编码。

## 关键术语表
**MAPS (Motivation-Aware Personalized Search)**：动机感知个性化搜索框架，通过对齐用户咨询对话中的搜索动机提升检索与排序效果。
**MoAE (Mixture of Attention Experts)**：混合注意力专家池化网络，由参数化、自注意力、搜索中心交叉注意力三类专家组成，通过门控选择Top-K专家加权输出文本嵌入。
**General Alignment**：通用对齐，通过双向对比损失在共享语义空间中拉近token与商品ID，桥接自然语言与世界知识鸿沟。
**Personalized Alignment**：个性化对齐，以当前查询为锚，通过双向Transformer从咨询历史与查询历史中提取动机感知嵌入，并与用户偏好融合。
**Search-Centered Cross-Attention Pooling**：搜索中心交叉注意力池化，以当前查询文本作为Query，使咨询/评论等长文本聚焦于与查询相关的语义片段。
**Consultation-Search Linkage**：咨询-搜索关联，指用户在发起搜索前进行的AI客服对话，蕴含潜在搜索动机，是本文核心信号源。
**InfoNCE Loss**：信息噪声对比损失，本文用于个性化对齐阶段，最大化正样本对相似性、最小化负样本对相似性。
**Token-Item Mapping**：token-商品映射，基于共现频率统计构建的词-商品关联表，用于通用对齐阶段的对比学习正样本构造。

## 可复现要素
- **数据集**：Commercial数据集为内部电商数据，未公开；Amazon数据集基于Amazon Reviews (Ni et al., 2019) 与PersonalWAB处理版 (Cai et al., 2024)，咨询文本由GPT-4o生成。
- **代码/权重**：代码开源在 https://github.com/E-qin/MAPS，补充材料同上。
- **关键超参**：$d=64$（ID维度）、$d_t=32$（文本维度）、序列最大长度30、映射阈值$t=3$、激活函数tanh（默认）、温度参数$\tau_1,\tau_2$与权重$\lambda_1,\lambda_2,\lambda_3,\lambda_4$（论文未列具体值，见附录B.2）；负样本数99。
