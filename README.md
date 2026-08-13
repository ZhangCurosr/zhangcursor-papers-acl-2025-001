# 📄 ACL 2025 会议论文库

ACL 2025（63rd Annual Meeting of the Association for Computational Linguistics）会议论文的解析归档。

<!-- 胶囊徽章带：论文数 / 最近更新 / 流水线 / License -->
![Papers](https://img.shields.io/badge/Papers-107-brightgreen?style=flat-square)
![Last commit](https://img.shields.io/github/last-commit/ZhangCurosr/zhangcursor-papers-acl-2025-001?style=flat-square)
![Pipeline](https://img.shields.io/github/actions/workflow/status/ZhangCurosr/paper-notes/mineru_batch.yml?label=daily%20pipeline&style=flat-square)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)

## 这是什么

- **内容**：ACL 2025 年会主会论文（含 Long / Short / Findings）。
- **来源**：https://aclanthology.org/events/acl-2025/
- **解析**：每篇论文经 MinerU 云端解析（版面还原 + 全文 Markdown + 图表提取）
- **归档**：论文标题命名目录，每篇包含 4 类完整产物（见下）

## 产物结构

```
{日期}/{论文标题}_{batch8}/
├── paper.pdf      # 原版 PDF
├── full.md        # 解析全文 Markdown
├── images/        # 论文中的图表（按哈希命名）
└── meta.json      # 元数据（来源 URL / 标题 / batch / 时间）
```

> 目录名 = 论文标题 slug + 提交批次短哈希，可直接按标题检索。

## 怎么用

无需安装。直接浏览本仓库目录树，或通过[总厂库索引](https://github.com/ZhangCurosr/zhangcursor-hub)按来源 / 标题 / 日期检索。

## 更新机制

由云端流水线自动维护（GitHub Actions，无需本地机器）：

| 时间 (UTC) | 动作 |
|---|---|
| 每天 02:00 / 14:00 | 自动抓取新论文（增量）→ MinerU 解析 → 推送本仓库 |
| 手动 | 仓库 Actions → “Dedup Repos” 一键去重 + 重建总厂库索引 |

## 相关仓库

- **[zhangcursor-hub](https://github.com/ZhangCurosr/zhangcursor-hub)** — 总厂库：统一索引全部论文
- **[paper-notes](https://github.com/ZhangCurosr/paper-notes)** — 流水线部署仓库（抓取 / 调度 / 同步代码）

## Limits

- 仅收录 MinerU 服务可达的论文源（本库为 ACL 2025）
- 单日抓取上限 250 篇，超出部分次日补抓
- 图表提取依赖 PDF 质量，个别论文可能无图
- 解析为机器生成结果，公式 / 表格偶有误差，请以原文为准

## License

MIT
