# PRD：Multi-Annotator Sentiment（hw0-3）

版本 1.0 · 2026-11-02

## 1. 背景与目标

课程作业 hw0-3：做一个三人情感标注流程。句子来自数据集，三位标注员各给一个标签，结果存进 SQLite 且不可修改；再用已保存的标注计算一致率、训练分类器并展示报告。

形式为 Tkinter 桌面程序，与作业样例界面一致：一个标注窗口，一个 Stats & Classifier 窗口。

## 2. 数据现状

`multi_annotator_dataset_list.csv` 共 1000 行，列为 `id, text, annotations`，`annotations` 是长度为 3 的 JSON 数组。

| 指标 | 值 |
| --- | --- |
| 行数 / 不同句子 | 1000 / 20 |
| 三人一致 / 两人一致 / 三人各不同 | 685 / 210 / 105 |
| 多数票标签分布 | positive 250 · sarcastic 227 · negative 212 · neutral 206 |

由此得出两点设计约束：

- 每句话平均出现约 50 次，且同一句的标签不稳定（例如 "I hate how it performs." 有整行标成 positive）。按句子取多数标签，准确率上限约 80%，分类器报告需如实说明这一点。
- 105 行三人各选一个标签，不存在多数票。参考实现按字母序硬选一个，结果没有意义；本项目把这类记录判为 `tie`。

## 3. 术语

| 术语 | 定义 |
| --- | --- |
| 记录 | 一句话 + 三位标注员的标签，对应表中一行。导入的 CSV 行和界面保存的标注都是记录 |
| Resolved label | 至少两人一致时取该标签，否则为 `tie` |
| Agreement | 多数票数 ÷ 3，取值为 100%、66.67% 或 33.33% |
| 平均一致率 | 所有记录 Agreement 的平均 |

## 4. 需求

优先级：P0 为作业必需；P1 让结果可信、好用；P2 为锦上添花。

### 4.1 数据（D）

| ID | 优先级 | 需求 |
| --- | --- | --- |
| D-1 | P0 | `python import_data.py` 把 CSV 导入 `annotations.db`；重复运行不产生重复行，并打印导入行数和跳过行数 |
| D-2 | P0 | 表 `annotations(id, text, annotator_1..3, source, created_at)`，标签限定为四个值（CHECK 约束），`source` 为 `import` 或 `ui` |
| D-3 | P0 | 不可修改：SQLite 触发器拒绝对 `annotations` 的任何 UPDATE 和 DELETE，以及复用已有 id 的 INSERT（防 `INSERT OR REPLACE` 覆盖） |
| D-4 | P1 | 首次启动 `app.py` 时如果库为空，自动导入 CSV，免去单独运行导入的步骤 |
| D-5 | P1 | 导入时校验每一行，标签不在四个值内或数组长度不是 3 的行报出行号并整体失败，不写入半份数据 |

### 4.2 标注界面（U）

| ID | 优先级 | 需求 |
| --- | --- | --- |
| U-1 | P0 | 显示一句从数据库随机抽取的句子；"Generate sentence" 换一句，且不与当前句重复 |
| U-2 | P0 | Annotator 1/2/3 各一个下拉框，选项为 Positive、Neutral、Negative、Sarcastic，初始为未选 |
| U-3 | P0 | 三个下拉框都选好之前，"Save annotation" 不可点 |
| U-4 | P0 | 保存后写入一条新记录（`source=ui`），该句的下拉框和保存按钮锁定，提示已保存的记录号；只能换下一句继续 |
| U-5 | P1 | 状态栏提示当前步骤和还缺哪几位标注员；显示本次会话已保存条数 |
| U-6 | P1 | 统计窗口已打开时，保存后自动刷新统计 |

### 4.3 统计与分类器（S）

| ID | 优先级 | 需求 |
| --- | --- | --- |
| S-1 | P0 | 表格列出全部记录：Id、Text、三位标签、Resolved、Prediction、Confidence、Agreement、Source |
| S-2 | P0 | 顶部汇总：总记录数、不同句子数、平均一致率、分类器准确率 |
| S-3 | P0 | 用已保存的标注（排除 `tie`）训练分类器：TF-IDF（1–2 gram）+ 逻辑回归，分层 80/20 划分，显示 sklearn classification report |
| S-4 | P0 | "Rerun classifier" 按钮：重新读库、重新训练、刷新表格和报告 |
| S-5 | P1 | 报告附加"未见句子准确率"：按句子分组做 5 折 GroupKFold，测的是对没见过的句子的效果，并在报告中说明它与随机划分准确率的差别 |
| S-6 | P1 | 随机种子可调（默认 42），Rerun 使用当前种子，便于观察划分带来的波动 |
| S-7 | P1 | "By sentence" 分页：每个句子的记录数、各标签票数、多数标签、平均一致率 |
| S-8 | P1 | 表格按 Source 筛选（All / Imported / New），新标注一眼可见 |
| S-9 | P2 | "Export CSV"：导出每条记录的 Resolved、Agreement、Prediction、Confidence |

### 4.4 质量（Q）

| ID | 优先级 | 需求 |
| --- | --- | --- |
| Q-1 | P0 | pytest 覆盖导入、幂等、不可修改、多数票与 tie、分类器、保存按钮启用逻辑 |
| Q-2 | P1 | README：安装、运行、设计说明；仓库公开到 GitHub |
| Q-3 | P1 | 提交文档（Word）：需求对照、运行截图、关键代码 |

## 5. 非目标

- 不做标注员账号或登录，三位标注员在同一界面选择。
- 不做网页版。
- 不改动已保存的标注，包括撤销上一次保存。

## 6. 数据模型

```sql
CREATE TABLE annotations (
  id          INTEGER PRIMARY KEY,
  text        TEXT NOT NULL,
  annotator_1 TEXT NOT NULL CHECK (annotator_1 IN ('positive','neutral','negative','sarcastic')),
  annotator_2 TEXT NOT NULL CHECK (...),
  annotator_3 TEXT NOT NULL CHECK (...),
  source      TEXT NOT NULL CHECK (source IN ('import','ui')),
  created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TRIGGER annotations_no_update BEFORE UPDATE ON annotations
  BEGIN SELECT RAISE(ABORT, 'annotations are immutable'); END;
CREATE TRIGGER annotations_no_delete BEFORE DELETE ON annotations
  BEGIN SELECT RAISE(ABORT, 'annotations are immutable'); END;
```

导入保留 CSV 的 `id`（`INSERT OR IGNORE`，因此幂等）；界面保存的记录使用自增 id，从 1001 开始。

## 7. 模块划分

```
db.py           建表、触发器、导入、插入、查询、随机句子
import_data.py  命令行导入
analysis.py     多数票、一致率、分类器、分组评估、按句汇总
app.py          Tkinter：AnnotatorApp + StatsWindow
tests/          pytest
```

`analysis.py` 不依赖 Tkinter，界面只负责展示，逻辑都可以单测。

## 8. 验收标准

1. 全新目录运行 `python app.py`：自动导入 1000 行，显示一句随机句子。
2. 只选两位标注员时保存按钮不可点；三位都选后可点；保存后记录号为 1001，下拉框锁定。
3. 用 sqlite3 对 `annotations` 执行 UPDATE 或 DELETE 均报 `annotations are immutable`。
4. Stats 窗口：总记录数 = 1000 + 新保存条数；平均一致率 86.00%；id 1 显示为 `tie`、33.33%。
5. 报告显示四类的 precision / recall / f1 / support 与准确率；点 Rerun 后包含刚保存的记录。
6. `pytest` 全部通过。

## 9. 里程碑

1. P0：D-1～D-3、U-1～U-4、S-1～S-4、Q-1
2. P1：D-4、D-5、U-5、U-6、S-5～S-8、Q-2、Q-3
3. P2：S-9

## 10. 完成情况

全部完成（2026-11-02）。`pytest` 40/40 通过。

| ID | 状态 | 实现 / 验证 |
| --- | --- | --- |
| D-1 | ✅ | `import_data.py`；`test_import_loads_every_row`、`test_import_is_idempotent`、`test_import_script` |
| D-2 | ✅ | `db.SCHEMA`，CHECK 约束；`test_check_constraint_guards_raw_inserts` |
| D-3 | ✅ | 三个触发器。开发中发现 `INSERT OR REPLACE` 删除旧行时不触发 DELETE 触发器，补了 `annotations_no_replace`；`test_records_are_immutable`、`test_replace_cannot_overwrite` |
| D-4 | ✅ | `db.ensure_imported`，`app.main` 启动时调用 |
| D-5 | ✅ | 先校验全部行再写入；`test_bad_csv_imports_nothing` |
| U-1 | ✅ | `db.random_sentence(exclude=当前句)`；200 次抽样覆盖全部 20 句且不连续重复 |
| U-2～U-4 | ✅ | `AnnotatorApp`；`test_save_needs_all_three_annotators`、`test_saving_locks_the_record` |
| U-5、U-6 | ✅ | 状态栏列出缺哪几位标注员；保存后自动刷新已打开的统计窗口 |
| S-1～S-4 | ✅ | `StatsWindow`；`test_stats_window` |
| S-5 | ✅ | GroupKFold 未见句子准确率；另加"本划分最好可能准确率"（测试集内每句取最多标签），`test_accuracy_never_beats_the_ceiling` 10 个种子验证 |
| S-6～S-8 | ✅ | Seed 输入框、By sentence 分页、Source 筛选 |
| S-9 | ✅ | `export_csv` |
| Q-1～Q-3 | ✅ | 40 个测试；README；提交文档 `submission/hw0-3.docx` |

实测结果（seed 42）：平均一致率 86.00%，tie 105 条，分类器准确率约 78%，等于本划分的最好可能值；未见句子准确率约 13%。
