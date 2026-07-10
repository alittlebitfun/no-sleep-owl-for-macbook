# DesignGPT 量化文案 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 交付一版 97 字造物云宣传文案、内部工时统计依据，以及 2024—2026 年 IP 设计和家具设计参考测算表。

**Architecture:** 使用一份 Markdown 交付文件集中保存最终文案、统计口径、测算表和发布边界。通过字符计数与公式复算校验文案长度和全部百分比；公开来源仅用于说明平台背景及广告数据合规要求，参考测算值明确标注为内部建模数据。

**Tech Stack:** Markdown、Python 3 标准库、市场监管总局公开法规、造物云官网公开信息。

## Global Constraints

- 正文字符数与原文一致，均为 97 个字符，换行不计。
- 保留“国内领先”“90% 以上”“2/3”三个传播点，并给出明确限定范围。
- 参考对象为中国中小型实体设计团队，团队模型为 8 人。
- 90% 以上仅描述概念方案生成环节的人时降幅。
- 2/3 仅描述前期设计周期降幅。
- 2024—2026 年数字统一标注为参考测算值。
- 正式广告发布前须以真实工时台账和国内同类平台对测结果替换参考测算。
- 文案及说明统一使用直接、清晰的肯定句。

---

### Task 1: 生成等字数文案与统计依据

**Files:**
- Create: `deliverables/designgpt-metrics-evidence.md`

**Interfaces:**
- Consumes: `docs/superpowers/specs/2026-07-10-designgpt-metrics-copy-design.md` 中的口径和验收标准。
- Produces: 标记为 `FINAL_COPY_START` 与 `FINAL_COPY_END` 的 97 字正文，以及可直接用于内部材料的统计依据。

- [ ] **Step 1: 写入正文**

在交付文件中写入以下两行，换行不计入字符数：

```text
造物云是国内领先的AI商品概念设计平台,造物GPT(DesignGPT)为实体企业组建专业商品创新团队
概念方案生成效率提升90%以上,前期设计周期缩短2/3,以高效产品创新推动企业业务持续增长。
```

- [ ] **Step 2: 写入内部工时统计依据**

统计依据须覆盖统计对象、团队规模、项目阶段、对照方法、公式、证据材料、排除范围和发布条件，并明确当前表格属于参考测算。

- [ ] **Step 3: 校验字符数**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
text = Path('deliverables/designgpt-metrics-evidence.md').read_text()
copy = text.split('<!-- FINAL_COPY_START -->', 1)[1].split('<!-- FINAL_COPY_END -->', 1)[0]
copy = ''.join(line for line in copy.splitlines() if line.strip())
assert len(copy) == 97, len(copy)
print('copy_chars=97')
PY
```

Expected: `copy_chars=97`

### Task 2: 生成两张参考测算表

**Files:**
- Modify: `deliverables/designgpt-metrics-evidence.md`

**Interfaces:**
- Consumes: 8 人团队模型、工时降幅与周期降幅公式。
- Produces: IP 设计与家具设计两张 2024—2026 年度参考测算表。

- [ ] **Step 1: 写入 IP 设计表**

使用以下逐项目基准：2024 年 64→6.4 人时、15→5 天；2025 年 64→5.8 人时、15→4.5 天；2026 年 64→5.1 人时、15→4 天。参考项目数依次为 24、36、48。

- [ ] **Step 2: 写入家具设计表**

使用以下逐项目基准：2024 年 96→9.6 人时、21→7 天；2025 年 96→8.6 人时、21→6.3 天；2026 年 96→7.7 人时、21→5.6 天。参考项目数依次为 18、27、36。

- [ ] **Step 3: 复算两表**

Run:

```bash
python3 - <<'PY'
rows = [
    ('IP-2024', 24, 64, 6.4, 15, 5, 90.0, 66.7, 1382.4),
    ('IP-2025', 36, 64, 5.8, 15, 4.5, 90.9, 70.0, 2095.2),
    ('IP-2026', 48, 64, 5.1, 15, 4, 92.0, 73.3, 2827.2),
    ('家具-2024', 18, 96, 9.6, 21, 7, 90.0, 66.7, 1555.2),
    ('家具-2025', 27, 96, 8.6, 21, 6.3, 91.0, 70.0, 2359.8),
    ('家具-2026', 36, 96, 7.7, 21, 5.6, 92.0, 73.3, 3178.8),
]
for name, n, old_h, new_h, old_d, new_d, expected_h, expected_d, expected_saved in rows:
    h = round((old_h - new_h) / old_h * 100, 1)
    d = round((old_d - new_d) / old_d * 100, 1)
    saved = round((old_h - new_h) * n, 1)
    assert (h, d, saved) == (expected_h, expected_d, expected_saved), (name, h, d, saved)
print('metric_rows=6 verified')
PY
```

Expected: `metric_rows=6 verified`

### Task 3: 补齐来源与发布边界

**Files:**
- Modify: `deliverables/designgpt-metrics-evidence.md`

**Interfaces:**
- Consumes: 市场监管总局《广告引证内容执法指南》与造物云官网“关于我们”页面。
- Produces: 来源说明、领先对标条件、参考测算声明及正式发布检查单。

- [ ] **Step 1: 标注公开来源的证明范围**

造物云官网只用于支持平台背景、团队背景与服务企业数量，不用于证明“国内领先”“90% 以上”或“2/3”。

- [ ] **Step 2: 写入领先对标条件**

要求对不少于 6 个国内同类平台进行同任务对照，记录首个可用方案时间、8 小时方案数、专家盲评通过率、人工返工工时和流程覆盖率。

- [ ] **Step 3: 写入正式发布检查单**

发布前应具备原始项目简报、人员工时记录、平台日志、版本记录、交付文件、验收记录和竞品对照测试底稿。

- [ ] **Step 4: 运行交付检查**

Run:

```bash
rg -n "参考测算|正式发布|不少于 6 个|FINAL_COPY_START|FINAL_COPY_END|IP 设计|家具设计" deliverables/designgpt-metrics-evidence.md
git diff --check -- deliverables/designgpt-metrics-evidence.md
```

Expected: 所有关键词均命中，`git diff --check` 无输出。

### Task 4: 提交交付文件

**Files:**
- Create: `deliverables/designgpt-metrics-evidence.md`

**Interfaces:**
- Consumes: 已通过字符数、计算与来源边界校验的交付文件。
- Produces: 可审阅的 Git 提交。

- [ ] **Step 1: 提交文件**

Run:

```bash
git add deliverables/designgpt-metrics-evidence.md docs/superpowers/plans/2026-07-10-designgpt-metrics-copy.md
git commit -m "docs: add DesignGPT metrics evidence copy"
```

Expected: 提交成功，且只包含计划文件和交付文件。
