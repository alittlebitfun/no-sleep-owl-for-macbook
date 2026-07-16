# Bosideng 统一 57 词多标签训练实施计划

**目标：** 在 8×H20 上完成一个可交付的 57 维服装字典词分类器。单次推理输出全部标签的两位小数概率，并可按阈值渲染为最终完整 JSON。

**总时限：** 从正式数据构建开始计时不超过 8 小时；训练与独立测试均计入。

## 全局约束

- 词表为现有 canonical56 加 `无袖`，固定为 57 维；标签顺序写入版本化 schema。
- `假两件` 当前没有可信正样本，保留输出维度并标记 unsupported；不得将它的未知项伪造为负例。
- JD 每张图仅监督其 23 个已标注维度；字典集监督已出现的正标签和明确互斥组内的可信负标签；其余维度保持 unknown。
- 损失函数只在 `known_mask=1` 的位置计算 masked BCE。任何 unknown 均不得计为 0。
- 完全相同或视觉相同的图片先聚合全部正标签，再进入切分。exact SHA 和 exact pHash 用于监督合并；pHash Hamming 距离 ≤2 的连通分量用于 split 隔离。
- 跨来源冲突遵循：任一来源有可信正例时保留正例；没有可信正例且来源相互冲突时改为 unknown，并记录审计表。
- 安全互斥组仅包含：领型 `立领/翻领/无领`、肩部结构 `插肩袖/正肩袖/落肩袖`、外廓形 `H型/O型/X型/A型/茧型/箱型`、松量 `合体/宽松`、衣长 `长款/中长款/中款/短款`。`弯袖`不加入肩部互斥组；压胶工艺不互斥。
- train/validation/test 以全局视觉连通分量为原子切分，目标比例 80/10/10，并按标签支持度近似分层。三份集合间不允许 exact SHA、exact pHash、pHash≤2 连通分量交叉。
- 训练使用 Qwen3-VL-8B-Instruct + LoRA + 57 维线性分类头，优先加载已验证 JD18 checkpoint 中可兼容的 LoRA 和 18 个分类头行。
- 输出概率使用 sigmoid 原值，序列化为 `0.00` 至 `1.00` 两位小数字符串；完整 JSON 必须含 57 个标签且标签顺序固定。
- 远程任务用 systemd-run 或等价的 durable supervisor；每 50 step 写进度，每 250 step 和训练结束保存可恢复检查点。

## 验收口径

交付结论按以下硬指标分级：

- **success：** JSON 合法率 100%；known-label test micro F1 ≥ 0.88；JD23 test micro F1 ≥ 0.88；有正负支持标签的 macro F1 ≥ 0.75；字典正例 macro recall ≥ 0.85；可信互斥负例 specificity ≥ 0.90；三份集合视觉泄漏为 0。
- **partial：** JSON 合法率 100%、视觉泄漏为 0，且 known-label micro F1 ≥ 0.82，但至少一项 success 指标未达标。
- **fail：** JSON 合法率低于 100%、存在切分泄漏、known-label micro F1 < 0.82、训练/评估未完成，或交付包无法复现。

所有指标同时报告 validation 和独立 test；阈值只允许在 validation 上校准，test 仅运行一次正式评估。

## Task 1：实现并测试统一数据构造器

**产物：**

- `scripts/build_unified57_masked_dataset.py`
- `tests/test_build_unified57_masked_dataset.py`
- `configs/bosideng_unified57_schema.json`

**测试先行：**

1. 同一视觉的多条字典记录聚合为一行，多标签全部为正。
2. JD 只打开 23 个 known mask，未覆盖维度保持 unknown。
3. 安全互斥组生成可信负例，`弯袖`和压胶标签不被互斥逻辑误伤。
4. 跨源正负冲突时正例优先；纯冲突降为 unknown。
5. exact pHash 合并，pHash≤2 组件只用于切分隔离。
6. train/validation/test 无视觉组件交叉。
7. 每行 labels、known_mask 均严格为 57 维，schema 哈希一致。

## Task 2：构造全量数据并完成训练前审计

**输入：**

- JD 53,952 张有效去 SHA 重图及完整 23 维标注。
- 字典 Base 1,841 条记录，含 1,809 条 canonical56 与 32 条 `无袖`。

**产物：**

- `datasets/bosideng_unified57_v1/{train,val,test}.jsonl`
- `datasets/bosideng_unified57_v1/dataset_summary.json`
- `datasets/bosideng_unified57_v1/conflicts.csv`
- `datasets/bosideng_unified57_v1/leakage_check.json`
- `evaluation_results/unified57_data_audit/representative_samples.md`

**硬检查：** 记录数守恒、源数据覆盖率、每标签正/负/未知计数、跨源冲突数、重复组数、视觉泄漏为 0、所有图片可解码。发现任何硬检查失败时停止正式训练并修复。

## Task 3：实现 57 维迁移训练与冒烟测试

**产物：**

- `scripts/train_unified57_qwen3vl_multilabel.py`
- `tests/test_unified57_checkpoint_transfer.py`
- 远程 run 配置与启动脚本

**要求：**

1. 读取统一 schema 与 masked dataset。
2. 从 JD18 checkpoint 迁移 LoRA；按标签名迁移 18 个分类头行，其余 39 行正常初始化。
3. pos_weight 仅按训练集已知正负计数计算并设置上限，避免稀有词产生极端梯度。
4. 先运行 20 step 冒烟，验证 8 卡、loss 有限、梯度有限、吞吐满足时限、checkpoint 可恢复。
5. 记录总监督位数、每来源采样数、每标签实际参与 loss 的次数。

## Task 4：8 卡正式训练

**时间预算：** 数据构建与审计 ≤75 分钟；冒烟与修复 ≤30 分钟；正式训练 ≤5 小时；校准、独立测试和打包 ≤75 分钟。

**训练策略：**

- BF16 DDP，8×H20；以实测吞吐设置每卡 batch 和累积步数。
- 一个 epoch 上限覆盖完整训练视觉组；允许在 validation micro F1 连续两次无提升时提前停止。
- 训练预算优先保证每个 JD 图至少一次覆盖；字典样本采用标签均衡 sampler 提高长尾监督频次，同时记录实际重复次数。
- 所有 checkpoint、日志、进度和退出码保存到独立 run 目录。

## Task 5：阈值校准、独立测试与交付

**产物：**

- validation 上逐标签阈值与统一回退阈值。
- validation/test 的总指标、JD23 子集指标、字典正例指标、可信负例指标、逐标签指标及错误案例。
- 全 57 标签概率 JSON、selected-only JSON、最终提示词/输入输出契约。
- LoRA + 57 维分类头轻量交付包、推理脚本、README、SHA256 与复现验证文件。

**最终验证：**

1. 随机 32 张图从交付包重跑，与归档概率逐值比对。
2. 展示至少 6 张有代表性的 JD/字典图片、真值、57 维预测摘要和错误说明。
3. 确认训练与评估进程退出，汇报 node1 GPU 占用和最终 run 目录。

