# 作者运行流程审阅清单

## 已核实的数据关系

- `event_label=0`：Charlie Hebdo
- `event_label=1`：Ferguson
- `event_label=2`：Germanwings crash
- `event_label=3`：Ottawa shooting
- `event_label=4`：Sydney siege
- `clippool_16.xlsx` 的 19 行全部来自 Germanwings 的 event-2 pool。
- `clippool_50.xlsx` 的 50 行也全部来自同一 pool。
- 代码在 `pm.load_soft_prompt(...)` 后读取 clippool，并把它传给 `pretrain_amodel(...)`，所以你的记忆正确：它用于 soft prompt 预训练。

## A. PHEME：当前最完整、数据互相一致的快照

```powershell
$env:SHARP_BERT_MODEL = "bert-base-uncased"
$env:SHARP_LLM_MODEL = "<原始LLM或兼容模型>"
$env:SHARP_PHEME_EVENT = "2"
$env:SHARP_PROMPT_SEED = "<仓库>\data\pheme\prompt_seed\clippool_16.xlsx"
cd experiments\pheme
python model_adjust.py
```

执行关系：

1. 读取五事件的共享训练数据；event 2 有 94 条初始标注目标域样本。
2. 读取 `event_2_pool.xlsx` 的 280 条 Germanwings 候选样本。
3. 读取 `event_2_test.pkl` 与 `event_2_validate.pkl`。
4. 预训练检测器。
5. 使用 19 行 clippool 预训练 25-token soft prompt。
6. 按熵选择边界样本，训练 soft prompt，调用 LLM 增强文本。
7. 将选中/生成样本加入运行时训练文件，移出 pool，微调并评估。

其他 event id 的 pool/test/validate 已打包，但共享 train/source 张量与 clippool 属于 event 2；不能直接改一个数字就宣称复现了其他四个事件。

## B. LIAR：完整方法代码 + LIAR 张量 + 发布版 prompt seed

```powershell
cd DAAL
python model_adjust.py --less-frac 0.10 --llm-model $env:SHARP_LLM_MODEL
```

主体流程完整，LIAR 的 train/pool/validation/test 张量也齐全。历史工作区没有发现 LIAR clippool；发布版因此从 LIAR 自身 pool 以 `random_state=42` 确定性抽取 19 行作为可运行 seed。它不是论文历史 seed 的证据，可通过 `SHARP_PROMPT_SEED` 覆盖。

## C. Twitter15/16：数据、预处理和发布版增强流程齐全

```powershell
python experiments\cross_benchmark\twitter_processing.py
python experiments\cross_benchmark\process_data.py
```

原始 Twitter15/16 tree/source 文件、聚类 CSV、XLSX 和 PKL 均已统一打包。发布版已恢复两处 LLM 生成文本注入，并从 Twitter pool 以 `random_state=42` 确定性抽取 19 行作为可运行 seed；二者属于发布重建，不应冒充论文历史执行状态。

## 上传前仍需要作者确认

- [ ] 原始因果 LLM 的准确 checkpoint/量化版本。
- [ ] 是否还能找回论文实验使用的 LIAR soft-prompt 历史 seed；当前仅有发布重建 seed。
- [ ] 是否还能找回论文实验使用的 Twitter15/16 soft-prompt 历史 seed；当前仅有发布重建 seed。
- [ ] PHEME 其余四个事件是否通过重新运行预处理脚本逐次覆盖共享 train/source 文件。
- [ ] GitHub 仓库名、公开/私有、代码许可证和数据再分发权限。

## 验证边界

离线检查能证明目录、语法、表格规模、事件映射和 clippool 包含关系；没有原始 CUDA 环境与 LLM checkpoint 时，不能把它等同于论文数值复现实验。
