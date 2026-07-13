# Data audit summary

All counts below were computed directly from the reorganized local bundle.

## PHEME

| event_label | Event | Raw rows | Pool rows |
|---:|---|---:|---:|
| 0 | Charlie Hebdo | 2,079 | 280 |
| 1 | Ferguson | 1,143 | 280 |
| 2 | Germanwings crash | 469 | 280 |
| 3 | Ottawa shooting | 888 | 280 |
| 4 | Sydney siege | 1,209 | 280 |

The raw files contain 5,788 rows before invalid-label filtering. Event ids are constant within each raw file and match the ordered list in `add_event_label(...)`.

The retained shared PHEME training snapshot targets event 2:

- `train_data.xlsx`: 5,397 rows, including 94 initially labeled Germanwings records.
- `train2_data.xlsx`: 5,677 rows, adding the 280-row Germanwings pool.
- `test_data.xlsx`: 94 Germanwings records.

### Prompt seed

| File | Rows | Exact matches in event-2 pool | Event id | Class |
|---|---:|---:|---:|---:|
| `clippool_16.xlsx` | 19 | 19 | 2 | all 0 |
| `clippool_50.xlsx` | 50 | 50 | 2 | all 0 |

`clippool_50.xlsx` is pool positions 1–50. The 19-row file contains pool positions 1–19 with one record moved to the first row. This is a selected slice, not a reproducible `pandas.sample(random_state=...)` result found in the code.

## LIAR prepared data

| File | Rows | Class 0 | Class 1 | Event ids |
|---|---:|---:|---:|---:|
| `train2_data.xlsx` | 10,240 | 4,488 | 5,752 | 48 |
| `pool_data.xlsx` | 280 | 143 | 137 | 48 |
| `validate_data.xlsx` | 1,284 | 616 | 668 | 47 |
| `test_data.xlsx` | 1,267 | 553 | 714 | 47 |

`train2 + validation + test = 12,791`, matching the paper's LIAR total. In an authorized local reconstruction, place the official-format `train.tsv`, `valid.tsv`, and `test.tsv` under `data/liar/raw/`.

## Twitter15/16 prepared data

The private audit observed 1,154 records in `twitter15_16_best_eventlabel.csv`: 579 in class 0 and 575 in class 1 across eight derived event/domain clusters. Raw Twitter15/16 source/tree files and generated XLSX/PKL tensors are not redistributed.

## Important reproducibility boundary

Dataset preprocessing outputs were overwritten in the historical generic directory as experiments changed. The documented local layout separates them. Only the event-2 PHEME prompt seed was evidenced as a historical file by exact row membership; LIAR and Twitter15/16 seeds must be reconstructed deterministically from authorized local pools.
