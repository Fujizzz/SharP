# Version-selection audit

The author confirmed that the code was edited to accommodate PHEME, LIAR, and Twitter15/16 formats. The retained workspace therefore represents successive dataset-specific states rather than one universal runner.

## Core augmentation implementation

`DAAL/model_adjust.py` comes from `DAAL/model_adjust_true.py` because it contains the most complete method flow found:

- `--less-frac` label-budget control;
- entropy-based active selection;
- soft-prompt pre-training and fine-tuning;
- hard/soft prompt LLM generation;
- active generated-text insertion;
- detector validation and test loops.

Its retained generic prepared artifacts are LIAR: 12,791 total records and 48 subject/event domains. However, its hardcoded `clippool_16.xlsx` was a PHEME Germanwings seed, showing that the generic working directory was overwritten/mixed between experiments. The release replaces this unsafe default with a deterministic 19-row seed sampled from the LIAR pool. This supports execution but is not claimed as the historical seed.

## PHEME version

`experiments/pheme/model_adjust.py` originates from the older `DAAL_code/DAAL/model_adjust.py` with augmentation active. It has been aligned to the retained event-2 snapshot because:

- raw Germanwings rows have `event_label=2` and `if_marked_label=0`;
- shared `train_data.xlsx` contains 94 marked event-2 records;
- shared `train2_data.xlsx` adds a 280-row unmarked event-2 pool;
- `test_data.xlsx` is event 2;
- both clippool workbooks are exact subsets of the event-2 pool.

Per-event pool/test/validation artifacts for ids 0–4 are retained. The shared source/train tensors and prompt seed are event-2-specific, so the other four full runs must be regenerated from raw data rather than mixed with these shared files.

## Twitter15/16 version

The workspace's later `DAAL/model_adjust.py` is retained at `experiments/cross_benchmark/model_adjust.py` because it loads `twitter_*` artifacts and preserves the label-budget interface.

Its recovered snapshot had two generated-text insertion calls commented out, and no historical Twitter-specific prompt-seed workbook was found. The release re-enables the matching augmentation calls and supplies a deterministic 19-row seed from the Twitter pool. This is a runnable reconstruction, not evidence that the paper used this exact seed or source state.

## `DAAL/amodel.py`

The latest active workspace implementation was selected. Historical commented duplicates were removed while retaining the 25-token soft prompt, frozen LLM backbone, hard/soft prompt composition, generation, BERT alignment, and numerical-stability handling.

## Excluded intermediates

`fixed_model_adjust.py`, `model_adjust_fixed.py`, `updated_model_adjust.py`, and similarly named files are repair/intermediate snapshots. `model_adjust_updated_01.py` is empty. They add ambiguity without providing a clearer three-dataset reproduction path.
