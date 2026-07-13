# Cross-benchmark snapshots

This directory preserves the later LIAR and Twitter15/16 adaptations used for the paper's Group B study.

- `LIAR_clean.py`: converts the official LIAR TSV splits to binary fake/real CSVs and maps subjects to event/domain ids.
- `twitter_processing.py`: merges Twitter15/16 source posts, converts labels to binary fake/real, and clusters BERT embeddings into event/domain ids.
- `process_data.py`: creates Twitter train/validation/test/pool tables and PyTorch pickle files.
- `model_adjust.py`: Twitter15/16 adaptation of the main training script.
- `run.py`: low-resource sweep launcher for the adaptation.

Raw and prepared data are expected under `data/liar/` and `data/twitter15_16/` but are not redistributed. The scripts use those locations by default; environment variables can override them.

These files originate from experiment-specific snapshots. For the runnable release, both Twitter generated-text insertion points are enabled and a deterministic 19-row seed sampled from the Twitter pool is provided. The seed and enabled state are release reconstructions, so they must not be described as the exact historical execution state without further author evidence.
