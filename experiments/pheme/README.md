# Historical PHEME snapshot

This directory preserves the clearest PHEME-specific implementation found in `DAAL_code/DAAL/` and aligns it with the retained event-2 Germanwings snapshot.

Run from this directory so its historical relative paths resolve:

```bash
cd experiments/pheme
python model_adjust.py
```

Set `SHARP_BERT_MODEL`, `SHARP_LLM_MODEL`, `SHARP_PHEME_EVENT=2`, and `SHARP_PROMPT_SEED=data/pheme/prompt_seed/clippool_16.xlsx` first. Machine-specific paths were replaced, CUDA selection is CPU-safe, and outputs are written under `outputs/pheme_event_2/`.

Prepare per-event pool/test/validation files for all five event ids from an authorized upstream dataset copy. The audited shared train/source artifacts and clippool belonged to event 2. Rebuild the shared artifacts before running another event. See `docs/RUN_REVIEW.md` and `docs/DATA_SUMMARY.md` before publishing reproduction claims.
