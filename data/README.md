# External data preparation

SharP evaluates low-resource fake-news detection on PHEME, LIAR, and Twitter15/16 and uses several independent lexical resources. Dataset text, prepared tensors, prompt seeds, and third-party lexicons are intentionally excluded from this public Git history because each resource has its own redistribution terms.

## Upstream sources

| Resource | Upstream reference | Expected local root |
| --- | --- | --- |
| PHEME | [PHEME rumour dataset](https://doi.org/10.6084/m9.figshare.4010619.v1) | `data/pheme/` |
| LIAR | [ACL Anthology paper and dataset reference](https://aclanthology.org/P17-2067/) | `data/liar/` |
| Twitter15/16 | Obtain from the original rumour-detection dataset maintainers and follow applicable platform terms | `data/twitter15_16/` |
| Lexicons | Obtain NRC and other lexical resources from their respective maintainers | `data/resources/lexicons/` |

## Expected layout

```text
data/
├── pheme/{raw,processed,prompt_seed}/
├── liar/{raw,processed,prompt_seed}/
├── twitter15_16/{raw,processed,prompt_seed}/
└── resources/lexicons/
```

The preprocessing entry points are under `experiments/pheme/` and `experiments/cross_benchmark/`. After preparing an authorized local copy, run:

```bash
python scripts/build_release_prompt_seeds.py
python scripts/smoke_check.py
```

The first command builds deterministic LIAR and Twitter15/16 release seeds from their local pools. The second verifies file structure, Python syntax, schemas, dataset totals, event mappings, and prompt-seed membership.

See [`../docs/DATA_POLICY.md`](../docs/DATA_POLICY.md) and [`../docs/DATA_SUMMARY.md`](../docs/DATA_SUMMARY.md) for provenance and audit boundaries.
