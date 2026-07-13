from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SEED_SIZE = 19
RANDOM_STATE = 42

SPECS = {
    "liar": (
        ROOT / "data" / "liar" / "processed" / "pool_data.xlsx",
        ROOT / "data" / "liar" / "prompt_seed" / "clippool_16_release.xlsx",
    ),
    "twitter15_16": (
        ROOT / "data" / "twitter15_16" / "processed" / "twitter_pool_data.xlsx",
        ROOT / "data" / "twitter15_16" / "prompt_seed" / "clippool_16_release.xlsx",
    ),
}

REQUIRED_COLUMNS = {
    "id",
    "label",
    "affection",
    "content",
    "if_marked_label",
    "event_label",
}


def build_seed(dataset: str, pool_path: Path, output_path: Path) -> None:
    pool = pd.read_excel(pool_path)
    missing = REQUIRED_COLUMNS.difference(pool.columns)
    if missing:
        raise SystemExit(f"[FAIL] {dataset} pool is missing columns: {sorted(missing)}")
    if len(pool) < SEED_SIZE:
        raise SystemExit(f"[FAIL] {dataset} pool has only {len(pool)} rows")

    seed = pool.sample(n=SEED_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seed.to_excel(output_path, index=False)

    reread = pd.read_excel(output_path)
    pool_keys = set(map(tuple, pool[list(REQUIRED_COLUMNS)].astype(str).to_numpy()))
    seed_keys = set(map(tuple, reread[list(REQUIRED_COLUMNS)].astype(str).to_numpy()))
    if len(reread) != SEED_SIZE or not seed_keys.issubset(pool_keys):
        raise SystemExit(f"[FAIL] {dataset} seed verification failed")

    print(
        f"[OK] {dataset}: {len(reread)} deterministic pool rows -> "
        f"{output_path.relative_to(ROOT)}"
    )


for name, (pool, output) in SPECS.items():
    build_seed(name, pool, output)
