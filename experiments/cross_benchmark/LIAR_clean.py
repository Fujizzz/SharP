#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocess LIAR dataset for SharP/DAAL pipeline (binary classification, with subject->event_label mapping + auto Top-K).

功能：
1) 读取 LIAR train.tsv / valid.tsv / test.tsv
2) 6类标签 -> 二分类 (3真 vs 3假)
3) 基于 train split 的 subject 频次分布，自动选择 Top-K：
   - 满足 TARGET_COVERAGE 覆盖率（累计频次占比）
   - 且 Top-K 中最小类频次 >= MIN_COUNT
   - K 不超过 MAX_K
4) event_label 映射为：__UNK__ = 0, __OTHER__ = 1, Top-K subjects = 2..K+1
5) 输出：liar_clean_{train,valid,test}.csv + subject2event_label.json + subject_stats.csv
"""

import os
import json
import re
from pathlib import Path
import pandas as pd

# ===== 路径配置 =====
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
RAW_DIR = os.environ.get("SHARP_LIAR_RAW", str(REPO_ROOT / "data" / "liar" / "raw"))
OUT_DIR = os.environ.get("SHARP_LIAR_OUT", str(REPO_ROOT / "data" / "liar" / "processed"))

TRAIN_FILE = "train.tsv"
VALID_FILE = "valid.tsv"
TEST_FILE  = "test.tsv"

os.makedirs(OUT_DIR, exist_ok=True)

# ===== Top-K 自动选择策略参数 =====
TARGET_COVERAGE = 0.85   # 头部subject累计覆盖率目标（0~1）
MIN_COUNT = 20           # Top-K中最尾部那个类至少要有这么多样本（避免保留极长尾）
MAX_K = 200              # K上限（防止类别过多）
# subject 归一化方式：
#   "first": 多标签只取第一个subject（更适合单标签event/domain分类）
#   "set":   多标签按集合归一为一个字符串（组合会很多，域类数可能爆炸）
SUBJECT_MODE = "first"   # 推荐 first

LIAR_COLUMNS = [
    "id_json",
    "raw_label",
    "statement",
    "subject",
    "speaker",
    "job_title",
    "state_info",
    "party",
    "cnt_barely_true",
    "cnt_false",
    "cnt_half_true",
    "cnt_mostly_true",
    "cnt_pants_fire",
    "context"
]

NORMALIZE_LABEL_MAP = {
    "pants-fire": "pants-fire",
    "pants on fire": "pants-fire",
    "pants-on-fire": "pants-fire",

    "false": "false",
    "mostly-false": "false",

    "barely-true": "barely-true",
    "barely true": "barely-true",

    "half-true": "half-true",
    "half true": "half-true",

    "mostly-true": "mostly-true",
    "mostly true": "mostly-true",

    "true": "true"
}

BINARY_LABEL_MAP = {
    "pants-fire": 0,
    "false": 0,
    "barely-true": 0,
    "half-true": 1,
    "mostly-true": 1,
    "true": 1
}

UNK_SUBJECT = "__UNK__"
OTHER_SUBJECT = "__OTHER__"


def _split_subjects(s: str):
    """把原始 subject 字符串拆分为 token 列表（小写、去空、统一分隔符）。"""
    s = str(s).strip().lower()
    if not s:
        return []
    # ; | / -> ,
    s = re.sub(r"[;|/]+", ",", s)
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return parts


def normalize_subject(x) -> str:
    """
    规范化 subject 字段。
    SUBJECT_MODE="first": 多标签仅取第一个 token
    SUBJECT_MODE="set":   多标签按 set 去重+排序后 join
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return UNK_SUBJECT

    parts = _split_subjects(x)
    if not parts:
        return UNK_SUBJECT

    if SUBJECT_MODE == "first":
        return parts[0]
    elif SUBJECT_MODE == "set":
        parts = sorted(set(parts))
        return ",".join(parts)
    else:
        raise ValueError(f"Unknown SUBJECT_MODE={SUBJECT_MODE}, expected 'first' or 'set'")


def load_raw_split(path: str) -> pd.DataFrame:
    print(f"[INFO] Loading raw LIAR split from {path}")
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=LIAR_COLUMNS,
        encoding="utf-8"
    )

    # 标签清洗
    raw = df["raw_label"].astype(str).str.strip().str.lower()
    df["norm_label"] = raw.map(lambda lab: NORMALIZE_LABEL_MAP.get(lab, lab))

    # 过滤不在 6 类里的样本
    valid_labels = set(BINARY_LABEL_MAP.keys())
    before = len(df)
    df = df[df["norm_label"].isin(valid_labels)].copy()
    after = len(df)
    print(f"[INFO] Rows before label filter: {before}, after: {after}")

    # 二分类标签
    df["label_binary"] = df["norm_label"].map(BINARY_LABEL_MAP).astype(int)

    # subject 规范化
    df["subject_norm"] = df["subject"].apply(normalize_subject)

    return df


def analyze_subjects_and_choose_k(train_df: pd.DataFrame):
    """
    在 train 上统计 subject_norm 分布并自动选择 K。
    返回：
      best_k, top_subjects(list), stats_df(DataFrame with freq/cum_coverage)
    """
    freq = train_df["subject_norm"].astype(str).value_counts(dropna=False)
    total = int(freq.sum())

    stats = pd.DataFrame({
        "subject_norm": freq.index.astype(str),
        "count": freq.values.astype(int),
    })
    stats["ratio"] = stats["count"] / total
    stats["cum_count"] = stats["count"].cumsum()
    stats["cum_coverage"] = stats["cum_count"] / total
    stats["rank"] = range(1, len(stats) + 1)

    # 找到满足 coverage 的最小 k
    k_cov = int((stats["cum_coverage"] >= TARGET_COVERAGE).idxmax()) + 1 if (stats["cum_coverage"] >= TARGET_COVERAGE).any() else len(stats)

    # 找到满足 MIN_COUNT 的最大 k（即保留到 count>=MIN_COUNT 的最后一个）
    eligible = stats[stats["count"] >= MIN_COUNT]
    k_min_count = int(eligible["rank"].max()) if len(eligible) > 0 else 0

    # 综合：既要覆盖率达标，又要尾部不太小
    # 逻辑：先取 k_cov，但如果 k_cov > k_min_count（说明为了覆盖率被迫纳入很多小类）
    # 就把k收缩到 k_min_count；覆盖率会下降，但更利于稳定训练。
    if k_min_count == 0:
        # 全是长尾（或样本太少），退化：取 min(k_cov, MAX_K)
        best_k = min(k_cov, MAX_K)
        reason = "MIN_COUNT not satisfied by any class; fallback to coverage-only"
    else:
        best_k = min(k_cov, k_min_count, MAX_K)
        reason = "min(K for coverage, K for min_count, MAX_K)"

    top_subjects = stats.head(best_k)["subject_norm"].tolist()

    # 打印摘要
    coverage = float(stats.loc[best_k - 1, "cum_coverage"]) if best_k > 0 else 0.0
    kth_count = int(stats.loc[best_k - 1, "count"]) if best_k > 0 else 0
    uniq = len(stats)

    print("\n[SUBJECT STATS]")
    print(f"  SUBJECT_MODE      = {SUBJECT_MODE}")
    print(f"  unique subjects   = {uniq}")
    print(f"  total samples     = {total}")
    print(f"  TARGET_COVERAGE   = {TARGET_COVERAGE}")
    print(f"  MIN_COUNT         = {MIN_COUNT}")
    print(f"  MAX_K             = {MAX_K}")
    print(f"  chosen best_k     = {best_k}  ({reason})")
    print(f"  best_k coverage   = {coverage:.4f}")
    print(f"  count of K-th cls = {kth_count}")
    print("  top-10 subjects (subject_norm, count):")
    for i in range(min(10, len(stats))):
        print(f"    {i+1:02d}. {stats.loc[i, 'subject_norm']}  ({int(stats.loc[i, 'count'])})")
    print("")

    return best_k, top_subjects, stats


def build_subject2event_map_topk(top_subjects):
    """
    构建映射：
      __UNK__   -> 0
      __OTHER__ -> 1
      top_subjects -> 2..K+1（保持按频次顺序）
    """
    mapping = {UNK_SUBJECT: 0, OTHER_SUBJECT: 1}
    for i, subj in enumerate(top_subjects):
        # 避免 top_subjects 里恰好包含 __UNK__/__OTHER__
        if subj in mapping:
            continue
        mapping[subj] = len(mapping)
    return mapping


def convert_to_model_format(df: pd.DataFrame, subject2event: dict, top_set: set) -> pd.DataFrame:
    df = df.copy()

    # id
    df["id"] = df["id_json"].astype(str).str.replace(".json", "", regex=False)

    # content
    df["content"] = df["statement"].astype(str)

    # if_marked_label
    df["if_marked_label"] = 1

    # event_label: Top-K -> 映射；UNK -> 0；其它 -> OTHER(1)
    def map_event(subj_norm: str) -> int:
        subj_norm = str(subj_norm)
        if subj_norm == UNK_SUBJECT:
            return subject2event[UNK_SUBJECT]
        if subj_norm in top_set:
            return subject2event.get(subj_norm, subject2event[OTHER_SUBJECT])
        return subject2event[OTHER_SUBJECT]

    df["event_label"] = df["subject_norm"].apply(map_event).astype(int)

    out_df = df[["id", "label_binary", "content", "if_marked_label", "event_label"]].copy()
    out_df.rename(columns={"label_binary": "label"}, inplace=True)
    return out_df


def main():
    train_path = os.path.join(RAW_DIR, TRAIN_FILE)
    valid_path = os.path.join(RAW_DIR, VALID_FILE)
    test_path  = os.path.join(RAW_DIR, TEST_FILE)

    for p, name in [(train_path, "Train"), (valid_path, "Valid"), (test_path, "Test")]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{name} file not found: {p}")

    # 1) load
    train_raw = load_raw_split(train_path)
    valid_raw = load_raw_split(valid_path)
    test_raw  = load_raw_split(test_path)

    # 2) analyze train subjects, choose best_k
    best_k, top_subjects, stats_df = analyze_subjects_and_choose_k(train_raw)
    top_set = set(top_subjects)

    # 3) save subject stats
    stats_out_path = os.path.join(OUT_DIR, "subject_stats.csv")
    stats_df.to_csv(stats_out_path, index=False, encoding="utf-8")
    print(f"[INFO] Saved subject stats to {stats_out_path}")

    # 4) build mapping table
    subject2event = build_subject2event_map_topk(top_subjects)

    map_payload = {
        "meta": {
            "SUBJECT_MODE": SUBJECT_MODE,
            "TARGET_COVERAGE": TARGET_COVERAGE,
            "MIN_COUNT": MIN_COUNT,
            "MAX_K": MAX_K,
            "chosen_k": best_k,
            "num_event_classes": len(set(subject2event.values())),  # includes UNK & OTHER
        },
        "subject2event": subject2event
    }

    map_out_path = os.path.join(OUT_DIR, "subject2event_label.json")
    with open(map_out_path, "w", encoding="utf-8") as f:
        json.dump(map_payload, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved subject2event mapping to {map_out_path}")
    print(f"[INFO] event_label classes = {len(subject2event)} (including {UNK_SUBJECT}=0, {OTHER_SUBJECT}=1, TopK start from 2)")

    # 5) convert to model format
    train_clean = convert_to_model_format(train_raw, subject2event, top_set)
    valid_clean = convert_to_model_format(valid_raw, subject2event, top_set)
    test_clean  = convert_to_model_format(test_raw, subject2event, top_set)

    # 6) save
    train_out_path = os.path.join(OUT_DIR, "liar_aug_train.csv")
    valid_out_path = os.path.join(OUT_DIR, "liar_aug_valid.csv")
    test_out_path  = os.path.join(OUT_DIR, "liar_aug_test.csv")

    train_clean.to_csv(train_out_path, index=False, encoding="utf-8")
    valid_clean.to_csv(valid_out_path, index=False, encoding="utf-8")
    test_clean.to_csv(test_out_path, index=False, encoding="utf-8")

    print(f"[INFO] Saved cleaned train to {train_out_path}, rows = {len(train_clean)}")
    print(f"[INFO] Saved cleaned valid to {valid_out_path}, rows = {len(valid_clean)}")
    print(f"[INFO] Saved cleaned test  to {test_out_path}, rows = {len(test_clean)}")

    # 7) quick check distribution after mapping
    mapped_counts = train_clean["event_label"].value_counts().sort_index()
    print("\n[INFO] Train event_label distribution (first 20 labels):")
    for lab, cnt in mapped_counts.head(20).items():
        print(f"  event_label={int(lab):4d}  count={int(cnt)}")
    print(f"[INFO] Train __OTHER__ count = {int(mapped_counts.get(subject2event[OTHER_SUBJECT], 0))}")
    print(f"[INFO] Train __UNK__   count = {int(mapped_counts.get(subject2event[UNK_SUBJECT], 0))}\n")


if __name__ == "__main__":
    main()
