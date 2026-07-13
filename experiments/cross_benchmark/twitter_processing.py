import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score


# =======================
# ✅ 配置区（全写死）
# =======================
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DATA_ROOT = os.environ.get("SHARP_TWITTER_RAW", str(REPO_ROOT / "data" / "twitter15_16" / "raw"))
OUT_CSV = os.environ.get(
    "SHARP_TWITTER_OUT",
    str(REPO_ROOT / "data" / "twitter15_16" / "processed" / "twitter15_16_best_eventlabel.csv"),
)

# 真假标签：只保留 true/false
DROP_UNLABELED = True
POSITIVE_IS_FALSE = True   # false->1(fake), true->0(real)

# BERT 向量
BERT_PATH = os.environ.get("SHARP_BERT_MODEL", "bert-base-uncased")
BERT_BATCH_SIZE = 32
BERT_MAX_LEN = 128

# 自动选K（你现在1154条，建议K别太大）
K_CANDIDATES = [8, 10, 12, 15, 18, 20, 25, 30]

# 小簇阈值：小于它的簇统一并入 OTHER
MIN_CLUSTER_SIZE = 20

# silhouette计算用的采样数（加速）
SIL_SAMPLE = 900

SEED = 42

# 输出额外统计
OUT_COUNTS_CSV  = OUT_CSV.replace(".csv", "_counts.csv")
OUT_SUMMARY_TXT = OUT_CSV.replace(".csv", "_summary.txt")


# =======================
# 工具函数
# =======================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def clean_text(s: str) -> str:
    # 轻量清洗：去多余空白（你也可以在这里扩展去URL/@等）
    s = (s or "").replace("\t", " ").replace("\n", " ")
    s = " ".join(s.split())
    return s

def load_split(split_dir: str, split_name: str) -> pd.DataFrame:
    label_path = os.path.join(split_dir, "label.txt")
    src_path = os.path.join(split_dir, "source_tweets.txt")
    if not os.path.exists(label_path): raise FileNotFoundError(label_path)
    if not os.path.exists(src_path): raise FileNotFoundError(src_path)

    labels = []
    with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            raw_label, tid = line.split(":", 1)
            labels.append((tid.strip(), raw_label.strip().lower()))

    texts = []
    with open(src_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            tid, content = parts[0].strip(), clean_text(parts[1].strip())
            texts.append((tid, content))

    lab_df = pd.DataFrame(labels, columns=["id", "raw_label"])
    txt_df = pd.DataFrame(texts, columns=["id", "content"])
    df = lab_df.merge(txt_df, on="id", how="left")
    df["split"] = split_name
    return df

def make_veracity_binary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["if_marked_label"] = df["raw_label"].isin(["true", "false"]).astype(int)

    mapping = {"false": 1, "true": 0} if POSITIVE_IS_FALSE else {"false": 0, "true": 1}
    df["label"] = df["raw_label"].map(mapping)

    if DROP_UNLABELED:
        df = df[df["if_marked_label"] == 1].reset_index(drop=True)
        df["label"] = df["label"].astype(int)
    else:
        df["label"] = df["label"].fillna(-1).astype(int)

    df["content"] = df["content"].fillna("").astype(str)
    df = df[df["content"].str.len() > 0].reset_index(drop=True)
    return df

def compute_bert_embeddings(texts, bert_path, batch_size=32, max_length=128, device=None):
    import torch
    from transformers import AutoTokenizer, AutoModel

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    model = AutoModel.from_pretrained(bert_path)
    model.to(device)
    model.eval()

    all_emb = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc=f"BERT embed ({device})"):
            batch = texts[i:i+batch_size]
            enc = tokenizer(
                batch, padding=True, truncation=True,
                max_length=max_length, return_tensors="pt"
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)

            last = out.last_hidden_state  # [B,T,H]
            mask = enc["attention_mask"].unsqueeze(-1)  # [B,T,1]
            summed = (last * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1)
            emb = (summed / denom).cpu().numpy()  # mean pooling
            all_emb.append(emb)

    X = np.vstack(all_emb).astype(np.float32)

    # ✅ L2 normalize：聚类更接近 cosine 行为
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.clip(norm, 1e-12, None)
    return X

def cluster_stats(labels: np.ndarray):
    counts = pd.Series(labels).value_counts().sort_index()
    total = int(counts.sum())
    uniq = int(counts.shape[0])
    vals = counts.values.astype(int)

    if uniq == 0:
        return counts, {"uniq": 0, "total": total, "median": 0, "min": 0, "max": 0, "tail_ratio": 1.0}

    tail = int((vals < MIN_CLUSTER_SIZE).sum())
    tail_ratio = tail / uniq

    return counts, {
        "uniq": uniq,
        "total": total,
        "median": int(np.median(vals)),
        "min": int(vals.min()),
        "max": int(vals.max()),
        "tail_ratio": float(tail_ratio),
    }

def pick_best_k(X: np.ndarray):
    n = X.shape[0]
    sample_idx = np.arange(n)
    if n > SIL_SAMPLE:
        rng = np.random.RandomState(SEED)
        sample_idx = rng.choice(n, size=SIL_SAMPLE, replace=False)

    best = None
    log_lines = []
    for k in K_CANDIDATES:
        km = MiniBatchKMeans(
            n_clusters=k, random_state=SEED,
            batch_size=1024, n_init="auto"
        )
        labels = km.fit_predict(X)

        counts, st = cluster_stats(labels)

        # silhouette 用 cosine（在 L2 normalize 后更稳定）
        try:
            sil = silhouette_score(X[sample_idx], labels[sample_idx], metric="cosine")
        except Exception:
            sil = -1.0

        # ✅ 训练友好目标：sil 越大越好；尾部越少越好；median 太小也惩罚
        # 你可以认为这是“稳定域标签”的评分
        median_penalty = 0.0 if st["median"] >= MIN_CLUSTER_SIZE else (MIN_CLUSTER_SIZE - st["median"]) / MIN_CLUSTER_SIZE
        score = sil - 0.8 * st["tail_ratio"] - 0.4 * median_penalty

        log_lines.append(
            f"K={k:<3d} sil={sil:.4f} uniq={st['uniq']:<3d} "
            f"max/med/min={st['max']}/{st['median']}/{st['min']} tail_ratio(<{MIN_CLUSTER_SIZE})={st['tail_ratio']:.3f} score={score:.4f}"
        )

        if best is None or score > best["score"]:
            best = {"k": k, "score": score, "sil": sil, "stats": st, "log": log_lines}

    return best, log_lines

def apply_other_bucket(labels: np.ndarray):
    # 小簇合并为 OTHER（新标签=K）
    counts = pd.Series(labels).value_counts()
    small = set(counts[counts < MIN_CLUSTER_SIZE].index.tolist())
    out = labels.copy()
    # OTHER label 设为当前最大label+1（通常等于K，但更安全）
    other_label = int(out.max()) + 1
    mask = np.isin(out, list(small))
    out[mask] = other_label
    return out, other_label

def save_summary(text: str):
    os.makedirs(os.path.dirname(os.path.abspath(OUT_SUMMARY_TXT)), exist_ok=True)
    with open(OUT_SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write(text + "\n")


# =======================
# 主流程
# =======================
def main():
    set_seed(SEED)

    # 1) 读数据
    t15 = load_split(os.path.join(DATA_ROOT, "twitter15"), "twitter15")
    t16 = load_split(os.path.join(DATA_ROOT, "twitter16"), "twitter16")
    df = pd.concat([t15, t16], ignore_index=True)
    df = make_veracity_binary(df)

    # 2) 向量
    X = compute_bert_embeddings(df["content"].tolist(), BERT_PATH, BERT_BATCH_SIZE, BERT_MAX_LEN)

    # 3) 自动选K（以“训练友好”为目标）
    best, log_lines = pick_best_k(X)
    best_k = best["k"]

    # 4) 训练最终聚类器
    km = MiniBatchKMeans(n_clusters=best_k, random_state=SEED, batch_size=1024, n_init="auto")
    labels = km.fit_predict(X).astype(int)

    # 5) 小簇并入 OTHER（让域分类训练稳定）
    labels2, other_label = apply_other_bucket(labels)
    df["event_label"] = labels2

    # 6) 输出 CSV
    out = df[["id", "label", "content", "if_marked_label", "event_label"]].copy()
    os.makedirs(os.path.dirname(os.path.abspath(OUT_CSV)), exist_ok=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    # 7) 输出计数表
    counts = out["event_label"].value_counts().sort_index()
    count_table = counts.reset_index()
    count_table.columns = ["event_label", "num_samples"]
    count_table.to_csv(OUT_COUNTS_CSV, index=False, encoding="utf-8-sig")

    # 8) summary
    uniq = int(counts.shape[0])
    total = int(counts.sum())
    vals = counts.values.astype(int)
    max_v, min_v, med_v = int(vals.max()), int(vals.min()), int(np.median(vals))

    summary = []
    summary.append("========== Auto-K Search Log ==========")
    summary.extend(log_lines)
    summary.append("")
    summary.append("========== Selected K (train-friendly) ==========")
    summary.append(f"Selected K = {best_k} (before OTHER)")
    summary.append(f"OTHER label = {other_label}  (clusters < {MIN_CLUSTER_SIZE} merged into OTHER)")
    summary.append("")
    summary.append("========== Final Cluster Stats (after OTHER merge) ==========")
    summary.append(f"Unique event_label: {uniq}")
    summary.append(f"Total samples: {total}")
    summary.append(f"Max/Median/Min per event_label: {max_v} / {med_v} / {min_v}")
    summary.append("")
    summary.append("========== Counts (first 40) ==========")
    summary.append(count_table.head(40).to_string(index=False))

    summary_text = "\n".join(summary)
    save_summary(summary_text)

    print(f"[OK] saved main CSV:     {OUT_CSV}")
    print(f"[OK] saved counts CSV:   {OUT_COUNTS_CSV}")
    print(f"[OK] saved summary TXT:  {OUT_SUMMARY_TXT}\n")
    print(summary_text)

if __name__ == "__main__":
    main()
