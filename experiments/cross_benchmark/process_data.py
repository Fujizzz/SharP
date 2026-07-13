import os
import pickle
from pathlib import Path
import sys
import numpy as np
import torch
import random
from transformers import BertModel, BertTokenizer
from idea1_features.building_features import manual_features
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(34075)

# ===== 全局：用于保证 train/pool 不重叠 =====
GLOBAL_POOL_IDX = None   # 存放从 train 中抽出的 pool 的行索引（基于 reset_index 后的 index）


# ==========================
# ✅ Twitter 配置（避免覆盖 LIAR）
# ==========================
TWITTER_CSV_PATH = os.environ.get(
    "SHARP_TWITTER_CSV",
    str(REPO_ROOT / "data" / "twitter15_16" / "processed" / "twitter15_16_best_eventlabel.csv"),
)
OUT_DIR = os.environ.get(
    "SHARP_TWITTER_PROCESSED",
    str(REPO_ROOT / "data" / "twitter15_16" / "processed"),
)
os.makedirs(OUT_DIR, exist_ok=True)

XLSX_POOL   = os.path.join(OUT_DIR, "twitter_pool_data.xlsx")
XLSX_TRAIN  = os.path.join(OUT_DIR, "twitter_train_data.xlsx")
XLSX_TRAIN2 = os.path.join(OUT_DIR, "twitter_train2_data.xlsx")
XLSX_VALID  = os.path.join(OUT_DIR, "twitter_validate_data.xlsx")
XLSX_TEST   = os.path.join(OUT_DIR, "twitter_test_data.xlsx")

PKL_SOURCE        = os.path.join(OUT_DIR, "twitter_source_data.pkl")
PKL_SOURCE_POOL   = os.path.join(OUT_DIR, "twitter_source_pool_data.pkl")
PKL_SOURCE_EXTEND = os.path.join(OUT_DIR, "twitter_source_extend_data.pkl")
PKL_DESTINATION   = os.path.join(OUT_DIR, "twitter_destination_data.pkl")   # test
PKL_VALIDATE      = os.path.join(OUT_DIR, "twitter_validate_data.pkl")

# 数据划分比例（从总数据里划 train/valid/test）
SPLIT_TRAIN = 0.80
SPLIT_VALID = 0.10
SPLIT_TEST  = 0.10

# pool 抽样大小（从 train 中抽）
POOL_SIZE = 280

# 分层划分用：按 event_label（域）分层，保证每个域都有样本
SPLIT_SEED = 1234



class Vocab:
    UNK = '[UNK]'

    def __init__(self, vocab_path):
        self.stoi = {}
        self.itos = []
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for i, word in enumerate(f):
                w = word.strip('\n')
                self.stoi[w] = i
                self.itos.append(w)

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get(Vocab.UNK))

    def __len__(self):
        return len(self.itos)


def build_vocab(vocab_path):
    return Vocab(vocab_path)


def read_data_twitter(filepath):
    """
    读取 twitter15_16_best_eventlabel.csv
    兼容：id列可能不存在/叫别的名字/有 Unnamed: 0
    最终保证输出含: id,label,content,if_marked_label,event_label
    """
    df = pd.read_csv(filepath, encoding='utf-8-sig')  # ✅ 比 unicode_escape 更适合你这种 csv
    # 统一清理列名：去空格、去BOM
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]

    print("[read_data_twitter] columns:", df.columns.tolist())

    # ---- 1) 处理 id 列（自动识别候选）----
    if "id" not in df.columns:
        # 常见候选名
        candidates = ["tweet_id", "tweetid", "source_tweet_id", "source_id", "tid", "Id", "ID"]
        found = None
        for cand in candidates:
            if cand in df.columns:
                found = cand
                break

        if found is not None:
            df = df.rename(columns={found: "id"})
            print(f"[read_data_twitter] rename `{found}` -> `id`")
        else:
            # 兜底：如果有 Unnamed: 0 这种索引列，用它当 id
            unnamed = [c for c in df.columns if c.lower().startswith("unnamed")]
            if len(unnamed) > 0:
                df = df.rename(columns={unnamed[0]: "id"})
                print(f"[read_data_twitter] rename `{unnamed[0]}` -> `id`")
            else:
                # 再兜底：直接生成唯一 id（不依赖外部列）
                df = df.copy()
                df.insert(0, "id", np.arange(len(df)).astype(int))
                print("[read_data_twitter] `id` not found, generated incremental ids.")

    # ---- 2) 必要列检查（除 id 外必须存在）----
    required = ["label", "content", "event_label"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"[read_data_twitter] Missing column `{c}` in: {filepath}")

    # if_marked_label 不强制要求：没有就补
    if "if_marked_label" not in df.columns:
        df["if_marked_label"] = 1

    # ---- 3) 清理 & 类型 ----
    df = df.dropna(subset=["label", "content", "event_label"])

    # label 兼容 "0"/"1"/"0.0"/"1.0"
    df["label"] = df["label"].astype(float).astype(int)
    df = df[df["label"].isin([0, 1])].copy()

    df["content"] = df["content"].astype(str)
    df["event_label"] = df["event_label"].astype(int)

    # pool 会覆盖为 0，这里统一先标 1
    df["if_marked_label"] = 1

    # 去重：优先按 id 去重；若 id 是生成的，也无所谓
    df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)
    return df

def stratified_split_by_event(df_all: pd.DataFrame, seed: int = 1234,
                             train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1):
    """
    按 event_label 分层划分，保证每个 event 都能分到 train/valid/test（尽量）
    """
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6
    rng = np.random.RandomState(seed)

    train_idx, valid_idx, test_idx = [], [], []

    for e in sorted(df_all["event_label"].unique().tolist()):
        df_e = df_all[df_all["event_label"] == e]
        idx = df_e.index.values.copy()
        rng.shuffle(idx)

        n = len(idx)
        if n == 1:
            # 极端情况：只有1条，直接放 train
            train_idx.extend(idx.tolist())
            continue

        n_train = int(round(train_ratio * n))
        n_valid = int(round(valid_ratio * n))
        # 兜底：至少给 train 一些
        n_train = max(1, min(n_train, n-1))
        # valid/test 兜底
        remaining = n - n_train
        n_valid = min(n_valid, max(0, remaining-1))
        n_test = n - n_train - n_valid

        train_idx.extend(idx[:n_train].tolist())
        valid_idx.extend(idx[n_train:n_train + n_valid].tolist())
        test_idx.extend(idx[n_train + n_valid:].tolist())

    df_train = df_all.loc[train_idx].reset_index(drop=True)
    df_valid = df_all.loc[valid_idx].reset_index(drop=True)
    df_test  = df_all.loc[test_idx].reset_index(drop=True)

    return df_train, df_valid, df_test


def _balanced_pool_indices(df_train_all: pd.DataFrame, pool_size: int, seed: int = 1234):
    """
    你的原函数：从 df_train_all 中抽 pool_size 条样本，领域均衡 + 真假均衡
    返回 df_train_all 的 index（要求 df_train_all 已 reset_index(drop=True)）
    """
    rng = np.random.RandomState(seed)

    n = len(df_train_all)
    if n == 0 or pool_size <= 0:
        return []
    if pool_size >= n:
        pool_size = max(1, int(0.2 * n))

    events = sorted(df_train_all['event_label'].unique().tolist())
    E = len(events)
    if E == 0:
        return rng.choice(df_train_all.index.values, size=min(pool_size, n), replace=False).tolist()

    base = pool_size // E
    rem = pool_size % E

    chosen = set()

    for i, e in enumerate(events):
        target_e = base + (1 if i < rem else 0)
        df_e = df_train_all[df_train_all['event_label'] == e]
        if len(df_e) == 0:
            continue

        df_e0 = df_e[df_e['label'] == 0]
        df_e1 = df_e[df_e['label'] == 1]
        half = target_e // 2

        take0 = min(len(df_e0), half)
        take1 = min(len(df_e1), half)
        remaining = target_e - take0 - take1

        idx0 = rng.choice(df_e0.index.values, size=take0, replace=False).tolist() if take0 > 0 else []
        idx1 = rng.choice(df_e1.index.values, size=take1, replace=False).tolist() if take1 > 0 else []

        picked = idx0 + idx1
        chosen.update(picked)

        if remaining > 0:
            rest_idx = df_e.index.difference(pd.Index(list(chosen)))
            if len(rest_idx) > 0:
                take_rest = min(remaining, len(rest_idx))
                extra = rng.choice(rest_idx.values, size=take_rest, replace=False).tolist()
                chosen.update(extra)

    if len(chosen) < pool_size:
        remaining_df = df_train_all.loc[df_train_all.index.difference(pd.Index(list(chosen)))]
        if len(remaining_df) == 0:
            return sorted(list(chosen))

        def event_counts(chosen_idx):
            s = df_train_all.loc[list(chosen_idx), 'event_label']
            return s.value_counts().to_dict()

        def event_label_counts(chosen_idx):
            s = df_train_all.loc[list(chosen_idx), ['event_label', 'label']]
            return s.groupby(['event_label', 'label']).size().to_dict()

        while len(chosen) < pool_size:
            remaining_df = df_train_all.loc[df_train_all.index.difference(pd.Index(list(chosen)))]
            if len(remaining_df) == 0:
                break

            ec = event_counts(chosen)
            elc = event_label_counts(chosen)

            candidates_events = remaining_df['event_label'].unique().tolist()
            candidates_events.sort(key=lambda x: ec.get(x, 0))

            picked_one = False
            for e in candidates_events:
                df_e = remaining_df[remaining_df['event_label'] == e]
                if len(df_e) == 0:
                    continue

                c0 = elc.get((e, 0), 0)
                c1 = elc.get((e, 1), 0)
                prefer_label = 0 if c0 <= c1 else 1

                df_pref = df_e[df_e['label'] == prefer_label]
                if len(df_pref) > 0:
                    idx = int(rng.choice(df_pref.index.values, size=1, replace=False)[0])
                    chosen.add(idx)
                    picked_one = True
                    break
                else:
                    idx = int(rng.choice(df_e.index.values, size=1, replace=False)[0])
                    chosen.add(idx)
                    picked_one = True
                    break

            if not picked_one:
                idx = int(rng.choice(remaining_df.index.values, size=1, replace=False)[0])
                chosen.add(idx)

    chosen = sorted(list(chosen))[:pool_size]
    return chosen


class LoadSingleSentenceClassificationDataset:
    def __init__(self,
                 vocab_path="./vocab.txt",
                 tokenizer=None,
                 batch_size=64,
                 max_sen_len=54,
                 max_position_embeddings=512,
                 pad_index=0,
                 ):
        self.tokenizer = tokenizer
        self.vocab = build_vocab(vocab_path)
        self.PAD_IDX = self.vocab['[PAD]']
        self.SEP_IDX = self.vocab['[SEP]']
        self.CLS_IDX = self.vocab['[CLS]']
        self.batch_size = batch_size
        self.max_position_embeddings = max_position_embeddings
        if isinstance(max_sen_len, int) and max_sen_len > max_position_embeddings:
            max_sen_len = max_position_embeddings
        self.max_sen_len = max_sen_len

    def to_BiGRU_features(self, df, flag):
        """
        计算 manual_features => affection（list of list）
        """
        n_jobs = 1
        dataset = 'Rumor'
        segments_number = 8
        emo_rep = 'frequency'
        data = {"content": df['content'], "flag": flag}
        content_features = manual_features(
            n_jobs=n_jobs,
            path=str(REPO_ROOT / 'data' / 'resources' / 'lexicons'),
            model_name=dataset,
            segments_number=segments_number,
            emo_rep=emo_rep
        ).transform(data)
        return content_features.tolist()

    def ensure_affection(self, df, flag):
        """
        若 df 没有 affection，则计算并加入。
        """
        if 'affection' not in df.columns:
            aff = self.to_BiGRU_features(df, flag)
            df = df.copy()
            df['affection'] = [str(a) for a in aff]  # 存成字符串，兼容你后面 eval
        return df

    def data_process_twitter(self, twitter_csv_path: str, pool_size=280):
        """
        Twitter 新逻辑：
        - 读总 CSV
        - 按 event_label 分层划 train/valid/test
        - pool 从 train 抽（不重叠 + 领域/真假均衡）
        - train1 = train minus pool
        - train2 = train1 + pool
        - 对 train/pool/valid/test 分别补齐 affection
        - 保存 xlsx（加 twitter 前缀，单独目录）
        """
        global GLOBAL_POOL_IDX

        df_all = read_data_twitter(twitter_csv_path)

        # 1) split
        df_train_all, df_valid, df_test = stratified_split_by_event(
            df_all,
            seed=SPLIT_SEED,
            train_ratio=SPLIT_TRAIN,
            valid_ratio=SPLIT_VALID,
            test_ratio=SPLIT_TEST
        )
        df_train_all = df_train_all.reset_index(drop=True)

        # 2) pool from train
        pool_idx = _balanced_pool_indices(df_train_all, pool_size=pool_size, seed=1234)
        GLOBAL_POOL_IDX = set(pool_idx)

        df_pool = df_train_all.loc[pool_idx].copy().reset_index(drop=True)
        df_pool['if_marked_label'] = 0

        keep_idx = [i for i in df_train_all.index.tolist() if i not in GLOBAL_POOL_IDX]
        df_train = df_train_all.loc[keep_idx].copy().reset_index(drop=True)
        df_train['if_marked_label'] = 1

        df_train2 = pd.concat([df_train, df_pool], axis=0).reset_index(drop=True)

        # 3) ensure affection (关键：twitter CSV 没有 affection)
        df_train  = self.ensure_affection(df_train,  flag="twitter_train")
        df_pool   = self.ensure_affection(df_pool,   flag="twitter_pool")
        df_train2 = self.ensure_affection(df_train2, flag="twitter_train2")
        df_valid  = self.ensure_affection(df_valid,  flag="twitter_valid")
        df_test   = self.ensure_affection(df_test,   flag="twitter_test")

        # 4) save xlsx（防覆盖）
        df_pool.to_excel(XLSX_POOL, index=False)
        df_train.to_excel(XLSX_TRAIN, index=False)
        df_train2.to_excel(XLSX_TRAIN2, index=False)
        df_valid.to_excel(XLSX_VALID, index=False)
        df_test.to_excel(XLSX_TEST, index=False)

        print(f"[TWITTER SPLIT] all={len(df_all)} | train_all={len(df_train_all)} | train={len(df_train)} | pool={len(df_pool)} | valid={len(df_valid)} | test={len(df_test)}")
        print(f"[TWITTER SPLIT] unique_events(all/train/valid/test): {df_all['event_label'].nunique()} / {df_train_all['event_label'].nunique()} / {df_valid['event_label'].nunique()} / {df_test['event_label'].nunique()}")
        print("[POOL] event_label counts(top10):", df_pool['event_label'].value_counts().head(10).to_dict())
        print("[POOL] label counts:", df_pool['label'].value_counts().to_dict())

        return df_train, df_pool, df_train2, df_valid, df_test

    def to_bert_input_new(self, df, df_index):
        """
        你现在的新版签名：不再需要 flag 参数
        """
        text, mask = [], []
        eve_label, label = [], []
        if_marked_label = []

        # affection 必须存在（已在 ensure_affection 补齐）
        affection = df['affection'].tolist()

        for i in range(len(df)):
            s = df.loc[df.index[i], 'content']
            l = int(df.loc[df.index[i], 'label'])
            e = int(df.loc[df.index[i], 'event_label'])
            il = int(df.loc[df.index[i], 'if_marked_label'])

            if not isinstance(s, str):
                s = str(s)

            tmp = [self.CLS_IDX]
            tmp += [self.vocab[token] for token in self.tokenizer.tokenize(s)]

            if len(tmp) > self.max_sen_len - 1:
                tmp = tmp[:(self.max_sen_len - 1)]
                tmp += [self.SEP_IDX]
            else:
                tmp += [self.SEP_IDX]
                tmp = tmp + [self.PAD_IDX for _ in range(self.max_sen_len - len(tmp))]

            attn_mask = [1 if num != 0 else 0 for num in tmp]
            text.append(torch.tensor(tmp, dtype=torch.long))
            mask.append(torch.tensor(attn_mask, dtype=torch.long))

            label.append(torch.tensor(l, dtype=torch.long))
            eve_label.append(torch.tensor(e, dtype=torch.long))
            if_marked_label.append(torch.tensor(il, dtype=torch.long))

        # affection 是字符串版 list，转回数组
        affection = [eval(a) for a in affection]
        affection = torch.from_numpy(np.array(affection))

        data = {
            "text": text,
            "mask": mask,
            "affection": affection,
            "label": label,
            "event_label": eve_label,
            "if_marked_label": if_marked_label,
            "data_index": df_index
        }
        return data


if __name__ == '__main__':
    # ===== 1. BERT 路径 =====
    path = os.environ.get('SHARP_BERT_MODEL', 'bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained(path)
    _ = BertModel.from_pretrained(path)
    vocab_path = str(REPO_ROOT / "DAAL" / "vocab.txt")

    mydic = LoadSingleSentenceClassificationDataset(vocab_path, tokenizer)

    print("开始构建 twitter dataframe 数据集（从一个总 CSV 分层划分 train/valid/test；pool 从 train 抽且不重叠）")

    df_train, df_pool, df_train2, df_valid, df_test = mydic.data_process_twitter(
        TWITTER_CSV_PATH,
        pool_size=POOL_SIZE
    )

    # 索引
    index_pool   = list(df_pool.index)
    index_train  = list(df_train.index)
    index_train2 = list(df_train2.index)
    index_valid  = list(df_valid.index)
    index_test   = list(df_test.index)

    # 生成 BERT 输入 + affection（注意：新版 to_bert_input_new 不要再传 flag）
    source_data        = mydic.to_bert_input_new(df_train,  index_train)
    source_pool_data   = mydic.to_bert_input_new(df_pool,   index_pool)
    source_extend_data = mydic.to_bert_input_new(df_train2, index_train2)
    destination_data   = mydic.to_bert_input_new(df_test,   index_test)
    validate_data      = mydic.to_bert_input_new(df_valid,  index_valid)

    # 保存（防覆盖：全部加 twitter_ 前缀）
    with open(PKL_SOURCE, 'wb') as f:
        pickle.dump(source_data, f)
    with open(PKL_SOURCE_POOL, 'wb') as f:
        pickle.dump(source_pool_data, f)
    with open(PKL_SOURCE_EXTEND, 'wb') as f:
        pickle.dump(source_extend_data, f)
    with open(PKL_DESTINATION, 'wb') as f:
        pickle.dump(destination_data, f)
    with open(PKL_VALIDATE, 'wb') as f:
        pickle.dump(validate_data, f)

    print("全部 twitter pkl 文件已生成：twitter_source / twitter_source_pool / twitter_source_extend / twitter_destination(test) / twitter_validate")
