from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from transformers import BertTokenizer, BertModel

import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import torch
from timeit import default_timer as timer
from contextlib import redirect_stdout, redirect_stderr
import os
import re

# 114

# from awq import AutoAWQForCausalLM

import os

import torch.nn.functional as F


def _to_plain_text(tokenizer, text):
    """Allow text to be str or a 1D/2D tensor of token ids."""
    if isinstance(text, str):
        return text.strip()

    if torch.is_tensor(text):
        ids = text.detach().tolist()
        if len(ids) > 0 and isinstance(ids[0], list):  # [B, L]
            ids = ids[0]
        pad = tokenizer.pad_token_id
        eos = tokenizer.eos_token_id
        filtered = []
        for i in ids:
            if pad is not None and i == pad:
                continue
            if eos is not None and i == eos:
                continue
            filtered.append(int(i))
        s = tokenizer.decode(filtered, skip_special_tokens=True)
        return s.strip()

    return str(text).strip()


def _clean_paraphrases(raw_lines, original, k):
    """
    Minimal clean:
    - drop empty lines
    - drop question/instruction/explanation lines
    - drop lines identical to original
    - de-duplicate while preserving order
    - return up to k lines
    """
    bad_subs = [
        "how would you", "rewrite the following", "preserving its original meaning",
        "don't give", "do not", "task:", "output", "explanation", "sequence",
        "here are", "sure", "as an ai", "i can't", "i cannot",
        "in other words", "for example", "this makes", "samples:",
        "please", "continue this essay", "based on the given",
    ]

    def norm(s):
        return re.sub(r"\s+", " ", s.strip().lower())

    orig_n = norm(original)
    seen = set()
    cleaned = []

    for t in raw_lines:
        if t is None:
            continue
        line = t.strip().strip('"').strip("'").strip()
        if not line:
            continue

        # drop questions
        if "?" in line:
            continue

        low = line.lower()
        if any(b in low for b in bad_subs):
            continue

        # strip leading numbering like "1) " / "1. " / "- "
        line = re.sub(r"^\s*[\-\*\u2022]?\s*\d*\s*[\)\].:-]?\s*", "", line).strip()
        if not line:
            continue

        # identical to original -> skip
        if norm(line) == orig_n:
            continue

        # de-dup
        key = norm(line)
        if key in seen:
            continue
        seen.add(key)

        cleaned.append(line)
        if len(cleaned) >= k:
            break

    return cleaned


class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
        # self.hard_prompts = hard_prompts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]  # , self.hard_prompts[idx]


from idea1_features.building_features import manual_features

from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


class PM(nn.Module):
    def __init__(self, device, model_name):

        # 注意先初始化再传入mynet
        super(PM, self).__init__()

        self.model_name = model_name
        if torch.cuda.is_available():
            print("GPU available")
        else:
            print("GPU not OK")

        self.device = device  # torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False,
                                                       trust_remote_code=True)  # 加载本地部署的tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, output_hidden_states=False,
                                                          trust_remote_code=True)  # 加载本地部署的模型

        # if torch.cuda.device_count() > 1:
        #     print("Using", torch.cuda.device_count(), "GPUs")
        #     self.model = nn.DataParallel(self.model)
        # self.model.to(self.device)
        #
        #
        # self.model = DDP(self.model, device_ids=[self.device])
        # # 启用模型的gradient checkpointing（梯度检查点）
        # self.model.module.gradient_checkpointing_enable()

        # Move model to the requested device (for AWQ models, this must be CUDA to avoid slow CPU inference)
        self.model.to(self.device)
        print(self.model.device)
        self.losses = []
        self.epoch_losses = []
        self.epoch_mean_losses = []

        self.bert_path = os.environ.get('SHARP_BERT_MODEL', 'bert-base-uncased')
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.bert_model = BertModel.from_pretrained(self.bert_path)
        self.bert_model.to(self.device)

        # 检查模型配置对象的属性名称
        print(dir(self.model.config))

        # 使用 pipeline 构建文本生成任务
        # self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        # NOTE: transformers.pipeline expects an int device id in many versions (0/-1), not torch.device.
        pipe_device = -1
        if isinstance(self.device, torch.device) and self.device.type == "cuda":
            pipe_device = int(self.device.index) if self.device.index is not None else 0
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=pipe_device)

        self.task = ["more common words", "more professional words", "more intense words", "more neutral words"]
        self.num = 1
        self.tasknum = 0

        self.soft_prompt = None  # 初始设为空
        self.lr = 0.1
        # 检查是否存在权重文件

        self.last_epoch = 0

        print("soft_prompt:", self.soft_prompt)
        print('------------------------------------')
        return

    def load_soft_prompt(self, start_epoch=5, obj="pre", lr=0.1):
        """
        Load or initialize soft prompt, and build an optimizer ONLY for the soft prompt.

        Key fixes:
        - Train soft_prompt in FP32 (avoid FP16 overflow -> NaN).
        - Auto-detect NaN/Inf checkpoint and re-init.
        - Clamp lr to a safe range (0.1 will almost surely explode for FP16/AWQ setting).
        - Freeze base model params; only soft_prompt is trainable.
        """
        import os
        import torch
        import torch.nn as nn
        import torch.optim as optim

        if obj == "pre":
            obj = ""

        device = self.device

        # ---- lr safety clamp ----
        raw_lr = float(lr)
        safe_lr = min(raw_lr, 1e-3)  # 0.1 这种会直接把 prompt 炸成 NaN
        if safe_lr != raw_lr:
            print(f"[WARN] lr={raw_lr} is too large for soft-prompt; clamped to {safe_lr}")
        self.lr = safe_lr

        ckpt_path = f"weight/pre_soft_prompt_word_epoch{start_epoch}{obj}.pth"

        # ---- freeze base model; train only soft prompt ----
        for p in self.model.parameters():
            p.requires_grad_(False)

        def _reinit_prompt():
            embedding_dim = int(self.model.config.hidden_size)
            sp = torch.randn(1, 25, embedding_dim, device=device, dtype=torch.float32) * 0.02
            return nn.Parameter(sp, requires_grad=True)

        # ---- load or init ----
        loaded_ok = False
        if start_epoch != 0 and os.path.exists(ckpt_path):
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                soft_prompt, last_epoch, self.losses, self.epoch_mean_losses = ckpt

                if isinstance(soft_prompt, nn.Parameter):
                    soft_prompt = soft_prompt.detach()

                soft_prompt = soft_prompt.to(device=device, dtype=torch.float32).contiguous()

                if torch.isfinite(soft_prompt).all():
                    self.soft_prompt = nn.Parameter(soft_prompt, requires_grad=True)
                    self.last_epoch = int(last_epoch)
                    start_epoch = self.last_epoch + 1
                    loaded_ok = True
                    print(f"Loaded soft prompt from {ckpt_path}. Last epoch: {self.last_epoch}")
                else:
                    print(f"[WARN] soft prompt in {ckpt_path} contains NaN/Inf -> reinit")
            except Exception as e:
                print(f"[WARN] failed to load soft prompt from {ckpt_path}: {e} -> reinit")

        if not loaded_ok:
            self.soft_prompt = _reinit_prompt()
            self.last_epoch = 0
            start_epoch = 1
            print("Initialized soft prompt with random values (FP32, std=0.02).")

        # ---- optimizer ONLY for soft prompt (no weight decay) ----
        self.optimizer = optim.AdamW(
            [self.soft_prompt],
            lr=self.lr,
            weight_decay=0.0,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        return

    def get_embeddings(self, text):

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # 或者使用自定义的 [PAD]

        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        input_embeddings = self.model.get_input_embeddings()(input_ids)

        # print(input_embeddings.size())

        # 平均池化来获取句子的嵌入向量
        # input_embeddings = torch.mean(input_embeddings, dim=1)
        # # 获取模型的嵌入层
        # embeddings = self.model.get_input_embeddings()(input_ids)

        return input_embeddings, attention_mask

    def get_last(self, combined_embeddings, attention_mask):
        outputs = self.model(inputs_embeds=combined_embeddings, attention_mask=attention_mask,
                             output_hidden_states=True)

        # print("outputs:", outputs)
        last_hidden_state = outputs.hidden_states[-1]
        # print("last_hidden_state:", last_hidden_state.size()) # torch.Size([1, 4306, 4096])

        # last_hidden_state = torch.mean(last_hidden_state, dim=1)

        return last_hidden_state

    def to_bert_embbeding(self, output_embeddings):

        dtype = next(self.model.parameters()).dtype

        # print('output_embeddings', output_embeddings.size()) #torch.Size([1, 4883, 4096])
        self.output_embeddings = output_embeddings
        # output_embeddings = output_embeddings[:, -1, :]
        output_embeddings = output_embeddings.to(self.device)
        output_embeddings = output_embeddings.to(dtype)

        # # 获取原来样本的emb
        # input_embeddings, attention_mask = self.get_embeddings(input_texts)
        # print('input_embeddings', input_embeddings.size())
        # # 最后一层
        # self.input_embeddings = input_embeddings
        # input_embeddings = input_embeddings[:, -1, :]
        # input_embeddings = input_embeddings.to(self.device)
        # input_embeddings = input_embeddings.to(dtype)

        # 平均池化操作来训练softpromt
        # input_embeddings = torch.mean(input_embeddings, dim=1)
        # output_embeddings = torch.mean(output_embeddings, dim=1)

        # logits = self.model.lm_head(output_embeddings)  # [batch_size, seq_len, vocab_size]
        #
        # probabilities = F.softmax(logits, dim=-1)  # [batch_size, seq_len, vocab_size]
        #
        # # 概率分布与 BERT 嵌入表相乘，得到生成文本的嵌入表示
        # output_embeddings = torch.matmul(probabilities,
        #                                  self.aligned_bert_embeddings)  # [batch_size, seq_len, hidden_size]

        return output_embeddings

    def to_BiGRU_input(self, df, flag):

        # 拿到每句话的文本后，将其对应转化为24维的向量,返回到上一个函数中，一起封装成为data形式
        # print("先来拿到BiGRU的信息")
        # print(flag)
        n_jobs = 1
        dataset = 'Rumor'
        segments_number = 8  # 不是10个单词，而是指 把每个句子分成了10段，每段用24维的向量来表示
        emo_rep = 'frequency'  # 分段数和先前的提取特征是不影响的，因为它是先提取了每个单词的特征，再综合成了一个句子几段
        content_features = []
        data = {"content": df['content'], "flag": flag}
        content_features = manual_features(n_jobs=n_jobs, path="../data/resources/lexicons", model_name=dataset,
                                           segments_number=segments_number, emo_rep=emo_rep).transform(data)
        # print(content_features.shape)                   # (4689, 10, 24)
        content_features = content_features.tolist()
        affection = np.array(content_features)
        affection = torch.from_numpy(affection)
        return affection

    # def _to_text(self, x):
    #     """把各种类型安全转成 str（重点：Tensor 不会触发 bool 判断）"""
    #     if x is None:
    #         return ""
    #     if isinstance(x, str):
    #         return x
    #     # list/tuple of tokens/strings
    #     if isinstance(x, (list, tuple)):
    #         try:
    #             return " ".join([self._to_text(i) for i in x])
    #         except Exception:
    #             return str(x)
    #
    #     if torch.is_tensor(x):
    #         # token ids -> decode
    #         try:
    #             if x.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
    #                 ids = x.detach().cpu()
    #                 if ids.dim() == 2:
    #                     ids = ids[0]
    #                 return self.tokenizer.decode(ids.tolist(), skip_special_tokens=True)
    #         except Exception:
    #             pass
    #         # scalar tensor
    #         try:
    #             if x.dim() == 0:
    #                 return str(x.item())
    #             if x.numel() == 1:
    #                 return str(x.view(-1)[0].item())
    #         except Exception:
    #             pass
    #         # embedding tensor：没法还原文本，返回空串，让 sanitize 走兜底
    #         return ""
    #
    #     return str(x)
    #
    # def _sanitize_source(self, s, max_chars=600):
    #     """把一条样本清洗成“适合改写”的单句/短句（避免你日志里那种拼了一堆字段/多条样本）"""
    #     s = self._to_text(s)
    #     s = (s if isinstance(s, str) else str(s))
    #     s = s.replace("\r", "\n").strip()
    #
    #     if not s:
    #         return ""
    #
    #     # 只取第一行
    #     s = s.splitlines()[0].strip()
    #
    #     # 含 tab 的通常是 LIAR/自拼字段：只取第一列（claim）
    #     if "\t" in s:
    #         s = s.split("\t")[0].strip()
    #
    #     # 如果误把多条样本拼进来（出现 "xxxx.json"），截断
    #     m = re.search(r"\s+\d+\.json\b", s)
    #     if m and m.start() > 20:
    #         s = s[:m.start()].strip()
    #
    #     # 太长截断（防止 prompt 超长、truncation 警告）
    #     if len(s) > max_chars:
    #         s = s[:max_chars].rsplit(" ", 1)[0].strip()
    #
    #     return s
    #
    # def _normalize(self, s):
    #     s = (s or "").lower()
    #     s = re.sub(r"[^a-z0-9\s]", " ", s)
    #     s = re.sub(r"\s+", " ", s).strip()
    #     return s
    #
    # def _token_overlap_ratio(self, a, b):
    #     a_t = self._normalize(a).split()
    #     b_t = self._normalize(b).split()
    #     if not a_t or not b_t:
    #         return 1.0
    #     inter = len(set(a_t) & set(b_t))
    #     return inter / max(1, len(set(b_t)))
    #
    # def _clean_generated_line(self, s):
    #     """把模型输出清成“单行句子”，剔除标签/括号说明/包裹符号等"""
    #     s = self._to_text(s)
    #     s = (s or "").replace("\r", "\n").strip()
    #     if not s:
    #         return ""
    #
    #     # 找第一条非空行
    #     lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    #     if not lines:
    #         return ""
    #     s = lines[0]
    #
    #     # 去除常见前缀
    #     s = re.sub(r"^(answer|output|rewrite|rewritten|claim|daire|accuracy and style)\s*[:：\-–]\s*", "", s, flags=re.I)
    #
    #     # 去掉 <<< >>> / 引号 / 代码块符号
    #     s = re.sub(r"^<{2,}\s*", "", s)
    #     s = re.sub(r"\s*>{2,}$", "", s)
    #     s = s.strip().strip("`").strip().strip('"').strip("'").strip()
    #
    #     # 去掉末尾括号里的“说明”
    #     s = re.sub(r"\s*\([^)]*(meaning|rewrite|unchanged|preserv|keep the meaning)[^)]*\)\s*$", "", s,
    #                flags=re.I).strip()
    #
    #     return s
    #
    # def _is_good_rewrite(self, cand, src):
    #     """过滤：空/太短/问句/提示语/复读"""
    #     if not cand:
    #         return False
    #     src_n = self._normalize(src)
    #     cand_n = self._normalize(cand)
    #
    #     if not cand_n:
    #         return False
    #
    #     # 明显垃圾短词
    #     if len(cand_n.split()) < max(4, int(0.4 * max(1, len(src_n.split())))):
    #         return False
    #
    #     # 不要模型反问/指令
    #     low = cand.lower()
    #     bad_phrases = ["how would you", "rewrite the following", "output only", "follow the", "explanation", "sequence"]
    #     if any(p in low for p in bad_phrases):
    #         return False
    #
    #     # 不能与原句完全相同
    #     if cand_n == src_n:
    #         return False
    #
    #     # 不能“几乎全复制”
    #     if self._token_overlap_ratio(cand, src) > 0.93:
    #         return False
    #
    #     return True
    #
    # def _build_prompt(self, src, style, num):
    #     # 强约束输出格式：恰好 num 行，每行一句改写，不要编号/标签
    #     system_msg = "You are a rewriting engine. You MUST follow the output format exactly."
    #
    #     user_msg = (
    #         f"Task: rewrite the text in a {style} style.\n"
    #         f"Hard rules:\n"
    #         f"1) Output language: English only.\n"
    #         f"2) Preserve meaning. Do NOT add new facts.\n"
    #         f"3) Keep ALL numbers, dates, and named entities EXACTLY the same as the input.\n"
    #         f"4) Output exactly {num} lines.\n"
    #         f"5) Each line must be ONE sentence (no lists, no multi-sentence lines).\n"
    #         f"6) No labels/prefixes (no 'Rewrite:', 'Original:', 'Sentence:', '<<ASSISTANT>>', etc.).\n"
    #         f"7) No questions, no explanations, no extra text.\n"
    #         f"Text: {src}"
    #     )
    #
    #     if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None):
    #         messages = [{"role": "system", "content": system_msg},
    #                     {"role": "user", "content": user_msg}]
    #         prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    #     else:
    #         prompt = system_msg + "\n" + user_msg + "\nAnswer:\n"
    #
    #     return prompt
    #
    # # =============== 可直接替换：chat ===============
    # def chat(self, tasknum, num, text, pro_type="text", out_type="text", if_soft=0):
    #     """
    #     ⭐重要约定（避免 unpack 报错）：
    #     - pro_type == "text" : 返回 outtext(str)
    #     - pro_type == "vec"  : 永远返回 4-tuple (combined_embeddings, last, combined_mask, outtext)
    #       - out_type == "vec"  -> 计算 last，outtext=""
    #       - out_type == "text" -> last=None，outtext 为生成文本
    #     """
    #     self.num = num
    #     self.tasknum = tasknum
    #     toc = timer()
    #
    #     style = self.task[int(self.tasknum)]
    #
    #     src = self._sanitize_source(text)
    #     if not src:
    #         if pro_type == "vec":
    #             return None, None, None, "[EMPTY]"
    #         return "[EMPTY]"
    #
    #     prompt = self._build_prompt(src, style, num)
    #
    #     # generation config（两轮尝试：第二轮更“激进”避免复读/空）
    #     def _gen_kwargs(temp, rep_pen):
    #         d = dict(
    #             max_new_tokens=72,
    #             min_new_tokens=6,  # 防止直接 EOS -> decode 为空
    #             do_sample=True,
    #             temperature=temp,
    #             top_k=50,
    #             top_p=0.95,
    #             repetition_penalty=rep_pen,
    #             no_repeat_ngram_size=3,
    #             num_return_sequences=1,  # 尽量不走“候选池”，失败再重试
    #         )
    #         pad_id = self.tokenizer.pad_token_id
    #         if pad_id is None:
    #             pad_id = self.tokenizer.eos_token_id
    #         if pad_id is not None:
    #             d["pad_token_id"] = pad_id
    #         if self.tokenizer.eos_token_id is not None:
    #             d["eos_token_id"] = self.tokenizer.eos_token_id
    #         return d
    #
    #     def _postprocess_multi(text_block):
    #         # 按行切分，清洗 + 过滤复读/垃圾
    #         lines = [ln.strip() for ln in (text_block or "").splitlines() if ln.strip()]
    #         outs = []
    #         for ln in lines:
    #             cand = self._clean_generated_line(ln)
    #             if self._is_good_rewrite(cand, src) and cand not in outs:
    #                 outs.append(cand)
    #             if len(outs) >= num:
    #                 break
    #         return outs
    #
    #     # ===== pro_type == text：用 input_ids 生成，并只 decode 新 token，最稳 =====
    #     if pro_type == "text":
    #         inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
    #         input_len = inputs["input_ids"].shape[1]
    #
    #         outs = []
    #         for (temp, rep_pen) in [(0.9, 1.08), (1.15, 1.12)]:
    #             with torch.no_grad():
    #                 gen_ids = self.model.generate(**inputs, **_gen_kwargs(temp, rep_pen))
    #             cand = self.tokenizer.decode(gen_ids[0][input_len:], skip_special_tokens=True)
    #             cand = self._clean_generated_line(cand)
    #             if self._is_good_rewrite(cand, src) and cand not in outs:
    #                 outs.append(cand)
    #             if len(outs) >= num:
    #                 break
    #
    #         if not outs:
    #             outs = [src]  # 最坏兜底：不崩溃
    #
    #         # 若 num>1：再补齐（再次调用比“候选池”更接近你想要的逻辑）
    #         while len(outs) < num:
    #             outs.append(outs[-1])
    #
    #         outtext = "\n".join(outs[:num])
    #         print("生成文本用时: ", timer() - toc)
    #         print("outtext是——————————————————————————————————————————")
    #         print(outtext)
    #         print("outtext在上面——————————————————————————————————————————")
    #         return outtext
    #
    #     # ===== pro_type == vec：拼 embedding + (optional) soft prompt =====
    #     dtype = next(self.model.parameters()).dtype
    #
    #     init_emb, init_mask = self.get_embeddings(prompt)
    #     init_emb = init_emb.to(dtype).to(self.device)
    #
    #     if if_soft == 1:
    #         soft_emb = self.soft_prompt.to(dtype).to(self.device)
    #         bsz = init_emb.size(0)
    #         soft_len = soft_emb.size(1)
    #         soft_mask = torch.ones((bsz, soft_len), device=self.device, dtype=init_mask.dtype)
    #
    #         # ✅ attention_mask 顺序必须和 embeddings 一致（修复你原来的错位）
    #         combined_embeddings = torch.cat([init_emb, soft_emb], dim=1)
    #         combined_mask = torch.cat([init_mask.to(self.device), soft_mask.to(self.device)], dim=1)
    #     else:
    #         combined_embeddings = init_emb
    #         combined_mask = init_mask
    #
    #     if out_type == "vec":
    #         last = self.get_last(combined_embeddings, combined_mask)
    #         return combined_embeddings, last, combined_mask, ""
    #
    #     # out_type == "text"：生成文本（注意：inputs_embeds 情况下“切掉 prompt_len”可能导致空，这里做双策略兜底）
    #     prompt_len = combined_embeddings.shape[1]
    #     outs = []
    #
    #     for (temp, rep_pen) in [(0.9, 1.08), (1.15, 1.12)]:
    #         with torch.no_grad():
    #             gen_ids = self.model.generate(
    #                 inputs_embeds=combined_embeddings,
    #                 attention_mask=combined_mask,
    #                 **_gen_kwargs(temp, rep_pen)
    #             )
    #
    #         # gen_ids 形状可能是 [1, L]；也可能 [num_return_sequences, L]
    #         if gen_ids.dim() == 1:
    #             gen_ids = gen_ids.unsqueeze(0)
    #
    #         # decode 策略：先按 prompt_len 切；若为空，再 decode 全部
    #         raw = self.tokenizer.decode(gen_ids[0][prompt_len:], skip_special_tokens=True)
    #         if not (raw or "").strip():
    #             raw = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    #
    #         cands = _postprocess_multi(raw)
    #         outs.extend([c for c in cands if c not in outs])
    #         if len(outs) >= num:
    #             break
    #
    #     if not outs:
    #         outs = [src]
    #
    #     while len(outs) < num:
    #         outs.append(outs[-1])
    #
    #     outtext = "\n".join(outs[:num])
    #     print("生成文本用时: ", timer() - toc)
    #     print("outtext是——————————————————————————————————————————")
    #     print(outtext)
    #     print("outtext在上面——————————————————————————————————————————")
    #     return combined_embeddings, None, combined_mask, outtext
    #
    # # =============== 可直接替换：gen_text ===============
    # def gen_text(self, df_tmp=None, num=2, samples=3, prompt="prepared",
    #              pro_type="text", out_type="text", start_epoch=1, if_soft=0):
    #
    #     print("_____________________________________________________")
    #     aff_list = []
    #
    #     if df_tmp is None:
    #         df_tmp = pd.DataFrame(columns=["id", "label", "affection", "content", "if_marked_label", "event_label"])
    #         print("gentext传进来的df是空的")
    #
    #     if prompt == "prepared":
    #         data = {
    #             "id": ["5.8E+17", "5.8E+17", "5.8E+17"],
    #             "label": [0, 0, 0],
    #             "affection": ["x", "x", "x"],
    #             "content": [
    #                 "Pray for #4U9525 http://t.co/II7Rl24ffH",
    #                 "Airbus A320 #4U9525 crash: Flight tracking data here: http://t.co/9W6hfTGYQV #airbus http://t.co/RchXQsoqdJ",
    #                 "Flightradar24 has this as the plane's last position. #4U9525 https://t.co/KffO4s4N6G"
    #             ],
    #             "if_marked_label": [0, 0, 0],
    #             "event_label": [2, 2, 2]
    #         }
    #         df_tmp = pd.DataFrame(data)
    #
    #     gen_texts = pd.DataFrame(columns=["id", "label", "affection", "content", "if_marked_label", "event_label"])
    #
    #     if prompt == "normal":
    #         max_i = min(samples, len(df_tmp))
    #
    #         for i in range(max_i):
    #             print(f"提取第{i}个样本开始——————————————————————————————————————")
    #
    #             raw_init = df_tmp["content"].iloc[i]
    #             init_text = self._sanitize_source(raw_init)
    #             print(f"原文本是什么样的init_text{i}", raw_init)
    #
    #             ret = self.chat(
    #                 tasknum=3,
    #                 num=num,
    #                 text=init_text,  # ✅ 用清洗后的单句进入生成
    #                 pro_type=pro_type,
    #                 out_type=out_type,
    #                 if_soft=if_soft
    #             )
    #
    #             # ✅ 兼容：vec 模式返回 4-tuple
    #             if isinstance(ret, tuple) and len(ret) == 4:
    #                 outtext = ret[3]
    #             else:
    #                 outtext = ret
    #
    #             print("outtext:", outtext)
    #
    #             # 解析输出：按行取
    #             lines = [s.strip() for s in (outtext or "").splitlines() if s.strip()]
    #             lines = [self._clean_generated_line(s) for s in lines]
    #             lines = [s for s in lines if self._is_good_rewrite(s, init_text)]
    #
    #             # 兜底：保证至少 1 条，避免 DataLoader num_samples=0
    #             if not lines:
    #                 print(f"[WARN] sample {i}: generation empty/bad -> fallback to source once to avoid crash.")
    #                 lines = [init_text]
    #
    #             # 限制最多 num 条（实际可能不足 num）
    #             generated_texts = lines[:num]
    #
    #             print("生成出来的几个句子是什么样的")
    #             print("提取出来的文本是：————————————————————————————————————————————")
    #             for generated_sentence in generated_texts:
    #                 print(generated_sentence)
    #             print(f"一个文本生成句子结束，一共目标{num}个，实际{len(generated_texts)}个")
    #             print("提取出来的文本在上面：————————————————————————————————————————")
    #
    #             # 情感特征
    #             print("情感提取开始")
    #             with open(os.devnull, "w") as fnull:
    #                 with redirect_stdout(fnull), redirect_stderr(fnull):
    #                     affection = self.to_BiGRU_input(
    #                         pd.DataFrame({"content": generated_texts}),
    #                         flag="generated"
    #                     )
    #             print("情感提取结束")
    #
    #             for j, txt in enumerate(generated_texts):
    #                 new_sample = df_tmp.iloc[i].copy()
    #                 new_sample["content"] = txt
    #
    #                 print(f"new_sample['content']{j}: ", txt)
    #                 print(f"这次的new_sample{j}是怎么样的", new_sample)
    #
    #                 new_sample = new_sample.to_frame().T
    #                 gen_texts = pd.concat([gen_texts, new_sample], ignore_index=True)
    #
    #             print(f"提取第{i}个样本结束——————————————————————————————————————————————————————")
    #
    #     return gen_texts

    # def chat(self, tasknum, num, text, pro_type="text", out_type="text", if_soft=0):
    #     self.num = num
    #     self.tasknum = tasknum
    #     toc = timer()
    #     # Make sure `text` is a plain string. In your training code, it may come in as token ids (Tensor).
    #     if not isinstance(text, str):
    #         try:
    #             if torch.is_tensor(text):
    #                 ids = text.detach().to('cpu').tolist()
    #                 text = self.bert_tokenizer.decode(ids, skip_special_tokens=True)
    #             elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
    #                 text = self.bert_tokenizer.decode(list(text), skip_special_tokens=True)
    #             else:
    #                 text = str(text)
    #         except Exception:
    #             text = str(text)
    #
    #     style = self.task[int(self.tasknum)]
    #     init_text = (
    #         f"Rewrite the text using {style} alternatives with minimal substitutions.\n"
    #         "Keep meaning and facts unchanged.\n"
    #         "Do NOT add explanations, analysis, or extra sentences.\n"
    #         "Do NOT output any labels such as Original/Rewritten/Assistant/Text.\n"
    #         "Do NOT change numbers, dates, names, or percentages.\n"
    #         f"Output exactly {self.num} rewrites, one per line, rewrites only.\n"
    #         f"{text}\n"
    #     )
    #
    #     # init_text = f"Replace some of words in the sentence '{text}'with {self.task[int(self.tasknum)]} ." \
    #     #             f"output {self.num} sentences. don't give explanation/sequence or any other things except pure imitated sentences"
    #
    #
    #
    #     # init_text = self.pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    #     if pro_type == "text":
    #         # Text -> text generation (used for augmentation)
    #         outputs = self.pipe(
    #             init_text,
    #             max_new_tokens=256,
    #             do_sample=True,
    #             temperature=0.7,
    #             top_k=50,
    #             top_p=0.95,
    #         )
    #         outtext = outputs[0]["generated_text"]
    #         # pipeline usually returns prompt+completion; strip the prompt part to avoid repetition
    #         if isinstance(outtext, str) and outtext.startswith(init_text):
    #             outtext = outtext[len(init_text):].lstrip()
    #         return outtext
    #
    #     elif pro_type == "vec":
    #         # 通过vec生成vec
    #
    #         dtype = next(self.model.parameters()).dtype
    #
    #         # init_text
    #         init_text_embedding, init_text_attention_mask = self.get_embeddings(init_text)
    #         init_text_embedding = init_text_embedding.to(dtype).to(self.device)
    #         # print('init_text_embedding', init_text_embedding.size()) # torch.Size([1, 4882, 4096])
    #         # print('init_text_attention_mask', init_text_attention_mask.size()) # torch.Size([1, 4882])
    #
    #         # 确保soft_prompt和attention_mask的维度匹配
    #         # soft prompt prefix
    #         if if_soft == 1 and self.soft_prompt is None:
    #             raise RuntimeError("soft_prompt is None. Call load_soft_prompt() before using if_soft=1")
    #
    #         soft_text_embedding = self.soft_prompt
    #         batch_size = init_text_embedding.size(0)
    #         if soft_text_embedding is not None:
    #             if soft_text_embedding.device != self.device:
    #                 soft_text_embedding = soft_text_embedding.to(self.device)
    #             if soft_text_embedding.dtype != dtype:
    #                 soft_text_embedding = soft_text_embedding.to(dtype)
    #             if soft_text_embedding.size(0) != batch_size:
    #                 soft_text_embedding = soft_text_embedding.expand(batch_size, -1, -1).contiguous()
    #
    #         soft_prompt_length = self.soft_prompt.size(1) if self.soft_prompt is not None else 0
    #         prompt_length = init_text_embedding.size(1)
    #         soft_text_attention_mask = torch.ones(
    #             (batch_size, soft_prompt_length),
    #             device=self.device,
    #             dtype=init_text_attention_mask.dtype,
    #         )
    #         # print("soft_text_embedding", soft_text_embedding.size())
    #         # print("soft_text_attention_mask", soft_text_attention_mask.size())
    #
    #         combined_attention_mask = []
    #         combined_embeddings = []
    #
    #         # --- soft prompt prefix ---
    #         if if_soft == 1:
    #             if self.soft_prompt is None:
    #                 raise RuntimeError("soft_prompt is None. Call load_soft_prompt() before using if_soft=1")
    #
    #             # soft_prompt: (1, P, H) -> (B, P, H), and dtype align to init_text_embedding
    #             soft_text_embedding = self.soft_prompt
    #             if soft_text_embedding.device != self.device:
    #                 soft_text_embedding = soft_text_embedding.to(self.device)
    #
    #             # 对齐 dtype：用 init_text_embedding（就是文本embedding）来对齐
    #             if soft_text_embedding.dtype != init_text_embedding.dtype:
    #                 soft_text_embedding = soft_text_embedding.to(dtype=init_text_embedding.dtype)
    #
    #             batch_size = init_text_embedding.size(0)
    #             if soft_text_embedding.size(0) != batch_size:
    #                 soft_text_embedding = soft_text_embedding.expand(batch_size, -1, -1).contiguous()
    #
    #             soft_prompt_length = soft_text_embedding.size(1)
    #             soft_text_attention_mask = torch.ones(
    #                 (batch_size, soft_prompt_length),
    #                 device=self.device,
    #                 dtype=init_text_attention_mask.dtype,
    #             )
    #
    #             combined_embeddings = torch.cat([soft_text_embedding, init_text_embedding], dim=1)
    #             combined_attention_mask = torch.cat(
    #                 [soft_text_attention_mask, init_text_attention_mask.to(self.device)], dim=1
    #             )
    #         else:
    #             combined_embeddings = init_text_embedding
    #             combined_attention_mask = init_text_attention_mask.to(self.device)
    #         if out_type == "vec":
    #             last = self.get_last(combined_embeddings, combined_attention_mask)
    #             # print("生成向量用时: ", timer() - toc)
    #             outtext = ""
    #             return combined_embeddings, last, combined_attention_mask, outtext
    #
    #
    #         # elif out_type == "text":
    #         #     # input来生成
    #         #     with torch.no_grad():
    #         #         generated_ids = self.model.generate(
    #         #             inputs_embeds=combined_embeddings,
    #         #
    #         #             attention_mask=combined_attention_mask,
    #         #             # position_ids=position_ids,  # 添加 position_ids 参数
    #         #             max_length=200,
    #         #             temperature=1.0,
    #         #             top_k=50,
    #         #             top_p=0.95,
    #         #             # pad_token_id=self.tokenizer.eos_token_id  # 显式设置 pad_token_id
    #         #         )
    #         #         # generated_texts = [self.tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
    #         #         # print("text:", generated_texts)
    #         #
    #         #         # generated_ids = self.model.generate(inputs_embeds=combined_embeddings, max_length=100,
    #         #         #                                     num_beams=55, early_stopping=True)
    #         #         outtext = [self.tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
    #         #         print("生成文本用时: ", timer() - toc)
    #         #
    #         #         # outputs[0]["generated_text"]
    #         #
    #         #         print("outtext是——————————————————————————————————————————")
    #         #         print(outtext)
    #         #         print("outtext在上面——————————————————————————————————————————")
    #         #         return outtext[0]
    #         elif out_type == "text":
    #             # ---- generate text from combined_embeddings ----
    #             pad_id = self.tokenizer.pad_token_id
    #             if pad_id is None:
    #                 pad_id = self.tokenizer.eos_token_id
    #
    #             # eos: 尽量包含 chat end token
    #             eos_ids = [self.tokenizer.eos_token_id]
    #             for tok in ["<|im_end|>", "<|endoftext|>"]:
    #                 try:
    #                     tid = self.tokenizer.convert_tokens_to_ids(tok)
    #                     if isinstance(tid, int) and tid != self.tokenizer.unk_token_id:
    #                         eos_ids.append(tid)
    #                 except Exception:
    #                     pass
    #             eos_ids = list(dict.fromkeys([i for i in eos_ids if i is not None]))
    #
    #             # 禁掉常见 chat 特殊 token，避免输出 <|user|> 等
    #             bad_tokens = ["<|user|>", "<|assistant|>", "<|system|>", "<|im_start|>", "<|im_end|>"]
    #             bad_words_ids = []
    #             for t in bad_tokens:
    #                 try:
    #                     tid = self.tokenizer.convert_tokens_to_ids(t)
    #                     if isinstance(tid, int) and tid != self.tokenizer.unk_token_id:
    #                         bad_words_ids.append([tid])
    #                 except Exception:
    #                     pass
    #
    #             seq_len = combined_embeddings.size(1)
    #             dummy_input_ids = torch.full(
    #                 (combined_embeddings.size(0), seq_len),
    #                 pad_id,
    #                 device=self.device,
    #                 dtype=torch.long
    #             )
    #
    #             # gen_ids = self.model.generate(
    #             #     input_ids=dummy_input_ids,
    #             #     inputs_embeds=combined_embeddings,
    #             #     attention_mask=combined_attention_mask,
    #             #     max_new_tokens=128,
    #             #     min_new_tokens=20,
    #             #     do_sample=True,
    #             #     temperature=0.4,
    #             #     top_p=0.9,
    #             #     top_k=50,
    #             #     repetition_penalty=1.08,
    #             #     no_repeat_ngram_size=3,
    #             #     eos_token_id=eos_ids if len(eos_ids) > 1 else eos_ids[0],
    #             #     pad_token_id=pad_id,
    #             #     bad_words_ids=bad_words_ids if len(bad_words_ids) > 0 else None,
    #             # )
    #             bad_phrases = [
    #                 "<<ASSISTANT>>", "<</TEXT>>", "<<TEXT>>",
    #                 "Original:", "ORIGINAL:", "Rewritten:", "Rewrite", "Rewrites:",
    #                 "This information is", "To rewrite", "we could say", "This can be rewritten",
    #                 "According to the text",
    #             ]
    #             bad_words_ids = []
    #             for ph in bad_phrases:
    #                 ids = self.tokenizer(ph, add_special_tokens=False).input_ids
    #                 if ids:
    #                     bad_words_ids.append(ids)
    #
    #             gen_ids = self.model.generate(
    #                 input_ids=dummy_input_ids,
    #                 inputs_embeds=combined_embeddings,
    #                 attention_mask=combined_attention_mask,
    #                 max_new_tokens=80,
    #                 do_sample=True,
    #                 temperature=0.35,  # 更保守，减少跑题/解释
    #                 top_p=0.8,
    #                 top_k=50,
    #                 repetition_penalty=1.1,
    #                 no_repeat_ngram_size=3,
    #                 bad_words_ids=bad_words_ids if bad_words_ids else None,
    #                 pad_token_id=pad_id,
    #                 eos_token_id=eos_ids if len(eos_ids) > 1 else eos_ids[0],
    #             )
    #
    #             # 只解码新增 tokens（避免把 prompt 混进输出）
    #             new_ids = gen_ids[:, seq_len:]
    #             raw = self.tokenizer.decode(new_ids[0], skip_special_tokens=True).strip()
    #
    #             # ---- post-process: 按行抽取“纯句子”，过滤垃圾 ----
    #             raw = re.sub(r"<\|.*?\|>", "", raw).strip()  # 再保险去掉残留特殊标记
    #             lines = [ln.strip(" \t-•") for ln in raw.splitlines()]
    #             lines = [ln for ln in lines if len(ln) >= 8 and re.search(r"[A-Za-z]", ln)]
    #
    #             # 过滤明显“解释/教学/提问”风格
    #             bad_phrases = [
    #                 "here is", "sample output", "so you just need", "to practice", "original :",
    #                 "can you", "provide some examples", "how does", "from your text"
    #             ]
    #             clean = []
    #             for ln in lines:
    #                 low = ln.lower()
    #                 if any(bp in low for bp in bad_phrases):
    #                     continue
    #                 # 去掉行首编号
    #                 ln = re.sub(r"^\s*\d+\s*[\)\.\-:]\s*", "", ln).strip()
    #                 if ln:
    #                     clean.append(ln)
    #
    #             # 只返回 num 条；不足就返回现有的（至少不会出现 '.' 或 instruction）
    #             clean = clean[: self.num]
    #             outtext = "\n".join(clean) if clean else raw
    #             return outtext
    #
    #             # inputs_embeds -> text generation. Use dummy input_ids so we can reliably slice prompt vs new tokens.
    #             # with torch.no_grad():
    #             #     pad_id = self.tokenizer.pad_token_id
    #             #     if pad_id is None:
    #             #         pad_id = self.tokenizer.eos_token_id
    #             #
    #             #     seq_len = combined_embeddings.size(1)
    #             #     dummy_input_ids = torch.full(
    #             #         (combined_embeddings.size(0), seq_len),
    #             #         fill_value=int(pad_id) if pad_id is not None else 0,
    #             #         device=self.device,
    #             #         dtype=torch.long,
    #             #     )
    #             #
    #             #     generated_ids = self.model.generate(
    #             #         input_ids=dummy_input_ids,
    #             #         inputs_embeds=combined_embeddings,
    #             #         attention_mask=combined_attention_mask,
    #             #         max_new_tokens=128,
    #             #         do_sample=True,
    #             #         temperature=1.0,
    #             #         top_k=50,
    #             #         top_p=0.95,
    #             #         pad_token_id=pad_id,
    #             #         eos_token_id=self.tokenizer.eos_token_id,
    #             #         use_cache=True,
    #             #     )
    #             #
    #             #     new_ids = generated_ids[:, seq_len:]
    #             #     if new_ids.numel() == 0:
    #             #         outtext = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    #             #     else:
    #             #         outtext = self.tokenizer.decode(new_ids[0], skip_special_tokens=True,
    #             #                                         clean_up_tokenization_spaces=True)
    #
    #             print("生成文本用时: ", timer() - toc)
    #             print("outtext是——————————————————————————————————————————")
    #             print(outtext)
    #             print("outtext在上面——————————————————————————————————————————")
    #             # return outtext.strip()
    #         # elif pro_type == "vec" and out_type == "vec_text":
    #         #
    #         #     last = self.get_last(combined_embeddings, combined_attention_mask)
    #         #     # print("生成向量用时: ", timer() - toc)
    #         #
    #         #     # input来生成
    #         #     with torch.no_grad():
    #         #         generated_ids = self.model.generate(
    #         #             inputs_embeds=combined_embeddings,
    #         #
    #         #             attention_mask=combined_attention_mask,
    #         #             # position_ids=position_ids,  # 添加 position_ids 参数
    #         #             max_length=200,
    #         #             temperature=1.0,
    #         #             top_k=50,
    #         #             top_p=0.95,
    #         #             # pad_token_id=self.tokenizer.eos_token_id  # 显式设置 pad_token_id
    #         #         )
    #         #         # generated_texts = [self.tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
    #         #         # print("text:", generated_texts)
    #         #
    #         #         # generated_ids = self.model.generate(inputs_embeds=combined_embeddings, max_length=100,
    #         #         #                                     num_beams=55, early_stopping=True)
    #         #         outtext = [self.tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
    #         #         print("生成文本用时: ", timer() - toc)
    #         #
    #         #         # outputs[0]["generated_text"]
    #         #
    #         #
    #         #         print("outtext是——————————————————————————————————————————")
    #         #         print(outtext)
    #         #         print("outtext在上面—"
    #         #               ""
    #         #               ""
    #         #               "—————————————————————————————————————————")
    #         #
    #         #         return combined_embeddings, last, combined_attention_mask, outtext
    #
    # def gen_text(self, df_tmp=None, num=2, samples=3, prompt="prepared", pro_type="text", out_type="text",
    #              start_epoch=1, if_soft=0):
    #     # 根据samples个原文本生成n个样本，返回之前选出来的样本df_tmp
    #     # print(df_tmp)
    #     # print(df_tmp["content"])
    #     # print(df_tmp["affection"])
    #     print("_____________________________________________________")
    #     aff_list = []
    #     if df_tmp is None:
    #         # 创建一个空的 DataFrame
    #         df_tmp = pd.DataFrame(columns=["id", "label", "affection", "content", "if_marked_label", "event_label"])
    #         print("gentext传进来的df是空的")
    #
    #     # 不生成了，直接得到gen_texts
    #     if prompt == "prepared":
    #         # 定义数据
    #         data = {
    #             "id": ["5.8E+17", "5.8E+17", "5.8E+17"],
    #             "label": [0, 0, 0],
    #             "affection": ["x", "x", "x"],
    #             "content": [
    #                 "Pray for #4U9525 http://t.co/II7Rl24ffH",
    #                 "Airbus A320 #4U9525 crash: Flight tracking data here: http://t.co/9W6hfTGYQV #airbus http://t.co/RchXQsoqdJ",
    #                 "Flightradar24 has this as the plane's last position. #4U9525 https://t.co/KffO4s4N6G"
    #             ],
    #             "if_marked_label": [0, 0, 0],
    #             "event_label": [2, 2, 2]
    #         }
    #         # 转换为DataFrame
    #         df_tmp = pd.DataFrame(data)
    #
    #     gen_texts = pd.DataFrame(columns=["id", "label", "affection", "content", "if_marked_label", "event_label"])
    #
    #     if prompt == "normal":
    #
    #         for i in range(samples):
    #             print(f"提取第{i}个样本开始——————————————————————————————————————")
    #             # 生成文本
    #
    #             init_text = df_tmp['content'].iloc[i]
    #             print(f"原文本是什么样的init_text{i}", init_text)
    #
    #             outtext = self.chat(tasknum=3
    #                                 , num=num, text=init_text, pro_type=pro_type, out_type=out_type,
    #                                 if_soft=if_soft)  # 使用整数索引来访问 DataFrame
    #
    #             print("outtext:", outtext)
    #
    #             sentences = outtext.split("\n")
    #             # sentences = sentences[5:]
    #             # print("spilt之后的结果",sentences)
    #             tmp_sen = []
    #             # sentences = [x for x in sentences if len(x) > int(len(x) * 0.8)]
    #             sentences = sorted(sentences, key=len, reverse=True)[:num]
    #             # for sentence in range(0, len(sentences)):
    #             #     if len(sentence)>int(len(text)*0.7):
    #             #        tmp_sen.append(sentence)
    #
    #             # sentences = sentences[0::2]
    #             # sentences = re.findall(r'~~~(.*?)~~~', outtext)
    #             if sentences:
    #
    #                 # 怎么从输出结果里拿到句子
    #                 print("生成出来的几个句子是什么样的")
    #                 print("提取出来的文本是：————————————————————————————————————————————")
    #                 for generated_sentence in sentences:
    #                     print(generated_sentence)
    #
    #                 print(f"一个文本生成句子结束，一共{num}个")
    #                 print("提取出来的文本在上面：————————————————————————————————————————")
    #
    #             else:
    #                 # raise ValueError("No sentences found in the output text.")
    #                 # print("没有找到~~~")
    #                 print("no sentences")
    #             # print(sentences)
    #             generated_texts = sentences
    #
    #             # 创建一个新的样本，将生成的文本添加到 DataFrame 中
    #             # 将新生成的文本转换为情感特征
    #             print("情感提取开始")
    #             affection = []
    #             with open(os.devnull, 'w') as fnull:
    #                 with redirect_stdout(fnull), redirect_stderr(fnull):
    #                     affection = self.to_BiGRU_input(pd.DataFrame({'content': generated_texts}),
    #                                                     flag='generated')
    #
    #             affection_list = affection.tolist()
    #             # print(affection_list[0], type(affection_list[0]))
    #
    #             affection_list_str = [str(x) for x in affection_list]
    #
    #             # affection_list_str = ["\n".join(["\t".join(map(str, row.numpy())) for row in tensor]) for tensor in
    #             #                    affection_list]
    #
    #             # # 打印结果
    #             # for i, tensor_str in enumerate(affection_list_str):
    #             #     print(f"Tensor {i} as string:\n{tensor_str}\n")
    #
    #             # print(affection_list_str[0], type(affection_list_str[0]))
    #             # print("_____________________________________________________")
    #
    #             aff_list = aff_list + affection_list_str
    #             # print(aff_list)
    #
    #             # 将情感特征添加到新生成的文本所在的行中
    #             # 假设 new_content_features 是一个列表，其中包含了生成文本的情感特征
    #             print("情感提取结束")
    #
    #             for j, text in enumerate(generated_texts):
    #                 if j >= num:
    #                     break
    #                 new_sample = df_tmp.iloc[i].copy()  # 复制第 i 行的数据
    #                 #
    #                 new_sample['content'] = text  # 将生成的文本添加到新样本的 'content' 列中
    #
    #                 # aff = str(affection[j].numpy())
    #                 # new_sample['affection'] = aff  # 把每个情感都替换原来的感情
    #
    #                 print(f"new_sample['content']{j}: ", text)
    #                 print(f"这次的new_sample{j}是怎么样的", new_sample)
    #
    #                 # aff = affection[j]
    #                 # print(aff, type(aff))
    #                 # aff = aff.numpy()
    #                 # aff = str(aff)
    #                 # print(f"new_sample['affection']{j}: ", aff)
    #                 # print(new_sample.to_frame().T)
    #                 new_sample = new_sample.to_frame().T
    #                 # 忘保存了
    #                 # 将新样本添加到 DataFrame 的末尾
    #                 gen_texts = pd.concat([gen_texts, new_sample], ignore_index=True)
    #             (f"提取第{i}个样本结束——————————————————————————————————————————————————————")
    #
    #     # gen_texts["affection"] = aff_list
    #     # print(aff_list)
    #
    #     return gen_texts  # 返回生成样本后的 DataFrame

    def chat(self, tasknum, num, text, pro_type="text", out_type="text", if_soft=0):
        import re
        import torch

        self.num = int(num)
        self.tasknum = int(tasknum)

        # ---- ensure text is plain str ----
        if not isinstance(text, str):
            try:
                if torch.is_tensor(text):
                    ids = text.detach().to('cpu').tolist()
                    text = self.bert_tokenizer.decode(ids, skip_special_tokens=True)
                elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                    text = self.bert_tokenizer.decode(list(text), skip_special_tokens=True)
                else:
                    text = str(text)
            except Exception:
                text = str(text)

        src = (text or "").strip()
        if not src:
            return ""

        style = self.task[self.tasknum]

        # ---- short prompt: fewer chances to "teach" ----
        # 注意：不写 Output: / Text: 之类标签，避免模型学着输出标签
        def _build_prompt(s, st):
            return (
                f"Rewrite this ONE sentence using {st} word choices (minimal substitutions). "
                f"Keep meaning/facts; keep names/numbers unchanged. "
                f"Return ONE sentence only, no extra text.\n"
                f"{s}"
            )

        # ---- postprocess: cut off any leakage like "Text:" / "Rewrite:" ----
        _CUT_PAT = re.compile(
            r"(\bText\s*:|\bRewrite\s*:|\bRewritten\s*:|\bRewrites\s*:|\bOriginal\s*:|\bSentence\s*:|\bBased on\b|<<|<</TEXT>>|<</TEXT>|<\|)",
            flags=re.IGNORECASE
        )

        def _clean_one(raw, src):
            """
            Robust cleaner for one generation result.
            Fix: sometimes raw is empty / only newlines / None -> splitlines()[0] crashes.
            Strategy:
              - Normalize raw to str
              - Remove empty lines
              - If still empty: fallback to src (or empty string)
            """
            # ---------- normalize ----------
            if raw is None:
                raw = ""
            if not isinstance(raw, str):
                raw = str(raw)

            raw = raw.strip()

            # ---------- guard: empty raw ----------
            if raw == "":
                # 回退到 src，避免整个流程崩
                if isinstance(src, str) and src.strip():
                    return src.strip()
                return ""

            # ---------- safe first line extraction ----------
            # 去掉空行，取第一条非空行
            lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
            if not lines:
                if isinstance(src, str) and src.strip():
                    return src.strip()
                return ""

            raw = lines[0]

            # ---------- 你原来的清洗逻辑（保留/延续） ----------
            # 下面这些是常见的清洗，你如果原函数里还有其它规则，请接着放在这里
            # 比如去掉引号、去掉特殊前缀、去掉多余空格等
            raw = raw.strip().strip('"').strip("'").strip()

            # 如果清完又空了，再次兜底
            if raw == "":
                if isinstance(src, str) and src.strip():
                    return src.strip()
                return ""

            return raw

        # ---- minimal bad_words_ids (fast): only block chat role tokens ----
        if not hasattr(self, "_min_bad_words_ids"):
            bad_tokens = ["<|user|>", "<|assistant|>", "<|system|>", "<|im_start|>", "<|im_end|>"]
            bw = []
            for t in bad_tokens:
                try:
                    tid = self.tokenizer.convert_tokens_to_ids(t)
                    if isinstance(tid, int) and tid != self.tokenizer.unk_token_id:
                        bw.append([tid])
                except Exception:
                    pass
            self._min_bad_words_ids = bw if bw else None

        # ---- eos ids (cached) ----
        if not hasattr(self, "_eos_ids"):
            eos_ids = []
            try:
                if self.tokenizer.eos_token_id is not None:
                    eos_ids.append(self.tokenizer.eos_token_id)
            except Exception:
                pass
            for tok in ["<|im_end|>", "<|endoftext|>"]:
                try:
                    tid = self.tokenizer.convert_tokens_to_ids(tok)
                    if isinstance(tid, int) and tid != self.tokenizer.unk_token_id:
                        eos_ids.append(tid)
                except Exception:
                    pass
            # unique & remove None
            eos_ids = [x for x in dict.fromkeys(eos_ids) if x is not None]
            self._eos_ids = eos_ids

        # ---- generation config (speed-first) ----
        # 关键：max_new_tokens 小很多，会显著减少“扩写/跑题/夹带示例”
        GEN = dict(
            max_new_tokens=48,
            do_sample=True,
            temperature=0.25,
            top_p=0.85,
            top_k=0,  # top_k=0 通常更快也更稳（只用 nucleus）
            repetition_penalty=1.05,
            num_beams=1,
            use_cache=True,
        )

        # ---- pro_type == text (pipeline) ----
        if pro_type == "text":
            outs = []
            for _ in range(self.num):
                prompt = _build_prompt(src, style)
                with torch.inference_mode():
                    # return_full_text=False 可以避免把 prompt 拼回输出（若你的 transformers 版本支持）
                    try:
                        outputs = self.pipe(prompt, return_full_text=False, **GEN)
                        raw = outputs[0]["generated_text"]
                    except TypeError:
                        outputs = self.pipe(prompt, **GEN)
                        raw = outputs[0]["generated_text"]
                        # strip prompt if pipeline returned full text
                        if isinstance(raw, str) and raw.startswith(prompt):
                            raw = raw[len(prompt):].lstrip()

                one = _clean_one(raw, src)
                if one:
                    outs.append(one)

            return "\n".join(outs)

        # ---- pro_type == vec ----
        elif pro_type == "vec":
            dtype = next(self.model.parameters()).dtype

            init_text = _build_prompt(src, style)
            init_text_embedding, init_text_attention_mask = self.get_embeddings(init_text)
            init_text_embedding = init_text_embedding.to(self.device, dtype=dtype)
            init_text_attention_mask = init_text_attention_mask.to(self.device)

            # soft prompt prefix
            if if_soft == 1 and self.soft_prompt is None:
                raise RuntimeError("soft_prompt is None. Call load_soft_prompt() before using if_soft=1")

            if if_soft == 1:
                soft = self.soft_prompt.to(self.device, dtype=dtype)
                if soft.size(0) != init_text_embedding.size(0):
                    soft = soft.expand(init_text_embedding.size(0), -1, -1).contiguous()
                soft_mask = torch.ones(
                    (init_text_embedding.size(0), soft.size(1)),
                    device=self.device,
                    dtype=init_text_attention_mask.dtype,
                )
                combined_embeddings = torch.cat([soft, init_text_embedding], dim=1)
                combined_attention_mask = torch.cat([soft_mask, init_text_attention_mask], dim=1)
            else:
                combined_embeddings = init_text_embedding
                combined_attention_mask = init_text_attention_mask

            if out_type == "vec":
                last = self.get_last(combined_embeddings, combined_attention_mask)
                return combined_embeddings, last, combined_attention_mask, ""

            # out_type == "text"
            pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            seq_len = combined_embeddings.size(1)
            dummy_input_ids = torch.full(
                (combined_embeddings.size(0), seq_len),
                int(pad_id) if pad_id is not None else 0,
                device=self.device,
                dtype=torch.long
            )

            outs = []
            for _ in range(self.num):
                with torch.inference_mode():
                    gen_ids = self.model.generate(
                        input_ids=dummy_input_ids,
                        inputs_embeds=combined_embeddings,
                        attention_mask=combined_attention_mask,
                        pad_token_id=pad_id,
                        eos_token_id=self._eos_ids if len(self._eos_ids) > 1 else (
                            self._eos_ids[0] if self._eos_ids else None),
                        bad_words_ids=self._min_bad_words_ids,
                        **GEN
                    )

                new_ids = gen_ids[:, seq_len:]
                raw = self.tokenizer.decode(new_ids[0], skip_special_tokens=True).strip()
                one = _clean_one(raw, src)
                if one:
                    outs.append(one)

            return "\n".join(outs)

        else:
            raise ValueError(f"Unknown pro_type={pro_type}")

    def gen_text(self, df_tmp=None, num=2, samples=3, prompt="prepared",
                 pro_type="text", out_type="text",
                 start_epoch=1, if_soft=0, tasknum=3):
        import os
        import pandas as pd
        from contextlib import redirect_stdout, redirect_stderr

        print("_____________________________________________________")
        if df_tmp is None:
            df_tmp = pd.DataFrame(columns=["id", "label", "affection", "content", "if_marked_label", "event_label"])
            print("gentext传进来的df是空的")

        if prompt == "prepared":
            data = {
                "id": ["5.8E+17", "5.8E+17", "5.8E+17"],
                "label": [0, 0, 0],
                "affection": ["x", "x", "x"],
                "content": [
                    "Pray for #4U9525 http://t.co/II7Rl24ffH",
                    "Airbus A320 #4U9525 crash: Flight tracking data here: http://t.co/9W6hfTGYQV #airbus http://t.co/RchXQsoqdJ",
                    "Flightradar24 has this as the plane's last position. #4U9525 https://t.co/KffO4s4N6G"
                ],
                "if_marked_label": [0, 0, 0],
                "event_label": [2, 2, 2]
            }
            df_tmp = pd.DataFrame(data)

        gen_texts = pd.DataFrame(columns=["id", "label", "affection", "content", "if_marked_label", "event_label"])

        if prompt != "normal":
            return gen_texts

        # 生成时：每次只要 1 句，重复 num 次（比一次生成多句更稳）
        for i in range(min(samples, len(df_tmp))):
            print(f"提取第{i}个样本开始——————————————————————————————————————")
            init_text = str(df_tmp['content'].iloc[i])
            print(f"原文本是什么样的init_text{i}", init_text)

            generated_texts = []
            # 生成 num 句（不足就用已有的）
            for _ in range(num):
                out = self.chat(tasknum=tasknum, num=1, text=init_text,
                                pro_type=pro_type, out_type=out_type, if_soft=if_soft)
                out = (out or "").strip()
                if out:
                    # chat(num=1) 仍可能返回带换行（保险）
                    out = out.splitlines()[0].strip()
                    if out and out not in generated_texts:
                        generated_texts.append(out)

            # 展示
            if generated_texts:
                print("生成出来的句子：")
                print("提取出来的文本是：————————————————————————————————————————————")
                for s in generated_texts:
                    print(s)
                print(f"一个文本生成句子结束，一共{len(generated_texts)}个")
                print("提取出来的文本在上面：————————————————————————————————————————")
            else:
                print("no sentences")
                continue

            # 情感提取（保持你原逻辑：每个原样本一批生成句子一起提取）
            print("情感提取开始")
            with open(os.devnull, 'w') as fnull:
                with redirect_stdout(fnull), redirect_stderr(fnull):
                    affection = self.to_BiGRU_input(pd.DataFrame({'content': generated_texts}), flag='generated')
            print("情感提取结束")

            # 写入 DataFrame
            for j, txt in enumerate(generated_texts):
                new_sample = df_tmp.iloc[i].copy()
                new_sample['content'] = txt
                new_sample = new_sample.to_frame().T
                gen_texts = pd.concat([gen_texts, new_sample], ignore_index=True)

            (f"提取第{i}个样本结束——————————————————————————————————————————————————————")

        return gen_texts

    def plot_losses(self, title, num_per_epoch):

        plt.figure(figsize=(10, 5))
        plt.plot(self.losses, label='All Losses')
        plt.xlabel('Times of Training')
        plt.ylabel('Loss')
        plt.title(f'All Losses Epochs of {title}')
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(self.epoch_mean_losses, label='Loss Per Epoch')
        # print(self.epoch_mean_losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Loss Over Epochs of {title}')
        plt.legend()
        plt.grid(True)
        plt.show()


#
