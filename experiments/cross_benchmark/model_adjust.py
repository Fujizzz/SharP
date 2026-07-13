# -*- codeing = utf-8 -*-
# @Time : 2022-12-10 16:17
# @Author : 张超然
# @File ： model.py
# @Software: PyCharm
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
MAIN_CODE_DIR = REPO_ROOT / "DAAL"
if str(MAIN_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_CODE_DIR))

import torch
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sympy.physics.units import length
from torch.utils.data import Dataset, random_split
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from Focal_loss import focal_loss
import pickle
import random
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from os.path import exists
from torch import nn, optim
from torch.autograd import Function
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, BertConfig
import pandas as pd
import process_data as ProData
import amodel as amodel
from Samper import *
import csv
import os
from timeit import default_timer as timer
import seaborn as sns
import torch.distributed as dist
import torch.multiprocessing as mp
import json
import argparse
import os
import sys
from datetime import datetime







df_pool_len = pd.read_excel('../../data/twitter15_16/processed/twitter_pool_data.xlsx')
count = len(df_pool_len)
num = 32

fine_num = 0
easy = 0


# 114

def is_empty(x):
    if x is None:
        return True
    if isinstance(x, np.ndarray):
        return x.size == 0
    try:
        return len(x) == 0
    except TypeError:
        return False


# 继承自pytorch框架下的数据集基类
class Rumor_Data(Dataset):
    # dataset类——创建适应任意模型的数据集接口
    def __init__(self, dataset):
        self.text = dataset['text']
        self.mask = dataset['mask']
        self.affection = dataset['affection']
        self.label = dataset['label']
        self.event_label = dataset['event_label']
        self.if_marked_label = dataset['if_marked_label']
        self.data_index = dataset['data_index']
        # print('TEXT: %d, mask: %d, affection %d, label: %d,event_label: %d,if_marked_label: %d, index: %d'% (len(self.text),len(self.mask), len(self.affection), len(self.label),len(self.event_label),len(self.if_marked_label), len(self.data_index)) )

    def __len__(self):
        # __len__是指数据集长度
        return len(self.label)

    def __getitem__(self, idx):
        # __getitem__就是获取样本对，模型直接通过这一函数获得一对样本对{x:y:z:w...}
        return self.text[idx], self.mask[idx], self.affection[idx], self.label[idx], self.event_label[idx], \
        self.if_marked_label[idx], self.data_index[idx]


def log_performance(epoch, num_epochs, loss, class_loss, domain_loss, mark_loss, dynamic_lr, train_acc, log_file):
    performance_info = {
        'epoch': epoch + 1,
        'num_epochs': num_epochs,
        'loss': loss,
        'class_loss': class_loss,
        'domain_loss': domain_loss,
        'mark_loss': mark_loss,
        'dynamic_lr': dynamic_lr,
        'train_acc': train_acc
    }
    with open(log_file, 'a') as f:
        f.write(json.dumps(performance_info) + '\n')


# #梯度翻转层
# class ReverseLayerF(Function):
#     @staticmethod
#     def forward(self, x):
#         #lambd应该是用来缩放梯度的，放大程度越大，更能接近目标，更容易过拟合
#         self.lambd = 1
#
#         #传入下一层，不改变x值
#         return x.view_as(x)
#
#     @staticmethod
#     def backward(self, grad_output):
#         #在翻转参数上添加了一个-，表示将后面传来的参数取负数，再传递到前层
#         #借此实现反着梯度方向优化模型参数
#         return (grad_output * -self.lambd)


class ReverseLayerF(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL)
    forward: identity
    backward: multiply gradients by -1
    """

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output


def grad_reverse(x):
    # apply函数：判断变形金刚是否激活，运行
    return ReverseLayerF.apply(x)


def to_np(x):
    return x.data.cpu().numpy()


class Config(object):
    """配置参数"""

    def __init__(self, device):
        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # 设备
        self.device = device

        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_epochs = 3  # epoch数
        self.batch_size = 32  # mini-batch大小
        # 每句话处理成的长度(短填长切)
        self.pad_size = 64
        self.learning_rate = 3e-5  # 学习率

        self.bert_path = os.environ.get('SHARP_BERT_MODEL', 'bert-base-uncased')

        # bert切词器
        # 注意：你原代码里传了 weights_only=False，这里保持一致（避免环境差异带来行为变化）
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path, weights_only=False)

        # bert隐藏层个数（维度）?
        self.hidden_size = 16
        self.dropout = 0.3
        self.num_filters = 256
        self.filter_size = (2, 3, 4)

        # =========================================================
        # ✅ 领域鉴别器(event_label)类别数：自动适配任意数据集
        #    - event_num: 连续类别数 C
        #    - event_label2id: 原始label -> 连续id(0..C-1)
        #    - target_event_label_raw: 旧目标域默认取最小label（可自行覆盖）
        # =========================================================
        self.event_num = 5
        self.event_labels_raw = None
        self.event_label2id = None
        self.id2event_label = None
        self.target_event_label_raw = None
        self._unseen_event_labels = set()

        # =========================================================
        # ✅ gogogo() 中用于抽样的比例参数（可被命令行覆盖）
        #   用法：python model_adjust.py --less-frac 0.3
        # =========================================================
        self.less_frac = 0.4

        self._init_event_info_from_excel('../../data/twitter15_16/processed/twitter_train_data.xlsx')

    def _init_event_info_from_excel(self, xlsx_path: str):
        """从 train_data.xlsx 的 event_label 列推断类别数并建立映射。"""
        try:
            df = pd.read_excel(xlsx_path)
            if 'event_label' not in df.columns:
                print(f"[WARN] {xlsx_path} has no 'event_label' column; fallback event_num={self.event_num}")
                self._build_event_maps(list(range(self.event_num)))
                self.target_event_label_raw = 0
                return

            labels = df['event_label'].dropna().astype(int).unique().tolist()
            if len(labels) == 0:
                print(f"[WARN] {xlsx_path} 'event_label' is empty; fallback event_num={self.event_num}")
                self._build_event_maps(list(range(self.event_num)))
                self.target_event_label_raw = 0
                return

            labels = sorted([int(x) for x in labels])
            self.event_num = len(labels)
            self._build_event_maps(labels)

            # 默认把“旧目标域 event_label”设置为最小 label
            self.target_event_label_raw = labels[0]

            print(
                f"[INFO] event_num inferred = {self.event_num}, labels(raw) head={labels[:10]}{'...' if len(labels) > 10 else ''}")
            print(f"[INFO] target_event_label_raw = {self.target_event_label_raw}")
        except Exception as e:
            print(f"[WARN] Failed to infer event_num from {xlsx_path}: {e}. fallback event_num={self.event_num}")
            self._build_event_maps(list(range(self.event_num)))
            self.target_event_label_raw = 0

    def _build_event_maps(self, labels_raw):
        labels_raw = [int(x) for x in labels_raw]
        self.event_labels_raw = labels_raw
        self.event_label2id = {lab: i for i, lab in enumerate(labels_raw)}
        self.id2event_label = {i: lab for lab, i in self.event_label2id.items()}

    def map_event_label_to_id(self, raw_label: int) -> int:
        """原始 event_label -> 连续 id(0..C-1)。若出现未知标签，映射到 0（避免越界）并仅提醒一次。"""
        raw_label = int(raw_label)
        if not self.event_label2id:
            return raw_label

        if raw_label not in self.event_label2id:
            if not hasattr(self, "_unseen_event_labels"):
                self._unseen_event_labels = set()
            if raw_label not in self._unseen_event_labels:
                print(f"[WARN] unseen event_label={raw_label} not in mapping; mapped to 0")
                self._unseen_event_labels.add(raw_label)
            return 0

        return int(self.event_label2id[raw_label])


def load_state_dict_flexible(model: nn.Module, state_dict: dict, verbose: bool = True):
    """
    ✅ 兼容加载：当 event_label 类别数变化导致 domain_classifier 维度不同，
    这里会自动跳过 shape 不匹配的参数（例如 domain_classifier 的最后一层），保证其余权重可正常加载。
    """
    model_sd = model.state_dict()
    filtered = {}
    skipped = []

    for k, v in state_dict.items():
        if k in model_sd:
            try:
                if hasattr(v, "shape") and hasattr(model_sd[k], "shape") and tuple(v.shape) == tuple(model_sd[k].shape):
                    filtered[k] = v
                else:
                    skipped.append(k)
            except Exception:
                skipped.append(k)
        else:
            skipped.append(k)

    missing, unexpected = model.load_state_dict(filtered, strict=False)

    if verbose:
        if skipped:
            print(
                f"[INFO] load_state_dict_flexible skipped {len(skipped)} keys (shape/key mismatch). head={skipped[:5]}")
        if missing:
            print(f"[INFO] load_state_dict_flexible missing {len(missing)} keys after load. head={missing[:5]}")
        if unexpected:
            print(
                f"[INFO] load_state_dict_flexible unexpected {len(unexpected)} keys after load. head={unexpected[:5]}")

    return skipped, missing, unexpected


# 注意：此处只是建立了神经网络群，还没有对这个群下达任何指令，如：train,predict等
class MyNet(nn.Module):
    def __init__(self, config):
        super(MyNet, self).__init__()

        # -------- basic params --------
        self.batch_size = 16
        self.hidden_size = 32
        self.event_num = int(getattr(config, "event_num", 5))  # ✅ 自动适配 event_label 类别数
        self.fusion_dim = 64  # ✅ 关键：融合维度固定 64（匹配 BiGRU 输出）

        # -------- BERT encoder --------
        model_config = BertConfig.from_pretrained(config.bert_path, output_hidden_states=True)
        self.bert = BertModel.from_pretrained(config.bert_path, config=model_config)

        for p in self.bert.parameters():
            p.requires_grad = True

        self.dropout = nn.Dropout(getattr(config, "dropout", 0.1))

        # LLM hidden (4096) -> BERT hidden (768)
        self.llm_proj = nn.Linear(4096, 768)

        # ✅ 可选：domain 分支只训练 domain head，避免冲炸主干
        self.detach_domain_feat = True

        # -------- TEXT-CNN branch 1 --------
        self.convs1 = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, 768)) for k in config.filter_size]
        )
        self.text_relu1_1 = nn.LeakyReLU(True)

        # ✅ 关键：fc 输出维度不要用 batch_size，必须是特征维度（这里用 64）
        self.fc1 = nn.Linear(config.num_filters * len(config.filter_size), self.fusion_dim)
        self.fc1_2 = nn.Linear(self.fusion_dim, self.fusion_dim)
        self.text_relu1_2 = nn.LeakyReLU(True)

        # -------- TEXT-CNN branch 2 --------
        self.convs2 = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, 768)) for k in config.filter_size]
        )
        self.text_relu2_1 = nn.LeakyReLU(True)

        self.fc2 = nn.Linear(config.num_filters * len(config.filter_size), self.fusion_dim)
        self.fc2_2 = nn.Linear(self.fusion_dim, self.fusion_dim)
        self.text_relu2_2 = nn.LeakyReLU(True)

        # -------- GRU layers --------
        self.gru1 = nn.GRU(24, 32, batch_first=True, bidirectional=True)  # -> [B, T, 64]
        self.gru2 = nn.GRU(24, 32, batch_first=True, bidirectional=True)

        # -------- Branch 1: Fake/Real classifier (logits for CE) --------
        self.class_classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 2)  # ✅ logits, no softmax
        )

        # -------- Branch 2: Domain classifier (logits for CE) --------
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, self.hidden_size),
            nn.LeakyReLU(True),
            nn.Dropout(getattr(config, "dropout", 0.1)),
            nn.Linear(self.hidden_size, self.event_num)  # ✅ logits, no softmax
        )

        # -------- Branch 3: infer_discriminator (unused in your shown forward) --------
        self.infer_discriminator = nn.Sequential(
            nn.Linear(self.fusion_dim, self.hidden_size),
            nn.ReLU(True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(True),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )

        # -------- Branch 4: affection_discriminator (marked/unmarked) --------
        self.affection_discriminator = nn.Sequential(
            nn.Linear(self.fusion_dim, self.hidden_size),
            nn.ReLU(False),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )

    def conv_and_pool(self, x, conv):
        # x: [B, 1, seq_len, 768]
        x = conv(x)          # -> [B, num_filters, seq_len-k+1, 1]
        x = F.relu(x)
        x = x.squeeze(3)     # -> [B, num_filters, seq_len-k+1]
        x = F.max_pool1d(x, x.size(2))
        x = x.squeeze(2)     # -> [B, num_filters]
        return x

    def _fusion(self, fc_out, gru_out):
        """
        ✅ 修复点：用 batch-wise 融合，不用 torch.mm
        fc_out: [B,64], gru_out: [B,64]  ->  [B,64]
        """
        return fc_out * gru_out

    def forward(self, x1, x2, x3, x4, x5, x6, flag, embedding=None):
        # flag==0: pretrain/test multi-branch
        # flag==1: finetune fake/real branch
        # flag==3: finetune marked branch（你代码里用 affection_discriminator）
        # flag==5: use external embedding -> proj -> bert(inputs_embeds)
        # =========================
        # ✅ 关键修复：flag 归一化
        #   - load_dataset 用 flag=2 表示“后续微调数据”
        #   - 但 MyNet.forward 只支持 0/1/3/5
        #   - 所以把 2 映射为 1（真假鉴别分支）
        # =========================
        if isinstance(flag, torch.Tensor):
            # 兼容标量 / 形状 (1,) 等
            flag_val = int(flag.view(-1)[0].item())
        else:
            flag_val = int(flag)

        if flag_val == 2:
            flag_val = 1

        flag = flag_val  # 后面逻辑继续用 int


        dev = next(self.parameters()).device

        if flag == 0:
            context1, mask1 = x1.to(dev), x2.to(dev)
            context2, mask2 = x3.to(dev), x4.to(dev)
            gru_inputs1 = x5.to(dev).float()
            gru_inputs2 = x6.to(dev).float()

            o1 = self.bert(context1, attention_mask=mask1)
            o2 = self.bert(context2, attention_mask=mask2)

            # ----- branch 1 feature -----
            emb1 = o1.last_hidden_state                     # [B, L, 768]
            new1 = emb1.unsqueeze(1)                        # [B, 1, L, 768]
            out1 = torch.cat([self.conv_and_pool(new1, c) for c in self.convs1], dim=1)
            out1 = self.text_relu1_1(out1)
            fc_out1 = self.fc1(out1)                        # [B, 64]

            gru_out1, _ = self.gru1(gru_inputs1, None)       # [B, T, 64]
            gru_out1 = gru_out1[:, -1, :]                    # [B, 64]

            fused1 = self._fusion(fc_out1, gru_out1)         # ✅ [B, 64]
            final_out1 = self.text_relu1_2(self.fc1_2(fused1))

            # ----- branch 2 feature (domain/marked) -----
            emb2 = o2.last_hidden_state
            new2 = emb2.unsqueeze(1)
            out2 = torch.cat([self.conv_and_pool(new2, c) for c in self.convs2], dim=1)
            out2 = self.text_relu2_1(out2)
            fc_out2 = self.fc2(out2)                         # [B, 64]

            gru_out2, _ = self.gru2(gru_inputs2, None)
            gru_out2 = gru_out2[:, -1, :]                    # [B, 64]

            fused2 = self._fusion(fc_out2, gru_out2)         # ✅ [B, 64]
            final_out2 = self.text_relu2_2(self.fc2_2(fused2))

            # Branch 1 logits
            class_logits = self.class_classifier(final_out1)  # [B,2], logits

            # Branch 2 domain logits (GRL)
            dom_feat = final_out2.detach() if self.detach_domain_feat else final_out2
            reverse_feature = grad_reverse(dom_feat)          # 你已有 GRL
            domain_logits = self.domain_classifier(reverse_feature)  # [B,event_num], logits

            # Branch 4 marked prob
            marked_prob = self.affection_discriminator(final_out2)   # [B,1], sigmoid prob

            return class_logits, domain_logits, marked_prob

        elif flag == 1:
            context1, mask1 = x1.to(dev), x2.to(dev)
            gru_inputs1 = x5.to(dev).float()

            o1 = self.bert(context1, attention_mask=mask1)
            emb1 = o1.last_hidden_state
            new1 = emb1.unsqueeze(1)

            out1 = torch.cat([self.conv_and_pool(new1, c) for c in self.convs1], dim=1)
            out1 = self.text_relu1_1(out1)
            fc_out1 = self.fc1(out1)                         # [B,64]

            gru_out1, _ = self.gru1(gru_inputs1, None)
            gru_out1 = gru_out1[:, -1, :]                    # [B,64]

            fused1 = self._fusion(fc_out1, gru_out1)          # ✅ [B,64]
            final_out1 = self.text_relu1_2(self.fc1_2(fused1))

            logits = self.class_classifier(final_out1)        # [B,2], logits
            return [logits, final_out1]

        elif flag == 3:
            # 你的 train_finetune3_epoch 用的是 models(..., flag=3) 直接拿输出做 BCELoss
            # 这里沿用 “marked_prob” 的逻辑：只需要 final_out2 -> affection_discriminator
            context, mask = x1.to(dev), x2.to(dev)
            gru_inputs = x5.to(dev).float()

            o = self.bert(context, attention_mask=mask)
            emb = o.last_hidden_state
            new = emb.unsqueeze(1)

            out = torch.cat([self.conv_and_pool(new, c) for c in self.convs2], dim=1)
            out = self.text_relu2_1(out)
            fc_out = self.fc2(out)                           # [B,64]

            gru_out, _ = self.gru2(gru_inputs, None)
            gru_out = gru_out[:, -1, :]                      # [B,64]

            fused = self._fusion(fc_out, gru_out)            # ✅ [B,64]
            final = self.text_relu2_2(self.fc2_2(fused))

            marked_prob = self.affection_discriminator(final)  # [B,1]
            return marked_prob

        elif flag == 5:
            # embedding: [B, seq_len, 4096]
            if embedding is None:
                raise ValueError("flag==5 requires `embedding`")

            emb = embedding.to(dev)
            emb = emb.to(dtype=self.llm_proj.weight.dtype)
            emb = self.llm_proj(emb)  # -> [B, seq_len, 768]

            # attention mask for inputs_embeds (all ones)
            attn_mask = torch.ones(emb.size()[:2], device=dev, dtype=torch.long)

            # 如果 seq_len 很长，分块走 bert，再拼接
            seq_len = emb.size(1)
            max_length = 512
            chunks = [emb[:, i:i + max_length, :] for i in range(0, seq_len, max_length)]
            mask_chunks = [attn_mask[:, i:i + max_length] for i in range(0, seq_len, max_length)]

            outs = []
            for c, m in zip(chunks, mask_chunks):
                o = self.bert(inputs_embeds=c, attention_mask=m)
                outs.append(o.last_hidden_state)
            emb_out = torch.cat(outs, dim=1)  # [B, seq_len, 768]

            # conv expects [B,1,L,768]
            new = emb_out.unsqueeze(1)
            out1 = torch.cat([self.conv_and_pool(new, conv) for conv in self.convs1], dim=1)
            out1 = self.text_relu1_1(out1)
            fc_out1 = self.fc1(out1)  # [B,64]

            gru_inputs1 = x5.to(dev).float()
            gru_out1, _ = self.gru1(gru_inputs1, None)
            gru_out1 = gru_out1[:, -1, :]  # [B,64]

            fused = self._fusion(fc_out1, gru_out1)          # ✅ [B,64]
            final = self.text_relu1_2(self.fc1_2(fused))

            logits = self.class_classifier(final)            # [B,2]
            return [logits, final]

        else:
            raise ValueError(f"Unsupported flag: {flag}")

def modify_if_marked_label(add_data):
    # 相当于把目标域，选择出来的有标记的数据的mark标记位置为1, add_data即为每次选择出来的数据, 类型是dataframe
    add_data['if_marked_label'] = add_data['if_marked_label'].replace({0: 1})

    # 记得在这里可以填充人工上标签的部分
    pass


def func1(amount, num):
    # 生成和固定为amount，个数为num的列表
    list1 = []
    for i in range(0, num - 1):
        a = random.randint(0, amount)  # 生成 n-1 个随机节点
        list1.append(a)
    list1.sort()  # 节点排序
    list1.append(amount)  # 设置第 n 个节点为amount，即总金额

    list2 = []
    for i in range(len(list1)):
        if i == 0:
            b = list1[i]  # 第一段长度为第 1 个节点 - 0
        else:
            b = list1[i] - list1[i - 1]  # 其余段为第 n 个节点 - 第 n-1 个节点
        list2.append(b)
    return list2


def vis_rep_vec(data, title, method):
    # print(f"Data shape before reshape: {data.shape}")
    # if data.ndim == 1:
    #     data = data.reshape(-1, 1)
    # print(f"Data shape after reshape: {data.shape}")
    data = data.detach().cpu().numpy()

    method = method.upper()  # 将 method 转换为大写

    if method == 'PCA':
        reducer = PCA(n_components=2)
        reduced_data = reducer.fit_transform(data)
        plt.title(f'PCA of {title}')
    elif method == 'TSNE':
        # 检查样本数量，确保 perplexity 合理
        n_samples = data.shape[0]
        perplexity = min(30, n_samples - 1)  # 设置合理的 perplexity
        reducer = TSNE(n_components=2, perplexity=perplexity)
        reduced_data = reducer.fit_transform(data)
        plt.title(f'TSNE of {title}')
    else:
        raise ValueError("Method should be either 'PCA' or 'TSNE'")

    plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()

    # if method == 'pca':
    #     pca = PCA(n_components=2)
    #     data_2d = pca.fit_transform(data)
    #
    #     # 画图
    #     plt.figure(figsize=(8, 6))
    #     plt.scatter(data_2d[:, 0], data_2d[:, 1], c='blue', marker='o', edgecolor='k', s=5)
    #     plt.title(f'PCA of {title}')
    #     plt.xlabel('Principal Component 1')
    #     plt.ylabel('Principal Component 2')
    #     plt.grid(True)
    #     plt.show()
    # #
    # #
    # #
    # if method == 'tsne':
    #     tsne = TSNE(n_components=2, random_state=42)
    #     data_2d = tsne.fit_transform(data)
    #     perplexity = min(30, len(data) - 1)  # 确保 perplexity 小于样本数量
    #
    #     # 画图
    #     plt.figure(figsize=(8, 6))
    #     plt.scatter(data_2d[:, 0], data_2d[:, 1], c='blue', marker='o', edgecolor='k', s=50)
    #     plt.title('t-SNE of Emotional Feature Vectors')
    #     plt.xlabel('Dimension 1')
    #     plt.ylabel('Dimension 2')
    #     plt.grid(True)
    #     plt.show()

    return


def load_pm(models, df_tmp, prompt="nothing", if_soft=0, samples=2, num=1):
    global pm
    start_time = timer()
    # 循环生成样本并添加到 DataFrame 中
    # for _ in range(len(df_tmp)):
    # 根据index个原文本生成n个样本，返回之前选出来的样本df_tmp

    # select_text = df_tmp["content"]
    print("if_soft现在是否使用了soft", if_soft)
    if prompt == "prepared" or prompt == "normal":
        # df_tmp = df_tmp.iloc[0:3]
        # df_tmp[]

        inputdata = mk_my_dataset(df_tmp)
        dataloader1 = DataLoader(dataset=inputdata, batch_size=16, shuffle=True, drop_last=False)
        input_score, input_vector = get_rep_vec1(models, dataloader1, flag=1)

        gen_texts = pm.gen_text(df_tmp, samples=samples, num=num, prompt=prompt, pro_type="vec", out_type="text",
                                if_soft=if_soft)

        outputdata = mk_my_dataset(gen_texts)
        dataloader2 = DataLoader(dataset=outputdata, batch_size=16, shuffle=True, drop_last=False)
        output_score, output_vector = get_rep_vec1(models, dataloader2, flag=1)

        input_texts = df_tmp['content'].tolist()  # 输入文本列表
        output_texts = gen_texts['content'].tolist()  # 生成文本列表

        # 将输入文本、生成文本和特征向量转换为可以保存的数据格式
        data = []

        # 假设 `input_vector` 和 `output_vector` 是一个二维的 NumPy 数组，每一行是一个向量
        for input_text, output_text, input_vec, output_vec in zip(input_texts, output_texts, input_vector,
                                                                  output_vector):
            data.append({
                'Input Text': input_text,
                'Output Text': output_text,
                'Input Vector': str(input_vec.tolist()),  # 将向量转换为字符串以便保存
                'Output Vector': str(output_vec.tolist())  # 将向量转换为字符串以便保存
            })

        # 将数据转换为 DataFrame
        df = pd.DataFrame(data)

        # 以追加模式保存到 CSV 文件
        # 如果文件已存在，确保不重复写入标题
        file_path = os.path.join(os.path.dirname(__file__), 'input_output_vectors.csv')

        # 判断文件是否存在
        write_header = not os.path.exists(file_path)

        # 将 DataFrame 保存到 CSV 文件，追加模式
        df.to_csv(file_path, index=False, mode='a', header=write_header)

        # print(gen_texts)
        # print(df_tmp.columns)
        # print(df_tmp["content"])
        # print(gen_texts["affection"])

        # gen_texts = gen_texts.reset_index(drop=True)
        #
        # index_gen = list(gen_texts.index)
        #
        #
        #
        # index_gen_data = mydict.to_bert_input_new(gen_texts, index_gen)
        #
        # gen_data = Rumor_Data(index_gen_data)
        #
        # gen_loader = DataLoader(dataset=gen_data,
        #                            batch_size=16,
        #                            shuffle=True,
        #                            drop_last=True)
        #
        # global fine_num
        #
        # get_rep_vec(model, gen_loader, fine_num=fine_num)

        df_tmp = pd.concat([gen_texts, df_tmp, ], axis=0)

    spend_time = timer() - start_time
    print(f"pm模块用了{spend_time}时间")
    return df_tmp


# 从之前封装好的 pkl 中得到读取data---W
def load_data(flag, models, data_index, fine_num=0):
    global easy
    # 注意看，W是只是一个缓存，代表着pickle文件，仅此而已
    # 如果是初始化训练，第一次的话，就直接把数据加载进来了d
    if len(data_index) == 0:
        if flag == "train1":
            data_path = '../../data/twitter15_16/processed/twitter_source_data.pkl'
            f = open(data_path, 'rb')
            w = pickle.load(f)
        elif flag == "train2":
            data_path = '../../data/twitter15_16/processed/twitter_source_extend_data.pkl'
            f = open(data_path, 'rb')
            w = pickle.load(f)

        elif flag == "test":
            data_path = '../../data/twitter15_16/processed/twitter_destination_data.pkl'
            f = open(data_path, 'rb')
            w = pickle.load(f)
        elif flag == "validate":
            data_path = '../../data/twitter15_16/processed/twitter_validate_data.pkl'
            f = open(data_path, 'rb')
            w = pickle.load(f)
        elif flag == "pool":
            data_path = '../../data/twitter15_16/processed/twitter_source_pool_data.pkl'
            f = open(data_path, 'rb')
            w = pickle.load(f)
    # 对于之后的任意次：2，3，4，5次
    else:
        print("data_index---info(len, max):", len(data_index), np.max(data_index))
        if flag == "fine_train":
            # 把data_index的数据 从pool_data中搞出来，增加到train1中，并改变if_marked_label0到1
            df = pd.read_excel('../../data/twitter15_16/processed/twitter_pool_data.xlsx')  # 即完整的目标域数据  加载出test的字典张量数据
            print("pool_data文件的长度：", len(df))
            df_tmp = df.iloc[data_index]  # 这是在初始数据上，根据append_index(是之前文件的index)加进去和删除的数据
            modify_if_marked_label(df_tmp)

            # #！！！！！！！！！轻量化处理
            # easy = 0
            # if easy:
            #     df_tmp = df_tmp.sample(frac=0.1, random_state=42)  # 随机选择 10% 的数据

            print("new_train_data文件的长度：", len(df_tmp))
            selected_dataset = mk_my_dataset(df_tmp)

            # 每个样本都放到amodel里面拿到embedding
            print(f"mynet的第{fine_num}次微调的soft微调开始")
            finetune_amodel(pm, model, selected_dataset, epochs=1, obj="fine")

            # 在这里插入llm
            print(f"mynet的第{fine_num}次微调的样本模仿生成开始，load_pm")






            df_tmp = load_pm(models, df_tmp, prompt="normal", if_soft=1, samples=5, num=2)






            df_source_extend = pd.read_excel('../../data/twitter15_16/processed/twitter_train_data.xlsx')  # 源 + 0.1目标

            # ✅ 不写死 event_label==0：旧目标域标签来自 config.target_event_label_raw（默认最小label）
            target_raw = getattr(config, 'target_event_label_raw', None)
            if target_raw is None:
                try:
                    target_raw = int(df_source_extend['event_label'].dropna().astype(int).min())
                except Exception:
                    target_raw = 0

            df_old_add = df_source_extend[df_source_extend['event_label'].astype(int) == int(target_raw)]
            if len(df_old_add) == 0:
                print(f"[WARN] No samples found in train_data.xlsx with event_label={target_raw}. Skip df_old_add.")
                df_old_add = df_source_extend.iloc[0:0]
            else:
                df_old_add = df_old_add.sample(frac=0.1, random_state=42)
            df_new_train = pd.concat([df_tmp, df_old_add, ], axis=0)  # 新的训练数据=0.1+0.05
            df_new_train = df_new_train.reset_index(drop=True)  # 重置了索引
            df_new_train.to_excel('../../data/twitter15_16/processed/new_train_data.xlsx',
                                  index=False)  # 这里现在是 0.1的目标+之后选择出来的0.05的目标
            print("new_train_data文件的长度：", len(df_new_train))
            index_train = list(df_new_train.index)  # 1...n
            new_train_data = mydict.to_bert_input_new(df_new_train, index_train)
            w = new_train_data
        elif flag == "fine_pool":
            df_pool = pd.read_excel('../../data/twitter15_16/processed/twitter_pool_data.xlsx')
            df_new_pool = df_pool.drop(df_pool.index[data_index])  # 训练备选数据删去后index后得到真正的pool样本 这个索引应该还是原来的索引
            df_new_pool = df_new_pool.reset_index(drop=True)  # 重置了索引
            df_new_pool.to_excel('../../data/twitter15_16/processed/new_pool_data.xlsx', index=False)  # 这是删去0.05的目标
            index_pool = list(df_new_pool.index)  # 返回回去的已经是重置之后的索引了
            new_test_data = mydict.to_bert_input_new(df_new_pool, index_pool)
            w = new_test_data



        elif flag == "fine_train_after":
            df1 = pd.read_excel('../../data/twitter15_16/processed/new_pool_data.xlsx')
            df_train_after = df1.iloc[data_index]  # 选出来
            modify_if_marked_label(df_train_after)

            # ！！！！！！！！！轻量化处理
            # easy = 0
            # if easy:

            #     df_train_after = df_train_after(frac=0.1, random_state=42)  # 随机选择 10% 的数据

            print("new_train_data文件的长度：", len(df_train_after))
            selected_dataset = mk_my_dataset(df_train_after)

            # 每个样本都放到amodel里面拿到embedding
            print(f"mynet的第{fine_num}次微调的soft微调开始")
            finetune_amodel(pm, model, selected_dataset, epochs=1, obj="fine")

            # 模仿生成
            print(f"mynet的第{fine_num}次微调的样本模仿生成开始，load_pm")





            df_train_after = load_pm(models, df_train_after, prompt="normal", if_soft=1, samples=5, num=2)







            df_train_before = pd.read_excel('../../data/twitter15_16/processed/new_train_data.xlsx')
            df_train_after = pd.concat([df_train_before, df_train_after], axis=0)
            df_train_after = df_train_after.reset_index(drop=True)
            print("new_train_data文件的长度：", len(df_train_after))
            df_train_after.to_excel('../../data/twitter15_16/processed/new_train_data.xlsx', index=False)
            # df_train_after.to_excel('../../data/twitter15_16/processed/new_train_after_data.xlsx', index=False)  # 这里现在是 0.1的目标+之后选择出来的0.05的目标
            index_train = list(df_train_after.index)
            new_train_after_data = mydict.to_bert_input_new(df_train_after, index_train)
            w = new_train_after_data
        elif flag == "fine_pool_after":
            df2 = pd.read_excel('../../data/twitter15_16/processed/new_pool_data.xlsx')
            print("new_test_data文件的长度：", len(df2))
            df_pool_after = df2.drop(df2.index[data_index])  # 删除掉
            df_pool_after = df_pool_after.reset_index(drop=True)

            df_pool_after.to_excel('../../data/twitter15_16/processed/new_pool_data.xlsx',
                                   index=False)  # 这里现在是 0.1的目标+之后选择出来的0.05的目标
            index_pool = list(df_pool_after.index)
            new_pool_after_data = mydict.to_bert_input_new(df_pool_after, index_pool)
            w = new_pool_after_data
    return w


# def load_data(flag, models, data_index, fine_num=0):
#     w = None  # 初始化w变量，确保它在函数的任何路径下都被赋值
#
#     # 如果data_index是一个非空的可迭代对象
#     if isinstance(data_index, (list, np.ndarray)) and len(data_index) == 0:
#         # 根据flag加载数据
#         if flag == "train1":
#             data_path = '../../data/twitter15_16/processed/source_data.pkl'
#             with open(data_path, 'rb') as f:
#                 w = pickle.load(f)
#         elif flag == "train2":
#             data_path = '../../data/twitter15_16/processed/source_extend_data.pkl'
#             with open(data_path, 'rb') as f:
#                 w = pickle.load(f)
#         elif flag == "test":
#             data_path = '../../data/twitter15_16/processed/sampled_data/event_label_4_.pkl'
#             with open(data_path, 'rb') as f:
#                 w = pickle.load(f)
#         elif flag == "validate":
#             data_path = '../../data/twitter15_16/processed/sampled_data/event_label_4_validate_.pkl'
#             with open(data_path, 'rb') as f:
#                 w = pickle.load(f)
#         elif flag == "pool":
#             data_path = '../../data/twitter15_16/processed/source_pool_data.pkl'
#             with open(data_path, 'rb') as f:
#                 w = pickle.load(f)
#     else:
#         print("data_index---info(len, max):", len(data_index), np.max(data_index))
#
#         # 处理fine_train情况
#         if flag == "fine_train":
#             df = pd.read_excel('../../data/twitter15_16/processed/sampled_data/pool_data_4.xlsx')  # 加载目标域数据
#             print("pool_data文件的长度：", len(df))
#             df_tmp = df.iloc[data_index]  # 根据索引选择数据
#             modify_if_marked_label(df_tmp)
#
#             print("new_train_data文件的长度：", len(df_tmp))
#             selected_dataset = mk_my_dataset(df_tmp)
#
#             # 每个样本都放到amodel里面拿到embedding
#             print(f"mynet的第{fine_num}次微调的soft微调开始")
#             finetune_amodel(pm, model, selected_dataset, epochs=1, obj="fine")
#
#             # 模仿生成
#             print(f"mynet的第{fine_num}次微调的样本模仿生成开始，load_pm")
#             df_tmp = load_pm(models, df_tmp, prompt="normal", if_soft=1, samples=6, num=2)
#
#             df_source_extend = pd.read_excel('../../data/twitter15_16/processed/train_data.xlsx')
#             df_old_add = df_source_extend[df_source_extend['event_label'] == 2]
#             df_old_add = df_old_add.sample(frac=0.1, random_state=42)
#
#             # 合并数据
#             df_new_train = pd.concat([df_tmp, df_old_add], axis=0)
#             df_new_train = df_new_train.reset_index(drop=True)
#             df_new_train.to_excel('../../data/twitter15_16/processed/new_train_data.xlsx', index=False)
#             print("new_train_data文件的长度：", len(df_new_train))
#             index_train = list(df_new_train.index)
#             new_train_data = mydict.to_bert_input_new(df_new_train, index_train)
#             w = new_train_data
#
#         elif flag == "fine_pool":
#             df_pool = pd.read_excel('../../data/twitter15_16/processed/sampled_data/pool_data_4.xlsx')
#             df_new_pool = df_pool.drop(df_pool.index[data_index])  # 从池数据中去除选择的数据
#             df_new_pool = df_new_pool.reset_index(drop=True)
#             df_new_pool.to_excel('../../data/twitter15_16/processed/new_pool_data.xlsx', index=False)
#             index_pool = list(df_new_pool.index)
#             new_test_data = mydict.to_bert_input_new(df_new_pool, index_pool)
#             w = new_test_data
#
#         elif flag == "fine_train_after":
#             df1 = pd.read_excel('../../data/twitter15_16/processed/new_pool_data.xlsx')
#             df_train_after = df1.iloc[data_index]  # 选择数据
#             modify_if_marked_label(df_train_after)
#
#             print("new_train_data文件的长度：", len(df_train_after))
#             selected_dataset = mk_my_dataset(df_train_after)
#
#             print(f"mynet的第{fine_num}次微调的soft微调开始")
#             finetune_amodel(pm, model, selected_dataset, epochs=1, obj="fine")
#
#             # 合并数据
#             df_train_before = pd.read_excel('../../data/twitter15_16/processed/new_train_data.xlsx')
#             df_train_after = pd.concat([df_train_before, df_train_after], axis=0)
#             df_train_after = df_train_after.reset_index(drop=True)
#             print("new_train_data文件的长度：", len(df_train_after))
#             df_train_after.to_excel('../../data/twitter15_16/processed/new_train_data.xlsx', index=False)
#             index_train = list(df_train_after.index)
#             new_train_after_data = mydict.to_bert_input_new(df_train_after, index_train)
#             w = new_train_after_data
#
#         elif flag == "fine_pool_after":
#             df2 = pd.read_excel('../../data/twitter15_16/processed/new_pool_data.xlsx')
#             print("new_test_data文件的长度：", len(df2))
#             df_pool_after = df2.drop(df2.index[data_index])
#             df_pool_after = df_pool_after.reset_index(drop=True)
#             df_pool_after.to_excel('../../data/twitter15_16/processed/new_pool_data.xlsx', index=False)
#             index_pool = list(df_pool_after.index)
#             new_pool_after_data = mydict.to_bert_input_new(df_pool_after, index_pool)
#             w = new_pool_after_data
#
#     # 如果w仍然没有被赋值，则抛出异常
#     if w is None:
#         raise ValueError("The variable 'w' was not assigned properly.")
#
#     return w


path = os.environ.get('SHARP_BERT_MODEL', 'bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained(path)
vocab_path = "./vocab.txt"
mydict = ProData.LoadSingleSentenceClassificationDataset(vocab_path, tokenizer)
config = Config(device=device)


# model2 = MyNet(config).to(device)

# 加载 dataset
def load_dataset(data_index, flag, fine_num=0, models=""):
    '''
    文件说明：
    '../../data/twitter15_16/processed/train_data.xlsx'
    '../../data/twitter15_16/processed/train2_data.xlsx'
    '../../data/twitter15_16/processed/test_data.xlsx'      这三个文件都是预处理数据的，即不能删改的

    '../../data/twitter15_16/processed/new_train_data.xlsx'
    '../../data/twitter15_16/processed/new_test_data.xlsx'  这两个文件都是微调时候，动态产生的文件，可以调整
    '''

    if flag == 0:  # 返回预训练时期的训练的测试原始数据
        train1 = load_data("train1", models, data_index)
        train2 = load_data("train2", models, data_index)

        pool = load_data("pool", models, data_index)
        test = load_data("test", models, data_index)
        validate = load_data("validate", models, data_index)
        # 加载训练数据
        print("loading data----------------------------------预训练阶段")
        train_dataset = Rumor_Data(train1)
        train_dataset2 = Rumor_Data(train2)
        # train_loader2是长的，即全部，train_loader是只说打了标签的

        pool_dataset = Rumor_Data(pool)

        # 加载验证集数据
        validate_dataset = Rumor_Data(validate)
        # 加载测试数据
        test_dataset = Rumor_Data(test)
        print(len(test_dataset))

        return train_dataset, train_dataset2, validate_dataset, test_dataset, pool_dataset
    if flag == 1:
        # 加载微调时期的训练数据和测试数据
        print("loading data----------------------------------第1次微调")
        fine_train = load_data("fine_train", models, data_index, fine_num=fine_num)  # 这里的长度是对的
        print(len(fine_train))
        fine_pool = load_data("fine_pool", models, data_index)  # 这里有问题，还是1093
        print(len(fine_pool))
        # 加载微调训练集
        fine_train_dataset = Rumor_Data(fine_train)
        # get_rep_vec(models=model, train_dataset=fine_train_dataset)
        # 加载微调测试集
        fine_pool_dataset = Rumor_Data(fine_pool)
        return fine_train_dataset, fine_pool_dataset
    if flag == 2:
        print(f"loading data----------------------------------后{fine_num}次微调")
        fine_train_after = load_data("fine_train_after", models, data_index, fine_num=fine_num)  # 这里的长度是对的

        fine_pool_after = load_data("fine_pool_after", models, data_index)  # 这里有问题，还是1093
        # 加载微调训练集
        fine_train_dataset = Rumor_Data(fine_train_after)
        # get_rep_vec(models=model, train_dataset=fine_train_dataset, fine_num=0)
        # 加载微调测试集
        fine_pool_dataset = Rumor_Data(fine_pool_after)
        return fine_train_dataset, fine_pool_dataset


# 定义预训练的训练方法，并测试是否能使用，最后选出最优的
def train(models, config, train_dataset, train_dataset2, test_dataset, pool_dataset, flag):
    criterion = nn.BCELoss()
    criterion2 = nn.CrossEntropyLoss()
    start_epoch = 0  # ！！！！！！！！！！！！！！！

    # 第一阶段结束后的文件
    final_file_name = ''
    best_dir = 'null'
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=16,
        shuffle=False,
        drop_last=False
    )
    pool_loader = DataLoader(
        dataset=pool_dataset,
        batch_size=16,
        shuffle=True,
        drop_last=False
    )
    print("预训练阶段：")
    print(f"test_dataset长度：{len(test_dataset)} | test_loader的长度：{len(test_loader)}")
    print(f"pool_dataset长度：{len(pool_dataset)} | pool_loader的长度：{len(pool_loader)}")

    # tst(models, config, train_dataset, test_loader, best_dir, flag, owner="DRCD not Pretrained")

    #  预训练！！
    if exists(final_file_name):
        print("加载已经最终完成的第一阶段预训练的模型")
        checkpoint = torch.load(final_file_name)
        load_state_dict_flexible(models, checkpoint['model'])
    else:

        # 训练4次！！！！！！！！！！！！！！
        print("训练模型！！!")
        for epoch in range(start_epoch, start_epoch + 8):
            train_epoch(models, train_dataset, train_dataset2, criterion, criterion2, epoch, flag,
                        test_dataset=test_dataset)

    # 上面的意思是，如果已经有模型了直接拿过来，否则自己马上在train——epoch用数据训练一个
    # 加载好模型之后，测试一下数据集，马上就选择最好的数据,这是第一轮

    # Select the most informative pool samples after pre-training.
    # tst(models, config, train_dataset, test_loader, best_dir, flag, owner="DRCD Pretrained")

    # 这个append_data是筛选之后得出来的   索引！！
    # flag = 0是废话
    # train_dataset穿进去也没用到，废话
    # 只有pool_loader是有效内容，是被筛选的
    select_data_indices = select_best_data(models, config, pool_loader, train_dataset, pool_dataset, flag=0)

    # select_data_subset = Subset(pool_dataset, select_data_indices)
    #
    # select_data_loader = DataLoader(dataset=select_data_subset,
    #                                 batch_size=16,
    #                                 shuffle=True,
    #                                 drop_last=True)
    #
    # # LLM,生成上面的类似样本SA
    # model.llm(select_data_loader)
    #
    # # SA和A合并

    return select_data_indices


# 预训练的每一轮的意思
def train_epoch(models, train_dataset, train_dataset2, criterion, criterion2, epoch, flag=0, test_dataset=None):
    """
    预训练一轮：
      - 第1分支：真假分类（用 train_loader 的 batch1：x1/x2/x5）
      - 第2分支：domain(event_label) + marked(if_marked)（用 train_loader2 的 batch2：x3/x4/x6）
    关键修复：domain 的 event_labels 必须与 x3/x4/x6 同一个 batch（即 event_labels2）
    """
    print("第epoch：", epoch)

    p = float(epoch) / 100
    dynamic_lr = 0.001 / (1. + 10 * p) ** 0.8

    # --------- 分类权重（避免坍塌到单类）---------
    try:
        y_all = torch.as_tensor(train_dataset.label).long().view(-1)
        cnt = torch.bincount(y_all, minlength=2).float().clamp(min=1.0)
        cls_w = (cnt.sum() / (2.0 * cnt)).to(device)  # [w0, w1]
    except Exception:
        cls_w = None

    criterion_cls = torch.nn.CrossEntropyLoss(weight=cls_w)

    # --------- 域损失权重（不要让 domain 压制分类）---------
    # 你之前 domain loss ≈ ln(C) 很大（比如 C=48 时约 3.87），乘 2 会直接“淹没”分类梯度
    # 这里给一个温和 schedule：前几轮更侧重分类
    base_lambda_domain = 0.2
    lambda_domain = base_lambda_domain * min(1.0, epoch / 3.0)  # 0 -> 0.2
    lambda_mark = 1.0

    optimizer = torch.optim.Adam([
        {'params': models.bert.parameters()},  # base lr=3e-5
        {'params': models.class_classifier.parameters(), 'lr': dynamic_lr},
        {'params': models.domain_classifier.parameters(), 'lr': dynamic_lr},
        {'params': models.infer_discriminator.parameters(), 'lr': dynamic_lr},
        {'params': models.gru1.parameters(), 'lr': dynamic_lr},
        {'params': models.text_relu1_1.parameters(), 'lr': dynamic_lr},
        {'params': models.text_relu1_2.parameters(), 'lr': dynamic_lr},
        {'params': models.text_relu2_1.parameters(), 'lr': dynamic_lr},
        {'params': models.text_relu2_2.parameters(), 'lr': dynamic_lr},
        {'params': models.affection_discriminator.parameters(), 'lr': dynamic_lr},
        {'params': models.gru2.parameters(), 'lr': dynamic_lr},
        {'params': models.convs1.parameters(), 'lr': dynamic_lr},
        {'params': models.convs2.parameters(), 'lr': dynamic_lr},
        {'params': models.fc1.parameters(), 'lr': dynamic_lr},
        {'params': models.fc1_2.parameters(), 'lr': dynamic_lr},
        {'params': models.fc2.parameters(), 'lr': dynamic_lr},
        {'params': models.fc2_2.parameters(), 'lr': dynamic_lr},
        # 如果你 MyNet 里有 llm_proj（flag=5 会用到），建议也加上
        # {'params': models.llm_proj.parameters(), 'lr': dynamic_lr},
    ], lr=3e-5)

    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, drop_last=True)
    train_loader2 = DataLoader(dataset=train_dataset2, batch_size=16, shuffle=True, drop_last=True)

    epoch_loss = 0.0
    epoch_acc = 0.0
    total_len = 0

    class_loss_sum = 0.0
    domain_loss_sum = 0.0
    mark_loss_sum = 0.0

    models.train()

    iter1 = iter(train_loader)
    flag_t = torch.tensor(int(flag), dtype=torch.long, device=device)

    for i, (train_text2, train_mask2, train_affection2, train_labels2,
            event_labels2, train_marked_label2, train_data_index2) in enumerate(train_loader2):

        # 取 batch1（用于真假分类）
        try:
            (train_text1, train_mask1, train_affection1, train_labels1,
             event_labels1, train_marked_label1, train_data_index1) = next(iter1)
        except StopIteration:
            iter1 = iter(train_loader)
            (train_text1, train_mask1, train_affection1, train_labels1,
             event_labels1, train_marked_label1, train_data_index1) = next(iter1)

        optimizer.zero_grad(set_to_none=True)

        # -------- inputs --------
        x1 = train_text1.to(device)
        x2 = train_mask1.to(device)
        x3 = train_text2.to(device)
        x4 = train_mask2.to(device)
        x5 = train_affection1.to(device)
        x6 = train_affection2.to(device)

        # forward(flag==0): predict(class_logits), domain_outputs, marked_prob
        predict, domain_outputs, marked_prob = models(x1, x2, x3, x4, x5, x6, flag_t)

        # -------- labels --------
        y_cls = train_labels1.long().to(device)

        # ✅ 关键修复：domain 标签必须来自 batch2（对应 x3/x4/x6）
        y_dom = event_labels2.long()
        if hasattr(config, 'map_event_label_to_id') and callable(getattr(config, 'map_event_label_to_id')):
            _ev = y_dom.detach().cpu().tolist()
            y_dom = torch.tensor([config.map_event_label_to_id(int(x)) for x in _ev], dtype=torch.long)
        y_dom = y_dom.to(device)

        y_mark = train_marked_label2.float().unsqueeze(1).to(device)

        # -------- losses --------
        class_loss = criterion_cls(predict, y_cls)
        domain_loss = criterion2(domain_outputs, y_dom)
        mark_loss = criterion(marked_prob, y_mark)

        loss = class_loss + lambda_domain * domain_loss + lambda_mark * mark_loss

        if not torch.isfinite(loss):
            print("[WARN] loss NaN/Inf, skip batch")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(models.parameters(), 1.0)
        optimizer.step()

        # -------- stats --------
        with torch.no_grad():
            pred_cls = torch.argmax(predict, dim=1)
            correct = (pred_cls == y_cls).sum().item()
            bs = y_cls.size(0)

        epoch_loss += float(loss.item()) * bs
        class_loss_sum += float(class_loss.item()) * bs
        domain_loss_sum += float(domain_loss.item()) * bs
        mark_loss_sum += float(mark_loss.item()) * bs
        epoch_acc += correct
        total_len += bs

    if total_len > 0:
        epoch_loss /= total_len
        class_loss_sum /= total_len
        domain_loss_sum /= total_len
        mark_loss_sum /= total_len
        train_acc = epoch_acc / total_len
    else:
        train_acc = 0.0

    print(f"Epoch [{epoch+1}/10],  Loss: {epoch_loss:.4f}, "
          f"Class Loss: {class_loss_sum:.4f}, domain loss: {domain_loss_sum:.4f}, "
          f"if_marked loss: {mark_loss_sum:.4f}, Dynamic_lr: {dynamic_lr:.4f}, Train_Acc: {train_acc:.4f}")


# 微调时候的训练
def train_finetune(models, train_dataset, train_dataset2, test_dataset, pool_dataset, fine_num, flag):
    print("微调时train_dataset长度", len(train_dataset))

    # 即在微调训练的时候，就把目前的训练集和测试集再仍到第三个分支(0.1+0.05 : ...), 让它识别到下一次更不像本次新训练数据的测试数据
    append_data = []
    criterion = nn.BCELoss()
    criterion2 = nn.CrossEntropyLoss()
    all_accuracy = []
    all_auc_roc = []
    all_f1 = []
    all_precision = []
    all_recall = []

    # optimizer = torch.optim.Adam([
    #     {'params': models.pre_model.parameters()},  # 学习率为3e-5
    #     {'params': models.class_classifier.parameters(), 'lr': 0.001},
    #     {'params': models.domain_classifier.parameters(), 'lr': 0.005},
    #     {'params': models.infer_discriminator.parameters(), 'lr': 0.001},
    #     {'params': models.gru.parameters(), 'lr': 0.005},
    #     {'params': models.affection_discriminator.parameters(), 'lr': 0.005},
    # ], lr=3e-5)

    freeze_layers = ['pre_model']

    for name, param in models.named_parameters():  # 打开Bert层
        for ele in freeze_layers:
            if ele in name:
                param.requires_grad = True
                break
    global count
    epoch_2 = 5
    if int(count) - num < num:
        epoch_2 = 5

    for i in range(epoch_2):  # 之前的模型是5(保存了的)
        # 微调第二个分支真假鉴别
        all_real, all_fake, forward1 = train_finetune_epoch(models, train_dataset, criterion2, i, flag)

        best_dir = "null"
        test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False, drop_last=False)
        pool_loader = DataLoader(dataset=pool_dataset, batch_size=16, shuffle=True, drop_last=False)

        accuracy, auc_roc, f1, precision, recall, test_confusion_matrix, init_per = tst(models, config, train_dataset,
                                                                                        test_loader, best_dir, flag)
        # all_accuracy.append(accuracy)
        # all_auc_roc.append(auc_roc)
        # all_f1.append(f1)
        # all_precision.append(precision)
        # all_recall.append(recall)

        print(f"真假鉴别训练了第{i}次")
    append_data1 = select_best_data(models, config, pool_loader, train_dataset, pool_dataset, flag)

    for name, param in models.named_parameters():  # 冻结Bert层
        for ele in freeze_layers:
            if ele in name:
                param.requires_grad = False
                break
    for i in range(5):
        # 微调第三个分支
        # optimizer1 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001) # 0.00001
        # ExpLR = torch.optim.lr_scheduler.ExponentialLR(optimizer1, gamma=0.98)
        train_finetune3_epoch(models, train_dataset, test_dataset, criterion, i, flag)
        best_dir = "null"
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=16,
                                 shuffle=True,
                                 drop_last=True)
        pool_loader = DataLoader(dataset=pool_dataset,
                                 batch_size=16,
                                 shuffle=True,
                                 drop_last=True)
        # accuracy, auc_roc, f1, precision, recall , test_confusion_matrix,init_per = tst(models, config, train_dataset, test_loader, best_dir, flag)
        # all_accuracy.append(accuracy)
        # all_auc_roc.append(auc_roc)
        # all_f1.append(f1)
        # all_precision.append(precision)
        # all_recall.append(recall)

    append_data1 = select_best_data(models, config, pool_loader, train_dataset, pool_dataset, flag)

    if is_empty(append_data1):
        print("[STOP] train_finetune: no samples selected; return empty to gogogo.")
        return [], forward1

    owner = f"DRCD Ft{fine_num}"
    accuracy, auc_roc, f1, precision, recall, test_confusion_matrix, init_per = tst(models, config, train_dataset,
                                                                                    test_loader, best_dir, flag,
                                                                                    owner=owner)

    global accuracy_list
    global aucroc_list
    global f1_list
    global precision_list
    global recall_list

    accuracy_list.append(accuracy)
    aucroc_list.append(auc_roc)
    f1_list.append(f1)
    precision_list.append(precision)
    recall_list.append(recall)

    # all_accuracy.append(accuracy)
    # all_auc_roc.append(auc_roc)
    # all_f1.append(f1)
    # all_precision.append(precision)
    # all_recall.append(recall)
    #
    # #报告，和前面无关，前面那行就是return的内容
    # max_index = all_f1.index(max(all_f1))
    # max_acc = all_accuracy[max_index]
    # max_auc_roc = all_auc_roc[max_index]
    # max_f1 = all_f1[max_index]
    # max_precision = all_precision[max_index]
    # max_recall = all_recall[max_index]

    # my_go(max_acc, max_auc_roc, max_f1, max_precision, max_recall)
    # print(f"第{fine_num}次微调的性能是:")
    # print('max_acc, max_auc_roc, max_f1, max_precision, max_recall')
    # print(max_acc, max_auc_roc, max_f1, max_precision, max_recall)

    # the_per(max_acc, max_f1, max_precision, max_recall, max_auc_roc, test_confusion_matrix,
    #                           test_true, test_score_convert)

    # get_rep_vec(models, train_dataset, fine_num=fine_num)
    print("运行了getrep")
    glo_per()

    # flag = 4
    # with torch.no_grad():
    #     for i, (test_text, test_mask, test_affection, test_labels, event_labels, test_marked_label, test_data_index) in enumerate(test_loader):
    #         x1 = test_text
    #         x2 = test_mask
    #         x3 = test_affection
    #         x1 = x1.to(device)
    #         x2 = x2.to(device)
    #         x3 = x3.to(device)
    #         out = models(x1, x2, x1, x2, x3, flag)  # out相当于Bert的句子向量输出
    #         sents_vec.append(out['pooler_output'].detach().cpu().numpy().tolist())
    #         sents_index.append(test_data_index.cpu().numpy().tolist())
    #     sents_vec = [np.array(xi) for x in sents_vec for xi in x]     # 1856
    #     sents_index = [xi for x in sents_index for xi in x]
    #     bert_features = pd.DataFrame(sents_vec)
    #     index = pd.DataFrame(sents_index)
    #     # 设置超参数（聚类数目K）搜索范围
    #     KS = 3
    #     CH_scores = []
    #     si_scores = []
    #     ch, si, append_data1 = K_cluster_analysis(KS, bert_features, index, append_data1) # bert_features, index均是1868，append_data是1868的排序
    #     append_data1 = np.array(append_data1)
    return append_data1, forward1


# 定义测试方法
def tst(models, config, train_dataset, test_loader, best_dir, flag, owner=""):
    """
    修复点：
    1) test_loader 可能 len==0（drop_last=True + 样本数<batch_size），防止 np.concatenate 空列表崩溃
    2) 兼容输出是 logits 或概率（必要时 softmax）
    3) AUC 在 test_true 只有一个类别时会报错 -> 返回 nan 并给出警告
    """
    models.eval()

    # ========= 兜底：空 loader =========
    if test_loader is None or len(test_loader) == 0:
        n = 0
        try:
            n = len(getattr(test_loader, "dataset", []))
        except Exception:
            pass
        print(f"[WARN][tst] test_loader is empty (dataset_len={n}). Owner={owner}. "
              f"Likely caused by drop_last=True with small test set or empty test_dataset.")
        empty_cm = np.zeros((2, 2), dtype=int)
        init_per = [np.array([], dtype=int), np.array([], dtype=int), np.empty((0, 2), dtype=float)]
        return 0.0, float("nan"), 0.0, 0.0, 0.0, empty_cm, init_per

    test_score_list = []
    test_pred_list = []
    test_true_list = []
    total_len = 0

    with torch.no_grad():
        for i, (test_text, test_mask, test_affection, test_labels,
                event_labels, test_marked_label, test_data_index) in enumerate(test_loader):

            test_text = test_text.to(device)
            test_mask = test_mask.to(device)
            test_affection = test_affection.to(device)

            # 你这里的 flag 通常是 1（真假分支）
            flag_t = np.array(flag)
            flag_t = torch.from_numpy(flag_t).long().to(device)

            forward1 = models(test_text, test_mask, test_text, test_mask, test_affection, test_affection, flag_t)
            predict_label = forward1[0]

            # ✅ predict_label 统一成 [B,2]
            if predict_label.dim() == 1:
                predict_label = predict_label.view(1, -1)

            predict_label_np = to_np(predict_label)
            if predict_label_np.ndim == 1:
                predict_label_np = predict_label_np.reshape(1, -1)

            # ✅ 如果看起来像 logits（有负数/不归一），就 softmax 一下
            if predict_label_np.shape[1] == 2:
                row_sum = predict_label_np.sum(axis=1, keepdims=True)
                looks_like_prob = (predict_label_np.min() >= 0.0) and (predict_label_np.max() <= 1.0) and np.allclose(row_sum, 1.0, atol=1e-3)
                if not looks_like_prob:
                    x = predict_label_np - predict_label_np.max(axis=1, keepdims=True)
                    ex = np.exp(x)
                    predict_label_np = ex / (ex.sum(axis=1, keepdims=True) + 1e-12)

            test_argmax = np.argmax(predict_label_np, axis=1)

            y_true = to_np(test_labels)
            if y_true.ndim > 1:
                y_true = y_true.squeeze()
            y_true = y_true.astype(int).reshape(-1)

            test_score_list.append(predict_label_np)
            test_pred_list.append(test_argmax.reshape(-1))
            test_true_list.append(y_true)

            total_len += len(y_true)

    # ========= 拼接前再兜底一次 =========
    if len(test_score_list) == 0:
        print(f"[WARN][tst] collected 0 batches (unexpected). Owner={owner}.")
        empty_cm = np.zeros((2, 2), dtype=int)
        init_per = [np.array([], dtype=int), np.array([], dtype=int), np.empty((0, 2), dtype=float)]
        return 0.0, float("nan"), 0.0, 0.0, 0.0, empty_cm, init_per

    test_score = np.concatenate(test_score_list, axis=0)
    test_pred = np.concatenate(test_pred_list, axis=0)
    test_true = np.concatenate(test_true_list, axis=0)

    # ===== 指标 =====
    test_accuracy = metrics.accuracy_score(test_true, test_pred)
    test_f1 = metrics.f1_score(test_true, test_pred, zero_division=0)
    test_precision = metrics.precision_score(test_true, test_pred, zero_division=0)
    test_recall = metrics.recall_score(test_true, test_pred, zero_division=0)

    # AUC：test_true 只有一个类别会报错
    test_aucroc = float("nan")
    if len(np.unique(test_true)) >= 2 and test_score.shape[1] >= 2:
        test_score_convert = test_score[:, 1]
        try:
            test_aucroc = metrics.roc_auc_score(test_true, test_score_convert, average='macro')
        except Exception as e:
            print(f"[WARN][tst] roc_auc_score failed: {e}. Set AUC=nan.")
    else:
        print(f"[WARN][tst] AUC skipped because test_true has <2 classes or score dim<2. Set AUC=nan.")
        test_score_convert = test_score[:, 1] if test_score.shape[1] >= 2 else np.zeros((len(test_true),), dtype=float)

    test_confusion_matrix = metrics.confusion_matrix(test_true, test_pred)

    test_precision2, test_recall2, test_f12, _ = precision_recall_fscore_support(
        test_true, test_pred, average="micro", zero_division=0
    )

    print("Classification Acc: %.4f, AUC-ROC: %s" % (test_accuracy, f"{test_aucroc:.4f}" if np.isfinite(test_aucroc) else "nan"))
    print("Classification report:\n%s\n" % (metrics.classification_report(test_true, test_pred, zero_division=0)))
    print("Classification confusion matrix:\n%s\n" % (test_confusion_matrix))
    print("test_f1, test_precision, test_recall", test_f1, test_precision, test_recall)
    print("micro下的f1值, precision, recall", test_f12, test_precision2, test_recall2)

    the_per(test_true, test_pred, test_score, owner=owner if owner else "tst")
    init_per = [test_true, test_pred, test_score]

    print('结果输出')
    return test_accuracy, test_aucroc, test_f1, test_precision, test_recall, test_confusion_matrix, init_per


accuracy_list = []
aucroc_list = []
f1_list = []
precision_list = []
recall_list = []


# 这是补充出俩的函数
def my_go(max_acc, max_auc_roc, max_f1, max_precision, max_recall):
    global accuracy_list
    global aucroc_list
    global f1_list
    global precision_list
    global recall_list

    file = open('result.csv', 'a', newline='')
    writer = csv.writer(file)
    writer.writerow([max_acc, max_auc_roc, max_f1, max_precision, max_recall])
    file.close()

    accuracy_list.append(max_acc)
    aucroc_list.append(max_auc_roc)
    f1_list.append(max_f1)
    precision_list.append(max_precision)
    recall_list.append(max_recall)


def the_per(test_true, test_pred, test_score, owner):
    # 计算各项指标
    test_accuracy = metrics.accuracy_score(test_true, test_pred)
    test_f1 = metrics.f1_score(test_true, test_pred)
    test_precision = metrics.precision_score(test_true, test_pred)
    test_recall = metrics.recall_score(test_true, test_pred)
    test_score_convert = [x[1] for x in test_score]
    test_aucroc = metrics.roc_auc_score(test_true, test_score_convert, average='macro')
    test_confusion_matrix = metrics.confusion_matrix(test_true, test_pred)
    precision, recall, _ = metrics.precision_recall_curve(test_true, test_score_convert)

    classification_report = metrics.classification_report(test_true, test_pred, output_dict=True)
    classification_report_df = pd.DataFrame(classification_report).transpose()

    # 创建一个大的图
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle(f'Model Performance Metrics of {owner}', fontsize=16)

    # 准确率条形图
    axes[0, 0].bar(['Accuracy'], [test_accuracy])
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_title('Model Accuracy')

    # AUC-ROC曲线
    fpr, tpr, _ = metrics.roc_curve(test_true, test_score_convert)
    axes[0, 1].plot(fpr, tpr, marker='.')
    axes[0, 1].plot([0, 1], [0, 1], linestyle='--')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title(f'ROC Curve (AUC = {test_aucroc:.4f})')

    # 混淆矩阵热力图
    sns.heatmap(test_confusion_matrix, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('True')
    axes[1, 0].set_title('Confusion Matrix')

    # F1、精确率、召回率的条形图
    metrics_dict = {
        'F1 Score': test_f1,
        'Precision': test_precision,
        'Recall': test_recall
    }
    axes[1, 1].bar(metrics_dict.keys(), metrics_dict.values())
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title('F1 Score, Precision, Recall')

    # Precision-Recall曲线
    axes[2, 0].plot(recall, precision, marker='.')
    axes[2, 0].set_xlabel('Recall')
    axes[2, 0].set_ylabel('Precision')
    axes[2, 0].set_title('Precision-Recall Curve')

    # 分类报告热力图
    sns.heatmap(classification_report_df.iloc[:-1, :].T, annot=True, cmap='Blues', ax=axes[2, 1])
    axes[2, 1].set_title('Classification Report')

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整布局以适应标题
    plt.show()


def glo_per():
    global accuracy_list
    global aucroc_list
    global f1_list
    global precision_list
    global recall_list

    epochs = range(1, len(accuracy_list) + 1)

    plt.figure(figsize=(14, 8))

    plt.subplot(2, 3, 1)
    plt.plot(epochs, accuracy_list, 'b', label='Accuracy')
    plt.title(f'Accuracy over Finetunings')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(epochs, aucroc_list, 'r', label='AUC-ROC')
    plt.title(f'AUC-ROC over Finetunings')
    plt.xlabel('Epochs')
    plt.ylabel('AUC-ROC')
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.plot(epochs, f1_list, 'g', label='F1 Score')
    plt.title(f'F1 Score over Finetunings')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.subplot(2, 3, 4)
    plt.plot(epochs, precision_list, 'm', label='Precision')
    plt.title(f'Precision over Finetunings')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.plot(epochs, recall_list, 'c', label='Recall')
    plt.title(f'Recall over Finetunings')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.legend()

    plt.tight_layout()
    plt.show()


# 微调第一个分支的函数
def train_finetune_epoch(models, train_dataset, criterion, epoch, flag=1):
    # ✅ 保险：防止之前 soft finetune 把模型冻住没恢复（你现在就是这个问题）
    torch.set_grad_enabled(True)
    models.train()
    for p in models.parameters():
        p.requires_grad_(True)

    flag = 1
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, drop_last=True)

    flag = np.array(flag)
    flag = torch.from_numpy(flag).long().to(device)

    p_ratio = float(epoch) / 100
    dynamic_lr = 0.0001 / (1. + 10 * p_ratio) ** 0.8

    optimizer = torch.optim.Adam([
        {'params': models.bert.parameters()},
        {'params': models.class_classifier.parameters(), 'lr': dynamic_lr},
        {'params': models.domain_classifier.parameters(), 'lr': dynamic_lr},
        {'params': models.infer_discriminator.parameters(), 'lr': dynamic_lr},
        {'params': models.gru1.parameters(), 'lr': dynamic_lr},
        {'params': models.text_relu1_1.parameters(), 'lr': dynamic_lr},
        {'params': models.text_relu1_2.parameters(), 'lr': dynamic_lr},
        {'params': models.text_relu2_1.parameters(), 'lr': dynamic_lr},
        {'params': models.text_relu2_2.parameters(), 'lr': dynamic_lr},
        {'params': models.affection_discriminator.parameters(), 'lr': dynamic_lr},
        {'params': models.gru2.parameters(), 'lr': dynamic_lr},
        {'params': models.convs1.parameters(), 'lr': dynamic_lr},
        {'params': models.convs2.parameters(), 'lr': dynamic_lr},
        {'params': models.fc1.parameters(), 'lr': dynamic_lr},
        {'params': models.fc1_2.parameters(), 'lr': dynamic_lr},
        {'params': models.fc2.parameters(), 'lr': dynamic_lr},
        {'params': models.fc2_2.parameters(), 'lr': dynamic_lr},
    ], lr=1e-5)

    all_result = []
    all_label = []

    for i, (train_text, train_mask, train_affection, train_labels, event_labels, train_marked_label, train_data_index) in enumerate(train_loader):
        optimizer.zero_grad(set_to_none=True)

        train_text = train_text.to(device)
        train_mask = train_mask.to(device)
        train_affection = train_affection.to(device)

        forward1 = models(train_text, train_mask, train_text, train_mask, train_affection, train_affection, flag)
        predict = forward1[0]
        representative_vector = forward1[1]

        # 如果这里 predict 不带 grad，说明又被冻住了
        if not predict.requires_grad:
            raise RuntimeError(
                "[ERROR] predict does not require grad. "
                "Your model is still frozen. Check finetune_amodel() restore logic."
            )

        all_result.extend(representative_vector.detach().cpu().numpy())
        all_label.extend(train_labels.detach().cpu().numpy())

        train_labels = train_labels.long().to(device)
        class_loss = criterion(predict, train_labels)
        class_loss.backward()
        optimizer.step()

    all_result = np.array(all_result)
    all_label = np.array(all_label)

    all_result_real = all_result[all_label == 0]
    all_result_fake = all_result[all_label == 1]
    print(1)
    return all_result_real, all_result_fake, forward1


# 微调第三个分支的函数
def train_finetune3_epoch(models, train_dataset, test_dataset, criterion, epoch, flag):
    models.train()
    cifar_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
    train_loader = DataLoader(
        dataset=cifar_dataset,
        batch_size=16,
        shuffle=True,
        drop_last=False
    )

    flag = 3
    flag = np.array(flag)
    flag = torch.from_numpy(flag).long()
    flag = flag.to(device)
    p = float(epoch) / 100
    dynamic_lr = 0.0001 / (1. + 10 * p) ** 0.8
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=dynamic_lr)
    for i, (train_text, train_mask, train_affection, train_labels, event_labels, train_marked_label,
            train_data_index) in enumerate(train_loader):
        optimizer.zero_grad()
        train_text = train_text.to(device)
        train_mask = train_mask.to(device)
        train_affection = train_affection.to(device)
        # forward=2调用
        label_out2 = models(train_text, train_mask, train_text, train_mask, train_affection, train_affection, flag)
        train_marked_label = train_marked_label.float()
        train_marked_label = train_marked_label.unsqueeze(1)
        train_marked_label1 = train_marked_label
        train_marked_label = train_marked_label.to(device)
        train_marked_label1 = train_marked_label1.to(device)
        # mark_loss = criterion(label_out, train_marked_label)
        mark_loss2 = criterion(label_out2, train_marked_label1)
        loss = mark_loss2
        loss.backward()
        optimizer.step()
        # ExpLR.step()


def select_best_data(models, config, pool_loader, train_dataset, pool_dataset, flag):
    flag = 0
    global num
    flag = np.array(flag)
    flag = torch.from_numpy(flag).long()
    flag = flag.to(device)
    # 相当于对已经训练好的模型，来选出那个是最不像的
    all_preds_mark2 = []  # 数据是否被标注
    all_preds_value = []  # 每个数据的最大概率
    all_prefict = []  # 每个数据得到的predict
    all_predict2 = []  # 经过第一层采样的样本得到的
    all_index = []
    with torch.no_grad():
        # 分batch进行测试
        total_batches = len(pool_loader)
        for i, (pool_text, pool_mask, pool_affection, pool_labels, event_labels, pool_marked_label,
                pool_data_index) in enumerate(pool_loader):
            # if i == total_batches - 1:
            #     print("select best data的pool_loader 已遍历完")
            # else:
            #     print(f"select best data的Processing batch {i + 1}/{total_batches}")
            x1 = pool_text
            x2 = pool_mask
            x3 = pool_affection
            x1 = x1.to(device)
            x2 = x2.to(device)
            x3 = x3.to(device)

            # 这三个分别是接收的score, domain_output, lable_output2
            # forward=0调用
            predict_label, predict_domain, predict_marked_label2 = models(x1, x2, x1, x2, x3, x3, flag)
            # print("predict_marked_label的形状:", predict_marked_label.shape)     torch.Size([32, 1])
            preds_mark2 = predict_marked_label2.cpu().data
            predict_label = predict_label.cpu().data
            _, test_argmax = torch.max(predict_label, 1)  # _对应的是最大概率，test_argmax为最大值对应的索引

            all_preds_mark2.extend(preds_mark2)
            all_index.extend(pool_data_index)
            all_preds_value.append(_)
            all_prefict.append(predict_label)

        if not all_preds_mark2:  # 检查 all_preds_mark2 是否为空
            print("all_preds_mark2 为空 选择样本失败，返回空列表")
            return []

        all_preds_mark2 = torch.stack(all_preds_mark2)
        all_preds_mark2 = all_preds_mark2.view(-1)

        all_preds_value = torch.cat(all_preds_value)  # torch.cat, 不增加维度而续接
        all_preds_value = all_preds_value.view(-1)
        # all_prefict = torch.cat(all_prefict)

        all_prefict = torch.cat(all_prefict)

        all_preds_mark2 *= -1
        all_preds_value *= -1
        # # 一层采样：找最不像的样本的办法
        first_querry_pool_indices = select_data(all_preds_mark2, all_index, 4 * num)
        first_querry_pool_indices = [i for i in first_querry_pool_indices if i < len(all_preds_mark2)]

        # 如果筛选后的索引数量少于所需数量，则调整 num
        if len(first_querry_pool_indices) < num:
            num = len(first_querry_pool_indices)

        # 二层采样：基于2倍的最不像的样本来找到分类边界的样本
        second_querry_pool = np.asarray(all_prefict)[first_querry_pool_indices]
        if len(second_querry_pool) < num:
            num = len(second_querry_pool)
        uncertaintyentropysampler = UncertaintyEntropySampling(2, 0)
        second_querry_pool_indices = uncertaintyentropysampler.query(second_querry_pool, first_querry_pool_indices, num)
        # print("这是重采样后的样本：", second_querry_pool_indices)
        # 熵方法
        # uncertaintyentropysampler = UncertaintyEntropySampling(2, 0)
        # append_data = uncertaintyentropysampler.query(all_prefict, all_index, num)
        # 不确定方法
        # uncertaintysampler = UncertaintySampling(2, 0)
        # append_data2 = uncertaintysampler.query(all_preds_value, all_index, num)
        # print("这次pool_data数量是:", count)
        # append_data = random_index(count, num)
        # core-set方法
        # coresetsampler = CoreSetSampling(2, 0)
        # append_data = coresetsampler.greedy_k_center(train_dataset, pool_dataset, num)
        print(second_querry_pool_indices)
        return second_querry_pool_indices


def select_data(predict_marked_label2, data_index, num1):
    # 这里传入的 data_index 是数据文件 里面的index列
    pool_data2 = predict_marked_label2

    pool_data2 = normalize(pool_data2, p=1.0, dim=0)

    # print("备选数据池内, pool_data: ", len(pool_data2))
    if len(pool_data2) < num1:
        num1 = len(pool_data2)
        print(f"备选数据池内, pool_data:{len(pool_data2)}, num1 > pooldata2")

    _, querry_indices = torch.topk(pool_data2, num1)  # 取一个tensor的topk元素
    querry_pool_indices = np.asarray(data_index)[querry_indices]  # 返回其索引
    return querry_pool_indices


def random_index(max, num):
    l = [i for i in range(max)]
    print("max:", max)
    random_pool_indices = random.sample(l, num)
    return random_pool_indices


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print(seed)


# 设置随机数种子
# setup_seed(3407)  (在训练2的基础)5 + 2 0.005 到达0.9223
# setup_seed(20)
setup_seed(3000)


def gogogo(config, models):
    # ---- local imports: 避免你忘了在文件顶部加 import ----
    from collections import defaultdict
    from torch.utils.data import Subset

    global count
    global num
    global pm
    global easy
    global fine_num

    def _map_ev(raw):
        """raw event_label -> continuous id (0..C-1) if mapping exists"""
        if hasattr(config, "map_event_label_to_id"):
            return int(config.map_event_label_to_id(int(raw)))
        return int(raw)

    def _balanced_ratio_split(train_ds, frac=0.2, seed=42, label_index=4, verbose=True):
        """
        按总比例 frac 抽样，但加入 event_label 类别均衡约束：
        - desired_total = int(frac * N)
        - 先把预算尽量均分到每个类别（覆盖尽可能多类别）
        - 某类别不够用时，将剩余预算分配给还有余量的类别（round-robin）
        返回：less_subset(用于训练), left_subset(剩余), stats
        """
        N = len(train_ds)
        if N <= 0:
            return Subset(train_ds, []), Subset(train_ds, []), {}

        # 1) 按 event_label 分组
        groups = defaultdict(list)
        for idx in range(N):
            sample = train_ds[idx]
            ev_raw = sample[label_index]
            ev_raw = int(ev_raw.item()) if hasattr(ev_raw, "item") else int(ev_raw)
            ev = _map_ev(ev_raw)
            groups[ev].append(idx)

        labels = sorted(groups.keys())
        K = len(labels)
        desired_total = int(frac * N)
        desired_total = max(1, min(desired_total, N))

        base = desired_total // K
        rem = desired_total % K

        # 2) 初始目标：尽量均分
        target = {}
        for j, lab in enumerate(labels):
            t = base + (1 if j < rem else 0)
            target[lab] = min(t, len(groups[lab]))

        # 3) 如果因为某些类别太短导致没用满预算，把预算分配给有余量的类别
        used = sum(target.values())
        left_budget = desired_total - used
        if left_budget > 0:
            # 还能补的类别
            can_grow = [lab for lab in labels if target[lab] < len(groups[lab])]
            ptr = 0
            while left_budget > 0 and len(can_grow) > 0:
                lab = can_grow[ptr % len(can_grow)]
                if target[lab] < len(groups[lab]):
                    target[lab] += 1
                    left_budget -= 1
                ptr += 1
                can_grow = [l for l in labels if target[l] < len(groups[l])]

        # 4) 采样
        g = torch.Generator().manual_seed(seed)
        less_indices = []
        left_indices = []
        stats = {}

        for lab in labels:
            idxs = groups[lab]
            # 打乱每类内部顺序
            perm = torch.randperm(len(idxs), generator=g).tolist()
            idxs = [idxs[p] for p in perm]

            take = target[lab]
            picked = idxs[:take]
            rest = idxs[take:]

            less_indices.extend(picked)
            left_indices.extend(rest)

            stats[lab] = {"total": len(idxs), "less_train": len(picked), "left": len(rest)}

        # 打乱 less_indices（避免按类别块状排列）
        g_all = torch.Generator().manual_seed(seed)
        perm_all = torch.randperm(len(less_indices), generator=g_all).tolist()
        less_indices = [less_indices[i] for i in perm_all]

        less_subset = Subset(train_ds, less_indices)
        left_subset = Subset(train_ds, left_indices)

        if verbose:
            print("\n📊 **采样统计信息（按比例 + 类别均衡）**")
            for lab in labels:
                c = stats[lab]
                print(f"🔹 event_label {lab}: 总={c['total']}, 进less_train={c['less_train']}, 剩余={c['left']}")
            print("\n✅ **最终数据集大小**")
            print(f"less_train_dataset 总大小: {len(less_subset)} / {len(train_ds)} (frac≈{len(less_subset)/len(train_ds):.4f})")
            print(f"left_dataset 总大小: {len(left_subset)}")

        return less_subset, left_subset, stats

    # ========================= 正式流程开始 =========================
    append_data_index = []
    append_data_tmp = []

    # ---------------- (A) 预训练阶段：flag=0（数据加载 & 模型 forward 均为 0） ----------------
    flag_data = 0
    print(f"flag=0时预训练阶段的pooldata有{count}个（旧count，仅供参考）")

    train_dataset, train_dataset2, validate_dataset, test_dataset, pool_dataset = load_dataset(append_data_index, flag_data)

    # ✅ 关键：这里的 test_dataset 才是真正的测试集，后面绝对不改、不拼、不替换
    print(f"✅ [CHECK] 原始 test_dataset 大小: {len(test_dataset)} | 类型: {type(test_dataset)}")
    if len(test_dataset) > 0:
        try:
            print(f"✅ [CHECK] test_dataset[0] = {test_dataset[0]}")
        except Exception:
            pass

    # 同步 count 到真实 pool 大小（避免你之前 count 读 excel 导致不一致）
    count = len(pool_dataset)
    print(f"✅ [CHECK] 真实 pool_dataset 大小: {count}")

    # ---------------- (B) 在“训练集”上做 less_frac 抽样（不碰 test_dataset） ----------------
    less_frac = float(getattr(config, "less_frac", 1.0))  # 没配就不抽样
    if 0.0 < less_frac < 1.0:
        print(f"🔧 使用 less_frac={less_frac} 对 train_dataset 做均衡抽样（不影响 test_dataset）")
        train_dataset, _, _ = _balanced_ratio_split(train_dataset, frac=less_frac, seed=42, label_index=4, verbose=True)

        # 可选：如果你也想对 train_dataset2 同样抽样，打开下面开关
        if bool(getattr(config, "less_frac_apply_to_train2", False)):
            print("🔧 同步对 train_dataset2 做均衡抽样（less_frac_apply_to_train2=True）")
            train_dataset2, _, _ = _balanced_ratio_split(train_dataset2, frac=less_frac, seed=43, label_index=4, verbose=False)
    else:
        print(f"🔧 less_frac={less_frac}（不抽样）")

    # ---------------- (C) pool 按比例使用（默认 1.0 = 全部） ----------------
    pool_frac = float(getattr(config, "pool_frac", 1.0))
    if 0.0 < pool_frac < 1.0:
        pool_keep = max(1, int(pool_frac * len(pool_dataset)))
        pool_drop = len(pool_dataset) - pool_keep
        pool_dataset, _ = random_split(pool_dataset, [pool_keep, pool_drop])
        count = len(pool_dataset)
        print(f"🔧 pool_frac={pool_frac} => pool_dataset 使用 {count} 条")

    # ---------------- (D) 跑预训练 train()：flag 仍然用 0 ----------------
    models.train()
    append_data_index = train(models, config, train_dataset, train_dataset2, test_dataset, pool_dataset, flag_data)

    # ---------------- (E) pm soft prompt 预训练 ----------------
    pm.load_soft_prompt(start_epoch=0, obj="pre", lr=0.1)

    easy = 1
    if easy:
        my_dataset = mk_my_dataset(pd.read_excel(config.prompt_seed))
    print(my_dataset)

    print("soft的预训练过程开始pretrain_amodel")
    pretrain_amodel(pm, models, my_dataset, epochs=20, obj="pre")

    # ---------------- (F) 多轮微调：数据加载 flag=1/2（仅用于 load_dataset），模型 forward flag 固定用 1 ----------------
    for i in range(50):
        fine_num = i

        if count < num:
            print(f"pooldata已经清空或不足num={num}，总共微调 {i} 次")
            return

        print(f"现在是第{i}次微调————————————————————————————————————————————————————————")

        models.train()

        # 这里的 flag_data 只决定 load_dataset 走 fine_train / fine_train_after 等分支
        if i == 0:
            flag_data = 1
            train_dataset_ft, pool_dataset_ft = load_dataset(append_data_index, flag_data, fine_num=i + 1, models=models)
        else:
            flag_data = 2
            train_dataset_ft, pool_dataset_ft = load_dataset(append_data_tmp, flag_data, fine_num=i + 1, models=models)

        # 更新 pool 计数
        dis = count - len(pool_dataset_ft)
        count = len(pool_dataset_ft)
        print(f"这次运行是flag_data={flag_data}，第{i+1}次微调选出了{dis}个样本，是否等于num{num}? {dis==num}，当前pool剩余{count}个")

        # ✅ 关键：train_finetune / tst / forward 需要的是“真假鉴别”分支 flag=1
        model_flag = 1

        append_data_tmp, forward1 = train_finetune(
            models,
            train_dataset_ft,
            train_dataset2,
            test_dataset,          # ✅ 永远用原始 test_dataset
            pool_dataset_ft,
            i + 1,
            model_flag
        )

        if is_empty(append_data_tmp):
            print(f"[STOP] round {i + 1}: select_best_data returned empty => end training.")
            break


def mk_my_dataset(my_data):
    # 传进来的是一个df，在前面需要把dir转化成df
    # my_data = pd.read_excel(dir)
    print("my_dataset文件的长度：", len(my_data))
    index_my_data = list(my_data.index)
    my_data = mydict.to_bert_input_new(my_data, index_my_data)
    my_dataset = Rumor_Data(my_data)
    return my_dataset


def get_rep_vec1(models, train_loader, flag=1, fine_num=0, embedding=[]):
    # models.eval()  # 将模型设置为评估模式，防止对模型参数进行更新
    models.train()
    # train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=False, drop_last=False)
    # criterion2 = nn.CrossEntropyLoss()
    all_representative_vectors = []
    all_predicts = []

    # with torch.no_grad():  # 禁用梯度计算，以提高计算效率
    for i, data in enumerate(train_loader):
        # print("flag:", flag, "train_loader长度", len(train_loader))
        train_text, train_mask, train_affection, train_labels, event_labels, train_marked_label, data_index = data

        # to device
        train_text = train_text.to(device)
        train_mask = train_mask.to(device)
        train_affection = train_affection.to(device)

        # print("flag=1的运行次数", i)
        forward = models(train_text, train_mask, train_text, train_mask, train_affection,
                         train_affection, flag, embedding=embedding)

        predict = forward[0]
        representative_vector = forward[1]

        # 把所有样本的结果都保存下来，方便画图
        all_representative_vectors.extend(representative_vector)
        all_predicts.extend(predict)

    # vis_rep_vec(title=f"第{fine_num}次微调样本的PCA", data=representative_vector, method="PCA")
    # vis_rep_vec(title=f"第{fine_num}次微调样本的TSNE", data=representative_vector, method="TSNE")

    return all_predicts, all_representative_vectors


# def get_rep_vec1(models, train_loader, flag=1, fine_num=0, embedding=[]):
#     models.eval()  # teacher 不需要训练
#     all_representative_vectors = []
#     all_predicts = []
#
#     with torch.no_grad():  # 关闭梯度
#         for i, data in enumerate(train_loader):
#             train_text, train_mask, train_affection, train_labels, event_labels, train_marked_label, data_index = data
#
#             train_text = train_text.to(device)
#             train_mask = train_mask.to(device)
#             train_affection = train_affection.to(device)
#
#             forward = models(train_text, train_mask, train_text, train_mask,
#                              train_affection, train_affection, flag, embedding=embedding)
#
#             predict = forward[0]
#             representative_vector = forward[1]
#
#             all_representative_vectors.extend(representative_vector)
#             all_predicts.extend(predict)
#
#     models.train()  # 用完恢复 train 状态
#     return all_predicts, all_representative_vectors


def get_rep_vec5(models, train_loader, flag=1, fine_num=0, embedding=[]):
    # models.eval()  # 将模型设置为评估模式，防止对模型参数进行更新
    models.train()
    # train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=False, drop_last=False)
    # criterion2 = nn.CrossEntropyLoss()
    all_representative_vectors = []
    all_predicts = []

    # with torch.no_grad():  # 禁用梯度计算，以提高计算效率
    for i, data in enumerate(train_loader):
        # print("flag:", flag, "train_loader长度", len(train_loader))
        train_text, train_mask, train_affection, train_labels, event_labels, train_marked_label, data_index = data

        # to device
        train_text = train_text.to(device)
        train_mask = train_mask.to(device)
        train_affection = train_affection.to(device)

        for j, text in enumerate(train_text):
            # 生成文本
            combined_embeddings, last, combined_mask, out_texts = pm.chat(tasknum=0, num=1, text=text,
                                                                          pro_type="vec",
                                                                          out_type="vec",
                                                                          if_soft=1)  # 使用整数索引来访问 DataFrame
            # print(f"flag=5的运行次数i{i}_j{j}")
            print('last:', last.size())
            output_embeddings = last
            bert_embbeding = pm.to_bert_embbeding(output_embeddings)
            # print("bert_embbeding.shape", bert_embbeding.shape) # torch.Size([1, 4306, 4096])
            embedding = bert_embbeding

            text_single = train_text[j].unsqueeze(0)
            mask_single = train_mask[j].unsqueeze(0)
            aff_single = train_affection[j].unsqueeze(0)

            forward = models(text_single, mask_single, text_single, mask_single, aff_single,
                             aff_single, flag, embedding=embedding)

            predict = forward[0]
            representative_vector = forward[1]

            # 把所有样本的结果都保存下来，方便画图
            all_representative_vectors.extend(representative_vector)
            all_predicts.extend(predict)

    # vis_rep_vec(title=f"第{fine_num}次微调样本的PCA", data=representative_vector, method="PCA")
    # vis_rep_vec(title=f"第{fine_num}次微调样本的TSNE", data=representative_vector, method="TSNE")

    return all_predicts, all_representative_vectors


# def finetune_amodel(pm, models, train_dataset, epochs=50, obj="fine"):
#     print("")
#     dataloader = DataLoader(dataset=train_dataset,
#                             batch_size=16,
#                             shuffle=True,
#                             drop_last=True)
#     # 没有样本就直接跳过这一轮微调
#     if len(dataloader) == 0:
#         print("[WARN] finetune_amodel: no samples in this round, skip fine-tuning.")
#         return
#
#     # pm.optimizer = optim.AdamW([pm.soft_prompt], lr=pm.lr)
#     for epoch in range(pm.last_epoch, pm.last_epoch + epochs):
#         toc = timer()
#         print(f"Finetuning Epoch {epoch},Finetuning")
#         pm.epoch_losses = []
#
#         all_socre1, all_rep_vec1 = get_rep_vec1(models, dataloader, flag=1)
#         all_socre5, all_rep_vec5 = get_rep_vec5(models, dataloader, flag=5)
#
#         print("all1", len(all_socre1), len(all_rep_vec1))
#         print("all5", len(all_socre5), len(all_rep_vec5))
#
#         # flag=5时的代码，集成到get_rep_vec里面了
#         for num, _ in enumerate(dataloader):
#             sam_time = timer()
#
#             # 拿loss1和loss2
#             # loss1需要用原版情感向量rep_vec1，和模仿出来的情感向量rep_vec5
#
#             rep_vec1 = all_rep_vec1[num]
#             rep_vec5 = all_rep_vec5[num]
#             loss1 = get_loss1(rep_vec1, rep_vec5)
#             print("loss1", loss1)
#             # 拿到了score，接下来就是用score计算loss2
#
#             score5 = all_socre5[num]
#             # loss2需要用score
#             loss2 = get_loss2(score5)
#             print("loss2成功", )
#
#             loss = calculate_combined_loss(loss1, loss2, alpha=0.5, beta=0.5)
#             print("计算loss成功", )
#
#             # ==== 关键修改：正常的反向传播流程 ====
#             pm.optimizer.zero_grad()  # 每个 batch 先清梯度
#             loss.backward()  # 不要 retain_graph
#             pm.optimizer.step()
#             # ====================================
#
#             # loss 回传完了，放回到 cpu 记录 / 画图
#             loss = loss.detach().cpu()  # 不再保留图
#
#             # 在每个 epoch 结束时清理显存缓存
#
#
#
#
#
#             pm.epoch_losses.append(loss)
#             # if num == int(length/5): # % 10 == 0:
#             #     self.plot_losses(title=f"samples {num}")
#             # print(f'loss of sample{num}: {loss}')
#             # print(f"sample{num}花费时间:", timer()-sam_time)
#
#         # loss.cpu().detach().numpy()
#
#         epoch_mean_loss = sum(pm.epoch_losses) / len(dataloader)
#         pm.epoch_mean_losses.append(epoch_mean_loss)
#         pm.losses.extend(pm.epoch_losses)
#
#         # pm.plot_losses(title=f"epoch {epoch}", num_per_epoch=len(dataloader))
#
#         # print("self.soft_prompt:", self.soft_prompt)
#         print(f"epoch{epoch}花费时间:", timer() - toc, "Loss:,", str(epoch_mean_loss))
#
#         state = {'model': pm.model.state_dict(),
#                  'optimizer': pm.optimizer.state_dict(),
#                  'epoch': epoch}
#         if obj == "pre":
#             obj = ""
#         soft_path = f"weight/pre_soft_prompt_word_epoch{epoch}{obj}.pth"
#         print(pm.soft_prompt)
#         torch.save((pm.soft_prompt, epoch, pm.losses, pm.epoch_mean_losses), soft_path)
#         print(f'Soft prompt saved at {soft_path}')
#         torch.cuda.empty_cache()

def finetune_amodel(pm, models, train_dataset, epochs=50, obj="fine"):
    """
    ✅ 只训练 pm.soft_prompt 的 soft-finetune
    ✅ 关键修复：
      1) 进入时冻结 models，但退出时必须恢复 requires_grad 和 train/eval 状态
      2) flag==5 必须传 embedding
      3) 每个样本单独喂入（embedding 是单样本的），避免 batch 维度不一致
    """
    dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
    if len(dataloader) == 0:
        print("[WARN] finetune_amodel: no samples, skip.")
        return

    # --------- 保存现场：训练/评估模式 + requires_grad 状态 ---------
    was_training = models.training
    orig_requires_grad = [p.requires_grad for p in models.parameters()]

    try:
        # --------- 冻结 models，仅训练 soft_prompt ---------
        models.eval()
        for p in models.parameters():
            p.requires_grad_(False)

        # soft_prompt 必须可训练
        if hasattr(pm, "soft_prompt"):
            pm.soft_prompt.requires_grad_(True)
        else:
            raise RuntimeError("[ERROR] pm has no attribute 'soft_prompt'")

        # 优化器只优化 soft_prompt
        if getattr(pm, "optimizer", None) is None or len(getattr(pm.optimizer, "param_groups", [])) == 0:
            lr = float(getattr(pm, "lr", 1e-3))
            # 你的日志里也提示过 soft prompt lr 太大，这里保守 clamp
            if lr > 1e-2:
                print(f"[WARN] lr={lr} is too large for soft-prompt; clamped to 0.001")
                lr = 1e-3
            pm.optimizer = torch.optim.AdamW([pm.soft_prompt], lr=lr)

        # --------- 开始 soft finetune ---------
        for epoch in range(getattr(pm, "last_epoch", 0), getattr(pm, "last_epoch", 0) + epochs):
            print(f"Finetuning Epoch {epoch}, Finetuning")
            epoch_losses = []

            for batch in dataloader:
                train_text, train_mask, train_affection, train_labels, event_labels, train_marked_label, train_data_index = batch
                train_text = train_text.to(device)
                train_mask = train_mask.to(device)
                train_affection = train_affection.to(device)

                # teacher rep：flag=1（不需要梯度）
                with torch.no_grad():
                    out_t = models(
                        train_text, train_mask,
                        train_text, train_mask,
                        train_affection, train_affection,
                        torch.tensor(1, device=device)
                    )
                    rep_t_batch = out_t[1].detach()  # [B, D]

                # student：逐样本生成 embedding（embedding 是单样本）
                B = train_text.size(0)
                for j in range(B):
                    text_j = train_text[j]              # [L]
                    text_j = text_j.unsqueeze(0)        # [1, L]
                    mask_j = train_mask[j].unsqueeze(0) # [1, L]
                    aff_j  = train_affection[j].unsqueeze(0)

                    # 这里必须让 embedding 的计算链路带梯度到 soft_prompt
                    # 注意：不要对 last/embedding 做 detach/cpu()
                    _, last, _, _ = pm.chat(
                        tasknum=0, num=1, text=train_text[j],
                        pro_type="vec", out_type="vec", if_soft=1
                    )
                    embedding = pm.to_bert_embbeding(last)  # 期望: [1, seq_len, 4096] 且带 grad

                    out_s = models(
                        text_j, mask_j,
                        text_j, mask_j,
                        aff_j, aff_j,
                        torch.tensor(5, device=device),
                        embedding=embedding
                    )
                    rep_s = out_s[1]   # [1, D]
                    score_s = out_s[0] # [1, 2]

                    rep_t = rep_t_batch[j].unsqueeze(0)  # [1, D]
                    loss1 = get_loss1(rep_t, rep_s)
                    loss2 = get_loss2(score_s)
                    loss = calculate_combined_loss(loss1, loss2, alpha=0.5, beta=0.5)

                    if not torch.isfinite(loss):
                        print("[WARN] loss NaN/Inf, skip")
                        continue

                    pm.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_([pm.soft_prompt], 1.0)
                    pm.optimizer.step()

                    epoch_losses.append(float(loss.detach().item()))

            mean_loss = sum(epoch_losses) / max(1, len(epoch_losses))
            if not hasattr(pm, "epoch_mean_losses"):
                pm.epoch_mean_losses = []
            if not hasattr(pm, "losses"):
                pm.losses = []
            pm.epoch_mean_losses.append(mean_loss)
            pm.losses.extend(epoch_losses)

            print(f"epoch{epoch} mean loss:", mean_loss)

    finally:
        # --------- ✅ 关键：恢复 models 的 requires_grad + train/eval 状态 ---------
        for p, rg in zip(models.parameters(), orig_requires_grad):
            p.requires_grad_(rg)
        models.train(was_training)


def pretrain_amodel(pm, models, train_dataset, epochs=50, obj="pre"):
    dataloader = DataLoader(dataset=train_dataset,
                            batch_size=16,
                            shuffle=True,
                            drop_last=True)

    pm.optimizer = optim.AdamW([pm.soft_prompt], lr=pm.lr)
    for epoch in range(pm.last_epoch, pm.last_epoch + epochs):
        print("soft预训练的epoch", epoch)
        toc = timer()
        print(f"Pretraining Epoch {epoch},pretraining")
        pm.epoch_losses = []

        all_socre1, all_rep_vec1 = get_rep_vec1(models, dataloader, flag=1)
        all_socre5, all_rep_vec5 = get_rep_vec5(models, dataloader, flag=5)

        # print("all1", len(all_socre1), len(all_rep_vec1))
        # print("all5", len(all_socre5), len(all_rep_vec5))

        # flag=5时的代码，集成到get_rep_vec里面了
        for num, _ in enumerate(dataloader):
            sam_time = timer()

            # 拿loss1和loss2
            # loss1需要用原版情感向量rep_vec1，和模仿出来的情感向量rep_vec5

            rep_vec1 = all_rep_vec1[num]
            rep_vec5 = all_rep_vec5[num]
            loss1 = get_loss1(rep_vec1, rep_vec5)
            print("loss1", loss1)

            # 拿到了score，接下来就是用score计算loss2
            # score5 = all_socre5[num]
            # loss2需要用score
            # loss2 = get_loss2(score5)

            # loss = calculate_combined_loss(loss1, loss2, alpha=0.7, beta=0.3)
            loss = loss1
            # 反向传播和参数更新
            pm.optimizer.zero_grad()

            # 计算 loss 后立刻检查
            if not torch.isfinite(loss):
                print("[WARN] loss is NaN/Inf, skip step")
                pm.optimizer.zero_grad(set_to_none=True)
                continue

            pm.optimizer.zero_grad(set_to_none=True)

            loss.backward()
            # 梯度裁剪：防爆
            torch.nn.utils.clip_grad_norm_([pm.soft_prompt], 1.0)

            pm.optimizer.step()

            loss = loss.detach().cpu()

            # 在每个 epoch 结束时清理显存缓存

            pm.epoch_losses.append(loss)
            # if num == int(length/5): # % 10 == 0:
            #     self.plot_losses(title=f"samples {num}")
            # print(f'loss of sample{num}: {loss}')
            # print(f"sample{num}花费时间:", timer()-sam_time)

        epoch_mean_loss = sum(pm.epoch_losses) / len(dataloader)
        pm.epoch_mean_losses.append(epoch_mean_loss)
        pm.losses.extend(pm.epoch_losses)

        # pm.plot_losses(title=f"epoch {epoch}", num_per_epoch=len(dataloader))

        # print("self.soft_prompt:", self.soft_prompt)
        print(f"epoch{epoch}花费时间:", timer() - toc, "Loss:,", str(epoch_mean_loss))

        state = {'model': pm.model.state_dict(),
                 'optimizer': pm.optimizer.state_dict(),
                 'epoch': epoch}
        if obj == "pre":
            obj = ""
        soft_path = f"weight/pre_soft_prompt_word_epoch{epoch}{obj}.pth"
        # print(pm.soft_prompt)
        torch.save((pm.soft_prompt, epoch, pm.losses, pm.epoch_mean_losses), soft_path)
        print(f'Soft prompt saved at {soft_path}')

        torch.cuda.empty_cache()


# def get_loss1(rep_vec1, rep_vec5):
#
#     # cosine_similarity = F.cosine_similarity(rep_vec1, rep_vec5, dim=-1)
#     # # print("cosine_similarity", cosine_similarity)
#     #
#     # # 过滤掉NaN值
#     # cosine_similarity = cosine_similarity[~torch.isnan(cosine_similarity)]
#     # cosine_similarity = torch.clamp(cosine_similarity, min=-1.0, max=1.0).mean()
#     # # print("cosine_similarity", cosine_similarity)
#     #
#     # # 计算均方误差损失
#     # mse_loss = F.mse_loss(rep_vec1, rep_vec5, reduction='mean')
#
#     cosine_sim = F.cosine_similarity(rep_vec1, rep_vec5, dim=-1)
#     # 计算损失：相似度越接近1，损失越小
#     loss = 1 - cosine_sim.mean()  # 越接近1表示越相似，损失越小
#
#
#     lambda_reg = 0.1  # 正则化权重
#     delta = 0.1  # 最小允许差异
#     euclidean_distance = torch.norm(rep_vec1 - rep_vec5, p=2, dim=-1)
#     regularization_term = torch.clamp(delta - euclidean_distance, min=0)  # 小于 delta 时施加惩罚
#     regularization_loss = lambda_reg * regularization_term.mean()
#
#     # 总损失
#     loss1 = loss + regularization_loss
#
#
#     return loss1

def get_loss1(rep_vec1, rep_vec5):
    """
    rep_vec1: 教师特征（原始样本 / 原始特征）
    rep_vec5: 学生特征（经过 soft prompt + 大模型生成后的特征）
    这里确保两者在同一 device 和相同 dtype，然后按你原来的
    余弦相似度 + 欧氏距离正则 的形式计算 loss。
    """

    # 1. 先把 rep_vec1 搬到 rep_vec5 所在的 device（rep_vec5 保持在 cuda 上以便反传）
    if rep_vec1.device != rep_vec5.device:
        rep_vec1 = rep_vec1.to(rep_vec5.device)

    # 2. 统一成 float（有时候是 half / double 会影响数值）
    rep_vec1 = rep_vec1.float()
    rep_vec5 = rep_vec5.float()

    # 3. 余弦相似度部分：越接近 1 越好，因此 1 - mean(cos)
    cosine_sim = F.cosine_similarity(rep_vec1, rep_vec5, dim=-1, eps=1e-8)

    cosine_sim = torch.clamp(cosine_sim, min=-1.0, max=1.0)  # 稍微保险一点
    loss_cos = 1.0 - cosine_sim.mean()

    # 4. 欧氏距离正则：希望两者至少有 delta 的差异（你原来的设计）
    lambda_reg = 0.1  # 正则化权重
    delta = 0.1  # 最小允许差异

    euclidean_distance = torch.norm(rep_vec1 - rep_vec5, p=2, dim=-1)  # [batch] or scalar
    regularization_term = torch.clamp(delta - euclidean_distance, min=0.0)  # 小于 delta 才惩罚
    regularization_loss = lambda_reg * regularization_term.mean()

    # 5. 总损失
    loss1 = loss_cos + regularization_loss
    return loss1


def get_loss2(score):
    # KL
    target = torch.full_like(score, 0.5)  # 理想置信度为 0.5
    loss2 = F.mse_loss(score, target)  # 使用 MSE 计算不确定性损失
    return loss2


def calculate_combined_loss(loss1, loss2, alpha=0.5, beta=0.5):
    """
    计算损失的加权和
    """
    # 加权求和
    combined_loss = alpha * loss1 + beta * loss2
    return combined_loss


if __name__ == '__main__':
    all_time = timer()

    # ---------------- argparse：支持单独运行一个参数 ----------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--less-frac", type=float, default=None,
                        help="Sampling ratio used in gogogo(). If not set, use config.less_frac.")
    parser.add_argument(
        "--llm-model",
        default=os.environ.get("SHARP_LLM_MODEL"),
        help="Hugging Face model id or local path for the causal LLM.",
    )
    parser.add_argument(
        "--prompt-seed",
        default=os.environ.get(
            "SHARP_PROMPT_SEED",
            str(REPO_ROOT / "data" / "twitter15_16" / "prompt_seed" / "clippool_16_release.xlsx"),
        ),
        help="Twitter-specific XLSX seed used to pre-train the soft prompt.",
    )
    args = parser.parse_args()

    if not args.llm_model:
        parser.error("--llm-model (or SHARP_LLM_MODEL) is required")
    if not os.path.isfile(args.prompt_seed):
        parser.error(f"Twitter15/16 prompt seed not found: {args.prompt_seed}")

    # ---------------- device ----------------
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    # ✅ 同步更新 config.device（你上面 config 是从 DAAL.model 导入 device 创建的）
    try:
        config.device = device
    except Exception:
        pass

    # ✅ 命令行覆盖 less_frac
    if args.less_frac is not None:
        config.less_frac = float(args.less_frac)
    config.prompt_seed = args.prompt_seed

    print(f"[RUN] device={device} | less_frac={getattr(config, 'less_frac', None)}")

    # ---------------- build model ----------------
    model = MyNet(config).to(device)

    # ---------------- build pm (LLM) ----------------
    pm = amodel.PM(device=device, model_name=args.llm_model)

    # ---------------- run ----------------
    gogogo(config, model)

    all_time = timer() - all_time
    print("整个代码运行花费时间：", all_time)
    print("over")
