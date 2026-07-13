# -*- codeing = utf-8 -*-
# @Time : 2022-12-10 16:17
# @Author : 张超然
# @File ： model.py
# @Software: PyCharm
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT / "DAAL"))
from torch.utils.data import Dataset, DataLoader, random_split,ConcatDataset
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sympy.physics.units import length
from torch.utils.data import Dataset, random_split
import torch
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
from transformers import BertTokenizer,BertModel,BertConfig
import pandas as pd
import process_data as ProData
import amodel
from Samper import *
import csv
from timeit import default_timer as timer
import seaborn as sns
import torch.distributed as dist
import torch.multiprocessing as mp
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
PHEME_EVENT_ID = int(os.environ.get('SHARP_PHEME_EVENT', '2'))
PHEME_DATA_DIR = REPO_ROOT / 'data' / 'pheme' / 'processed'
PHEME_EVENTS_DIR = PHEME_DATA_DIR / 'events'
PHEME_WORK_DIR = REPO_ROOT / 'outputs' / f'pheme_event_{PHEME_EVENT_ID}'
PHEME_WORK_DIR.mkdir(parents=True, exist_ok=True)
PHEME_POOL_FILE = PHEME_EVENTS_DIR / f'event_{PHEME_EVENT_ID}_pool.xlsx'
df_pool_len = pd.read_excel(PHEME_POOL_FILE)
count = len(df_pool_len)
num = 32

fine_num = 0
easy = 0

#114

#继承自pytorch框架下的数据集基类
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
        return self.text[idx], self.mask[idx], self.affection[idx], self.label[idx], self.event_label[idx], self.if_marked_label[idx], self.data_index[idx]

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

#梯度翻转层
class ReverseLayerF(Function):
    @staticmethod
    def forward(self, x):
        #lambd应该是用来缩放梯度的，放大程度越大，更能接近目标，更容易过拟合
        self.lambd = 1

        #传入下一层，不改变x值
        return x.view_as(x)

    @staticmethod
    def backward(self, grad_output):
        #在翻转参数上添加了一个-，表示将后面传来的参数取负数，再传递到前层
        #借此实现反着梯度方向优化模型参数
        return (grad_output * -self.lambd)

def grad_reverse(x):
    #apply函数：判断变形金刚是否激活，运行
    return ReverseLayerF.apply(x)


def to_np(x):
    return x.data.cpu().numpy()
class Config(object):
    """配置参数"""
    def __init__(self,device):

        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # 设备
        self.device = device

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_epochs = 3                                             # epoch数
        self.batch_size = 32                                            # mini-batch大小
        # 每句话处理成的长度(短填长切)
        # 似乎这个新闻只有一句话。。。。。。
        self.pad_size = 64
        self.learning_rate = 3e-5                                       # 学习率
        # 而下游的学习任务将其设置为 1e-4




        # bert预训练模型的位置                           没找到
        self.bert_path = os.environ.get('SHARP_BERT_MODEL', 'bert-base-uncased')


        # bert切词器
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path,weights_only=False)
        # bert隐藏层个数（维度）?
        self.hidden_size = 16
        self.dropout = 0.3
        self.num_filters = 256
        # 卷积核在序列维度上的尺寸 = n-gram大小 卷积核总数量=filter_size*num_filters
        self.filter_size = (2, 3, 4)

#注意：此处只是建立了神经网络群，还没有对这个群下达任何指令，如：train,predict等
class MyNet(nn.Module):
    def __init__(self, config):
        super(MyNet, self).__init__()
        # 相关参数设置
        self.batch_size = 16
        self.hidden_size = 32
        self.event_num = 5
        model_config = BertConfig.from_pretrained(config.bert_path, output_hidden_states=True)
        self.bert = BertModel.from_pretrained(config.bert_path, config = model_config)
        # 需要梯度：需要梯度来更新这个参数，也就是可以训练
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(config.dropout)




        # TEXT-CNN层
        #卷积层，处理文本信息
        self.convs1 = nn.ModuleList(
            # 并不意味着三层是串联的哦！！！！
            # 输入通道数,输出通道数（卷积核数），卷积核维度
            # 三个卷积层，每一位数字接受1个维度的信息，每层256个二维卷积核，2，3，4*768，768=bert词向量的长度（思考）
            [nn.Conv2d(1, config.num_filters, (k, 768)) for k in config.filter_size]
        )

        self.text_relu1_1 = nn.LeakyReLU(True)
        # 接收层 每一层卷积核的数量*卷积核的层数，恰好是卷积神经网络的“并联输出”？？？
        #config.num_filters=256每次卷积运算卷积核的个数，一个核输出一个值
        self.fc1 = nn.Linear(config.num_filters * len(config.filter_size), self.batch_size)
        self.fc1_2 = nn.Linear(64, 64)
        self.text_relu1_2 = nn.LeakyReLU(True)



        self.convs2 = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, 768)) for k in config.filter_size]
        )
        self.text_relu2_1 = nn.LeakyReLU(True)
        self.fc2 = nn.Linear(config.num_filters * len(config.filter_size), self.batch_size)
        self.fc2_2 = nn.Linear(64, 64)
        self.text_relu2_2 = nn.LeakyReLU(True)



        # GRU层
        self.gru1 = nn.GRU(24, 32, batch_first=True, bidirectional=True)   #32个隐藏层
        self.gru2 = nn.GRU(24, 32, batch_first=True, bidirectional=True)





        ###  第一个分支  Class  Classifier  真假新闻
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(64, 2))
        self.class_classifier.add_module('c_softmax', nn.Softmax(dim=1))


        ### 第二个分支   Domain Classifier   域判别器，似乎前面接的是双重特征提取器（DFE），64是DFE的输出长度？
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(64, self.hidden_size))    # [batch]
        self.domain_classifier.add_module('d_relu1', nn.LeakyReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(self.hidden_size, self.event_num))
        self.domain_classifier.add_module('d_softmax', nn.Softmax(dim=1))

        ### 第三个分支    区分是否为选中样本
        self.infer_discriminator = nn.Sequential()
        self.infer_discriminator.add_module('e_fc1', nn.Linear(64, self.hidden_size))
        self.infer_discriminator.add_module('e_relu1', nn.ReLU(True))
        self.infer_discriminator.add_module('e_fc2', nn.Linear(self.hidden_size, self.hidden_size))  # [32, 128]
        self.infer_discriminator.add_module('e_relu2', nn.ReLU(True))
        self.infer_discriminator.add_module('e_fc3', nn.Linear(self.hidden_size, 1))  # # [32, 1]
        self.infer_discriminator.add_module('e_softmax', nn.Sigmoid())

        # input_dim：向量的维度    hidden_size：代表GRU层的维度，GRU层有多少个神经元              每个句子是 [8 * 24]
        # n_layers：GRU的神经网络层数                                                        即输入是 [batchsize * [8 * 24]]

        # ### 第四个分支  事件二分类器（有无标签）
        self.affection_discriminator = nn.Sequential()
        self.affection_discriminator.add_module('f_fc1', nn.Linear(64, self.hidden_size))
        self.affection_discriminator.add_module('f_relu1', nn.ReLU(False))
        self.affection_discriminator.add_module('f_fc2', nn.Linear(self.hidden_size, 1))
        self.affection_discriminator.add_module('f_sigmoid', nn.Sigmoid())

    def conv_and_pool(self, x, conv): # x: [16, 1, 54, 768]  卷积核
        x = conv(x)
        #(batch_size, num_filters（卷积核数量）, output_height, 1)
        # print(x.shape) [16, 256, 53, 753]【16，256，53，1】
        x = F.relu(x)
        # print(x.shape)         # 【16，256，53，1】
        x = x.squeeze(3)
        # print(x.shape)         # 【16，256，53】
        x = F.max_pool1d(x, x.size(2))     #对（256，53）的最大值=（）
        x = x.squeeze(2)                    #【16，256】
        return x

    def forward(self, x1, x2, x3, x4, x5, x6, flag, embedding=[]):
        # x [ids , mask , label, 领域类别]
        # context1和mask1代表的是源域+目标域的部分数据(有标签)，context2和mask2代表的是所有数据(无标签)
        # flag是0: 是预训练模型+测试的时候，flag是1的话是微调模型第一个分支，flag是3的话是微调模型的第二个分支
        # flag = 0 和 flag = 3 是要涉及GRU部分
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.x4 = x4
        self.x5 = x5
        self.x6 = x6
        if flag == 0:
            context1 = x1               # 对应输入的句子 16个句子，每一个有54个词语              shape[batch_size * padding（填充）] [16, 54]
            mask1 = x2                  # 对padding负责挖空（将填充的用掩码掩起来）
            # context2和mask2代表的是 全部的数据，用来预测是否有标签的
            context2 = x3
            mask2 = x4
            gru_inputs1 = x5 # [16, 8, 24])
            gru_inputs2 = x6
            outputs1 = self.bert(context1, attention_mask=mask1)       # shape[batch_size * hidden_size(768)]
            #为啥呢 是把一句话分为batch_size（16）个长度为768的向量吗 NO
            #一批有batch_size（16）个训练单元，每一个数据为被解码为768
            outputs2 = self.bert(context2, attention_mask=mask2)       # shape[batch_size * hidden_size(768)]



            # ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
            # ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
            # ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
            # ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
            # ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
            # ————————————————————————————————————————————————
            # 数据经过解码器Encoder得到词向量与情感向量的乘积 （版本1）
            embadding_out1 = outputs1['last_hidden_state'] # ([16, 54, 768])
            # 似乎是提取位置在最后一个  可以代表全句子特征的向量
            #添加了一个通道维度
            new_embadding_out1 = embadding_out1.unsqueeze(1) # [16, 1, 54, 768]
            # 卷积核维度（2，768）（3，768）（4，768）
            # 分别把一个句子（54，768）卷成256个向量（每次256个卷积核），
            # 用的最大池化，过滤出向量中的最大值，contact (cat)连接
            out1 = torch.cat([self.conv_and_pool(new_embadding_out1, conv) for conv in self.convs1], 1)  #
            out1 = self.text_relu1_1(out1)
            fc_out1 = self.fc1(out1) #fc1的输出size为16,输入【config.num_filters（256） * len(config.filter_size)（3）】
            # 将（16，256，3）提取特征为（16，1），一个句子一个值，作为特征值               怎么与情感向量做点积呢？？？
            # 把BERT词向量经过卷积层和一层全连接层

            gru_inputs1 = gru_inputs1.float()               #（16，8，24）
            gru_out1, h1 = self.gru1(gru_inputs1, None)
            gru_out1 = gru_out1[:, -1, :]        #截取，顺带降维（16，64）           # 把情感向量拿到GRU中拿到对应的GRU输出

            mul_out1 = torch.mm(fc_out1, gru_out1)          # GRU输出与TEXT-CNN输出相乘
            final_out1 = self.fc1_2(mul_out1)               # 线性层（64，64）
            final_out1 = self.text_relu1_2(final_out1)       # torch.Size([16, 1])


            #————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
            # ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
            # ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
            # ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
            # ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
            # ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
            # 数据经过解码器Encoder得到词向量与情感向量的乘积 （版本2）
            embadding_out2 = outputs2['last_hidden_state']
            new_embadding_out2 = embadding_out2.unsqueeze(1)


            out2 = torch.cat([self.conv_and_pool(new_embadding_out2, conv) for conv in self.convs2], 1)
            out2 = self.text_relu2_1(out2)
            fc_out2 = self.fc2(out2)

            # gru_out
            gru_inputs2 = gru_inputs2.float()
            gru_out2, h2 = self.gru2(gru_inputs2, None)
            gru_out2 = gru_out2[:, -1, :]

            # final_out
            mul_out2 = torch.mm(fc_out2, gru_out2)
            final_out2 = self.fc2_2(mul_out2)
            final_out2 = self.text_relu2_2(final_out2)



            # 第1分支，真假新闻二分类器，返回0-1之间
            # 越接近0.5就是越模糊，越有含金量，有标记，训练标记的准确性
            score = self.class_classifier(final_out1)  # shape [batch_size,2]

            # 第2分支，领域鉴别器，softmax函数返回概率
            # 用的全体数据，无关label，只进行对抗域训练
            reverse_feature = grad_reverse(final_out2)  # 梯度反转，直接连接在DFE输出的后面
            domain_output = self.domain_classifier(reverse_feature) # shape [batch_size, event_num] 一批处理事件有5个概率

            # 第3分支，UL分类器，
            lable_output2 = self.affection_discriminator(final_out2)
            return score, domain_output, lable_output2

        # 有标签数据，求了双重特征，拿来判别真假新闻
        elif flag == 1:
            context1 = x1  # 对应输入的句子               shape[batch_size * padding] [16, 54]
            # 16个向量，每一个代表着句子中每个词语在词库里面的索引，填充为长度为54的向量，再用掩码掩盖住没有含义的位（置0）
            mask1 = x2
            gru_inputs1 = x5  # [16, 8, 24]16个句子，每一个被分成8部分，每部分有长度为24的特征维度


            outputs1 = self.bert(context1, attention_mask=mask1)  # shape[batch_size * hidden_size(768)]
            #hidden_size=768是BERT模型的一个参数，表示它使用了768个神经元来编码每个
            # 词语
            # 768—最后一层隐藏层的神经元个数，其参数能有效区展示词语特征
            embadding_out1 = outputs1['last_hidden_state'] # ([16, 54（句子长度）, 768（词语特征）])
            new_embadding_out1 = embadding_out1.unsqueeze(1)# [16, 1, 54, 768]
            # 卷积池化，降维，概括句子信息
            out1 = torch.cat([self.conv_and_pool(new_embadding_out1, conv) for conv in self.convs1], 1)  # [self.batch_size, 768]
            out1 = self.text_relu1_1(out1)
            fc_out1 = self.fc1(out1)                        # 把BERT词向量经过卷积层和一层全连接层


            gru_inputs1 = gru_inputs1.float()       #（16，8，24）8个时间步
            # print(gru_inputs1.shape)
            gru_out1, h1 = self.gru1(gru_inputs1, None)     #（16，8，64）八个时间步，2（双向）*32（隐藏层神经元数量）个参数
            #只选取最后一步的64个参数，融合了总体的语义特征，8个时间步变成了一个
            gru_out1 = gru_out1[:, -1, :]                   # 把情感向量拿到GRU中拿到对应的GRU输出


            mul_out1 = torch.mm(fc_out1, gru_out1)          # GRU输出与TEXT-CNN输出相乘
            final_out1 = self.fc1_2(mul_out1)
            final_out1 = self.text_relu1_2(final_out1)


            # cls1 = self.dropout(outputs1['pooler_output'])
            score = self.class_classifier(final_out1)  # shape [batch_size,2]，这里尝试把final_out1也输出了看看可视化的情况


            return [score,final_out1]

        # embbeding来获取特征向量
        elif flag == 5:
            context1 = x1  # 对应输入的句子               shape[batch_size * padding] [16, 54]
            # 16个向量，每一个代表着句子中每个词语在词库里面的索引，填充为长度为54的向量，再用掩码掩盖住没有含义的位（置0）
            mask1 = x2
            gru_inputs1 = x5  # [16, 8, 24]16个句子，每一个被分成8部分，每部分有长度为24的特征维度


            # print("embedding",type(embedding)) # 是个torch
            embedding = embedding.to(torch.float32)
            embedding_layer = torch.nn.Linear(4096,768).to(self.convs1[0].weight.dtype).to(device)
            # print("conv精度", self.convs1[0].weight.dtype)  # 查看第一个卷积层的权重精度
            # print("emlay精度", embedding_layer.weight.dtype)


            transformed_embedding = embedding_layer(embedding)
            transformed_embedding.to(device)



            seq_len = transformed_embedding.size(1)
            max_length = 512

            # 计算需要多少个部分
            num_chunks = (seq_len + max_length - 1) // max_length  # 向上取整
            chunks = [transformed_embedding[:, i * max_length:(i + 1) * max_length, :] for i in range(num_chunks)]

            # 通过 BERT 对每个部分分别进行推理
            outputs = []
            for chunk in chunks:
                output = self.bert(inputs_embeds=chunk)
                outputs.append(output['last_hidden_state'])

            # 拼接所有输出
            embadding_out1 = torch.cat(outputs, dim=1)  # 将不同块的输出拼接






            # outputs1 = self.bert(inputs_embeds=embedding)  # shape[batch_size * hidden_size(768)]
            # hidden_size=768是BERT模型的一个参数，表示它使用了768个神经元来编码每个
            # 词语
            # 768—最后一层隐藏层的神经元个数，其参数能有效区展示词语特征
            # embadding_out1 = outputs1['last_hidden_state']  # ([16, 54（句子长度）, 768（词语特征）])


            new_embadding_out1 = embadding_out1.unsqueeze(1)  # [16, 1, 54, 768]

            # # 拿到embedding 是new_embadding_out1
            # print("conv精度", self.convs1[0].weight.dtype)  # 查看第一个卷积层的权重精度
            # print("new精度", new_embadding_out1.dtype)
            #
            # new_embadding_out1 = new_embadding_out1.to(self.convs1[0].weight.dtype)
            # print("new精度", new_embadding_out1.dtype)
            # 卷积池化，降维，概括句子信息
            out1 = torch.cat([self.conv_and_pool(new_embadding_out1, conv) for conv in self.convs1],
                             1)  # [self.batch_size, 768]
            out1 = self.text_relu1_1(out1)
            fc_out1 = self.fc1(out1)  # 把BERT词向量经过卷积层和一层全连接层



            gru_inputs1 = gru_inputs1.float()  # （16，8，24）8个时间步
            gru_inputs1 = gru_inputs1.to(device)
            # print(gru_inputs1.shape)
            gru_out1, h1 = self.gru1(gru_inputs1, None)  # （16，8，64）八个时间步，2（双向）*32（隐藏层神经元数量）个参数
            # 只选取最后一步的64个参数，融合了总体的语义特征，8个时间步变成了一个
            gru_out1 = gru_out1[:, -1, :]  # 把情感向量拿到GRU中拿到对应的GRU输出




            mul_out1 = torch.mm(fc_out1, gru_out1)  # GRU输出与TEXT-CNN输出相乘
            final_out1 = self.fc1_2(mul_out1)
            final_out1 = self.text_relu1_2(final_out1)


            # cls1 = self.dropout(outputs1['pooler_output'])
            score = self.class_classifier(final_out1)  # shape [batch_size,2]，这里尝试把final_out1也输出了看看可视化的情况

            pm.rep_out = [score, final_out1]

            return [score, final_out1]








        #有标签数据，求了双重特征，拿来判别UL事件
        elif flag == 3:
            context1 = x1
            mask1 = x2
            gru_inputs2 = x5


            outputs1 = self.bert(context1, attention_mask=mask1)  # shape[batch_size * hidden_size(768)]


            embadding_out2 = outputs1['last_hidden_state']
            new_embadding_out2 = embadding_out2.unsqueeze(1)
            out2 = torch.cat([self.conv_and_pool(new_embadding_out2, conv) for conv in self.convs2], 1)
            out2 = self.text_relu2_1(out2)
            fc_out2 = self.fc2(out2)

            gru_inputs2 = gru_inputs2.float()
            gru_out2, h2 = self.gru2(gru_inputs2, None)
            gru_out2 = gru_out2[:, -1, :]

            mul_out2 = torch.mm(fc_out2, gru_out2)
            final_out2 = self.fc2_2(mul_out2)
            final_out2 = self.text_relu2_2(final_out2)


            lable_output1 = self.affection_discriminator(final_out2)
            return lable_output1# , final_out2

        #拿的是无标签的数据，求了双重特征，返回特征矩阵
        elif flag == 4:
            context2 = x3
            mask2 = x4
            gru_inputs2 = x6


            outputs2 = self.bert(context2, attention_mask=mask2)       # shape[batch_size * hidden_size(768)]

            embadding_out2 = outputs2['last_hidden_state']
            new_embadding_out2 = embadding_out2.unsqueeze(1)
            out2 = torch.cat([self.conv_and_pool(new_embadding_out2, conv) for conv in self.convs2], 1)
            out2 = self.text_relu2_1(out2)
            fc_out2 = self.fc2(out2)

            gru_inputs2 = gru_inputs2.float()
            gru_out2, h2 = self.gru2(gru_inputs2, None)
            gru_out2 = gru_out2[:, -1, :]

            mul_out2 = torch.mm(fc_out2, gru_out2)
            final_out2 = self.fc2_2(mul_out2)
            final_out2 = self.text_relu2_2(final_out2)

            return final_out2















def modify_if_marked_label(add_data):
    # 相当于把目标域，选择出来的有标记的数据的mark标记位置为1, add_data即为每次选择出来的数据, 类型是dataframe
    add_data['if_marked_label'] = add_data['if_marked_label'].replace({0: 1})

    # 记得在这里可以填充人工上标签的部分
    pass

def func1(amount,num):
    # 生成和固定为amount，个数为num的列表
    list1 = []
    for i in range(0,num-1):
        a = random.randint(0,amount)    # 生成 n-1 个随机节点
        list1.append(a)
    list1.sort()                        # 节点排序
    list1.append(amount)                # 设置第 n 个节点为amount，即总金额

    list2 = []
    for i in range(len(list1)):
        if i == 0:
            b = list1[i]                # 第一段长度为第 1 个节点 - 0
        else:
            b = list1[i] - list1[i-1]   # 其余段为第 n 个节点 - 第 n-1 个节点
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







def load_pm(models,df_tmp, prompt="nothing", if_soft=0 ,samples=2,num=1):
    global pm
    start_time = timer()
    # 循环生成样本并添加到 DataFrame 中
    # for _ in range(len(df_tmp)):
    # 根据index个原文本生成n个样本，返回之前选出来的样本df_tmp

    # select_text = df_tmp["content"]
    print("if_soft现在是否使用了soft",if_soft)
    if prompt == "prepared" or prompt == "normal":
        # df_tmp = df_tmp.iloc[0:3]
        # df_tmp[]

        inputdata = mk_my_dataset(df_tmp)
        dataloader1 = DataLoader(dataset=inputdata,
                            batch_size=16,
                            shuffle=True,
                            drop_last=True)
        input_score, input_vector = get_rep_vec1(models, dataloader1, flag=1)


        gen_texts = pm.gen_text(df_tmp, samples=samples, num=num, prompt=prompt, pro_type="vec", out_type="text", if_soft=if_soft)


        outputdata = mk_my_dataset(gen_texts)
        dataloader2 = DataLoader(dataset=outputdata,
                                 batch_size=16,
                                 shuffle=True,
                                 drop_last=True)
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
        file_path = str(REPO_ROOT / 'outputs' / 'pheme_input_output_vectors.csv')
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

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
def load_data(flag, models, data_index,fine_num=0):
    global easy
    # 注意看，W是只是一个缓存，代表着pickle文件，仅此而已
    # 如果是初始化训练，第一次的话，就直接把数据加载进来了d
    if len(data_index) == 0:
        if flag == "train1":
            data_path = '../../data/pheme/processed/source_data.pkl'
            f = open(data_path, 'rb')
            w = pickle.load(f)
        elif flag == "train2":
            data_path = '../../data/pheme/processed/source_extend_data.pkl'
            f = open(data_path, 'rb')
            w = pickle.load(f)

        elif flag == "test":
            data_path = PHEME_EVENTS_DIR / f'event_{PHEME_EVENT_ID}_test.pkl'
            f = open(data_path, 'rb')
            w = pickle.load(f)
        elif flag == "validate":
            data_path = PHEME_EVENTS_DIR / f'event_{PHEME_EVENT_ID}_validate.pkl'
            f = open(data_path, 'rb')
            w = pickle.load(f)
        elif flag == "pool":
            df_pool = pd.read_excel(PHEME_POOL_FILE)
            w = mydict.to_bert_input_new(df_pool, list(df_pool.index))
    # 对于之后的任意次：2，3，4，5次
    else:
        print("data_index---info(len, max):", len(data_index), np.max(data_index))
        if flag == "fine_train":
            # 把data_index的数据 从pool_data中搞出来，增加到train1中，并改变if_marked_label0到1
            df = pd.read_excel(PHEME_POOL_FILE)  # 即完整的目标域数据  加载出test的字典张量数据
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
            finetune_amodel(pm, model, selected_dataset,epochs=1,obj="fine")

            # 在这里插入llm
            print(f"mynet的第{fine_num}次微调的样本模仿生成开始，load_pm")
            df_tmp = load_pm(models,df_tmp, prompt="normal", if_soft=1, samples=5, num=2)


            df_source_extend = pd.read_excel('../../data/pheme/processed/train_data.xlsx')  # 源 + 0.1目标





            df_old_add = df_source_extend[df_source_extend['event_label'] == PHEME_EVENT_ID]             # 拿到初始化时候的目标域0.1数据
            df_old_add = df_old_add.sample(frac=0.1, random_state=42)





            df_new_train = pd.concat([df_tmp, df_old_add,], axis=0)                          # 新的训练数据=0.1+0.05
            df_new_train = df_new_train.reset_index(drop=True)                              # 重置了索引
            df_new_train.to_excel(PHEME_WORK_DIR / 'new_train_data.xlsx', index=False)  # 这里现在是 0.1的目标+之后选择出来的0.05的目标
            print("new_train_data文件的长度：", len(df_new_train))
            index_train = list(df_new_train.index)                                          # 1...n
            new_train_data = mydict.to_bert_input_new(df_new_train, index_train)
            w = new_train_data
        elif flag == "fine_pool":
            df_pool = pd.read_excel(PHEME_POOL_FILE)
            df_new_pool = df_pool.drop(df_pool.index[data_index])  # 训练备选数据删去后index后得到真正的pool样本 这个索引应该还是原来的索引
            df_new_pool = df_new_pool.reset_index(drop=True)        # 重置了索引
            df_new_pool.to_excel(PHEME_WORK_DIR / 'new_pool_data.xlsx', index=False)    # 这是删去0.05的目标
            index_pool = list(df_new_pool.index)                    # 返回回去的已经是重置之后的索引了
            new_test_data = mydict.to_bert_input_new(df_new_pool, index_pool)
            w = new_test_data



        elif flag == "fine_train_after":
            df1 = pd.read_excel(PHEME_WORK_DIR / 'new_pool_data.xlsx')
            df_train_after = df1.iloc[data_index]                     # 选出来
            modify_if_marked_label(df_train_after)

            # ！！！！！！！！！轻量化处理
            # easy = 0
            # if easy:
            #     df_train_after = df_train_after(frac=0.1, random_state=42)  # 随机选择 10% 的数据


            print("new_train_data文件的长度：", len(df_train_after))
            selected_dataset = mk_my_dataset(df_train_after)

            # 每个样本都放到amodel里面拿到embedding
            print(f"mynet的第{fine_num}次微调的soft微调开始")
            finetune_amodel(pm, model, selected_dataset,epochs=1,obj="fine")


            # 模仿生成
            print(f"mynet的第{fine_num}次微调的样本模仿生成开始，load_pm")
            df_train_after = load_pm(models,df_train_after, prompt="normal", if_soft=1, samples=5, num=2)





            df_train_before = pd.read_excel(PHEME_WORK_DIR / 'new_train_data.xlsx')
            df_train_after = pd.concat([df_train_before, df_train_after], axis=0)
            df_train_after = df_train_after.reset_index(drop=True)
            print("new_train_data文件的长度：", len(df_train_after))
            df_train_after.to_excel(PHEME_WORK_DIR / 'new_train_data.xlsx', index=False)
            # df_train_after.to_excel('../../data/pheme/processed/new_train_after_data.xlsx', index=False)  # 这里现在是 0.1的目标+之后选择出来的0.05的目标
            index_train = list(df_train_after.index)
            new_train_after_data = mydict.to_bert_input_new(df_train_after, index_train)
            w = new_train_after_data
        elif flag == "fine_pool_after":
            df2 = pd.read_excel(PHEME_WORK_DIR / 'new_pool_data.xlsx')
            print("new_test_data文件的长度：", len(df2))
            df_pool_after = df2.drop(df2.index[data_index])           # 删除掉
            df_pool_after = df_pool_after.reset_index(drop=True)

            df_pool_after.to_excel(PHEME_WORK_DIR / 'new_pool_data.xlsx', index=False)  # 这里现在是 0.1的目标+之后选择出来的0.05的目标
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
#             data_path = '../../data/pheme/processed/source_data.pkl'
#             with open(data_path, 'rb') as f:
#                 w = pickle.load(f)
#         elif flag == "train2":
#             data_path = '../../data/pheme/processed/source_extend_data.pkl'
#             with open(data_path, 'rb') as f:
#                 w = pickle.load(f)
#         elif flag == "test":
#             data_path = '../../data/pheme/processed/sampled_data/event_label_4_.pkl'
#             with open(data_path, 'rb') as f:
#                 w = pickle.load(f)
#         elif flag == "validate":
#             data_path = '../../data/pheme/processed/sampled_data/event_label_4_validate_.pkl'
#             with open(data_path, 'rb') as f:
#                 w = pickle.load(f)
#         elif flag == "pool":
#             data_path = '../../data/pheme/processed/source_pool_data.pkl'
#             with open(data_path, 'rb') as f:
#                 w = pickle.load(f)
#     else:
#         print("data_index---info(len, max):", len(data_index), np.max(data_index))
#
#         # 处理fine_train情况
#         if flag == "fine_train":
#             df = pd.read_excel('../../data/pheme/processed/sampled_data/pool_data_4.xlsx')  # 加载目标域数据
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
#             df_source_extend = pd.read_excel('../../data/pheme/processed/train_data.xlsx')
#             df_old_add = df_source_extend[df_source_extend['event_label'] == 2]
#             df_old_add = df_old_add.sample(frac=0.1, random_state=42)
#
#             # 合并数据
#             df_new_train = pd.concat([df_tmp, df_old_add], axis=0)
#             df_new_train = df_new_train.reset_index(drop=True)
#             df_new_train.to_excel(PHEME_WORK_DIR / 'new_train_data.xlsx', index=False)
#             print("new_train_data文件的长度：", len(df_new_train))
#             index_train = list(df_new_train.index)
#             new_train_data = mydict.to_bert_input_new(df_new_train, index_train)
#             w = new_train_data
#
#         elif flag == "fine_pool":
#             df_pool = pd.read_excel('../../data/pheme/processed/sampled_data/pool_data_4.xlsx')
#             df_new_pool = df_pool.drop(df_pool.index[data_index])  # 从池数据中去除选择的数据
#             df_new_pool = df_new_pool.reset_index(drop=True)
#             df_new_pool.to_excel(PHEME_WORK_DIR / 'new_pool_data.xlsx', index=False)
#             index_pool = list(df_new_pool.index)
#             new_test_data = mydict.to_bert_input_new(df_new_pool, index_pool)
#             w = new_test_data
#
#         elif flag == "fine_train_after":
#             df1 = pd.read_excel(PHEME_WORK_DIR / 'new_pool_data.xlsx')
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
#             df_train_before = pd.read_excel(PHEME_WORK_DIR / 'new_train_data.xlsx')
#             df_train_after = pd.concat([df_train_before, df_train_after], axis=0)
#             df_train_after = df_train_after.reset_index(drop=True)
#             print("new_train_data文件的长度：", len(df_train_after))
#             df_train_after.to_excel(PHEME_WORK_DIR / 'new_train_data.xlsx', index=False)
#             index_train = list(df_train_after.index)
#             new_train_after_data = mydict.to_bert_input_new(df_train_after, index_train)
#             w = new_train_after_data
#
#         elif flag == "fine_pool_after":
#             df2 = pd.read_excel(PHEME_WORK_DIR / 'new_pool_data.xlsx')
#             print("new_test_data文件的长度：", len(df2))
#             df_pool_after = df2.drop(df2.index[data_index])
#             df_pool_after = df_pool_after.reset_index(drop=True)
#             df_pool_after.to_excel(PHEME_WORK_DIR / 'new_pool_data.xlsx', index=False)
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
vocab_path = str(REPO_ROOT / 'DAAL' / 'vocab.txt')
mydict = ProData.LoadSingleSentenceClassificationDataset(vocab_path, tokenizer)
config = Config(device=device)
# model2 = MyNet(config).to(device)

# 加载 dataset
def load_dataset(data_index, flag, fine_num=0, models=""):
    '''
    文件说明：
    '../../data/pheme/processed/train_data.xlsx'
    '../../data/pheme/processed/train2_data.xlsx'
    '../../data/pheme/processed/test_data.xlsx'      这三个文件都是预处理数据的，即不能删改的

    '../../data/pheme/processed/new_train_data.xlsx'
    '../../data/pheme/processed/new_test_data.xlsx'  这两个文件都是微调时候，动态产生的文件，可以调整
    '''

    if flag == 0: # 返回预训练时期的训练的测试原始数据
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
        fine_train = load_data("fine_train",models, data_index, fine_num=fine_num)             # 这里的长度是对的
        print(len(fine_train))
        fine_pool = load_data("fine_pool",models, data_index)               # 这里有问题，还是1093
        print(len(fine_pool))
        # 加载微调训练集
        fine_train_dataset = Rumor_Data(fine_train)
        # get_rep_vec(models=model, train_dataset=fine_train_dataset)
        # 加载微调测试集
        fine_pool_dataset = Rumor_Data(fine_pool)
        return fine_train_dataset, fine_pool_dataset
    if flag == 2:
        print(f"loading data----------------------------------后{fine_num}次微调")
        fine_train_after = load_data("fine_train_after",models, data_index, fine_num=fine_num)  # 这里的长度是对的

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
    start_epoch = 0 #！！！！！！！！！！！！！！！

    # 第一阶段结束后的文件
    final_file_name = ''
    best_dir = 'null'
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=16,
                             shuffle=True,
                             drop_last=True)
    pool_loader = DataLoader(dataset=pool_dataset,
                             batch_size=16,
                             shuffle=True,
                             drop_last=True)
    print("预训练阶段：")
    print(f"test_loader的长度：{len(test_loader)}")
    print(f"pool_loader的长度：{len(pool_loader)}")

    # tst(models, config, train_dataset, test_loader, best_dir, flag, owner="DRCD not Pretrained")

    #  预训练！！
    if exists(final_file_name):
        print("加载已经最终完成的第一阶段预训练的模型")
        checkpoint = torch.load(final_file_name)
        models.load_state_dict(checkpoint['model'], strict=True)
    else:

        # 训练4次！！！！！！！！！！！！！！
        print("训练模型！！!")
        for epoch in range(start_epoch, start_epoch+8):
            train_epoch(models, train_dataset, train_dataset2, criterion, criterion2, epoch, flag,test_dataset=test_dataset)



    # 上面的意思是，如果已经有模型了直接拿过来，否则自己马上在train——epoch用数据训练一个
    # 加载好模型之后，测试一下数据集，马上就选择最好的数据,这是第一轮




    # 我测你码码 能不能用，能用就滚，正式把这些数据拿来筛选
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
def train_epoch(models, train_dataset, train_dataset2, criterion, criterion2, epoch, flag = 0,test_dataset=None):
    print("第epoch：", epoch)
    p = float(epoch) / 100
    dynamic_lr = 0.001 / (1. + 10 * p) ** 0.8
    optimizer = torch.optim.Adam([
        {'params': models.bert.parameters()},  # 学习率为3e-5
        {'params': models.class_classifier.parameters(), 'lr': dynamic_lr},  # 0.001
        {'params': models.domain_classifier.parameters(), 'lr': dynamic_lr},
        {'params': models.infer_discriminator.parameters(), 'lr': dynamic_lr},  # 0.005
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
    ], lr=3e-5)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=16,
                              shuffle=True,
                              drop_last = True)
    train_loader2 = DataLoader(dataset=train_dataset2,
                               batch_size=16,
                               shuffle=True,
                               drop_last = True)



    acc_vector = []
    cost_vector = []
    valid_acc_vector = []
    class_cost_vector = []
    domain_cost_vector = []
    mark_cost_vector = []
    epoch_loss, epoch_acc = 0., 0.
    total_len = 0
    models.train()
    iter2 = iter(train_loader)
    flag = np.array(flag)
    flag = torch.from_numpy(flag).long()
    flag = flag.to(device)



    # 分batch进行训练
    # 外面那层是多的，train_loader2
    # train_loader是只打了标签的部分数据
    for i, (train_text2, train_mask2, train_affection2, train_labels2, event_labels2, train_marked_label2, train_data_index2) in enumerate(train_loader2):


        (train_text1, train_mask1, train_affection1, train_labels1, event_labels1, train_marked_label1,train_data_index1) = iter2.__next__()
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        x1 = train_text1
        x2 = train_mask1
        x3 = train_text2
        x4 = train_mask2
        x5 = train_affection1  # torch.Size([32, 8, 24])
        x6 = train_affection2
        x1 = x1.to(device)
        x2 = x2.to(device)
        x3 = x3.to(device)
        x4 = x4.to(device)
        x5 = x5.to(device)
        x6 = x6.to(device)
        # forward=0调用
        predict, domain_outputs, lable_outputs2 = models(x1, x2, x3, x4, x5, x6, flag)  # predict torch.Size([32, 2])
        # print(predict.cpu().detach().numpy())
        train_labels = train_labels1.long()
        event_labels = event_labels1.long()
        # train_marked_labels_soft = train_marked_label2.long()
        train_marked_labels2 = train_marked_label2.float()
        train_labels = train_labels.unsqueeze(1)
        # train_marked_labels_soft = train_marked_labels_soft.unsqueeze(1)
        train_marked_labels2 = train_marked_labels2.unsqueeze(1)

        train_labels = train_labels.to(device)
        event_labels = event_labels.to(device)
        # train_marked_labels_soft = train_marked_labels_soft.to(device)
        train_marked_labels2 = train_marked_labels2.to(device)
        # print(train_marked_labels2)
        train_labels = train_labels.squeeze(dim = 1)
        # class_loss = criterion2(predict, train_labels)  # loss2(x,y.long())，cross损失函数里面要求，target的类型应该是long类型，input类型不做要求
        loss_fn = focal_loss(alpha=0.25, gamma=2, num_classes=2)
        class_loss = loss_fn(predict, train_labels)
        domain_loss = criterion2(domain_outputs, event_labels)
        # train_marked_labels_soft = train_marked_labels_soft.squeeze(dim = 1)
        # mark_loss = criterion(lable_outputs, train_marked_labels)
        mark_loss2 = criterion(lable_outputs2, train_marked_labels2)
        # 用 预测loss来作为标准
        # class_loss2 = criterion3(predict, train_labels.squeeze(dim=1))
        # loss_loss = LossPredLoss(pred_loss, class_loss2)
        loss = class_loss + 2 * domain_loss + mark_loss2
        # print("loss", loss)
        loss.backward()
        optimizer.step()

        _, argmax = torch.max(predict, 1)
        # print(argmax.cpu().detach().numpy())
        # print(train_labels.cpu().detach().numpy())

        acc = (train_labels == argmax.squeeze()).float().mean()
        epoch_loss += loss * len(train_labels)
        epoch_acc += acc * len(train_labels)
        # print(acc)             # 0.75 0.6

        total_len += len(train_labels)
        # print(acc)


        class_cost_vector.append(class_loss.item())
        domain_cost_vector.append(domain_loss.item())
        # mark_cost_vector.append(mark_loss.item())
        mark_cost_vector.append(mark_loss2.item())
        cost_vector.append(loss.item())
        # if i % 30 == 0:
        #     print(acc_vector)
        #     print("Loss: %.4f, Class Loss: %.4f, Domain loss: %.4f, Mark loss: %.4f, Dynamic_lr: %.4f,Train_Acc: %.4f"%(np.mean(cost_vector), np.mean(class_cost_vector),
        #          np.mean(domain_cost_vector), np.mean(mark_cost_vector), dynamic_lr, np.mean(acc_vector)))
        if (i + 1) % len(train_loader) == 0:
            iter2 = iter(train_loader)

    print('Epoch [%d/%d],  Loss: %.4f, Class Loss: %.4f, domain loss: %.4f, if_marked loss: %.4f, Dynamic_lr: %.4f, Train_Acc: %.4f'
            % (epoch + 1, 10, np.mean(cost_vector), np.mean(class_cost_vector),
                np.mean(domain_cost_vector), np.mean(mark_cost_vector), dynamic_lr,
                np.mean(acc_vector)))
    # # 保存模型和优化器参数
    # best_dir = "null"
    # test_loader = DataLoader(dataset=train_dataset2,
    #                          batch_size=16,
    #                          shuffle=True,
    #                          drop_last=True)

    # 保存模型和优化器参数
    best_dir = "null"
    # 这里原版是用train2的，但是不符合常理，用前后统一的测试集
    # test_loader = DataLoader(dataset=train_dataset2,
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=16,
                             shuffle=True,
                             drop_last=True)











    accuracy, auc_roc, f1, precision, recall, test_confusion_matrix, init_per = tst(models, config, train_dataset,
                                                                                    test_loader, best_dir, flag,
                                                                                    owner=f"DRCD Pretrain{epoch + 1}")

    state = {'model': models.state_dict(),
             'optimizer': optimizer.state_dict(),
             'epoch': epoch}





    output_dir = REPO_ROOT / 'outputs' / 'pheme_event1'
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = output_dir / ('pre-training-network-epoch' + str(epoch) + '.pth')
    torch.save(state, tmp_path)
    log_file = output_dir / 'performance_log.json'
    log_performance(epoch, 100, np.mean(cost_vector), np.mean(class_cost_vector),
                    np.mean(domain_cost_vector), np.mean(mark_cost_vector), dynamic_lr,
                    np.mean(acc_vector), log_file)







# 微调时候的训练
def train_finetune(models, train_dataset, train_dataset2, test_dataset, pool_dataset,fine_num, flag):
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





    for name, param in models.named_parameters():                # 打开Bert层
        for ele in freeze_layers:
            if ele in name:
                param.requires_grad = True
                break
    global count
    epoch_2 = 5
    if int(count) - num < num:
        epoch_2 = 5

    for i in range(epoch_2):              # 之前的模型是5(保存了的)
        # 微调第二个分支真假鉴别
        all_real, all_fake, forward1 = train_finetune_epoch(models, train_dataset, criterion2, i, flag)

        best_dir = "null"
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=16,
                                 shuffle=True,
                                 drop_last=True)
        pool_loader = DataLoader(dataset=pool_dataset,
                                 batch_size=16,
                                 shuffle=True,
                                 drop_last=True)


        accuracy, auc_roc, f1, precision, recall, test_confusion_matrix,init_per = tst(models, config, train_dataset, test_loader, best_dir, flag)
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





    owner = f"DRCD Ft{fine_num}"
    accuracy, auc_roc, f1, precision, recall, test_confusion_matrix,init_per = tst(models, config, train_dataset, test_loader, best_dir, flag, owner=owner)

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
def tst(models, config, train_dataset, test_loader, best_dir, flag, owner="the model"):
    # 仅作为测试功能的函数
    global count
    epoch_acc = 0.
    epoch_test_accuracy = 0.
    total_len = 0
    all_preds_mark = []
    all_preds_mark2 = []
    all_preds = []                                # 只拿到了最大一列的值
    all_prefict = []
    all_index = []
    if exists(best_dir):
        models = MyNet(config).to(device)
        models.load_state_dict(torch.load(best_dir))
        print('加载已保存模型！')
    if torch.cuda.is_available():
        models.cuda()
    models.eval()
    test_score = []
    test_pred = []
    test_true = []
    flag = 0
    flag = np.array(flag)
    flag = torch.from_numpy(flag).long()
    flag = flag.to(device)
    with torch.no_grad():
        # 分batch进行测试
        for i, (test_text, test_mask, test_affection, test_labels, event_labels, test_marked_label, test_data_index) in enumerate(test_loader):
            # print(f'测试样本{i}')
            x1 = test_text
            x2 = test_mask
            x3 = test_affection
            x1 = x1.to(device)
            x2 = x2.to(device)
            x3 = x3.to(device)
            # forward=0调用
            predict_label, predict_domain, predict_marked_label2 = models(x1, x2, x1, x2, x3, x3, flag)
            # print("predict_marked_label的形状:", predict_marked_label.shape)     torch.Size([32, 1])
            # preds_mark = predict_marked_label.cpu().data
            preds_mark2 = predict_marked_label2.cpu().data
            # 这里拿到的predict_label是经过了 softmax层的
            predict_label = predict_label.cpu().data
            # print("predict_label", predict_label)
            _, test_argmax = torch.max(predict_label, 1)                 # _是一行两个中的最大值，test_argmax是最大值对应的索引
            # 得到每个batch预测的标签
            if i == 0:
                test_score = to_np(predict_label.squeeze())
                test_pred = to_np(test_argmax.squeeze())
                test_true = to_np(test_labels.squeeze())
            else:
                test_score = np.concatenate((test_score, to_np(predict_label.squeeze())), axis=0)
                test_pred = np.concatenate((test_pred, to_np(test_argmax.squeeze())), axis=0)
                test_true = np.concatenate((test_true, to_np(test_labels.squeeze())), axis=0)
            # all_preds_mark.extend(preds_mark)            # 在列表末尾一次性追加另一个序列中的多个值
            # all_preds_mark2.extend(preds_mark2)
            # all_index.extend(test_data_index)
            #
            # all_preds.append(_)
            # all_prefict.append(predict_label)

            test_labels = test_labels.float()
            test_labels = test_labels.unsqueeze(1)
            test_labels = test_labels.to(device)
            total_len += len(test_labels)
        # 最后把 predict_marked_label 拼起来，然后选择最低的几个（最靠近0的几个）选择出来
        # all_preds_mark = torch.stack(all_preds_mark)    # all_preds_mark 把所有数据的0维拼接起来
        # all_preds_mark = all_preds_mark.view(-1)
        # all_preds_mark2 = torch.stack(all_preds_mark2)
        # all_preds_mark2 = all_preds_mark2.view(-1)

        # all_preds = torch.cat(all_preds)                # torch.cat, 把张量shape不相等的拼接起来
        # all_preds = all_preds.view(-1)
        # all_prefict = torch.cat(all_prefict)
        # print(all_preds.tolist())                                  最后都接近了0
        # need to multiply by -1 to be able to use torch.topk        负的好拿一些
        # all_preds_mark *= -1
        # all_preds_mark2 *= -1
        # all_preds *= -1



        # 计算准确率
        test_accuracy = metrics.accuracy_score(test_true, test_pred)
        # F1值 可以解释为精度和查全率的加权平均值，其中F1分数在1时达到最佳值，在0时达到最差值。
        test_f1 = metrics.f1_score(test_true, test_pred)  # average='macro'
        # precison_score：预测为正类且预测正确的数量/预测为正类的数量
        test_precision = metrics.precision_score(test_true, test_pred)
        # 召回率 被预测为正的样本占正样本总量的比例。Recall体现了模型对正样本的识别能力，Recall越高，模型对正样本的识别能力越强。
        test_recall = metrics.recall_score(test_true, test_pred)
        # test_score_convert 就是把模型输出结果的对于第二列，每个对于1的预测概率来输出了
        test_score_convert = [x[1] for x in test_score]
        # ROC曲线，围成面积(记作AUC）越大，说明性能越好
        test_aucroc = metrics.roc_auc_score(test_true, test_score_convert, average='macro')

        test_confusion_matrix = metrics.confusion_matrix(test_true, test_pred)


        test_precision2, test_recall2, test_f12, _ = precision_recall_fscore_support(test_true, test_pred, average = "micro")




        print("Classification Acc: %.4f, AUC-ROC: %.4f"
              % (test_accuracy, test_aucroc))
        print("Classification report:\n%s\n"
              % (metrics.classification_report(test_true, test_pred)))
        print("Classification confusion matrix:\n%s\n"
              % (test_confusion_matrix))
        print("test_f1, test_precision, test_recall", test_f1, test_precision, test_recall)
        print("micro下的f1值, precision, recall", test_f12, test_precision2, test_recall2)

        the_per(test_true, test_pred, test_score, owner=owner)
        init_per = [test_true, test_pred, test_score]




        print('结果输出')
        # VCCA的办法
        # append_data = select_data(all_preds_mark, all_preds_mark2, all_index, num)
        # 随机采样
        # append_data = random_index(count, num)

        # 不确定性采样
        # uncertaintysampler = UncertaintySampling(2, 0)
        # append_data = uncertaintysampler.query(all_preds, all_index, num)
        # 不确定熵采样
        # uncertaintyentropysampler = UncertaintyEntropySampling(2, 0)
        # append_data = uncertaintyentropysampler.query(all_prefict, all_index, num)
        # Core-set集采样
        # coresetsampler = CoreSetSampling(2, 0)
        # append_data = coresetsampler.greedy_k_center(train_dataset, test_dataset, num)
        # return append_data
        # 这里返回一个准确率，最终在微调函数中
        return test_accuracy, test_aucroc, test_f1, test_precision, test_recall, test_confusion_matrix,init_per



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


    file= open('result.csv','a',newline='')
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
def train_finetune_epoch(models, train_dataset, criterion, epoch, flag = 1):
    models.train()
    flag = 1
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=16,
                              shuffle=True,
                              drop_last=True)


    flag = np.array(flag)
    flag = torch.from_numpy(flag).long()
    flag = flag.to(device)
    p = float(epoch) / 100
    dynamic_lr = 0.0001 / (1. + 10 * p) ** 0.8
    optimizer = torch.optim.Adam([
        {'params': models.bert.parameters()},  # 学习率为3e-5
        {'params': models.class_classifier.parameters(), 'lr': dynamic_lr},  # 0.001
        {'params': models.domain_classifier.parameters(), 'lr': dynamic_lr},
        {'params': models.infer_discriminator.parameters(), 'lr': dynamic_lr},  # 0.005
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
    all_result_real = []
    all_result_fake = []
    all_label = []
    for i, (train_text, train_mask, train_affection, train_labels, event_labels, train_marked_label, train_data_index) in enumerate(train_loader):
        optimizer.zero_grad()
        train_text = train_text.to(device)
        train_mask = train_mask.to(device)
        train_affection = train_affection.to(device)
        # forward=1调用
        forward1 = models(train_text, train_mask, train_text, train_mask, train_affection, train_affection, flag)





        predict = forward1[0]
        representative_vector = forward1[1]

        # vis_rep_vec(title=f"第{fine_num}次微调样本的PCA", data=representative_vector, method="PCA")
        # vis_rep_vec(title=f"第{fine_num}次微调样本的TSNE", data=representative_vector, method="TSNE")

        # print("predict: ", predict)
        # print("representative_vector: ", representative_vector)
        all_result.extend(representative_vector.detach().cpu().numpy())
        all_label.extend(train_labels.detach().cpu().numpy())


        train_labels = train_labels.long()
        train_labels = train_labels.unsqueeze(1)
        train_labels = train_labels.to(device)
        train_labels = train_labels.squeeze(dim=1)
        class_loss = criterion(predict, train_labels)
        class_loss.backward()
        optimizer.step()

    # print("predict: ", predict.size(), predict)
    # print("representative_vector: ", representative_vector.size(), representative_vector)




    all_result = np.array(all_result)
    all_label = np.array(all_label)

    # print("all_result", all_result, len(all_result))
    # print("all_label", all_label, len(all_label))




    all_result_real = all_result[all_label == 0]
    all_result_fake = all_result[all_label == 1]
    print(1)

    return all_result_real, all_result_fake, forward1

# 微调第三个分支的函数
def train_finetune3_epoch(models, train_dataset, test_dataset, criterion, epoch,flag):
    models.train()
    cifar_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
    train_loader = DataLoader(dataset=cifar_dataset,
                              batch_size=16,
                              shuffle=True,
                              drop_last= True)
    flag = 3
    flag = np.array(flag)
    flag = torch.from_numpy(flag).long()
    flag = flag.to(device)
    p = float(epoch) / 100
    dynamic_lr = 0.0001 / (1. + 10 * p) ** 0.8
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=dynamic_lr)
    for i, (train_text, train_mask, train_affection, train_labels, event_labels, train_marked_label, train_data_index) in enumerate(train_loader):
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













def select_best_data(models, config, pool_loader,  train_dataset, pool_dataset, flag):
    flag = 0
    global num
    flag = np.array(flag)
    flag = torch.from_numpy(flag).long()
    flag = flag.to(device)
    # 相当于对已经训练好的模型，来选出那个是最不像的
    all_preds_mark2 = []      # 数据是否被标注
    all_preds_value = []      # 每个数据的最大概率
    all_prefict = []          # 每个数据得到的predict
    all_predict2 = []         # 经过第一层采样的样本得到的
    all_index = []
    with torch.no_grad():
        # 分batch进行测试
        total_batches = len(pool_loader)
        for i, (pool_text, pool_mask, pool_affection, pool_labels, event_labels, pool_marked_label, pool_data_index) in enumerate(pool_loader):
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
        first_querry_pool_indices = select_data(all_preds_mark2, all_index, 4*num)
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

def random_index(max,num):
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
    append_data_index = []
    append_data_tmp = []
    test_dataset = 0
    global count
    global num
    global pm
    global easy
    # 改变count为test长度
    # 改变选择数据的数目 count
    # df_source_extend['event_label']


    flag = 0  # 预训练模型阶段
    print(f"flag=0时预训练阶段的pooldata有{count}个")
    # return出来的是 train_dataset, train_dataset2, validate_dataset, test_dataset, pool_dataset

    train_dataset, train_dataset2, validate_dataset, test_dataset, pool_dataset = load_dataset(append_data_index, flag)
    print(f"🔍 测试集大小: {len(test_dataset)}")
    print(f"🔍 测试数据集类型: {type(test_dataset)}")
    print(f"🔍 `test_dataset` 大小: {len(test_dataset)}")
    # for i in range(3):
    #     sample = test_dataset[i]
    #     print(f"✅ 样本 {i}: {sample}")

    add_test_dataset = []  # 让其为空

    # 这个是把train分割之后得到更多的test
    add_test_size = int(0.02 * len(train_dataset))   #0.02
    test_size = len(train_dataset) - add_test_size
    add_test_dataset, discarded_dataset = random_split(train_dataset, [add_test_size, test_size])
    print(f"🔍 测试集大小: {len(test_dataset)}")
    print(f"🔍 测试数据集类型: {type(test_dataset)}")
    print(f"🔍 `test_dataset` 大小: {len(test_dataset)}")
    # 无论数据集类型是什么，直接打印第一条数据的全部内容
    print(f"🔍 第一条数据的全部内容: {test_dataset[0]}")

    # 这个是把train的比例调小,并且加上test，得到train2
    # 此处修改用于训练的样本数量，实现少样本情况下的检测
    #
    # less_train_size = int(0.01 * len(train_dataset))
    # left_size = len(train_dataset) - less_train_size
    # less_train_dataset, left_dataset = random_split(train_dataset, [less_train_size, left_size])
    # less_train_dataset2 = ConcatDataset([less_train_dataset, test_dataset])

    event_label_groups = {0: [], 1: [], 2: [], 3: [], 4: []}

    for sample in train_dataset:
        event_label = sample[4].item()  # 假设 event_label 在索引 5
        if event_label in event_label_groups:
            event_label_groups[event_label].append(sample)

    # 2️⃣ 按 15% 采样
    less_train_dataset_list = []
    left_dataset_list = []
    sample_counts = {}

    for label, samples in event_label_groups.items():
        if len(samples) > 0:  # 确保类别有数据
            less_size = int(0.5 * len(samples))
            left_size = len(samples) - less_size
            generator = torch.Generator().manual_seed(42)

            # 使用该生成器进行 random_split
            less_subset, left_subset = random_split(samples, [less_size, left_size], generator=generator)

            less_train_dataset_list.append(less_subset)
            left_dataset_list.append(left_subset)

            # 记录采样数量
            sample_counts[label] = {
                "total": len(samples),
                "less_train": less_size,
                "left": left_size
            }

    # 3️⃣ 合并所有 less_train_dataset
    less_train_dataset = ConcatDataset(less_train_dataset_list)
    left_dataset = ConcatDataset(left_dataset_list)

    # 4️⃣ 与 test_dataset 进行合并
    less_train_dataset2 = ConcatDataset([less_train_dataset, test_dataset])

    # 5️⃣ **打印采样后的数据量**
    print("\n📊 **采样统计信息**")
    for label, counts in sample_counts.items():
        print(
            f"🔹 event_label {label}: 总样本数 = {counts['total']}, 采样进 less_train = {counts['less_train']}, 剩余样本数 = {counts['left']}")

    # **打印最终的数据集大小**
    print("\n✅ **最终数据集大小**")
    print(f"less_train_dataset 总大小: {len(less_train_dataset)}")
    print(f"left_dataset 总大小: {len(left_dataset)}")
    print(f"less_train_dataset2 (含 test_dataset) 总大小: {len(less_train_dataset2)}")




    if 1:
        train_dataset, train_dataset2,test_dataset= less_train_dataset, less_train_dataset2, add_test_dataset
        # train_dataset, train_dataset2 = less_train_dataset, less_train_dataset2


    print(f"test_dataset 总大小: {len(test_dataset)}")
    print(f"🔍 第一条数据的全部内容: {test_dataset[0]}")
    train_size = int(1 * len(pool_dataset))
    test_size = len(pool_dataset) - train_size

    pool_dataset, discarded_dataset = random_split(pool_dataset, [train_size, test_size])




    '''
        文件说明：
        '../../data/pheme/processed/train_data.xlsx'
        '../../data/pheme/processed/train2_data.xlsx'
        '../../data/pheme/processed/test_data.xlsx'      这三个文件都是预处理数据的，即不能删改的

        '../../data/pheme/processed/new_train_data.xlsx'
        '../../data/pheme/processed/new_test_data.xlsx'  这两个文件都是微调时候，动态产生的文件，可以调整
    '''


    # 通过训练拿到了预训练的模型 和补充数据
    # 把需要用到的数据拿来训练 flag=0，所以train返回的是选出来数据的索引
    # train是进行预训练并测试数据集能不能用的意思，最后选出并返回了最优的样本的索引
    # train——epoch是预训练的epoch的意思
    models.train()
    append_data_index = train(models, config, train_dataset, train_dataset2, test_dataset, pool_dataset, flag)

    # 将整个样本池放到pm里面预训练
    pm.load_soft_prompt(start_epoch=0, obj="pre", lr=0.1)

    easy=1
    if easy:
        prompt_seed_value = os.environ.get('SHARP_PROMPT_SEED')
        if not prompt_seed_value and PHEME_EVENT_ID != 2:
            raise RuntimeError('Set SHARP_PROMPT_SEED for PHEME events other than event 2.')
        prompt_seed = Path(
            prompt_seed_value
            or REPO_ROOT / 'data' / 'pheme' / 'prompt_seed' / 'clippool_16.xlsx'
        )
        my_dataset = mk_my_dataset(pd.read_excel(prompt_seed))
    print(my_dataset)

    # my_dataset=pool_dataset[:50]

    print("soft的预训练过程开始pretrain_amodel")
    pretrain_amodel(pm, models, my_dataset, epochs=20, obj="pre")




    # 在微调的时候，第一次用的是初始化的0.1的数据+0.05新选择的数据，第二次在新选择出来后，训练数据能不能只要第二次新选择的；175:1039, 54:1085
    for i in range(50):
        global fine_num

        fine_num = i
        #调整pool的样本量
        # count = count - num
        if count < num: print("pooldata已经清空，总共微调i次");return;print(f"现在是第{i}次微调————————————————————————————————————————————————————————")

        models.train()
        if i == 0:
            flag = 1  # 微调阶段
            # 第一次微调用 0.1+0.05的数据
            # 读取选出来样本的意思
            train_dataset, pool_dataset = load_dataset(append_data_index, flag, fine_num=i+1, models=models)
            # print(append_data_tmp)


        else:
            # 第二次微调只要0.05的数据每次选出的
            # 读取上一次的样本的意思
            flag = 2 # 后续微调状态的意思
            # 要在生成之前微调，所以soft的微调插在里面↓↓↓
            train_dataset, pool_dataset = load_dataset(append_data_tmp, flag, fine_num=i+1, models=models)





        # 前面已经选出了dataset


        #样本池的数据更新
        dis = count - len(pool_dataset)
        count = len(pool_dataset)
        print(f"这次运行是flag={flag}，而第{i+1}次微调选出了{dis}个样本，是否等于num{num}?{dis==num}，预训练阶段的pool_data有{count}个")


        append_data_tmp, forward1 = train_finetune(models, train_dataset, train_dataset2, test_dataset, pool_dataset, i+1, flag)



def mk_my_dataset(my_data):
    # 传进来的是一个df，在前面需要把dir转化成df
    # my_data = pd.read_excel(dir)
    print("my_dataset文件的长度：", len(my_data))
    index_my_data = list(my_data.index)
    my_data = mydict.to_bert_input_new(my_data, index_my_data)
    my_dataset = Rumor_Data(my_data)
    return my_dataset




def get_rep_vec1(models, train_loader, flag = 1, fine_num=0, embedding=[]):
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



def get_rep_vec5(models, train_loader, flag = 1, fine_num=0, embedding=[]):
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





def finetune_amodel(pm, models, train_dataset, epochs=50, obj="fine"):
    print("")
    dataloader = DataLoader(dataset=train_dataset,
                            batch_size=16,
                            shuffle=True,
                            drop_last=True)

    # pm.optimizer = optim.AdamW([pm.soft_prompt], lr=pm.lr)
    for epoch in range(pm.last_epoch, pm.last_epoch + epochs):
        toc = timer()
        print(f"Finetuning Epoch {epoch},Finetuning")
        pm.epoch_losses = []

        all_socre1, all_rep_vec1 = get_rep_vec1(models, dataloader, flag=1)
        all_socre5, all_rep_vec5 = get_rep_vec5(models, dataloader, flag=5)

        print("all1", len(all_socre1), len(all_rep_vec1))
        print("all5", len(all_socre5), len(all_rep_vec5))

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

            score5 = all_socre5[num]
            # loss2需要用score
            loss2 = get_loss2(score5)
            print("loss2成功", )

            loss = calculate_combined_loss(loss1, loss2, alpha=0.5, beta=0.5)
            print("计算loss成功", )
            # 反向传播和参数更新
            loss.backward(retain_graph=True)
            print("回传成功",)
            pm.optimizer.step()

            # # loss回传完了，放回到cpu里面负责记录和画图
            loss.cpu().detach().numpy()




            # 在每个 epoch 结束时清理显存缓存





            pm.epoch_losses.append(loss)
            # if num == int(length/5): # % 10 == 0:
            #     self.plot_losses(title=f"samples {num}")
            # print(f'loss of sample{num}: {loss}')
            # print(f"sample{num}花费时间:", timer()-sam_time)

        # loss.cpu().detach().numpy()

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
        weight_dir = REPO_ROOT / 'outputs' / 'weight'
        weight_dir.mkdir(parents=True, exist_ok=True)
        soft_path = weight_dir / f"pre_soft_prompt_word_epoch{epoch}{obj}.pth"
        print(pm.soft_prompt)
        torch.save((pm.soft_prompt, epoch, pm.losses, pm.epoch_mean_losses), soft_path)
        print(f'Soft prompt saved at {soft_path}')
        torch.cuda.empty_cache()





def pretrain_amodel(pm, models, train_dataset, epochs=50,obj="pre"):
    dataloader = DataLoader(dataset=train_dataset,
                            batch_size=16,
                            shuffle=True,
                            drop_last=True)

    pm.optimizer = optim.AdamW([pm.soft_prompt], lr=pm.lr)
    for epoch in range(pm.last_epoch, pm.last_epoch + epochs):
        print("soft预训练的epoch",epoch)
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
            print("loss1",loss1)


            # 拿到了score，接下来就是用score计算loss2
            # score5 = all_socre5[num]
            # loss2需要用score
            # loss2 = get_loss2(score5)

            # loss = calculate_combined_loss(loss1, loss2, alpha=0.7, beta=0.3)
            loss = loss1
            # 反向传播和参数更新
            loss.backward()
            pm.optimizer.step()






            # # loss回传完了，放回到cpu里面负责记录和画图
            loss.cpu().detach().numpy()

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
        weight_dir = REPO_ROOT / 'outputs' / 'weight'
        weight_dir.mkdir(parents=True, exist_ok=True)
        soft_path = weight_dir / f"pre_soft_prompt_word_epoch{epoch}{obj}.pth"
        # print(pm.soft_prompt)
        torch.save((pm.soft_prompt, epoch, pm.losses, pm.epoch_mean_losses), soft_path)
        print(f'Soft prompt saved at {soft_path}')

        torch.cuda.empty_cache()




def get_loss1(rep_vec1, rep_vec5):

    # cosine_similarity = F.cosine_similarity(rep_vec1, rep_vec5, dim=-1)
    # # print("cosine_similarity", cosine_similarity)
    #
    # # 过滤掉NaN值
    # cosine_similarity = cosine_similarity[~torch.isnan(cosine_similarity)]
    # cosine_similarity = torch.clamp(cosine_similarity, min=-1.0, max=1.0).mean()
    # # print("cosine_similarity", cosine_similarity)
    #
    # # 计算均方误差损失
    # mse_loss = F.mse_loss(rep_vec1, rep_vec5, reduction='mean')

    cosine_sim = F.cosine_similarity(rep_vec1, rep_vec5, dim=-1)
    # 计算损失：相似度越接近1，损失越小
    loss = 1 - cosine_sim.mean()  # 越接近1表示越相似，损失越小


    lambda_reg = 0.1  # 正则化权重
    delta = 0.1  # 最小允许差异
    euclidean_distance = torch.norm(rep_vec1 - rep_vec5, p=2, dim=-1)
    regularization_term = torch.clamp(delta - euclidean_distance, min=0)  # 小于 delta 时施加惩罚
    regularization_loss = lambda_reg * regularization_term.mean()

    # 总损失
    loss1 = loss + regularization_loss


    return loss1





def get_loss2(score):
    #KL
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







    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    # 占用显存，假设需要占用 4GB
    # reserved_memory = 23 * 1024 * 1024 * 1024  # 4GB
    # x = torch.empty(int(reserved_memory // 4), dtype=torch.float32, device=device)
    #
    # print("Successfully reserved 4GB of GPU memory.")
    # input("Press Enter to release the memory...")






    # device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


    # # 给llm的实例化，再传进去mynet里面
    # from amodel import PM, train_pm

    # getattr(pm,"get")
    model = MyNet(config).to(device)

    llm_model = os.environ.get('SHARP_LLM_MODEL')
    if not llm_model:
        raise RuntimeError('Set SHARP_LLM_MODEL to a local Hugging Face model path or model ID.')
    pm = amodel.PM(device=device, model_name=llm_model)

    gogogo(config, model)
    all_time = timer()-all_time
    print("整个代码运行花费时间：", all_time)
    print("over")
