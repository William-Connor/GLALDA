import os
import pandas as pd
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import h5py
import time
config = {
    'hid':      64,
    'gan_in_channels':  1147,
    'gan_out_channels':  8,
    'n_heads': 4,
    'lr':  0.001,
    'dataset': 'dataset1',
    'fold': 3,
    'weight_decay':0.00000253,
    'att_drop_rate':0.4,
    'n_epochs': 1000,
    'hid1':64,
    'hid2':16,

    # 'mlp_layer':[512, 256, 16, 1],
    # 'batch_size': 32,
}


dataset='dataset1'
# hid = config['hid']
gan_in_channels = config['gan_in_channels']
gan_out_channels = config['gan_out_channels']
n_head = config['n_heads']
lr = config['lr']
weight_decay = config['weight_decay']
att_drop_rate = config['att_drop_rate']
# mlp_layers = config['mlp_layer']
fold=config['fold']
# 2697 * 2里面存储的id，（i，j），ndarray
positive_ij = np.load('data/ours/' + dataset + '/positive_ij.npy')
# 96183 * 2里面存储的id，（i，j），ndarray
negative_ij = np.load('data/ours/' + dataset + '/negative_ij.npy')
positive5foldsidx = np.load('data/ours/' + dataset + '/positive5foldsidx.npy', allow_pickle=True)
negative5foldsidx = np.load('data/ours/' +dataset + '/negative5foldsidx.npy', allow_pickle=True)

    ##测试集id
positive_test_ij = positive_ij[positive5foldsidx[fold]['test']]
positive_train_ij = positive_ij[positive5foldsidx[fold]['train']]
negative_test_ij = negative_ij[negative5foldsidx[fold]['test']]
negative_train_ij = negative_ij[negative5foldsidx[fold]['train']]
case_positive_test_ij = np.concatenate((positive_test_ij, positive_train_ij))#2697

case_negative_test_ij = np.concatenate((negative_test_ij, negative_train_ij))#96183

    # 94566个标签
test_target = torch.Tensor([1] * len(case_positive_test_ij ) + [0] * len(case_negative_test_ij))

    # 训练集id
positive_train_ij = positive_ij[positive5foldsidx[fold]['train']]
positive_test_ij = positive_ij[positive5foldsidx[fold]['test']]
case_positive_train_ij=np.concatenate((positive_train_ij, positive_test_ij))
case_negative_train_ij=negative_ij[negative5foldsidx[fold]['test']][:(len(positive_train_ij) + len(positive_test_ij))] # Attention!

    # 4314个标签
train_target =torch.Tensor([1] * (len(positive_train_ij) + len(positive_test_ij)) + [0] * len(case_negative_train_ij))


#########################

#上面是  训练集ij准备在最终预测结果去掉

##########


ds = 'dataset1'
lnc_di = pd.read_csv('data/ours/' + ds + '/lnc_di.csv', index_col=0)
diseases = lnc_di.columns
lncRNAs = lnc_di.index





new_results = pd.read_csv('dataset1.csv')#####去掉第一列的序号 0 1 2 3 等
df=new_results
ij=np.concatenate((case_positive_train_ij,case_negative_train_ij))
i = ij[:, 0].T
j = ij[:, 1].T
# 从 lncRNAs 和 diseases 中取数据
lncRNA_names = [lncRNAs[lncRNA] for lncRNA in i]
disease_names = [diseases[disease - len(lncRNAs)] for disease in j]
# 根据给定的 lncRNA 和 disease 组合，构建逻辑条件
conditions = []
print(df.columns)
for lncRNA_name, disease_name in zip(lncRNA_names, disease_names):
    # 删除符合条件的指定行，并替换原始df
    new_results_filterd=df.drop(df[(df['lncRNA']==lncRNA_name) & (df['disease']==disease_name)].index ,inplace = True)

# for disease, group in list(new_results[new_results['evidence'] == 1].groupby(['disease'])):
#     if len(group.values) > 10:
#         # 筛选出满足条件的数据
#         top_10 = group[(group['evidence'] == 1) & (group['label'] == 0) & (group['pred'] > 0.5)] \
#                      .sort_values(by='pred', ascending=False)[:10]
#         top_15 = group[(group['evidence'] == 1) & (group['label'] == 0) & (group['pred'] > 0.5)] \
#                      .sort_values(by='pred', ascending=False)[:15]
#         # top_20 = group[(group['evidence'] == 1) & (group['label'] == 0) & (group['pred'] > 0.5)] \
#         #              .sort_values(by='pred', ascending=False)[:20]
#         # 输出每个疾病的前 k 个满足条件的数量
#         evidence_1_10count = top_10.shape[0]
#         evidence_1_15count = top_15.shape[0]
#         print(f"For disease {disease}, {evidence_1_10count} cases satisfy,{evidence_1_15count} cases satisfy .")
#

for disease, group in new_results.groupby('disease'):
    # 筛选出满足条件的数据
    top_10 = group.head(10)
    # 统计前 k 行数据中满足条件的行数
    evidence_1_10count = len(top_10[(top_10['evidence'] == 1) & (top_10['label'] == 0) & (top_10['pred'] > 0.5)])
    top_15 = group.head(15)
    # 统计前 k 行数据中满足条件的行数
    evidence_1_15count = len(top_15[(top_15['evidence'] == 1) & (top_15['label'] == 0) & (top_15['pred'] > 0.5)])
    # top_10 = group[(group['evidence'] == 1) & (group['label'] == 0) & (group['pred'] > 0.5)] \
    #              .sort_values(by='pred', ascending=False)[:10]
    # top_15 = group[(group['evidence'] == 1) & (group['label'] == 0) & (group['pred'] > 0.5)] \
    #              .sort_values(by='pred', ascending=False)[:15]
    # top_20 = group[(group['evidence'] == 1) & (group['label'] == 0) & (group['pred'] > 0.5)] \
    #              .sort_values(by='pred', ascending=False)[:20]
    # 输出每个疾病的前 k 个满足条件的数量

    print(f"For disease {disease}, {evidence_1_10count} cases satisfy,{evidence_1_15count} cases satisfy .")






