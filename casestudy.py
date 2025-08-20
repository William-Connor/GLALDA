from __future__ import print_function
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import h5py
import time
import methods
from model import GANLDAModel
import copy
import pandas as pd
import itertools
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve, average_precision_score, accuracy_score, \
    precision_score, recall_score, f1_score, auc
# import wandb
import warnings
import matrix
import SNF
import os
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='GANLDA ten-fold')
parser.add_argument('--gpu',type=int,default=0)

args = parser.parse_args()

import pandas as pd


def analyze_results(preds, labels):
    ds = 'dataset1'
    lnc_di = pd.read_csv('data/ours/' + ds + '/lnc_di.csv', index_col=0)
    diseases = lnc_di.columns
    lncRNAs = lnc_di.index

    positive_ij = np.load('data/ours/' + ds + '/positive_ij.npy')
    negative_ij = np.load('data/ours/' + ds + '/negative_ij.npy')
    positive5foldsidx = np.load('data/ours/' + ds + '/positive5foldsidx.npy', allow_pickle=True)
    negative5foldsidx = np.load('data/ours/' + ds + '/negative5foldsidx.npy', allow_pickle=True)
    positive_test_ij = positive_ij[positive5foldsidx[0]['test']]
    positive_train_ij = positive_ij[positive5foldsidx[0]['train']]
    negative_test_ij = negative_ij[negative5foldsidx[0]['test']]
    negative_train_ij = negative_ij[negative5foldsidx[0]['train']]

    ij = np.concatenate((positive_test_ij, positive_train_ij, negative_test_ij, negative_train_ij))
    i = ij[:, 0].T
    j = ij[:, 1].T
    labels = labels.astype(int)
    prediction_results = pd.DataFrame({
        'lncRNA': np.array([lncRNAs[lncRNA] for lncRNA in i]),
        'disease': np.array([diseases[disease - len(lncRNAs)] for disease in j]),
        'pred': preds,
        'label': labels
    })

    evidence = pd.read_csv('data/ours/dataset2/union/di_lnc_union.csv', index_col='Unnamed: 0')
    evidence_diseases = evidence.index
    evidence_lncRNAs = evidence.columns

    new_results = pd.DataFrame(columns=['lncRNA', 'disease', 'pred', 'label', 'evidence'])
    for idx, row in prediction_results.iterrows():
        lncRNA = row['lncRNA']
        disease = row['disease']
        evd = 0
        if (lncRNA in evidence_lncRNAs) and (disease in evidence_diseases) and (evidence.loc[disease, lncRNA] == 1):
            evd = 1
        # new_results = new_results.append({
        #     'lncRNA': lncRNA,
        #     'disease': disease,
        #     'pred': row['pred'],
        #     'label': row['label'],
        #     'evidence': evd
        # }, ignore_index=True)
        temp_df = pd.DataFrame([{
            'lncRNA': lncRNA,
            'disease': disease,
            'pred': row['pred'],
            'label': row['label'],
            'evidence': evd
        }])
        new_results = pd.concat([new_results, temp_df], ignore_index=True)

    #new_results.sort_values(by='pred', ascending=False).to_csv('files/case study/dataset1.csv')
    # for di, case in list(new_results[new_results['evidence'] == 1].groupby(['disease'])):
    #     if len(case.values) > 10:
    #         print(di)
    #         new_results[(new_results['disease'] == di) & (new_results['label'] == 0)].sort_values(by='pred',
    #                                                                                               ascending=False).to_csv(
    #             'files/case study/' + di + '.csv')
        # 统计符合条件的案例数量
    num_cases = len(new_results[(
                (new_results['evidence'] == 1) & (new_results['label'] == 0) & (new_results['pred'] > 0.5))].values)


    return new_results, num_cases


# 使用示例
# new_results, num_cases = analyze_results(preds, labels)


def set_all_seeds(seed_value):
    # Set the seed for generating random numbers
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
# torch.manual_seed(1)  # 初始化随机种子
# set_all_seeds(42)###3407
gpu = torch.device(f"cuda:{args.gpu}")



#图transformer的输入和输出都是hid
#gan_in_channels是传入维度，会再次经过一个嵌入成维度hid
def compute_H_from_A(A, dimension):
    Ai = []
    Ai.append(A)
    for i in range(dimension - 1):
        tmp = np.dot(Ai[i], A)
        np.fill_diagonal(tmp, 0)
        tmp = tmp / np.max(tmp)
        Ai.append(copy.copy(tmp))
    Ai = np.array(Ai)
    H_tmp = Ai[-1]  # 获取最后一个元素
    H = torch.Tensor(H_tmp)
    H=H.to(gpu)
    return H
# def compute_H_from_All(A, dimension):
#     Ai = []
#     Ai.append(A)
#     for i in range(dimension - 1):
#         tmp = np.dot(Ai[i], A)
#         np.fill_diagonal(tmp, 0)
#         tmp = tmp / np.max(tmp)
#         Ai.append(copy.copy(tmp))
#     Ai = np.array(Ai)
#     H_tmp= np.concatenate(Ai, axis=1)
#     # H_tmp = Ai[-1]  # 获取最后一个元素
#     H = torch.Tensor(H_tmp)
#     H=H.to(gpu)
#     return H
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
# loss function
def loss_function(pre_adj, adj):
    #adj = torch.Tensor(adj)
    loss_fn = torch.nn.BCELoss()
    return loss_fn(pre_adj, adj)


###这些positive_ij和negative_ij对应是原始矩阵A的o或者1的id
def main(seed=None):
    if seed is not None:
        set_all_seeds(seed)
    # run=wandb.init()
    # 复杂图的临界矩阵
    A = np.load('data/ours/' + dataset + '/A.npy')

    # A = np.load('data/ours/' + dataset + '/A_' + str(fold) + '.npy')
    np.fill_diagonal(A, 1)  ##将对角线设为1 ，因为相似性为1
###### ！！！！！！！！！！！！！！！！！   ###获取 SNF
    # A=matrix.A_SNF(A)
    A = SNF.A_SNF(A)
    H=compute_H_from_A(A,3)



####lnc-dis
    lnc_dis_view=methods.lnc_dis_view(A)
    lnc_dis_fea=H[0:651 + 1]
    lnc_dis_fea=lnc_dis_fea.to(gpu)
####lnc特征视图
    lnc_feature_view=methods.lnc_feature_view(A)
    lnc_feature=H[0:239 + 1]
    lnc_feature=lnc_feature.to(gpu)
####dis特征视图
    dis_feature_view=methods.dis_feature_view(A)
    dis_feature=H[240:651 + 1]
    dis_feature=dis_feature.to(gpu)
#####lnc-mirna
    ####disease-mirna特征视图
    dis_mi_view = methods.dis_mi_view(A)
    dis_mi_fea = H[240:1146 + 1]
    dis_mi_fea = dis_mi_fea.to(gpu)

####miRNAz特征视图

    ###训练集和测试集划分
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

    # ganlda model
    ganlda_model = GANLDAModel(gan_in_channels,config['hid'], gan_out_channels, n_head, att_drop_rate,config, gpu)
    # wandb.watch(ganlda_model, log='all', log_graph=True)

    optimizer = torch.optim.Adam(ganlda_model.parameters(), lr=lr,
                                 weight_decay=weight_decay)  # 更新参数

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 800], gamma=0.8)

    max_AUC = 0
    best_result = None
    best_preds= None
    best_labels= None
    max_case=0
    max_result=None
    ########################################################
    ganlda_model.to(gpu)
    # H = H.to(gpu)
    train_target = train_target.to(gpu)
    test_target = test_target.to(gpu)

    ################################################
    for epoch in range(1, 6):
        #
        # out, loss = methods.train(train_target, ganlda_model, optimizer, lncx, disx,mix,
        #                               A)  # label, ganlda_model, optimizer, lncx, disx, adj
        out, loss = methods.train(case_positive_train_ij, case_negative_train_ij, train_target, ganlda_model, optimizer, H,
                                  A, lnc_dis_fea, lnc_dis_view, lnc_feature, lnc_feature_view, dis_feature,
                                  dis_feature_view, dis_mi_fea, dis_mi_view, config, gpu)
        #print('the ' + str(epoch) + ' times train_loss is ' + str(loss))
        print(f'the {epoch} times train_loss is {loss:.4f}')
        # print('the ' + str(epoch) + ' times test_loss is ' + str(loss))
        scheduler.step()

        ganlda_model.eval()
        pred = ganlda_model(case_positive_test_ij, case_negative_test_ij, H, A,lnc_dis_fea,lnc_dis_view,lnc_feature,lnc_feature_view,dis_feature,dis_feature_view,dis_mi_fea,dis_mi_view,config,gpu)
        test_loss = loss_function(pred, test_target)
        #print('the ' + str(epoch) + ' times test_loss is ' + str(test_loss))
        print(f'the {epoch} times test_loss is {test_loss:.4f}')
        preds = pred.cpu().data.numpy()

        labels = test_target.cpu().data.numpy()
        AUC = roc_auc_score(labels, preds)
        print("AUC =", AUC)
        precision, recall, _ = precision_recall_curve(labels, preds)
        AUPR = auc(recall, precision)
        print('AUPR =', AUPR)
        results,case_nums=analyze_results(preds,labels)

        file_name = f'files/case study/dataset1_case_nums_{case_nums}.csv'
        results.sort_values(by='pred', ascending=False).to_csv(file_name, index=False)
        print(f"文件 {file_name} 保存成功!")
    #     print(f"epoch：{epoch} case_nums:{case_nums}")
    #     if case_nums > max_case:
    #         max_case=case_nums
    #         max_result=results
    #         c
    # max_result.sort_values(by='pred', ascending=False).to_csv('files/case study/dataset1.csv')




if __name__ == "__main__":
    # 从0到10000中随机选择10个不同的种子
    main(9854)
