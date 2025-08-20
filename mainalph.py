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
def set_all_seeds(seed_value):
    # Set the seed for generating random numbers
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
torch.manual_seed(9854)  # 初始化随机种子
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
def main(alph):
    # if seed is not None:
    #     set_all_seeds(seed)
    # run=wandb.init()
    # 复杂图的临界矩阵

    A = np.load('data/ours/' + dataset + '/A_' + str(fold) + '.npy')
    np.fill_diagonal(A, 1)  ##将对角线设为1 ，因为相似性为1
###### ！！！！！！！！！！！！！！！！！   ###获取 SNF
    # A=matrix.A_SNF(A)
    A = SNF.A_SNF(A,alpha=alph)
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
    # 94026  2
    negative_test_ij_tmp = negative_ij[negative5foldsidx[fold]['test']]
    # 生成随机索引
    random_indices = np.random.choice(negative_test_ij_tmp.shape[0], size=len(positive_test_ij), replace=False)

    # 使用随机索引采样
    negative_test_ij = negative_test_ij_tmp[random_indices]
    # 94566个标签
    test_target = torch.Tensor([1] * len(positive_test_ij) + [0] * len(negative_test_ij))  # 标签

    # 训练集id
    # 2157*2
    positive_train_ij = positive_ij[positive5foldsidx[fold]['train']]
    negative_train_ij = negative_ij[negative5foldsidx[fold]['train']]
    # 4314个标签
    train_target = torch.Tensor([1] * len(positive_train_ij) + [0] * len(negative_train_ij))

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
    ########################################################
    ganlda_model.to(gpu)
    # H = H.to(gpu)
    train_target = train_target.to(gpu)
    test_target = test_target.to(gpu)

    ################################################
    for epoch in range(1, 500):
        #
        # out, loss = methods.train(train_target, ganlda_model, optimizer, lncx, disx,mix,
        #                               A)  # label, ganlda_model, optimizer, lncx, disx, adj
        out, loss = methods.train(positive_train_ij, negative_train_ij, train_target, ganlda_model, optimizer, H,
                                  A, lnc_dis_fea, lnc_dis_view, lnc_feature, lnc_feature_view, dis_feature,
                                  dis_feature_view, dis_mi_fea, dis_mi_view, config, gpu)
        #print('the ' + str(epoch) + ' times train_loss is ' + str(loss))
        print(f'the {epoch} times train_loss is {loss:.4f}')
        # print('the ' + str(epoch) + ' times test_loss is ' + str(loss))
        scheduler.step()

        ganlda_model.eval()
        pred = ganlda_model(positive_test_ij, negative_test_ij, H, A,lnc_dis_fea,lnc_dis_view,lnc_feature,lnc_feature_view,dis_feature,dis_feature_view,dis_mi_fea,dis_mi_view,config,gpu)
        test_loss = loss_function(pred, test_target)
        #print('the ' + str(epoch) + ' times test_loss is ' + str(test_loss))
        print(f'the {epoch} times test_loss is {test_loss:.4f}')
        preds = pred.cpu().data.numpy()

        labels = test_target.cpu().data.numpy()

        AUC = roc_auc_score(labels, preds)
        precision, recall, _ = precision_recall_curve(labels, preds)
        AUPR = auc(recall, precision)
        ##@@@@@@@@@@@@@
        if AUC > max_AUC:
            best_preds=preds
            best_labels=labels
        #@@@@@@@@@@
        preds = np.array([1 if p > 0.5 else 0 for p in preds])
        ACC = accuracy_score(labels, preds)
        P = precision_score(labels, preds)
        R = recall_score(labels, preds)
        F1 = f1_score(labels, preds)

        with open('output.txt', 'a+') as f:
            result = (AUC, AUPR, ACC, P, R, F1)
            f.write(f'{AUC} {AUPR} {ACC} {P} {R} {F1}\n')
            if AUC > max_AUC:
                max_AUC = AUC
                best_result = result

        with open('best_output.txt', 'w') as f:
            f.write(
                f'{best_result[0]} {best_result[1]} {best_result[2]} {best_result[3]} {best_result[4]} {best_result[5]}\n')


        print(AUC, AUPR, ACC, P, R, F1)
    # 输出AUC最大的结果
    with open('best_output.txt', 'a+') as f:
        f.write(
            f'{best_result[0]} {best_result[1]} {best_result[2]} {best_result[3]} {best_result[4]} {best_result[5]}\n')
    #methods.save_to_excel(best_labels, best_preds,max_AUC,k)






if __name__ == "__main__":
    best_auc = 0
    best_results = None

    alph_range = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    with open('alph.txt', 'w') as t_file:
        t_file.write('alph AUC AUPR ACC P R F1\n')  # 写入表头
        print("表头保存成功！")
    for alph in alph_range:
        # 在这里添加修改k的代码
        #
        # 更新配置参数中的k值
        main(alph)  # 调用主函数运行任务

        # 读取并记录AUC
        with open('best_output.txt', 'r') as f:
            lines = f.readlines()
            last_line = lines[-1].strip().split()
            current_auc = float(last_line[0])

        # 记录最佳结果
        if current_auc > best_auc:
            best_auc = current_auc
            best_results = last_line
        # 打开文件k.txt，保存当前k值
        with open('alph.txt', 'a') as k_file:
            k_file.write(f'{alph} ')  # 先保存k值
            print("alph 保存成功！")
            k_file.write(" ".join(last_line) + '\n')  # 再保存指标
            print("alph对应指标保存成功！")


    print(f'Best AUC: {best_auc}')
    print(f'Best Results (AUC, AUPR, ACC, P, R, F1): {best_results}')

    # main()