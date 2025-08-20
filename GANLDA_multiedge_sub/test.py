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
import wandb
import warnings
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='GANLDA ten-fold')
parser.add_argument('--gpu',type=int,default=0)

args = parser.parse_args()

torch.manual_seed(1)  # 初始化随机种子
gpu = torch.device(f"cuda:{args.gpu}")
#图transformer的输入和输出都是hid
#gan_in_channels是传入维度，会再次经过一个嵌入成维度hid
config = {
    'hid':      256,
    'gan_in_channels':  1147,
    'gan_out_channels':  8,
    'n_heads': 8,
    'lr':  0.001,
    'dataset': 'dataset1',
    'fold': 3,
    'weight_decay':0.000005081,
    'att_drop_rate':0.1,
    'n_epochs': 1000,
    'hid1':128,
    'hid2':521,

    # 'mlp_layer':[512, 256, 16, 1],
    # 'batch_size': 32,
}

# # if args.cuda:
# #     device = torch.device('cuda')
# # else:
# #     device = torch.device('cpu')
# #lr和weight_decay是Adam优化器的参数
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


# with h5py.File('lncRNA_disease_Associations.h5', 'r') as hf:
#     lncrna_disease_matrix = hf['rating'][:]  ## 240 * 412
#     lncrna_disease_matrix_val = lncrna_disease_matrix.copy()
# index_tuple = (np.where(lncrna_disease_matrix == 1))  ##获得值为1 的坐标(行，列)
# # one_list = list(zip(index_tuple[0], index_tuple[1]))
#
# # random.shuffle(one_list)  # 序列的所有元素随机排序
# # ceil()向上取整,将这个化为10份，一份多少==split,为后面10倍交叉验证做准备
# # split = math.ceil(len(one_list) / 10)  # 整个数据集划分10份，一份270个
# all_tpr = []
# all_fpr = []
# all_recall = []
# all_precision = []
# all_accuracy = []
# # load feature data
# with h5py.File('lncRNA_Features.h5', 'r') as hf:
#     lncx = hf['infor'][:]  # (240, 6066)
#     pca = PCA(n_components=gan_in_channels)  # gan_in_channels==128
#     lncx = pca.fit_transform(lncx)
#     lncx = torch.Tensor(lncx)  # (240, 128)
# with h5py.File('disease_Features.h5', 'r') as hf:
#     disx = hf['infor'][:]  # (412, 10621)
#     pca = PCA(n_components=gan_in_channels)
#     disx = pca.fit_transform(disx)
#     disx = torch.Tensor(disx)  # (412, 128)

# 10-fold start
# for i in range(0, len(one_list), split):
###这些positive_ij和negative_ij对应是原始矩阵A的o或者1的id
def main():
    # run=wandb.init()
    # 复杂图的临界矩阵

    A = np.load('data/ours/' + dataset + '/A_' + str(fold) + '.npy')
    np.fill_diagonal(A, 1)  ##将对角线设为1 ，因为相似性为1
    # lncrna disease mirna 相似性特征(这个相似性特征是最原始矩阵的，没有根据关联重新计算的)
    # lncRNA_FunSim=pd.read_csv('data/ours/' + dataset +'/lncrnaFunSim.csv',header=None)#240 240
    # disease_SemSim=pd.read_csv('data/ours/' + dataset +'/diseaseSemSim.csv',header=None)#412 412
    # miRNA_FunSim=disease_SemSim=pd.read_csv('data/ours/' + dataset +'/diseaseSemSim.csv',header=None)
    # miRNA_FunSim=pd.read_csv('data/ours/' + dataset +'/mirnaFunSim.csv',header=None)#495 495
    #
    # lncx=torch.Tensor(lncRNA_FunSim)
    # disx = torch.Tensor(disease_SemSim)
    # mix=torch.Tensor(miRNA_FunSim)
    # 使用矩阵的幂经过归一化作为特征，dimension为几次幂
    dimension = 3
    Ai = []
    Ai.append(A)
    for i in range(dimension - 1):
        tmp = np.dot(Ai[i], A)
        np.fill_diagonal(tmp, 0)
        tmp = tmp / np.max(tmp)
        Ai.append(copy.copy(tmp))
    Ai = np.array(Ai)  # 3个元素，每个都是1147 * 1147
    H_tmp = Ai[2]  # 1147 1147
    H = torch.Tensor(H_tmp)


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
    ########################################################
    ganlda_model.to(gpu)
    H = H.to(gpu)
    train_target = train_target.to(gpu)
    test_target = test_target.to(gpu)

    ################################################
    for epoch in range(1, 1000):
        #
        # out, loss = methods.train(train_target, ganlda_model, optimizer, lncx, disx,mix,
        #                               A)  # label, ganlda_model, optimizer, lncx, disx, adj
        out, loss = methods.train(positive_train_ij, negative_train_ij, train_target, ganlda_model, optimizer, H,
                                  A,lnc_dis_fea,lnc_dis_view,lnc_feature,lnc_feature_view,dis_feature,dis_feature_view,config,gpu)
        #print('the ' + str(epoch) + ' times train_loss is ' + str(loss))
        print(f'the {epoch} times train_loss is {loss:.4f}')
        # print('the ' + str(epoch) + ' times test_loss is ' + str(loss))
        scheduler.step()

        ganlda_model.eval()
        pred = ganlda_model(positive_test_ij, negative_test_ij, H, A,lnc_dis_fea,lnc_dis_view,lnc_feature,lnc_feature_view,dis_feature,dis_feature_view,config,gpu)

        test_loss = loss_function(pred, test_target)
        #print('the ' + str(epoch) + ' times test_loss is ' + str(test_loss))
        print(f'the {epoch} times test_loss is {test_loss:.4f}')
        preds = pred.cpu().data.numpy()

        labels = test_target.cpu().data.numpy()

        AUC = roc_auc_score(labels, preds)
        precision, recall, _ = precision_recall_curve(labels, preds)
        AUPR = auc(recall, precision)
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


if __name__ == "__main__":
    main()
