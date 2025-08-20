from __future__ import print_function
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import random

import methods
from model import GANLDAModel
import copy
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve, average_precision_score, accuracy_score, \
    precision_score, recall_score, f1_score, auc
# import wandb
import warnings
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
    #A = np.load('data/ours/' + dataset + '/A_' + str(fold) + '.npy')
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
    ###打乱顺序,negative_ij前面2697个和正样本组成训练集，后面的则是测试集
    np.random.shuffle(negative_ij)



    target=torch.Tensor([1] * len(positive_ij) + [0] * len(negative_ij))


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
    target = target.to(gpu)
    # H = H.to(gpu)
    # train_target = train_target.to(gpu)
    # test_target = test_target.to(gpu)

    ################################################
    for epoch in range(1, 1000):
        #
        # out, loss = methods.train(train_target, ganlda_model, optimizer, lncx, disx,mix,
        #                               A)  # label, ganlda_model, optimizer, lncx, disx, adj
        out, loss = methods.train_all(positive_ij, negative_ij, target, ganlda_model, optimizer, H,
                                  A, lnc_dis_fea, lnc_dis_view, lnc_feature, lnc_feature_view, dis_feature,
                                  dis_feature_view, dis_mi_fea, dis_mi_view, config, gpu)
        #print('the ' + str(epoch) + ' times train_loss is ' + str(loss))
        print(f'the {epoch} times train_loss is {loss:.4f}')
        # print('the ' + str(epoch) + ' times test_loss is ' + str(loss))
        scheduler.step()

        ganlda_model.eval()
        pred = ganlda_model(positive_ij, negative_ij, H, A,lnc_dis_fea,lnc_dis_view,lnc_feature,lnc_feature_view,dis_feature,dis_feature_view,dis_mi_fea,dis_mi_view,config,gpu)
        sore=pred[2697*2:]

        ids = negative_ij[2697:]
        preds = sore.cpu().data.numpy()
        i_coords = ids[:, 0]
        j_coords = ids[:, 1] - 240
        results = np.column_stack((i_coords, j_coords, preds))

        # Converting the results into a DataFrame
        df = pd.DataFrame(results, columns=['i', 'j', 'prediction_score'])
        folder_name = 'casestudy'

        # Check if the folder exists, and if not, create it
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # Save the DataFrame to an Excel file in the specified folder
        excel_path = os.path.join(folder_name, 'predictions.xlsx')
        df.to_excel(excel_path, index=False)

        print(f"File saved at {excel_path}")


if __name__ == "__main__":
    main(9854)
    # # 从0到10000中随机选择10个不同的种子
    # random_seeds = random.sample(range(10000), 100)
    # for seed in random_seeds:
    #     main(seed)
