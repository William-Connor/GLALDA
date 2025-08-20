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
import itertools
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve, average_precision_score, accuracy_score, precision_score, recall_score, f1_score, auc

parser = argparse.ArgumentParser(description='GANLDA ten-fold')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--gan_in', type=int, default=128,
                        help='PCA embedding size.')
parser.add_argument('--gan_out', type=int, default=8,
                        help='GAN embedding size.')
parser.add_argument('--n_head', type=int, default=8,
                        help='GAN head number.')
parser.add_argument('--lr', type=float, default=0.005,
                    help='Learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.00005,
                    help='Weight decay.')
parser.add_argument('--att_drop_rate', type=float, default=0.4,
                    help='GAT Dropout rate.')
parser.add_argument('--mlp_layers', nargs='?', type=list, default=[128,64,64,1],
                        help="Size of each mlp layer.")
parser.add_argument('--hid',type=int, default=64,
                        help='line in_hid')


args = parser.parse_args()
# args.cuda = not args.no_cuda and torch.cuda.is_available()
# torch.manual_seed(args.seed)#初始化随机种子
# device = torch.device('cpu')
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)  # 初始化随机种子

if args.cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')



hid=args.hid
gan_in_channels = args.gan_in
gan_out_channels = args.gan_out
n_head = args.n_head
lr = args.lr
weight_decay = args.weight_decay
attn_drop = args.att_drop_rate
mlp_layers = args.mlp_layers




with h5py.File('lncRNA_disease_Associations.h5', 'r') as hf:
    lncrna_disease_matrix = hf['rating'][:]## 240 * 412
    lncrna_disease_matrix_val = lncrna_disease_matrix.copy()
index_tuple = (np.where(lncrna_disease_matrix == 1))##获得值为1 的坐标(行，列)
one_list = list(zip(index_tuple[0], index_tuple[1]))

random.shuffle(one_list)#序列的所有元素随机排序
#ceil()向上取整,将这个化为10份，一份多少==split,为后面10倍交叉验证做准备
split = math.ceil(len(one_list) / 10)#整个数据集划分10份，一份270个

# load feature data
with h5py.File('lncRNA_Features.h5', 'r') as hf:
    lncx = hf['infor'][:]#(240, 6066)
    pca = PCA(n_components=gan_in_channels)#gan_in_channels==128
    lncx = pca.fit_transform(lncx)
    lncx = torch.Tensor(lncx)#(240, 128)
with h5py.File('disease_Features.h5', 'r') as hf:
    disx = hf['infor'][:]#(412, 10621)
    pca = PCA(n_components=gan_in_channels)
    disx = pca.fit_transform(disx)
    disx = torch.Tensor(disx)#(412, 128)

# 10-fold start
for i in range(0, len(one_list), split):
    
    # ganlda model
    ganlda_model = GANLDAModel(gan_in_channels,hid, gan_out_channels, n_head, attn_drop, mlp_layers)
    optimizer = torch.optim.Adam(ganlda_model.parameters(), lr=lr,
                                 weight_decay=weight_decay)#更新参数

    train_index = one_list[i:i + split]#one_list里面是矩阵中为1的行列坐标，270个
    new_lncrna_disease_matrix = lncrna_disease_matrix.copy()

    for index in train_index:
        new_lncrna_disease_matrix[index[0], index[1]] = 0  # train set，设为0
    roc_lncrna_disease_matrix = new_lncrna_disease_matrix + lncrna_disease_matrix

    rel_matrix = new_lncrna_disease_matrix
    row_n = rel_matrix.shape[0]
    col_n = rel_matrix.shape[1]
    temp_l = np.zeros((row_n, row_n))#240*240
    temp_d = np.zeros((col_n, col_n))#412*412
    ##np.hstack((temp_l, rel_matrix)：水平拼接成（240*240， 240*412）=240*（412+240）=240*652
    ##np.hstack((rel_matrix.T, temp_d))：垂直拼接成（412*240 ，412*412）=240*（412+240）=412*652
    ##adj=652 * 652
    adj = np.vstack((np.hstack((temp_l, rel_matrix)), np.hstack((rel_matrix.T, temp_d))))
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 800], gamma=0.8)

    for epoch in range(1, 2):
        out, loss = methods.train(rel_matrix, ganlda_model,optimizer, lncx, disx, adj) # label, ganlda_model, optimizer, lncx, disx, adj
        print('the ' + str(epoch) + ' times train_loss is ' + str(loss))

        #print('the ' + str(epoch) + ' times test_loss is ' + str(loss))
        scheduler.step()
    output=out.detach().numpy()
    preds=[]
    labels=[]
    for index in train_index:
        preds.append(output[index[0], index[1]])
        labels.append(lncrna_disease_matrix[index[0], index[1]])

    min=min(preds)
    preds = np.array([1 if p > 0.5 else 0 for p in preds])
    AUC = roc_auc_score(labels, preds)
    precision, recall, _ = precision_recall_curve(labels, preds)
    AUPR = auc(recall, precision)
    ACC = accuracy_score(labels, preds)
    P = precision_score(labels, preds)
    R = recall_score(labels, preds)
    F1 = f1_score(labels, preds)
    print(AUC, AUPR, ACC, P, R, F1)

    # output = out.cpu().detach().numpy()
    #
    # preds = out.flatten().detach().numpy()
    #
    # labels = rel_matrix.flatten()
    # AUC = roc_auc_score(labels, preds)
    # # the score matrix
    # score_matrix = output
    # AUC = roc_auc_score(labels, preds)
    # precision, recall, _ = precision_recall_curve(labels, preds)
    # AUPR = auc(recall, precision)
    # preds = np.array([1 if p > 0.5 else 0 for p in preds])
    # ACC = accuracy_score(labels, preds)
    # P = precision_score(labels, preds)
    # R = recall_score(labels, preds)
    # F1 = f1_score(labels, preds)
    #
    # print(AUC, AUPR, ACC, P, R, F1)
    

  
