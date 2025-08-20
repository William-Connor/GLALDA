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

parser = argparse.ArgumentParser(description='GANLDA init')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--gan_in', type=int, default=128,
                        help='PCA embedding size.')
parser.add_argument('--gan_out', type=int, default=8,
                        help='GAN embedding size.')
parser.add_argument('--n_head', type=int, default=1,
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
###############################################################################
#parser.add_argument('--in_degree',type=boolallow_zero_,default=True)
###############################################################################

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
lr = args.lr#学习率
weight_decay = args.weight_decay
attn_drop = args.att_drop_rate
mlp_layers = args.mlp_layers  #mlp_layers[128,64,64,1]



# load lncRNA-disease associations
with h5py.File('lncRNA_disease_Associations.h5', 'r') as hf:
    lncrna_disease_matrix = hf['rating'][:]## 240 * 412

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

# ganlda model
ganlda_model = GANLDAModel(gan_in_channels,hid, gan_out_channels, n_head, attn_drop, mlp_layers)
optimizer = torch.optim.Adam(ganlda_model.parameters(), lr=lr,
                             weight_decay=weight_decay)

rel_matrix = lncrna_disease_matrix## 240 * 412
row_n = rel_matrix.shape[0]## 240
col_n = rel_matrix.shape[1]## 412
temp_l = np.zeros((row_n, row_n))## 240 * 240
temp_d = np.zeros((col_n, col_n))## 412 * 412
##np.hstack((temp_l, rel_matrix)：水平拼接成（240*240， 240*412）=240*（412+240）=240*652
##np.hstack((rel_matrix.T, temp_d))：水平拼接成（412*240 ，412*412）=240*（412+240）=412*652
##adj=652 * 652
adj = np.vstack((np.hstack((temp_l, rel_matrix)), np.hstack((rel_matrix.T, temp_d))))

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 800], gamma=0.8)
bestloss = np.inf
for epoch in range(1, 1000):
    # train,lncx=(240, 128),disx=(412, 128)
    out, loss = methods.train(rel_matrix, ganlda_model, optimizer, lncx, disx,
                              adj)  # label, ganlda_model, optimizer, lncx, disx, adj
    print('the ' + str(epoch) + ' times loss is '+ str(loss))
    if bestloss > loss:
        bestloss = loss
    scheduler.step()
print(bestloss)
output = out.cpu().data.numpy()

# the score matrix
score_matrix = output
