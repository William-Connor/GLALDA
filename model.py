import torch
import torch.nn.functional as F
from torch.nn import Parameter
import math
import torch.nn as nn
import numpy as np
import dgl
from dgl.nn.pytorch import edge_softmax, GATConv
from graph_transformer_layer import GraphTransformerLayer
from graph_transformer_edge_layer import GraphTransformerLayer as GraphTransEdgeformerLayer
from methods import laplacian_positional_encoding
from scipy.sparse import coo_matrix
def uniform(tensor):
    if tensor is not None:
        nn.init.kaiming_uniform_(tensor)


class GANLayer(torch.nn.Module):
    def __init__(self, in_channels,hid, out_channels, n_head, gpu,attn_drop=0.4,
            ):
        super(GANLayer, self).__init__()
        self.num_layers = 1
        self.num_layers1 = 1  # 边的
        #hid是transformer输入之前要进行的线性变换
        # self.gat_layers = nn.ModuleList()
        # self.gat_layers1=nn.ModuleList()
        # just only one layer in our paper.
        self.in_channels=in_channels#1147
        self.hid=hid
        self.out_channels=out_channels
        self.gpu = gpu
        self.embedding_e = nn.Linear(1, hid)
        # self.linear = nn.Linear(1147, in_channels)
        self.linear = nn.Linear(1147, in_channels)
        self.embedding_lap_pos_enc = nn.Linear(hid, hid)
        self.embedding_h=nn.Linear(in_channels, hid) # node feat is an integer

        #GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout, self.layer_norm=Fa, self.batch_norm,  self.residual)
        #layer_norm=False, batch_norm=True, residual=True, use_bias=False

        # self.gat_layers.append(GraphTransformerLayer(hid, hid, n_head,
        #                                       attn_drop,gpu))
        # self.gat_layers1.append(GraphTransEdgeformerLayer(hid, hid, n_head,
        #                                              attn_drop))
        self.gat_layers = nn.ModuleList(
            [GraphTransformerLayer(hid, hid, n_head, attn_drop, gpu) for _ in range(self.num_layers)])
        self.gat_layers1 = nn.ModuleList(
            [GraphTransEdgeformerLayer(hid, hid, n_head, attn_drop) for _ in range(self.num_layers1)])

    def forward(self, H,adj,gpu):#lncx=(240, 128),disx=(412, 128)
        #adj: 图的邻接矩阵 维度[N, N] 非零即一，数据结构基本知识
        #表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留，返回下标
        # index_tuple = np.argwhere(adj == 1)
        #shape[0]  矩阵的行数
        lnc_size = 240
        dis_size = 412
        mi_size=495
        # z = torch.cat((lncrna_x, disease_x,mirna_x))#默认参数 0  ，竖着扩展拼接


        # 将numpy数组转为scipy的稀疏矩阵
        A_sparse = coo_matrix(adj)
        # 用dgl.from_scipy函数将稀疏矩阵转为dgl图
        g = dgl.from_scipy(A_sparse)
        g=g.to(gpu)
        # 获取非零元素的数量
        
        p = A_sparse.nnz
        # 将邻接矩阵的非零元素作为边特征
        e = A_sparse.data.reshape(-1, 1)
        e = torch.FloatTensor(e).to(gpu)
        #e = torch.ones(e.size(0), 1).to(gpu)
        e = self.embedding_e(e)
        
        
        
        
        H=self.linear(H)
        h = self.embedding_h(H)#1025

        #添加位置编码前，特征是否需要dropout
        #h = self.in_feat_dropout(h)

        #拉普拉斯位置编码
        h_lap_pos_enc=laplacian_positional_encoding(g,self.hid)#1025
        h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float().to(self.gpu))
        h = h + h_lap_pos_enc



        # z = self.gat_layers[0](g, h).flatten(1)##1147 1025

        #下面是特征归一化，视情况而添加操作
        # g.ndata['h'] = z
        # hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
        #h=self.gat_layers[0](g,h)
        z, e0 = self.gat_layers1[0](g, h, e)##1147 1025
        z=self.gat_layers[0](g,z,gpu)
        z=z.flatten(1)
        return z


##多层感知机
class MLPLayer(torch.nn.Module):

    def __init__(self, config):#mlp_layers[128,64,64,1]--list，传入layers参数
        super(MLPLayer, self).__init__()

        # self.mlp_layer1=torch.nn.Linear(config['hid'].hid*2, args.hid1)
        # self.mlp_layer2 = torch.nn.Linear(args.hid1, args.hid2)
        # self.mlp_layer3 = torch.nn.Linear(args.hid2,1)
        self.mlp_layer1 = torch.nn.Linear(config['hid'] * 2, config['hid1'])
        self.mlp_layer2 = torch.nn.Linear(config['hid1'], config['hid2'])
        self.mlp_layer3 = torch.nn.Linear(config['hid2'], 1)


    def forward(self, feature):


        z=feature
        # if self.num_layers < 3:
        #     z = self.mlp_layers[0](z)
        # else:
        #     #num_layer:0时z（98880，128）经过z = self.mlp_layers[num_layer](z)，变为（98880，64）
        #     # num_layer:1时z（98880，64）经过z = self.mlp_layers[num_layer](z)，变为（98880，64）
        #     # num_layer:2时z（98880，64）经过z = self.mlp_layers[num_layer](z)，变为（98880，1）
        #     for num_layer in range(0, self.num_layers - 2):
        #         z = self.mlp_layers[num_layer](z)
        #         F.elu(z)
        #     # num_layer:2时z（98880，64）经过z = self.mlp_layers[num_layer](z)，变为（98880，1）
        #     z = self.mlp_layers[self.num_layers - 2](z)
        z=self.mlp_layer1(z)
        z=self.mlp_layer2(z)
        z=self.mlp_layer3(z)

        output = z.flatten()
        return F.sigmoid(output)

def get_feature_matrix(H, ij_list):
    # 初始化一个空的list来存储所有的特征向量
    feature_list = []

    for i, j in ij_list:
        # 提取第i行和第j行的数据
        row_i = H[i, :]
        row_j = H[j, :]

        # 将这两行数据横向拼接成一个新的特征向量
        feature = torch.cat((row_i, row_j), dim=0)

        # 将这个特征向量添加到feature_list中
        feature_list.append(feature)

    # 将所有的特征向量竖向拼接成一个新的特征矩阵
    feature_matrix = torch.stack(feature_list, dim=0)

    return feature_matrix



class CrossAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(CrossAttentionModel, self).__init__()

        self.attention = nn.MultiheadAttention(input_dim, num_heads=num_heads)
        self.Q = nn.Linear(input_dim, hidden_dim)
        self.K = nn.Linear(input_dim, hidden_dim)
        self.V = nn.Linear(input_dim, hidden_dim)

    def forward(self, x1, x2):
        # 输入x1和x2的形状应为 (seq_len, batch_size, input_dim)

        # 使用线性层映射输入特征
        q = self.Q(x1)
        k = self.K(x2)
        v = self.V(x2)  # 这里使用x2两次作为k和v

        # 计算交叉注意力
        output, _ = self.attention(q, k, v)

        return output







def merge_and_average(z1, z3):
    # 提取lncrna特征和dis特征
    lncrna_features = z1[:240, :]
    dis_features_z1 = z1[240:, :]

    # 提取dis特征和mi特征
    dis_features_z3 = z3[:412, :]
    mi_features = z3[412:, :]

    # 将dis特征按行相加求平均（在GPU上进行）
    combined_dis_features = (dis_features_z1 + dis_features_z3) / 2.0

    # 将所有特征组合成一个新的矩阵
    combined_matrix = torch.vstack((lncrna_features, combined_dis_features, mi_features))

    return combined_matrix




class GANLDAModel(torch.nn.Module):
    def __init__(self, gan_in_channels, hid,gan_out_channels, n_head, attn_drop,config,gpu):
        super(GANLDAModel, self).__init__()
        self.complex_ganlayer = GANLayer(gan_in_channels,hid, gan_out_channels, n_head,gpu, attn_drop)
        #self.lnc_feature_layer = GANLayer(gan_in_channels, hid, gan_out_channels, n_head, gpu, attn_drop)
        #self.dis_feature_layer = GANLayer(gan_in_channels, hid, gan_out_channels, n_head, gpu, attn_drop)
        self.lnc_dis_feature_layer = GANLayer(gan_in_channels, hid, gan_out_channels, n_head, gpu, attn_drop)
        self.cross_attention=CrossAttentionModel(hid,hid,n_head)
        self.dis_mirna_feature_layer=GANLayer(gan_in_channels, hid, gan_out_channels, n_head, gpu, attn_drop)
        self.mlplayer = MLPLayer(config)

    # def forward(self, lncrna_x, disease_x,mirna_x, adj):
    def forward(self,positive_train_ij,negative_train_ij, H, adj,lnc_dis_fea,lnc_dis_view,lnc_feature,lnc_feature_view,dis_feature,dis_feature_view,dis_mi_fea,dis_mi_view,config,gpu):

        z = self.complex_ganlayer(H,adj,gpu)

        lnc_dis = self.lnc_dis_feature_layer(lnc_dis_fea, lnc_dis_view, gpu)
        # lnc_feature=self.lnc_feature_layer(lnc_feature,lnc_feature_view,gpu)
        # dis_feature=self.dis_feature_layer(dis_feature,dis_feature_view,gpu)
        dis_mi=self.dis_mirna_feature_layer(dis_mi_fea,dis_mi_view,gpu)
        #z_complex=z[0:651+1]
        z_complex=z
        z1=lnc_dis
        z3=dis_mi



        z_complex1=merge_and_average(z1,z3)
        #z2=torch.cat((lnc_feature, dis_feature), dim=0)

        #####共享参数的交叉注意力模型##########(z_complex,z1)是0.942
        #output1=self.cross_attention(z_complex,z1)
        output1 = self.cross_attention(z_complex, z_complex1)


        posi_feature=get_feature_matrix(output1,positive_train_ij)
        neg_feature=get_feature_matrix(output1,negative_train_ij)
        all_feature = torch.cat((posi_feature,neg_feature), dim=0)#竖着拼接

        out = self.mlplayer(all_feature)#F.sigmoid(output)
        return out
