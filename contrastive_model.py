
import torch
import torch.nn.functional as F
from torch.nn import Parameter
import math
import torch.nn as nn
import numpy as np
import dgl
from dgl.nn.pytorch import edge_softmax, GATConv
from graph_transformer_layer import GraphTransformerLayer
from methods import laplacian_positional_encoding
def uniform(tensor):
    if tensor is not None:
        nn.init.kaiming_uniform_(tensor)


#互补矩阵生成
def complement_matrix(adj_matrix):
    # 获取邻接矩阵的形状
    rows, cols = adj_matrix.shape

    # 创建一个全1矩阵，形状与邻接矩阵相同
    ones_matrix = np.ones((rows, cols))

    # 从全1矩阵中减去邻接矩阵
    complement = ones_matrix - adj_matrix
    return complement


class GANLayer(torch.nn.Module):
    def __init__(self, in_channels,hid, out_channels, n_head, attn_drop=0.4,
                 negative_slope=0.2, residual=False, activation=F.elu):
        super(GANLayer, self).__init__()
        #hid是transformer输入之前要进行的线性变换
        self.gat_layers = nn.ModuleList()
        # just only one layer in our paper.
        self.in_channels=in_channels
        self.hid=hid
        # out_feats： int。输出特征size。
        # num_heads： int。Multi - head Attention中heads的数量。
        # feat_drop = 0.： float。特征丢弃率。
        # attn_drop = 0.： float。注意力权重丢弃率。
        # negative_slope = 0.2： float。LeakyReLU的参数。
        # residual = False： bool。是否采用残差连接。
        # activation = None：用于更新后的节点的激活函数。
        #in_channels=128，out_channels=8，n_head=8
        ##mlp_layers[128,64,64,1]，attn_drop=0.4

        #GATConv默认输出为（N，H，D），N是节点数，H是head个数，D是out_feats大小

        # self.gat_layers.append(GATConv(
        #     in_channels, out_channels, n_head,
        #     attn_drop, attn_drop, negative_slope, residual, activation))
        # self.in_feat_dropout = nn.Dropout(attn_drop)
        self.embedding_lap_pos_enc = nn.Linear(in_channels, hid)
        self.embedding_h=nn.Linear(in_channels, hid) # node feat is an integer
        self.gat_layers.append(GraphTransformerLayer(hid, hid, n_head,
                                              attn_drop))

    def forward(self, lncrna_x, disease_x, adj):#lncx=(240, 128),disx=(412, 128)
        #adj: 图的邻接矩阵 维度[N, N] 非零即一，数据结构基本知识
        #表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留，返回下标
        index_tuple = np.argwhere(adj == 1)
        #shape[0]  矩阵的行数
        lnc_size = lncrna_x.shape[0]#240
        dis_size = disease_x.shape[0]#412
        z = torch.cat((lncrna_x, disease_x))#默认参数 0  ，竖着扩展拼接（240+412，128）=（652，128）
        #创建一个没有节点和边的空图。
        g = dgl.DGLGraph()
        ##图g 添加节点，节点数为lnc_size + dis_size=240+412=652个
        g.add_nodes(lnc_size + dis_size)
        edge_list = index_tuple  ##邻接矩阵有连接（关联为1）的下标
        ##取出头节点和尾节点 下 标
        src, dst = tuple(zip(*edge_list))#######具体示例看图片add_edge.png
        #添加边，前面已经添加过节点数目
        g.add_edges(src, dst)
        ###########################################################################
        g = dgl.add_self_loop(g)


############################################################################################################################
        


        ###############################################################################
        ##z是lncrna_x 和 disease_x行扩展拼接形成的矩阵，（240+412，128）=（652，128）
        #g则是由邻接矩阵有链接的节点组成的图，节点数目为lnc_size + dis_size=652（lncrna_x 和 disease_x的行数之和），即等于z的行数
        #因为gat_layers里只有一层网络，gat_layers[0]是第一层也是唯一的，即上面的添加的GATConv()
        #一开始z = torch.cat((lncrna_x, disease_x))#默认参数 0  ，行扩展拼接（240+412，128）=（652，128）
        #经过下面处理z=(652,64)
        h = self.embedding_h(z)#652*128
        #拉普拉斯位置编码
        h_lap_pos_enc=laplacian_positional_encoding(g,self.in_channels)#652*128  64
        h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float())
        h = h + h_lap_pos_enc

        #h = self.in_feat_dropout(h)
        #self.gat_layers[0](g, z):(652,8,8)
        # GATConv默认输出为（N，H，D），N是节点数，H是head个数，D是out_feats大小
        z = self.gat_layers[0](g, h).flatten(1)##(652,64),,,
        return z