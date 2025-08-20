import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import numpy as np

import dgl
import torch
import torch.nn as nn
import dgl.nn as dglnn
# from torch_geometric.nn import GCNConv
from dgl.nn import GraphConv as GCNConv


# GCNConv(in_dim, 64)

def extract_and_concat_node_features(graph, all_features, gpu):
    num_nodes = graph.number_of_nodes()  # 获取图中的总节点数量 rna + disease
    center_node_indices = list(range(num_nodes))
    center_node_representations = []

    for center_node_idx in center_node_indices:
        subgraph = dgl.node_subgraph(graph, [center_node_idx])  # 1阶
        node_features = all_features[subgraph.ndata[dgl.NID]]
        in_channels = node_features.shape[1]
        out_channels = 64  # 作为示例，设置输出维度为 64

        gcn_model = GCNConv(in_channels, in_channels)
        gcn_model = gcn_model.to(gpu)
        with torch.no_grad():
            node_representations = gcn_model(subgraph, node_features)
        center_node_representation = node_representations[0]  # 中心节点的表示

        ## 使用平均池化来将多个节点的表示池化为一个节点的表示
        pooled_representation = torch.mean(node_representations, dim=0)

        center_node_representations.append(center_node_representation)

    # 将所有中心节点的表示竖直拼接成特征矩阵
    result = torch.stack(center_node_representations)

    return result


import torch
import torch.nn as nn
import dgl.function as fn


class SelfAttentionModel(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super(SelfAttentionModel, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads

        # Query, Key, and Value linear projections
        self.linear_query = nn.Linear(in_dim, out_dim)
        self.linear_key = nn.Linear(in_dim, out_dim)
        self.linear_value = nn.Linear(in_dim, out_dim)

    def forward(self, g, x):
        # Compute Q, K, and V
        Q = self.linear_query(x)
        K = self.linear_key(x)
        V = self.linear_value(x)

        # Split Q, K, and V into multiple heads
        Q = Q.view(-1, self.num_heads, self.out_dim // self.num_heads)
        K = K.view(-1, self.num_heads, self.out_dim // self.num_heads)
        V = V.view(-1, self.num_heads, self.out_dim // self.num_heads)

        # Calculate attention scores
        g.apply_edges(fn.v_dot_u('K', 'Q', 'score'), edges=g.edges())

        # Normalize the scores
        g.send(g.edges(), fn.copy_edge('score', 'score'))
        g.recv(g.nodes(), fn.sum('score', 'score_sum'))
        g.send(g.edges(), fn.copy_edge('score', 'score_normalized'), g.sum('score_sum', 'score_sum'))

        # Weighted sum of values
        g.update_all(fn.u_mul_e('V', 'score_normalized', 'V_weighted_sum'), fn.sum('score_normalized', 'score_sum'))

        # Concatenate multi-head results
        V_weighted_sum = g.ndata['V_weighted_sum'].view(-1, self.out_dim)

        return V_weighted_sum

class TransformerEncoder(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super(TransformerEncoder, self).__init__()
        self.hidden_dim=hidden_dim
        self.num_heads=num_heads
        self.dropout=dropout
        # Multi-Head Self-Attention Layer
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)

        # Feed-Forward Neural Network
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.ReLU(),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )

        # Layer Normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-Head Self-Attention
        attention_output, _ = self.attention(x, x, x)
        x = x + self.dropout(attention_output)
        x = self.norm1(x)

        # Feed-Forward Neural Network
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)

        return x
###############################################################################

"""
    Graph Transformer Layer

"""

"""
    Util functions
"""


def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}

    return func


def scaled_exp(field, scale_constant):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field] / scale_constant).clamp(-5, 5))}

    return func


"""
    Single Attention Head
"""


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads

        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)

    def propagate_attention(self, g):
        # Compute attention score
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))  # , edges)
        g.apply_edges(scaled_exp('score', np.sqrt(self.out_dim)))

        # Send weighted values to target nodes
        eids = g.edges()
        g.send_and_recv(eids, fn.u_mul_e('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_e('score', 'score'), fn.sum('score', 'z'))

    def forward(self, g, h):

        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)

        # Reshaping into [num_nodes, num_heads, feat_dim] to
        # get projections for multi-head attention
        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)

        self.propagate_attention(g)

        head_out = g.ndata['wV'] / g.ndata['z']

        return head_out


class GraphTransformerLayer(nn.Module):
    """
        Param:
    """

    def __init__(self, in_dim, out_dim, num_heads, dropout, gpu, layer_norm=False, batch_norm=True, residual=True,
                 use_bias=False):
        super().__init__()
        self.gpu = gpu
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.in_feat_dropout = nn.Dropout(dropout)
        self.attention = MultiHeadAttentionLayer(in_dim, int(out_dim // num_heads), num_heads, use_bias)

        self.O = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(out_dim)

        # FFN
        self.FFN_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.layer_norm:
            self.layer_norm2 = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(out_dim)
        self.gcn = GCNConv(in_dim, out_dim)
        self.transformerEncoder = TransformerEncoder(in_dim, num_heads, dropout)
    def forward(self, g, h, gpu):

        #h = self.in_feat_dropout(h)
        num_nodes = g.number_of_nodes()  # 获取图中的总节点数量 rna + disease
        center_node_indices = list(range(num_nodes))
        center_node_representations = []

        for center_node_idx in center_node_indices:
            subgraph = dgl.node_subgraph(g, [center_node_idx])  # 1阶
            node_features = h[subgraph.ndata[dgl.NID]]
            # in_channels = node_features.shape[1]
            # out_channels = 64  # 作为示例，设置输出维度为 64

            # gcn_model = GCNConv(in_channels, in_channels)
            # gcn_model = gcn_model.to(gpu)

            #node_representations = self.gcn(subgraph, node_features)
            node_representations = self.transformerEncoder(node_features)
            center_node_representation = node_representations[0]  # 中心节点的表示

            ## 使用平均池化来将多个节点的表示池化为一个节点的表示
            pooled_representation = torch.mean(node_representations, dim=0)
            # 使用最大池化来将多个节点的表示池化为一个节点的表示
            max_pooled_representation, _ = torch.max(node_representations, dim=0)
            center_node_representations.append(center_node_representation)
            # center_node_representations.append(pooled_representation)
            #center_node_representations.append(max_pooled_representation)

        # 将所有中心节点的表示竖直拼接成特征矩阵
        h_sub = torch.stack(center_node_representations)

        # h_sub=extract_and_concat_node_features(g,h,gpu)

        h = h_sub
        h_in1 = h  # for first residual connection

        # multi-head attention out
        attn_out = self.attention(g, h)
        h = attn_out.view(-1, self.out_channels)

        h = F.dropout(h, self.dropout, training=self.training)

       # #@@@@需要 #h = self.O(h)
       #  h=self.transformerEncoder(h)
        if self.residual:
            h = h_in1 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm1(h)

        if self.batch_norm:
            h = self.batch_norm1(h)

        h_in2 = h  # for second residual connection

        # FFN
        h = self.FFN_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_layer2(h)

        if self.residual:
            h = h_in2 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm2(h)

        if self.batch_norm:
            h = self.batch_norm2(h)

        return h

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                                                                   self.in_channels,
                                                                                   self.out_channels, self.num_heads,
                                                                                   self.residual)