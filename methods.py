from __future__ import division
from __future__ import print_function
from sklearn.metrics.pairwise import cosine_similarity,pairwise_kernels
import torch
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve, average_precision_score, accuracy_score, \
    precision_score, recall_score, f1_score, auc
import copy
import pandas as pd
# def save_to_excel(labels, preds, filename="predictions.xlsx"):
#     labels = labels.flatten()
#     preds = preds.flatten()
#     df = pd.DataFrame({'Label': labels, 'Prediction': preds})
#     df.to_excel(filename, index=False)
import pandas as pd

import os
import pandas as pd


def save_to_excel(labels, preds, auc, seed, filename_prefix="predictions", folder_name="saved_predictions"):
    labels = labels.flatten()
    preds = preds.flatten()
    # 在文件名中包含随机种子和AUC值
    filename = f"{filename_prefix}_seed_{seed}_AUC_{auc:.4f}.xlsx"

    # 检查是否存在文件夹，如果不存在，则创建
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # 保存文件到指定的文件夹
    file_path = os.path.join(folder_name, filename)
    df = pd.DataFrame({'Label': labels, 'Prediction': preds})
    df.to_excel(file_path, index=False)
    print(f"File saved successfully with AUC {auc:.4f} and seed {seed} in the folder '{folder_name}'.")


def save_auc_roc_pr(labels, scores, auc_roc_file, pr_file):
    # 计算 ROC 曲线的数据
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # 计算 PR 曲线的数据
    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = average_precision_score(labels, scores)

    # 保存 ROC 数据到文件
    with open(auc_roc_file, 'w') as auc_roc_f:
        auc_roc_f.write("FPR\tTPR\n")
        for i in range(len(fpr)):
            auc_roc_f.write("{:.6f}\t{:.6f}\n".format(fpr[i], tpr[i]))

    # 保存 PR 数据到文件
    with open(pr_file, 'w') as pr_f:
        pr_f.write("Recall\tPrecision\n")
        for i in range(len(recall)):
            pr_f.write("{:.6f}\t{:.6f}\n".format(recall[i], precision[i]))


def fusion_similarity(similarity_matrix1, similarity_matrix2):
    # 确保两个输入矩阵具有相同的形状
    if similarity_matrix1.shape != similarity_matrix2.shape:
        raise ValueError("Input matrices must have the same shape.")

    # 创建一个新的矩阵用于保存融合后的相似性
    fused_similarity = np.maximum(similarity_matrix1, similarity_matrix2)

    return fused_similarity

#计算各个数据的cos相似性
def calculate_cosine_similarity(matrix):
    # 拆分输入矩阵为四个子矩阵
    gene_matrix = matrix[:, :5347].T
    gos_matrix = matrix[:, 5347:5347+487].T
    mirna_matrix = matrix[:, 5347+487:5347+487+232].T
    # mirna_matrix = matrix[:, 5347+487+232:]

    # 计算余弦相似性
    gene_cos = cosine_similarity(gene_matrix)
    gos_cos = cosine_similarity(gos_matrix)
    mirna_cos = cosine_similarity(mirna_matrix)


    return  gene_cos, gos_cos, mirna_cos
def calculate_kernel_similarity(matrix):
    # 拆分输入矩阵为四个子矩阵
    gene_matrix  = matrix[:, :5347]
    gos_matrix = matrix[:, 5347:5347+487]
    mirna_matrix  = matrix[:, 5347+487:5347+487+232]
    # mirna_matrix = matrix[:, 5347+487+232:]

    # 计算核相似性（使用RBF核函数）

    gene_kernel = pairwise_kernels(gene_matrix, metric='rbf')
    gos_kernel = pairwise_kernels(gos_matrix, metric='rbf')
    mirna_kernel = pairwise_kernels(mirna_matrix, metric='rbf')

    return  gene_kernel, gos_kernel, mirna_kernel
#
def matrix_graph(matrix):
    n, m = matrix.shape
    total_nodes = n + m
    # 创建一个零矩阵作为邻接矩阵
    adjacency_matrix = np.zeros((total_nodes, total_nodes), dtype=int)
    # 填充左上角的(n×m)部分为0
    adjacency_matrix[:n, :m] = 0
    # 填充右上角的(n×m)部分为1
    adjacency_matrix[:n, m:] = 1
    # 填充左下角的(n×m)部分为2
    adjacency_matrix[n:, :m] = 2
    # 填充右下角的(n×m)部分为3
    adjacency_matrix[n:, m:] = 3

    return adjacency_matrix
# loss function

def loss_function(pre_adj, adj):
    #adj = torch.Tensor(adj)
    loss_fn = torch.nn.BCELoss()
    return loss_fn(pre_adj, adj)

def lnc_feature_view(adj):
    data = adj[0:239 + 1, 0:239 + 1]  # 因为这个分割到n1结束，如歌要包括第n1行，则为n1
    return data
def dis_feature_view(adj):
    data = adj[240:651 + 1, 240:651 + 1]  # 因为这个分割到n1结束，如歌要包括第n1行，则为n1
    return data
def mi_feature_view(adj):
    data = adj[652:1146 + 1, 652:1146 + 1]  # 因为这个分割到n1结束，如歌要包括第n1行，则为n1
    return data
def lnc_dis_view(adj):
    data = adj[0:651 + 1, 0:651 + 1]  # 因为这个分割到n1结束，如歌要包括第n1行，则为n1
    return data
def lnc_mi_view(adj):
    data = adj[0:239 + 1, 652:1146 + 1]  # 因为这个分割到n1结束，如歌要包括第n1行，则为n1
    return data
# def dis_mi_view(adj):
#     data = adj[240:651 + 1, 652:1146 + 1]  # 因为这个分割到n1结束，如歌要包括第n1行，则为n1
#     return data
def dis_mi_view(adj):
    data = adj[240:1146+1, 240:1146+1]  # 因为这个分割到n1结束，如歌要包括第n1行，则为n1
    return data

# train method
def train(positive_train_ij,negative_train_ij,label, ganlda_model, optimizer, H, adj,
          lnc_dis_fea,lnc_dis_view,lnc_feature,lnc_feature_view,dis_feature,dis_feature_view,dis_mi_fea,dis_mi_view,config,gpu):

    # train
    optimizer.zero_grad()
    pred = ganlda_model(positive_train_ij,negative_train_ij,H, adj,lnc_dis_fea,lnc_dis_view,lnc_feature,lnc_feature_view,dis_feature,dis_feature_view,dis_mi_fea,dis_mi_view,config,gpu)
    loss = loss_function(pred, label)
    #就是将损失loss 向输入侧进行反向传播
    loss.backward()
    #这个方法会更新所有的参数,optimizer.step()是优化器对的值进行更新
    optimizer.step()

    return pred, loss
def test(label, ganlda_model,lncx, disx, adj):
    # train
    ganlda_model.eval()
    with torch.no_grad():
        pred = ganlda_model(lncx, disx, adj)
        loss = loss_function(pred, label)
    return pred, loss

    # 就是将损失loss 向输入侧进行反向传播

    # 这个方法会更新所有的参数,optimizer.step()是优化器对的值进行更新


# sort the score matrix
def sort_matrix(score_matrix, interact_matrix):
    sort_index = np.argsort(-score_matrix, axis=0)
    score_sorted = np.zeros(score_matrix.shape)
    y_sorted = np.zeros(interact_matrix.shape)
    for i in range(interact_matrix.shape[1]):
        score_sorted[:, i] = score_matrix[:, i][sort_index[:, i]]
        y_sorted[:, i] = interact_matrix[:, i][sort_index[:, i]]
    return y_sorted, score_sorted
import numpy as np
import networkx as nx
from scipy.linalg import eigh

def laplacian_positional_encoding(G, dim):
    # 计算邻接矩阵

    # A = G.adjacency_matrix().toarray().detach().cpu()
    #A = G.adjacency_matrix().to_dense().numpy()
    A = G.adjacency_matrix().to_dense().detach().cpu().numpy()
    # 计算度矩阵 D
    D = np.diag(np.sum(A, axis=1))

    # 计算 D^(-1/2)
    D_inv_sqrt = np.diag(1 / np.sqrt(np.diagonal(D)))

    # 计算拉普拉斯矩阵 L: I - D^(-1/2) * A * D^(-1/2)
    L = np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt

    # 计算 L 的特征值和特征向量
    eigvals, eigvecs = eigh(L)

    # 选取 dim 个最小非零特征值对应的特征向量
    eigvecs = eigvecs[:, 1:dim+1]
    eigvecs=torch.tensor(eigvecs)
    return eigvecs
