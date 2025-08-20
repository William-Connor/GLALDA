from __future__ import division
from __future__ import print_function
from sklearn.metrics.pairwise import cosine_similarity,pairwise_kernels
import torch
import copy
from sklearn.metrics.pairwise import cosine_similarity,pairwise_kernels
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix


# k 参数：这是用于构建 k-最近邻图的参数。k 指定每个样本在最近邻图中连接到多少个其他样本。通常情况下，
# 较小的 k 值可以捕捉到更细粒度的相似性，但可能会引入噪声。较大的 k 值可以捕捉更广泛的相似性，但可能忽略了细节。
#
# 选择 k 的一种方法是尝试不同的值，从较小的值开始，逐渐增加，然后通过验证集或交叉验证来选择最佳值。
# t 参数：这是用于在 SNF 中应用幂次转换的参数。t 控制了相似性矩阵的幂次，可以影响最终的融合结果。
#
# 通常情况下，较大的 t 值会强调潜在的相似性，而较小的 t 值可能会强调局部相似性。选择 t 的最佳值也可以通过尝试不同的值来确定。
#
# 对于 k 值，通常的范围可以从较小的值（例如 5 或 10）到较大的值（例如 20 或更大）进行尝试。您可以从一个较小的 k 值开始，
# 然后逐渐增加，以观察融合结果如何变化，并选择最适合您数据的值。
#
# 对于 t 值，通常的范围可以从 1 到 20 或更高。较小的 t 值可能会强调局部相似性，而较大的 t 值可能会强调全局相似性。您可以根据数据的性质来选择合适的 t 值。


def snf_fusion(lnc_sim, cos_sim, kernel_sim, k, t, alpha):
    # Step 1: Normalize the similarity matrices
    lnc_sim = normalize(lnc_sim, norm='l2', axis=1)
    cos_sim = normalize(cos_sim, norm='l2', axis=1)
    kernel_sim = normalize(kernel_sim, norm='l2', axis=1)
    print("！！k, t, alpha:", k, t, alpha)
    # Step 2: Create affinity matrices
    affinity_matrices = [lnc_sim, cos_sim, kernel_sim]

    # Step 3: Create k-nearest neighbor graphs
    knn_graphs = []
    for affinity_matrix in affinity_matrices:
        knn_graph = np.zeros_like(affinity_matrix)
        for i in range(len(affinity_matrix)):
            indices = np.argsort(affinity_matrix[i])[::-1][:k]
            knn_graph[i][indices] = affinity_matrix[i][indices]
        knn_graphs.append(knn_graph)

    # Step 4: Construct the affinity matrix W
    W = np.zeros_like(lnc_sim)
    for knn_graph in knn_graphs:
        W += knn_graph

    # Step 5: Apply the power normalization
    for i in range(len(W)):
        W[i] = np.power(W[i], t)

    # Step 6: Apply the exponential diffusion
    W = normalize(W, norm='l1', axis=1)
    for i in range(len(W)):
        W[i] = np.power(W[i], alpha)

    return W  # 返回融合后的相似性矩阵

# def snf(func_sim, cos_sim, kernel_sim, K=3, t=10, alpha=0.5):
#     # 将相似性矩阵存储在列表中
#     # similarity_list = [func_sim, cos_sim, kernel_sim]
#
#
#     snf = snf_fusion(similarity_list, K, t, alpha)
#
#     # # 执行SNF融合
#     # affinity_matrix = snf.fit()
#
#     return affinity_matrix


def lnc_sim_view(adj):
    data = adj[0:239 + 1, 0:239 + 1]  # 因为这个分割到n1结束，如歌要包括第n1行，则为n1
    return data
def dis_sim_view(adj):
    data = adj[240:651 + 1, 240:651 + 1]  # 因为这个分割到n1结束，如歌要包括第n1行，则为n1
    return data
def mi_sim_view(adj):
    data = adj[652:1146 + 1, 652:1146 + 1]  # 因为这个分割到n1结束，如歌要包括第n1行，则为n1
    return data
def lnc_dis(adj):
    data = adj[0:239 + 1, 240:651 + 1]  # 因为这个分割到n1结束，如歌要包括第n1行，则为n1
    return data
def lnc_mi(adj):
    data = adj[0:239 + 1, 652:1146 + 1]  # 因为这个分割到n1结束，如歌要包括第n1行，则为n1
    return data
def dis_mi(adj):
    data = adj[240:651 + 1, 652:1146 + 1]  # 因为这个分割到n1结束，如歌要包括第n1行，则为n1
    return data




def extract_and_fuse_matrices(adj,k, t, alpha):


    def lnc_sim_view(adj):
        data = adj[0:239 + 1, 0:239 + 1]  # 因为这个分割到n1结束，如歌要包括第n1行，则为n1
        return data

    def dis_sim_view(adj):
        data = adj[240:651 + 1, 240:651 + 1]  # 因为这个分割到n1结束，如歌要包括第n1行，则为n1
        return data

    def mi_sim_view(adj):
        data = adj[652:1146 + 1, 652:1146 + 1]  # 因为这个分割到n1结束，如歌要包括第n1行，则为n1
        return data

    def lnc_dis(adj):
        data = adj[0:239 + 1, 240:651 + 1]  # 因为这个分割到n1结束，如歌要包括第n1行，则为n1
        return data

    def lnc_mi(adj):
        data = adj[0:239 + 1, 652:1146 + 1]  # 因为这个分割到n1结束，如歌要包括第n1行，则为n1
        return data

    def dis_mi(adj):
        data = adj[240:651 + 1, 652:1146 + 1]  # 因为这个分割到n1结束，如歌要包括第n1行，则为n1
        return data

    # Extract submatrices
    lnc_sim = lnc_sim_view(adj)
    dis_sim = dis_sim_view(adj)
    mi_sim = mi_sim_view(adj)
    lnc_dis = lnc_dis(adj)
    dis_mi = dis_mi(adj)

    # Calculate cosine similarity
    def calculate_cosine_similarity(matrix):
        return cosine_similarity(matrix)

    lnc_cosine_sim = calculate_cosine_similarity(lnc_sim)
    dis_cosine_sim = calculate_cosine_similarity(dis_sim)
    mi_cosine_sim = calculate_cosine_similarity(mi_sim)

    # Calculate kernel similarity (you may need to implement your own kernel calculation)
    def calculate_kernel_similarity(matrix):
        return pairwise_kernels(matrix)

    lnc_kernel_sim = calculate_kernel_similarity(lnc_dis)
    dis_kernel_sim = calculate_kernel_similarity(lnc_dis.T)
    mi_kernel_sim = calculate_kernel_similarity(dis_mi.T)

    # Fusion of similarity matrices (you can use weighted averaging or other methods)
    def fuse0_similarity_matrices(sim1, sim2, sim3):
        # Implement fusion method (e.g., weighted averaging)
        return (sim1 + sim2 + sim3) / 3  # Simple weighted averaging

    def fuse_similarity_matrices(sim1, sim2, sim3):
        return snf_fusion(sim1,sim2,sim3,k, t, alpha)
    lnc_fused_sim=fuse_similarity_matrices(lnc_sim,lnc_cosine_sim,lnc_kernel_sim)
    dis_fused_sim=fuse_similarity_matrices(dis_sim,dis_cosine_sim,dis_kernel_sim)
    mi_fused_sim=fuse_similarity_matrices(mi_sim,mi_cosine_sim,mi_kernel_sim)
    # Return the fused similarity matrices
    return lnc_fused_sim, dis_fused_sim,mi_fused_sim

def cancatenate(lnclen, dilen, milen, lnc_di, lnc_mi, mi_di, lncSiNet, diSiNet, miSiNet):
    A = np.zeros((lnclen + dilen + milen, lnclen + dilen + milen))
    A[: lnclen, lnclen: lnclen + dilen] = lnc_di
    A[lnclen: lnclen + dilen, : lnclen] = lnc_di.T
    A[: lnclen, lnclen + dilen: ] = lnc_mi
    A[lnclen + dilen: , : lnclen] = lnc_mi.T
    A[lnclen: lnclen + dilen, lnclen + dilen: ] = mi_di.T
    A[lnclen + dilen: , lnclen: lnclen + dilen] = mi_di
    A[: lnclen, : lnclen] = lncSiNet
    A[lnclen: lnclen + dilen, lnclen: lnclen + dilen] = diSiNet
    A[lnclen + dilen: , lnclen + dilen: ] = miSiNet
    return A

def A_SNF(A,k=17, t=11, alpha=0.5):
    def lnc_dis(adj):
        data = adj[0:239 + 1, 240:651 + 1]  # 因为这个分割到n1结束，如歌要包括第n1行，则为n1
        return data

    def lnc_mi(adj):
        data = adj[0:239 + 1, 652:1146 + 1]  # 因为这个分割到n1结束，如歌要包括第n1行，则为n1
        return data

    def dis_mi(adj):
        data = adj[240:651 + 1, 652:1146 + 1]  # 因为这个分割到n1结束，如歌要包括第n1行，则为n1
        return data
    lnc_fused_sim, dis_fused_sim, mi_fused_sim=extract_and_fuse_matrices(A,k, t, alpha)

    lnc_dis = lnc_dis(A)
    lnc_mi=lnc_mi(A)
    dis_mi = dis_mi(A)
    mi_dis=dis_mi.T
    lnclen =lnc_dis.shape[0]
    dilen =dis_mi.shape[0]
    milen =dis_mi.shape[1]
    A_SNF=cancatenate(lnclen, dilen, milen, lnc_dis, lnc_mi, mi_dis, lnc_fused_sim, dis_fused_sim, mi_fused_sim)
    return A_SNF

