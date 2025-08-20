import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve, precision_recall_curve, auc
from sklearn import preprocessing
import csv

import pandas as pd

df = pd.read_excel('predictions_seed_9854_AUC_0.9492.xlsx')

# 取出Label列
labels = df['Label'].values

# 取出Prediction列
preds = df['Prediction'].values






ds ='dataset1'
lnc_di = pd.read_csv('data/ours/' + ds + '/lnc_di.csv', index_col=0)
diseases = lnc_di.columns
lncRNAs = lnc_di.index

positive_ij = np.load('data/ours/' + ds + '/positive_ij.npy')
negative_ij = np.load('data/ours/' + ds + '/negative_ij.npy')
positive5foldsidx = np.load('data/ours/' + ds + '/positive5foldsidx.npy', allow_pickle=True)
negative5foldsidx = np.load('data/ours/' + ds + '/negative5foldsidx.npy', allow_pickle=True)
positive_test_ij = positive_ij[positive5foldsidx[0]['test']]
positive_train_ij = positive_ij[positive5foldsidx[0]['train']]
negative_test_ij = negative_ij[negative5foldsidx[0]['test']]
negative_train_ij = negative_ij[negative5foldsidx[0]['train']]

ij = np.concatenate((positive_test_ij, positive_train_ij, negative_test_ij, negative_train_ij))
i = ij[:, 0].T ### 横坐标
j = ij[:, 1].T### 列坐标
labels = labels.astype(int)
prediction_results = pd.DataFrame({
        'lncRNA': np.array([lncRNAs[lncRNA] for lncRNA in i]),
        'disease': np.array([diseases[disease - len(lncRNAs)] for disease in j]),
        'pred': preds,
        'label': labels
    })


###387 * 3832

evidence = pd.read_csv('data/ours/dataset2/union/di_lnc_union.csv', index_col='Unnamed: 0')
evidence_diseases = evidence.index   ###doid 那一列
evidence_lncRNAs = evidence.columns   ###lncRNA名字

new_results = pd.DataFrame(columns=['lncRNA', 'disease', 'pred', 'label', 'evidence'])
for idx, row in prediction_results.iterrows():
    lncRNA = row['lncRNA']
    disease = row['disease']
    evd = 0
    if (lncRNA in evidence_lncRNAs) and (disease in evidence_diseases) and (evidence.loc[disease, lncRNA] == 1):
        evd = 1
    new_results = new_results.append({
        'lncRNA': lncRNA,
        'disease': disease,
        'pred': row['pred'],
        'label': row['label'],
        'evidence': evd
    }, ignore_index=True)
new_results.sort_values(by='pred', ascending=False).to_csv('files/case study/dataset1.csv')
for di, case in list(new_results[new_results['evidence'] == 1].groupby(['disease'])):
    if len(case.values) > 10:
        print(di)
        new_results[(new_results['disease'] == di) & (new_results['label'] == 0)].sort_values(by='pred', ascending=False).to_csv('files/case study/' + di + '.csv')
    dataset1_results = pd.read_csv('files/case study/dataset1/dataset1.csv', index_col=0)
    a=dataset1_results[
        ((dataset1_results['evidence'] == 1) & (dataset1_results['label'] == 0) & (dataset1_results['pred'] > 0.5))]