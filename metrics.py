# -*- coding: utf-8 -*-
# @Time    : 2024/12/30 16:11
# @Author  : Xuebing Wang
# @Email   : wangxuebing0212@163.com
# @FileName: metrics.py
# @Description: there is code description.
from sklearn.metrics import roc_auc_score

def cul_auc(pre,label):
    auc = roc_auc_score(pre, label)
    return auc

def cul_gauc():
    pass

def cul_DCG():
    pass

def cul_NDCG():
    pass

def cul_PNR():
    pass
if __name__ == '__main__':
    print('保持好心情！ ')
