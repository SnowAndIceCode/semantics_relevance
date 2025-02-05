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


def cul_PNR(scores,labels):
    """
    计算正逆序比（PNR）。

    :param labels: List[int] 标签，0表示不相关，1表示相关
    :param scores: List[float] 模型预测的分数，表示数据的相关性
    :return: 正逆序比（PNR）
    """
    n = len(labels)
    total_pairs = n * (n - 1) // 2  # 总对数 C(n, 2)
    inversions = 0

    # 遍历所有的对 (i, j)，i < j
    for i in range(n):
        for j in range(i + 1, n):
            # 如果模型预测排序和标签排序不同，则为逆序对
            if (scores[i] > scores[j] and labels[i] < labels[j]) or (scores[i] < scores[j] and labels[i] > labels[j]):
                inversions += 1

    # 计算正逆序比（PNR）
    pnr = inversions / total_pairs if total_pairs != 0 else 0
    return pnr

def cul_DCG():
    pass
def cul_NDCG():
    pass


if __name__ == '__main__':
    print('保持好心情！ ')
