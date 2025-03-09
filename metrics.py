# -*- coding: utf-8 -*-
# @Time    : 2024/12/30 16:11
# @Author  : Xuebing Wang
# @Email   : wangxuebing0212@163.com
# @FileName: metrics.py
# @Description: there is code description.
from sklearn.metrics import roc_auc_score
import numpy as np
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

    print(f"{total_pairs}:total_pairs")
    print(f"{inversions}:inversions")
    return pnr


import numpy as np


def calculate_pnr(scores, labels):
    """
    计算正逆序比（PNR）

    参数:
        scores (list/np.ndarray): 模型输出的概率值列表（0-1之间）
        labels (list/np.ndarray): 真实标签列表（取值为0,1,2,3）

    返回:
        float: PNR值
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    n = len(labels)

    if n < 2:
        return 0.0  # 样本不足时返回0

    # 生成i<j的掩码矩阵
    i_less_j = np.arange(n)[:, None] < np.arange(n)[None, :]

    # 计算标签差和分数差矩阵
    label_diff = labels[:, None] - labels[None, :]
    score_diff = scores[:, None] - scores[None, :]

    # 确定有效对（i<j且标签不同）
    valid_pairs = i_less_j & (label_diff != 0)

    # 计算正确排序的条件
    correct_order = (
            (label_diff > 0) & (score_diff > 0) |  # 正序情况1
            (label_diff < 0) & (score_diff < 0)  # 正序情况2
    )

    # 统计正序和逆序对数
    positive = np.sum(correct_order & valid_pairs)
    reverse = np.sum(~correct_order & valid_pairs)

    # 处理除零情况
    return positive / (reverse + 1e-9)



def dcg_at_k(relevances, k):
    relevances = relevances[:k]

    return np.sum([rel / np.log2(i + 2) for i, rel in enumerate(relevances)])



def cul_NDCG():
    pass


if __name__ == '__main__':
    print('保持好心情！ ')
    # Example usage:
    # relevance_scores = [3, 2, 3, 0, 1, 2]  # Relevance scores of documents in ranked order
    # k = 4  # Rank position to compute DCG for
    # dcg_value = dcg_at_k(relevance_scores, k)
    # print(f"DCG@{k}: {dcg_value}")
    scores = [0.9,0.9,0.1,0.8,0.3]
    labels = [3,3, 2, 1, 0]
    print(f"PNR: {cul_PNR(scores, labels):.4f}")