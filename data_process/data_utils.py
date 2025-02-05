# -*- coding: utf-8 -*-
# @Time    : 2024/12/30 14:45
# @Author  : Xuebing Wang
# @Email   : wangxuebing0212@163.com
# @FileName: data_utils.py
# @Description: there is code description.

import pandas as pd
def concat_csv(file_a,file_b):
    a_df = pd.read_csv(file_a)
    b_df = pd.read_csv(file_b)
    concat_df = pd.concat([a_df, b_df],axis=0)
    return concat_df

if __name__ == '__main__':
    print('保持好心情！ ')
    file_a = '/Users/a58/Documents/wxb/workspace/semantics_relevance/data_process/post-pretrain-data/post_tuning_q2q.csv'
    file_b = '/Users/a58/Documents/wxb/workspace/semantics_relevance/dataset/241111-241211-pointwise_simple_neg-click_posi.csv'
    save_path = '../dataset/241111-241211-pointwise_simple_neg-click_posi+q2q.csv'
    concat_df = concat_csv(file_a,file_b)
    concat_df.to_csv(save_path,index=False)
