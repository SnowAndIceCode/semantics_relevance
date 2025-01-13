# -*- coding: utf-8 -*-
# @Time    : 2025/1/6 10:16
# @Author  : Xuebing Wang
# @Email   : wangxuebing0212@163.com
# @FileName: skip-above-filter.py
# @Description: there is code description.

'''
    根据双塔模型的余弦值筛选出余弦值较低的样本
        以v10：0.7735 为阈值，低于这个阈值的样本，送到LLM中打标，筛选出真负样本。
'''
import pandas as pd
file_a = '20241201-20241231_skip-above.csv'
file_b = 'process_20241201-20241231_skip-above.csv'
a_df = pd.read_csv(file_a)
b_df = pd.read_csv(file_b)

# 找出共同的列名
common_columns = a_df.columns.intersection(b_df.columns)
print(common_columns)

b_df_unique = b_df.drop(columns=common_columns)
result_df = pd.concat([a_df, b_df_unique], axis=1)
# 处理可能存在的重复列（如果df2中也有非索引的相同名称的列）
# 这一步是为了确保即使有重复列名，也只有一份被保留
result_df = result_df.loc[:, ~result_df.columns.duplicated()]
# 列表，包含您想要保留的列名
columns_to_keep = ['gtid', 'query', 'infoid','doc','posi','sim']
filtered_df = result_df[columns_to_keep]

# 设定筛选条件的列名和阈值
target_column = 'sim'  # 替换为实际的目标列名
threshold_value = 0.7735  # 替换为您想要设置的阈值

# 使用布尔索引来筛选行
filtered_thr_df = filtered_df[result_df[target_column] < threshold_value]
print(filtered_thr_df.head(10))

filtered_thr_df.to_csv('skip-above-filter-by-thr.csv', index=False)

if __name__ == '__main__':
    print('保持好心情！ ')
