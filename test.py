# -*- coding: utf-8 -*-
# @Time    : 2025/1/2 17:47
# @Author  : Xuebing Wang
# @Email   : wangxuebing0212@163.com
# @FileName: test.py
# @Description: there is code description.

# sample = {'posi':'','neg':[]}
#
# print(sample["posi"]=='')
# print(sample["neg"]==[])
# import pandas as pd
#
# data_pd = pd.read_excel('test.xlsx')
# with open('test.txt',mode='wt') as fw:
#     for i in range(len(data_pd)):
#         data = '\t'.join(str(x) for x in data_pd.iloc[i].tolist())
#         fw.write(data+'\n')
import torch
import torch.nn as nn

loss = nn.BCEWithLogitsLoss()
input = torch.randn(3, requires_grad=True)
target = torch.empty(3).random_(2)
print(input)
print(target)
output = loss(input, target)
output.backward()

if __name__ == '__main__':
    print('保持好心情！ ')
