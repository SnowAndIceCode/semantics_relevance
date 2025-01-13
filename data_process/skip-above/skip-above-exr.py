# -*- coding: utf-8 -*-
# @Time    : 2025/1/2 15:07
# @Author  : Xuebing Wang
# @Email   : wangxuebing0212@163.com
# @FileName: skip-above-exr.py
# @Description: 抽取跳过浏览数据.

'''
    1. 根据gtid 确定是否抽取

'''
import pandas as pd
from tqdm import tqdm

# skip_path = './486954032_F1F0D9_skip-above.txt'
skip_path = './488850116_FC4593_skip-above.txt'
count_num = 0



all_data = {}
with open(skip_path) as fr:
    for line in tqdm(fr):
        if count_num == 0:
            count_num += 1
            continue
        count_num += 1
        # if count_num >= 100000:
        #     break
        data = line.strip().split('\t')
        if len(data) != 7 or data[3]=='-':
            print(f'data is error:{line.strip()}')
            continue
        try:
            gtid,query,infoid,info_title,click,gpos,rank = data[0],data[1],data[2],data[3],int(data[4]),int(data[5]),int(data[6])
        except:
            print(f"负值错误：{line.strip()}")
            continue
        if gtid not in all_data:
            all_data[gtid] = [{'gtid':gtid,
                           'query':query,
                           'infoid':infoid,
                           'info_title':info_title,
                           'click':click,
                           'gpos':gpos,
                           'rank':rank}]
        else:
            all_data[gtid].append({
            'gtid': gtid,
            'query': query,
            'infoid': infoid,
            'info_title': info_title,
            'click': click,
            'gpos': gpos,
            'rank': rank}
            )


skip_datas =[]
print(len(all_data.keys()))
for key in tqdm(all_data.keys()):
    sample = {'posi': '', 'neg': []}
    neg_data = []
    for item in all_data[key]:
        if item["click"] == 0:
            neg_data.append(item) # 可能负样本
        elif item["click"] == 1:
            if sample['posi'] == '' and neg_data == []: # 第一个帖子被点击了
                continue

            if sample['posi'] == '' and neg_data != []: #  点击前面有未点击样本
                sample['posi'] = item
                sample['neg'] = neg_data
                skip_datas.append(sample)
                sample = {'posi':'','neg':[]}
                neg_data = []


# [
#
# {
# "posi":{},
# "neg": [{}....{}]
# }
#
# ]

# print(len(skip_datas))
neg_datas =[]
for i in range(len(skip_datas)):
    neg_list = skip_datas[i]["neg"]
    posi_dict = skip_datas[i]["posi"]
    for neg_dict in neg_list:
        neg_dict['posi']=posi_dict
        neg_datas.append(neg_dict)

# print(neg_datas)
neg_df = pd.DataFrame(neg_datas)
neg_df.to_csv('./20241201-20241231_skip-above.csv',index=False)
if __name__ == '__main__':
    print('保持好心情！ ')
