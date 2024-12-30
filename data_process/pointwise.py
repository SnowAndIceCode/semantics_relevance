# -*- coding: utf-8 -*-
# @Time    : 2024/12/30 14:44
# @Author  : Xuebing Wang
# @Email   : wangxuebing0212@163.com
# @FileName: pointwise.py
# @Description: there is code description.

'''
    清洗pointwise类型数据：
        1. 点击为正样本
        2. 负样本为随机负样本
'''
import pandas as pd
from tqdm import tqdm
import random
import re

def contains_chinese(text):
    # 正则表达式匹配任何中文字符
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
    return bool(chinese_pattern.search(text))

def starts_with_punctuation(text):
    # 正则表达式匹配中英文标点符号和特殊字符
    punctuation_pattern = re.compile(r'^[.,;:!?(){}$$"\'\-—～！？。，；：“”‘’《》【】…·\\\\/*+$^_~]+$')
    if not text:
        return False
    first_char = text[0]
    return bool(punctuation_pattern.match(first_char))

def check_string(text):
    if starts_with_punctuation(text) and not contains_chinese(text):
        return 1 #
    else:
        return 0

def get_sample_neg(file_path,save_path):
    docs = []
    datas = []
    neg_datas = []
    with open(file_path) as fr:
        for line in fr:
            data = line.strip().split('\t')
            if len(data) != 3:
                print(f'lenth is error:{data}')
                continue
            query, doc = data[0].strip(), data[1].strip()
            if check_string(query):
                print(f'query is error:{line}')
                continue
            docs.append(doc.strip())
            datas.append({'query': query, 'doc': doc, 'isNext': 1})
    size = len(datas)
    random.shuffle(docs)
    for i in tqdm(range(size)):
        query = datas[i]['query']
        doc = docs[i]
        neg_datas.append({'query': query, 'doc': doc, 'isNext': 0})

    datas.extend(neg_datas)
    random.shuffle(datas)
    datas_df = pd.DataFrame(datas)
    datas_df.to_csv(save_path, index=False)
    print(datas_df.info())
if __name__ == '__main__':
    print('保持好心情！ ')
    file_path = 'raw_data/487030991_AB3B1A_相关性正样本数据挖掘.txt'
    save_path = '241111-241211-pointwise_simple_neg-click_posi.csv'
    get_sample_neg(file_path,save_path)

