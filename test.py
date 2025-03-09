# 需在GPU环境运行
# 加载数据集过程可能由于网络原因失败，请尝试重新运行代码
# from modelscope.metainfo import Trainers
# from modelscope.msdatasets import MsDataset
# from modelscope.trainers import build_trainer
# import tempfile
import os
import random
import json

# from datasets import load_dataset
# dataset = load_dataset(path='Shitao/bge-reranker-data',cache_dir="./dataset/beg_data")
# print(dataset)


# with open('./t2rank_100.distill.standard.jsonl') as fr:
    # for line in fr:
    #     data_json = json.loads(line.strip())
    #     break
    # print(data_json.keys())
    # print(len(data_json["pos"]))
    # print(data_json["query"])
    # print(data_json["pos"][0])

from transformers import AutoTokenizer
import torch
query = '58同城'
doc = '58同城厂家直销）免费上门定制方案丨办公家具丨会员打9折丨24小时在 家具'
tokenizer = AutoTokenizer.from_pretrained('/Users/a58/Documents/wxb/workspace/semantics_relevance/pretrain_models/tiansz/bert-base-chinese')
token = tokenizer(query,doc)

print(token)
