# -*- coding: utf-8 -*-
# @Time    : 2024/12/29 19:16
# @Author  : Xuebing Wang
# @Email   : wangxuebing0212@163.com
# @FileName: data_helper.py
# @Description: there is code description.

import random
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader,Dataset,random_split
from transformers import AutoTokenizer

'''
 v1:mask 有些问题
'''
class CustomDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length):
        self.data = self.load_data(data_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def load_data(self,data_file):
        count = 0
        datas = []
        data_df = pd.read_csv(data_file)
        for i in tqdm(range(len(data_df))):
            count += 1
            if count >10000:
                break
            query = data_df['query'].iloc[i]
            doc = data_df['doc'].iloc[i]
            label = data_df['isNext'].iloc[i]
            datas.append({"sentence1":query,"sentence2":doc,"isNext":label})
        return datas

    def __len__(self):
        return len(self.data)  # 每两个句子形成一组

    def __getitem__(self, idx):
        sample = self.data[idx]
        sent_a = sample["sentence1"]
        sent_b = sample["sentence2"]
        is_next = sample["isNext"]

        encoding = self.tokenizer(
            sent_a, sent_b,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        token_type_ids = encoding["token_type_ids"].squeeze(0)

        # Mask 部分单词用于 MLM 任务
        input_ids, labels = self.mask_tokens(input_ids)

        return input_ids, attention_mask, token_type_ids, labels, torch.tensor(is_next, dtype=torch.long)

    def mask_tokens(self, input_ids):
        labels = input_ids.clone() # [batch,seq]
        probability_matrix = torch.full(labels.shape, 0.15)
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(labels, already_has_special_tokens=True)

        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # 不计算未被mask的部分的loss

        # 80% mask token, 10% random token, 10% original token
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        return input_ids, labels


def get_dataLoader(args,dataset,batch_size=None, shuffle=False):
    # 可添加data_collate

    return DataLoader(
        dataset,
        batch_size=(batch_size if batch_size else args.batch_size),
        shuffle=shuffle
        # collate_fn=collote_fn
    )
if __name__ == '__main__':
    print('保持好心情！ ')
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    # texts = [{"sentence1":"这是一个简单的预训练示例。", "sentence2":"BERT是一个双向的Transformer模型。","isNext":True}]
    # dataset = CustomDataset(texts,tokenizer,64)
    # print(dataset.__getitem__(0))
    train_data_path = 'data_process/241111-241211-pointwise_simple_neg-click_posi.csv'
    dataset = CustomDataset(train_data_path, tokenizer, 64)
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])
    print(len(train_dataset))