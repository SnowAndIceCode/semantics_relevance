# -*- coding: utf-8 -*-
# @Time    : 2024/12/29 19:15
# @Author  : Xuebing Wang
# @Email   : wangxuebing0212@163.com
# @FileName: model.py
# @Description: there is code description.

import torch
import torch.nn as nn
from transformers import AutoModel,BertForMaskedLM,BertPreTrainedModel,BertModel,BertConfig,BertForSequenceClassification


'''
    MLM+pointwise+pairwise
'''

class PostTrainModel(BertPreTrainedModel):
    def __init__(self,config,num_layers=12):
        super().__init__(config)
        config.num_hidden_layers = num_layers
        self.bert = BertModel(config)
        self.cls_mlm = BertForMaskedLM(config).cls
        self.classification_head = nn.Linear(config.hidden_size, 1)

    def forward(self,input_ids,attention_mask,token_type_ids):

        outputs = self.bert(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output,pooled_output = outputs[:2] # 取出[CLS]后的隐藏状态
        prediction_scores = self.cls_mlm(sequence_output) # sequence_output：[b,seq,hidden_dim]
        classification_logits = self.classification_head(pooled_output)

        return prediction_scores, classification_logits.squeeze()


class FineTuningModel(BertPreTrainedModel):
    def __init__(self,config,num_layers=12):
        super().__init__(config)
        config.num_hidden_layers = num_layers
        self.bert = BertModel(config)
        self.classification_head = nn.Linear(config.hidden_size, 1)
    def forward(self,input_ids,attention_mask,token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output, pooled_output = outputs[:2]  # 取出[CLS]后的隐藏状态
        classification_logits = self.classification_head(pooled_output)
        return classification_logits.squeeze()

if __name__ == '__main__':
    from transformers import AutoTokenizer
    mode_path = '/Users/a58/Documents/wxb/workspace/semantics_relevance/pretrain_models/tiansz/bert-base-chinese'
    model =AutoModel.from_pretrained(mode_path)
    total_params = sum(p.numel() for p in model.parameters()) / 1_000_000
    print(total_params)
    # data = ['W','w']
    # tokenizer = AutoTokenizer.from_pretrained(mode_path)
    # inputs = tokenizer(data,padding="max_length",max_length=64,truncation=True,return_tensors='pt')
    # print(inputs)

    # config = BertConfig.from_pretrained(mode_path)
    # # model = PostTrainModel(config,12)
    # model = FineTuningModel(config,2)
    # print(model)
    # prediction_scores,classification_logits = model(**inputs)
    # print(prediction_scores.shape)
    # print(prediction_scores.view(-1, config.vocab_size)) # [bath*seq,vocab_size]
    # print(classification_logits.shape)
    # print(classification_logits)
    # print('保持好心情！ ')
