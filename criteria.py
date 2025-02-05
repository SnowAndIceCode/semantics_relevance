# -*- coding: utf-8 -*-
# @Time    : 2024/12/29 19:18
# @Author  : Xuebing Wang
# @Email   : wangxuebing0212@163.com
# @FileName: criteria.py
# @Description: there is code description.
import torch
import torch.nn as nn

class MultiTaskLossWithMRL(nn.Module):
    '''
    mlm+pointwise+pairwise
    {query,doc_neg,doc_posi}

    '''
    def __init__(self,margin=1.0,train_type='post_pretrain'):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss() # pointwise
        self.mrl_loss = nn.MarginRankingLoss(margin=margin)
        self.train_type = train_type

    def forward(self,prediction_scores,labels,classification_logits,next_sentence_labels):
        # pred:[batch*seq,vocab_size],
        mlm_loss = self.criterion(prediction_scores.view(-1, 21128), labels.view(-1))
        pointwise_loss = self.bce_loss(classification_logits,next_sentence_labels)
        pairwise_loss = self.criterion()
        losses = mlm_loss + pointwise_loss + pairwise_loss
        return losses


class DoubleLoss(nn.Module):
    '''
    mlm+pointwise+pairwise
    '''
    def __init__(self,train_type='post_pretrain'):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss() # pointwise
        self.train_type = train_type

    def forward(self,prediction_scores=None,labels=None,classification_logits=None,next_sentence_labels=None):
        # pred:[batch*seq,vocab_size],
        if self.train_type =='post_pretrain':
            mlm_loss = self.ce_loss(prediction_scores.view(-1, 21128), labels.view(-1))
            pointwise_loss = self.bce_loss(classification_logits,next_sentence_labels)
            losses = mlm_loss+pointwise_loss
            return {'total_loss':losses,'mlm_loss':mlm_loss,"pointwise_loss":pointwise_loss}
        elif self.train_type =='finetune':
            pointwise_loss = self.bce_loss(classification_logits, next_sentence_labels)
            return {'total_loss': pointwise_loss, 'mlm_loss': 0.0, "pointwise_loss": pointwise_loss}
        else:
            raise ValueError('train_type must be post_pretrain or finetune')
if __name__ == '__main__':
    print('保持好心情！ ')
