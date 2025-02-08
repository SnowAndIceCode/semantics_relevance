# -*- coding: utf-8 -*-
# @Time    : 2024/12/29 19:18
# @Author  : Xuebing Wang
# @Email   : wangxuebing0212@163.com
# @FileName: criteria.py
# @Description: there is code description.
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskLossWithRankNet(nn.Module):
    '''
    mlm+pointwise+pairwise
    {query,doc_neg,doc_posi}

    '''
    def __init__(self,train_type='post_pretrain'):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss() # pointwise
        self.train_type = train_type

    def cul_rankNet(self,posi_score,neg_score):
        y_loss = -F.logsigmoid(posi_score - neg_score)
        return y_loss.mean()

    def forward(self,neg_prediction_scores=None,neg_labels=None,posi_prediction_scores=None,posi_labels=None,neg_classification_logits=None,
                neg_nsp_label=None,posi_classification_logits=None,posi_nsp_label=None):
        # pred:[batch*seq,vocab_size],

        if self.train_type == 'post_pretrain':
            neg_mlm_loss = self.criterion(neg_prediction_scores.view(-1, 21128), neg_labels.view(-1))
            posi_mlm_loss = self.criterion(posi_prediction_scores.view(-1, 21128), posi_labels.view(-1))

            neg_pointwise_loss = self.bce_loss(neg_classification_logits, neg_nsp_label)
            posi_pointwise_loss = self.bce_loss(posi_classification_logits, posi_nsp_label)

            pairwise_loss = self.cul_rankNet(posi_classification_logits,neg_classification_logits)

            losses = neg_mlm_loss + posi_mlm_loss + neg_pointwise_loss + posi_pointwise_loss + pairwise_loss
            return {'total_loss': losses, 'mlm_loss': neg_mlm_loss + posi_mlm_loss, "pointwise_loss": neg_pointwise_loss + posi_pointwise_loss, "pairwise_loss": pairwise_loss}

        elif self.train_type == 'finetune':
            neg_pointwise_loss = self.bce_loss(neg_classification_logits, neg_nsp_label)
            posi_pointwise_loss = self.bce_loss(posi_classification_logits, posi_nsp_label)

            pairwise_loss = self.cul_rankNet(posi_classification_logits, neg_classification_logits)

            losses = neg_pointwise_loss + posi_pointwise_loss + pairwise_loss
            return {'total_loss': losses, 'mlm_loss': 0.0, "pointwise_loss":neg_pointwise_loss+posi_pointwise_loss,"pairwise_loss":pairwise_loss}
        else:
            raise ValueError('train_type must be post_pretrain or finetune')


class DoubleLoss(nn.Module):
    '''
    mlm+pointwise
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
