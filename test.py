# 需在GPU环境运行
# 加载数据集过程可能由于网络原因失败，请尝试重新运行代码
from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
import tempfile
import os

tmp_dir = tempfile.TemporaryDirectory().name
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

# load dataset
ds = MsDataset.load('dureader-retrieval-ranking', 'zyznull')
train_ds = ds['train'].to_hf_dataset()
dev_ds = ds['dev'].to_hf_dataset()
model_id = '/Users/a58/Documents/wxb/workspace/semantics_relevance/pretrain_models/iic/nlp_rom_passage-ranking_chinese-base'
print(len(next(iter(train_ds))["positive_passages"]))
print(len(next(iter(train_ds))["negative_passages"]))
# def cfg_modify_fn(cfg):
#     cfg.task = 'text-ranking'
#     cfg['preprocessor'] = {'type': 'text-ranking'}
#     cfg['dataset'] = {
#         'train': {
#             'type': 'bert',
#             'query_sequence': 'query',
#             'pos_sequence': 'positive_passages',
#             'neg_sequence': 'negative_passages',
#             'text_fileds': ['text'],
#             'qid_field': 'query_id'
#         },
#         'val': {
#             'type': 'bert',
#             'query_sequence': 'query',
#             'pos_sequence': 'positive_passages',
#             'neg_sequence': 'negative_passages',
#             'text_fileds': ['text'],
#             'qid_field': 'query_id'
#         },
#     }
#     cfg['train']['neg_samples'] = 4
#     cfg['evaluation']['dataloader']['batch_size_per_gpu'] = 30
#     cfg.train.max_epochs = 1
#     cfg.train.train_batch_size = 4
#     cfg.train.hooks = [{
#         'type': 'TextLoggerHook',
#         'interval': 2
#     }, {
#         'type': 'IterTimerHook'
#     }, {
#         'type': 'EvaluationHook',
#         'by_epoch': False,
#         'interval': 1000
#     }]
#     return cfg
# kwargs = dict(
#     model=model_id,
#     train_dataset=train_ds,
#     work_dir=tmp_dir,
#     eval_dataset=dev_ds,
#     cfg_modify_fn=cfg_modify_fn)
# trainer = build_trainer(name=Trainers.nlp_text_ranking_trainer, default_args=kwargs)
# print(trainer)
# trainer.train()