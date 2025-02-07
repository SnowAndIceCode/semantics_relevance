# -*- coding: utf-8 -*-
# @Time    : 2024/12/29 19:15
# @Author  : Xuebing Wang
# @Email   : wangxuebing0212@163.com
# @FileName: run_train.py
# @Description: there is code description.
'''
    bert pretrain:
        MLM:
        NSP:
'''
import random
import time
import numpy as np
import os
import logging
import math
import torch
from torch.utils.data import random_split
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, BertForPreTraining, AutoModelForSequenceClassification, BertConfig
from modeling import PostTrainModel, FineTuningModel
from transformers.file_utils import WEIGHTS_NAME, CONFIG_NAME
from transformers import AdamW, get_scheduler
from data_helper import CustomDataset, get_dataLoader, testDataset
from tqdm.auto import tqdm
from config import parse_args
from metrics import cul_auc
from criteria import DoubleLoss
import pandas as pd

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

hadoop_cmds = "/usr/lib/software/hadoop/bin/hadoop"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def upload_to_hdfs(local_path, hdfs_path):
    local_path_split = os.path.split(local_path)
    flag = False
    ordrer = "{} fs -mkdir -p {}".format(hadoop_cmds, hdfs_path)
    os.system(ordrer)
    ordrer = "{} fs -test -e {}/{}".format(hadoop_cmds, hdfs_path, local_path_split[1])
    if os.system(ordrer) == 0:
        ordrer = "{} fs -rm -r {}/{}-bak".format(hadoop_cmds, hdfs_path, local_path_split[1])
        os.system(ordrer)
        ordrer = "{0} fs -mv {1}/{2} {1}/{2}-bak".format(hadoop_cmds, hdfs_path, local_path_split[1])
        os.system(ordrer)
    ordrer = "{} fs -put {} {}/".format(hadoop_cmds, local_path, hdfs_path)
    if os.system(ordrer) == 0:
        flag = True
    return flag


def export_model(save_model_dir, model):
    # 从训练的bestmodel导出模型
    best_path = './{}/best_super_epoch.bin'.format(save_model_dir)
    # 加载保存最好的模型权重
    try:
        model.load_state_dict(torch.load(best_path))
    except Exception as e:
        print("load weights Error:{}".format(e))
    # 导出目录
    export_path = save_model_dir + "/export_model/"

    # 使用 os.path.exists() 检查目录是否存在
    if not os.path.exists(export_path):
        # 如果目录不存在，则创建它
        os.makedirs(export_path)
        print(f"Directory {export_path} created.")
    else:
        print(f"Directory {export_path} already exists.")

    try:
        # 保存整个模型
        torch.save(model, export_path + 'model.pth')
        logger.info(f'model is saving:{model}')
        print("save model pb to {} success...".format(os.path.abspath(export_path)))
        print('Export model finished..')
    except Exception as e:
        print("save weights Error:{}".format(e))


def load_model(args, device='cpu'):
    if args.train_type == 'finetune':
        logger.info(f"pretrain path:{args.load_path}")
        pretrain_model = PostTrainModel.from_pretrained(args.load_path)
        model = FineTuningModel(pretrain_model.config, args.num_layer).to(device)
        # 加载ckpt
        if args.is_load == "True":
            model.bert.load_state_dict(pretrain_model.bert.state_dict())  # 加载postpretrain bert
            model.classification_head.load_state_dict(
                pretrain_model.classification_head.state_dict())  # 加载postpretrain 分类头

            logger.info(f'load model is {args.load_path}')

    elif args.train_type == 'post_pretrain':
        config = BertConfig.from_pretrained(args.pretrain_path)
        model = PostTrainModel(config, args.num_layer).to(device)
        if args.is_load == "True":
            model_path = args.load_path + 'best_pytorch_model.bin'
            model.load_state_dict(torch.load(model_path))
            logger.info(f'load model is {model_path}')
    elif args.train_type == 'pretrain':
        model = BertForPreTraining.from_pretrained(args.pretrain_path)
    elif args.train_type == 'raw':
        model = AutoModel.from_pretrained(args.pretrain_path)
    else:
        raise ValueError('train_type must be in [fintune, post_pretrain, pretrain]')

    return model


def train(args):
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # set seed
    set_seed(args.seed)

    # tokenzier
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_path)

    train_dataset = CustomDataset(args.train_file, tokenizer, args.max_length,args.train_type)  # dataset
    test_dataset = testDataset(args.test_file, tokenizer, args.max_length)  # dataset
    # train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])
    train_dataloader = get_dataLoader(args, train_dataset, batch_size=args.train_batch_size, shuffle=True)  # dataloader
    test_dataloader = get_dataLoader(args, test_dataset, batch_size=args.train_batch_size, shuffle=False)

    # 加载model
    model = load_model(args, device)

    t_total = len(train_dataloader) * args.num_train_epochs  # total step
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    args.warmup_steps = int(t_total * args.warmup_proportion)

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon
    )

    lr_scheduler = get_scheduler(
        'linear',
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total
    )

    critertion = DoubleLoss(train_type=args.train_type)

    # 保存模型
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 训练参数
    step_per_epoch = math.ceil(len(train_dataset) / args.train_batch_size)
    # Train!
    logger.info("***** Running training *****")
    logger.info(f"Num examples - {len(train_dataset)}")
    logger.info(f"Num Epochs - {args.num_train_epochs}")
    logger.info(f"Total optimization steps - {t_total}")
    logger.info(f"Step of Per Epochs is - {step_per_epoch}")

    # criteria
    total_loss = 0.
    global_step = 0
    best_metrics = 0
    tic_train = time.time()

    model.train()
    for epoch in range(1, args.num_train_epochs + 1):
        for step, batch in enumerate(train_dataloader, start=1):
            if args.train_type == 'post_pretrain':
                input_ids, attention_mask, token_type_ids, labels, next_sentence_labels = [x.to(device) for x in batch]
                prediction_scores, classification_logits = model(input_ids, attention_mask, token_type_ids)
                loss_dict = critertion(prediction_scores, labels, classification_logits, next_sentence_labels)
            else:
                input_ids, attention_mask, token_type_ids, next_sentence_labels = [x.to(device) for x in batch]
                classification_logits = model(input_ids, attention_mask, token_type_ids)
                loss_dict = critertion(classification_logits=classification_logits, next_sentence_labels=next_sentence_labels)

            loss, mlm_loss, pointwise_loss = loss_dict['total_loss'], loss_dict['mlm_loss'], loss_dict['pointwise_loss']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            total_loss += loss.item()

            global_step += 1
            model_to_save = model.module if hasattr(model, 'module') else model
            if global_step % args.logging_steps == 0:
                time_diff = time.time() - tic_train
                logger.info(
                    "global step: %d, epoch: %d, batch: %d, total_loss: %.4f, mlm_loss: %.4f, pointwise_loss: %.4f,time cost: %.2fs" %
                    (global_step, epoch, step, loss, mlm_loss, pointwise_loss, time_diff))

            with open(args.log_path, 'a+') as fw:
                if global_step % args.save_steps == 0:
                    if args.test_file:
                        auc, datainfo = evaluate(args, model, test_dataloader, device)
                        logging.info(f"epoch:{epoch}--global_step:{global_step}--" + "auc: %.4f" % (auc))
                        fw.write(f"epoch:{epoch}--global_step:{global_step}--" + "auc: %.4f" % (auc) + '\n')
                        if args.save_best:
                            if best_metrics < auc:
                                best_metrics = auc
                                logging.info(f"epoch:{epoch}--global_step:{global_step}--" + "bestmodel: %.4f" % (auc))
                                fw.write(
                                    f"epoch:{epoch}--global_step:{global_step}--" + "bestmodel: %.4f" % (auc) + '\n')
                                logging.info("Saving model")
                                torch.save(model_to_save.state_dict(),
                                           os.path.join(args.output_dir, "best_" + WEIGHTS_NAME))
                                model_to_save.config.to_json_file(os.path.join(args.output_dir, "best_" + CONFIG_NAME))

                if global_step % step_per_epoch == 0:  # 每个epoch结束
                    auc, datainfo = evaluate(args, model, test_dataloader, device)
                    logging.info(f"epoch:{epoch}--global_step:{global_step}--" + "%.4f" % (auc))
                    fw.write(f"epoch:{epoch}--global_step:{global_step}--" + "%.4f" % (auc) + '\n')
                    torch.save(model_to_save.state_dict(),
                               os.path.join(args.output_dir, f"epoch-{epoch}-" + WEIGHTS_NAME))
                    model_to_save.config.to_json_file(os.path.join(args.output_dir, f"epoch-{epoch}-" + CONFIG_NAME))

    # 结果保存至hdfs
    if args.upload_hdfs == 'True':
        if args.to_hdfs != "":
            if not upload_to_hdfs(args.output_dir, args.to_hdfs):
                print("model upload to hdfs fail !\nplease check {}".format(args.to_hdfs))
            else:
                print("model upload to hdfs -> {}".format(args.to_hdfs))

        export_model(args.save_model_dir, model)  # 导出模型


@torch.no_grad()
def evaluate(args, model, dataloader, device):
    """
    Evaluate model performance on a given dataset.
    Compute spearman correlation coefficient.
    """
    model.eval()
    preds, nsp_labels = [], []
    all_sessionid, all_infoid, all_query, all_doc = [], [], [], []
    all_pred_result = []
    for batch in tqdm(dataloader):
        batch_data, batch_sessionid, batch_infoid, batch_query, batch_doc = batch[:-4], batch[-4], batch[-3], batch[-2],batch[-1]
        input_ids, attention_mask, token_type_ids, next_sentence_labels = [x.to(device) for x in batch_data]

        if args.train_type == 'post_pretrain':
            prediction_scores, classification_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        elif args.train_type == 'finetune':
            classification_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        else:
            logger.info(f'{args.train_type} is not exist!')
            # seq_relationship_logits = outputs.seq_relationship_logits
            # nsp_probs = F.softmax(seq_relationship_logits, dim=-1) # [b,2]
            # nsp_predictions = torch.argmax(nsp_probs, dim=-1)

        preds.extend(classification_logits.cpu().numpy())  # 预测结果

        nsp_label_cpu = next_sentence_labels.cpu()
        # 阈值转化
        if args.is_eval_thr == 'True' and args.threshold != None:
            # logger.info(f"eval 集合label根据threshold--{args.threshold}--进行划分")
            nsp_label_cpu = torch.where(nsp_label_cpu >= args.threshold, torch.tensor(1), torch.tensor(0))

        nsp_labels.extend(nsp_label_cpu.numpy())  # 标签

        all_sessionid.extend(batch_sessionid)
        all_infoid.extend(batch_infoid)
        all_query.extend(batch_query)
        all_doc.extend(batch_doc)

    # 计算auc
    preds = np.array(preds)
    labels_array = np.array(nsp_labels)
    auc = cul_auc(labels_array, preds)

    if args.mode == 'predict':
        for sessionid, infoid, query, doc, pred, label in zip(all_sessionid, all_infoid, all_query, all_doc, preds,
                                                              labels_array):
            all_pred_result.append(
                {'sessionid': sessionid, 'infoid': infoid, 'query': query, 'doc': doc, 'pred': pred, 'label': label})

    model.train()
    return auc, all_pred_result


def predict(args):
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # tokenzier
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_path)

    test_dataset = testDataset(args.test_file, tokenizer, args.max_length)  # dataset
    test_dataloader = get_dataLoader(args, test_dataset, batch_size=args.train_batch_size, shuffle=False)

    # 加载model
    model = load_model(args, device)
    logger.info(f'model:{model}')

    auc, data_info = evaluate(args, model, test_dataloader, device)
    logger.info(f'auc:{auc}')

    eval_df = pd.DataFrame(data_info)
    eval_df.to_csv(args.test_result, index=False)
    logger.info(f'预测结果保存至--->{args.test_result}')


if __name__ == '__main__':
    print('保持好心情！ ')
    args = parse_args()
    logger.info(f'开始执行-->{args.mode}')
    if args.mode == 'train':
        train(args)
    elif args.mode == 'predict':
        predict(args)
    else:
        logger.info('模式不可用，可选模式为： [train,predict]')
