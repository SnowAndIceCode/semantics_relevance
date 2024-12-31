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
import torch
from torch.utils.data import random_split
import torch.nn.functional  as F
from transformers import AutoTokenizer,BertForPreTraining
from transformers.file_utils import WEIGHTS_NAME, CONFIG_NAME
from transformers import AdamW,get_scheduler
from data_helper import CustomDataset,get_dataLoader
from tqdm.auto import tqdm
from config import parse_args
from metrics import cul_auc

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
        #保存整个模型
        torch.save(model,export_path+'model.pth')
        logger.info(f'model is saving:{model}')
        print("save model pb to {} success...".format(os.path.abspath(export_path)))
        print('Export model finished..')
    except Exception as e:
        print("save weights Error:{}".format(e))


def train(args):
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # set seed
    set_seed(args.seed)

    # tokenzier
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_path)


    train_dataset = CustomDataset(args.train_file, tokenizer, args.max_length)  # dataset
    test_dataset = CustomDataset(args.test_file, tokenizer, args.max_length)  # dataset
    # train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])
    train_dataloader = get_dataLoader(args, train_dataset,batch_size=args.train_batch_size, shuffle=True) # dataloader
    test_dataloader = get_dataLoader(args, test_dataset,batch_size=args.train_batch_size,shuffle=False)

    # 加载model
    model = BertForPreTraining.from_pretrained(args.pretrain_path).to(device)

    # 加载ckpt
    if args.is_load == "True":
        model.load_state_dict(torch.load(args.load_path))
        logger.info(f'load model is {args.load_path}')

    t_total = len(train_dataloader) * args.num_train_epochs # total step
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

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"Num examples - {len(train_dataset)}")
    logger.info(f"Num Epochs - {args.num_train_epochs}")
    logger.info(f"Total optimization steps - {t_total}")

    # criteria
    total_loss = 0.
    global_step = 0
    best_metrics = 0
    tic_train = time.time()

    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

    model.train()
    for epoch in range(1, args.num_train_epochs + 1):
        for step, batch in enumerate(train_dataloader, start=1):
            input_ids, attention_mask, token_type_ids, labels, next_sentence_labels = [x.to(device) for x in batch]
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
                next_sentence_label=next_sentence_labels
            )
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            total_loss += loss.item()

            global_step += 1
            if global_step % args.logging_steps == 0:
                time_diff = time.time() - tic_train
                logger.info("global step: %d, epoch: %d, batch: %d, loss: %.4f, time cost: %.2fs" %
                            (global_step, epoch, step, loss, time_diff))

            if (global_step % args.save_steps == 0 or global_step == t_total):
                model_to_save = model.module if hasattr(model, 'module') else model
                if args.eval_file:
                    auc = evaluate(model, test_dataloader, device)
                    logging.info("auc: %.4f" % (auc))
                    if args.save_best:
                        if best_metrics < auc:
                            best_metrics = auc
                            logging.info("Saving student model")
                            torch.save(model_to_save.state_dict(), output_model_file)
                            model_to_save.config.to_json_file(output_config_file)
                        continue
                logging.info("Saving student model")
                torch.save(model_to_save.state_dict(), output_model_file)
                model_to_save.config.to_json_file(output_config_file)

    # 结果保存至hdfs
    if args.upload_hdfs == 'True':
        if args.to_hdfs != "":
            if not upload_to_hdfs(args.output_dir, args.to_hdfs):
                print("model upload to hdfs fail !\nplease check {}".format(args.to_hdfs))
            else:
                print("model upload to hdfs -> {}".format(args.to_hdfs))

        export_model(args.save_model_dir, model) # 导出模型

@torch.no_grad()
def evaluate(model, dataloader, device):
    """
    Evaluate model performance on a given dataset.
    Compute spearman correlation coefficient.
    """
    model.eval()
    preds, nsp_labels = [], []
    data_info =[]
    for batch in tqdm(dataloader):
        batch_data,batch_info = batch[:-4],batch[-4:]
        input_ids, attention_mask, token_type_ids,next_sentence_labels = [x.to(device) for x in batch_data]
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        seq_relationship_logits = outputs.seq_relationship_logits
        pred = F.softmax(seq_relationship_logits, dim=-1)

        preds.extend(pred.cpu().numpy())
        nsp_labels.extend(next_sentence_labels.cpu().numpy())

    # 计算auc
    preds = np.array(preds)
    labels_array = np.array(nsp_labels)
    auc = cul_auc(labels_array, preds[:, 1])


    model.train()
    return auc,data_info

def predict(args):
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # tokenzier
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_path)

    test_dataset = CustomDataset(args.test_file, tokenizer, args.max_length)  # dataset
    test_dataloader = get_dataLoader(args, test_dataset,batch_size=args.train_batch_size,shuffle=False)

    # 加载model
    model = BertForPreTraining.from_pretrained(args.pretrain_path).to(device)

    # 加载ckpt
    if args.is_load == "True":
        model.load_state_dict(torch.load(args.load_path))
        logger.info(f'load model is {args.load_path}')

    auc,data_info = evaluate(model, test_dataloader, device)

if __name__ == '__main__':
    print('保持好心情！ ')
    args = parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'predict':
        predict(args)
    else:
        logger.info('模式不可用，可选模式为： [train,predict]')
