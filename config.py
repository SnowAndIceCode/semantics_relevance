# -*- coding: utf-8 -*-
# @Time    : 2024/12/29 19:16
# @Author  : Xuebing Wang
# @Email   : wangxuebing0212@163.com
# @FileName: config.py
# @Description: there is code description.

import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--train_file", default=None, type=str, required=True, help="The input training file.")
    parser.add_argument("--test_file", default=None, type=str, required=True, help="The input testing file.")
    parser.add_argument("--output_dir", default='./ckpt', type=str, required=True, help="模型保存路径", )

    parser.add_argument("--pretrain_path", default=None, type=str, required=True, help="The pretrain model path.")
    parser.add_argument("--model_type",default="bert", type=str, required=False)
    parser.add_argument('--is_load', default=False, type=str, help='是否加载模型权重')
    parser.add_argument("--load_path", default=None, type=str, required=True, help="加载模型路径")

    # Other parameters
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--train_batch_size", default=4, type=int)
    parser.add_argument("--max_length", default=64, type=int, required=True)
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")


    parser.add_argument("--adam_beta1", default=0.9, type=float,help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_beta2", default=0.98, type=float,help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")
    parser.add_argument("--weight_decay", default=0.01, type=float,help="Weight decay if we apply some.")

    parser.add_argument("--logging_steps",default=100,type=int, help="The interval steps to logging.")
    parser.add_argument("--save_steps",default=500,type=int,help="The interval steps to save checkpoints.")
    parser.add_argument("--save_best",action="store_true",help="Whether to save checkpoint on best evaluation performance.")

    # hdfs
    parser.add_argument('--upload_hdfs', default=False, type=str, help='是否将保存的模型存入hdfs')
    parser.add_argument("--to_hdfs", help="预测结果保存至hdfs的路径", default="")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    print('保持好心情！ ')
