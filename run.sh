#!/bin/bash


pip install -r requirements.txt &&
python run_train.py \
  --mode=train
  --train_file=dataset/241111-241211-pointwise_simple_neg-click_posi.csv  \
  --test_file=dataset/21Q4-22Q1Q2Q3-23Q1Q2Q3-evaled-all_process_new_dup_recorrect.csv \
  --output_dir=ckpt \
  --pretrain_path=pretrain_models/tiansz/bert-base-chinese \
  --is_load=False \
  --load_path=best_super_epoch.bin \
  --learning_rate=2e-5 \
  --num_train_epochs 3 \
  --train_batch_size=32 \
  --max_length=64 \
  --seed=42 \
  --logging_steps=10000 \
  --save_steps=10000 \
  --save_best \
  --to_hdfs=/home/wangxuebing/gte_v1/gte_v1_base_weight_18559_52 \
  --upload_hdfs=False