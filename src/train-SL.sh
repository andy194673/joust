#!/bin/bash

model_name=$1
model_dir='checkpoint/'$model_name

# dst option
oracle_dst=false; prev_bs=true
#oracle_dst=true; prev_bs=false # uncomment this line for oracle belief state setup

# running command
python main.py --mode='pretrain' --model_dir=$model_dir \
                --batch_size=100 --embed_size=150 --hidden_size=150 \
                --oracle_dst=$oracle_dst --attn_prev_bs=$prev_bs \
                --epoch=30 --no_improve_epoch=10
