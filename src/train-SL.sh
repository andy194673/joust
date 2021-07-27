#!/bin/bash

model_name="full_train_SL"
batch_size=100; dim=150

# dst option
oracle_dst=false; prev_bs=true
#oracle_dst=true; prev_bs=false # uncomment this line for oracle belief state setup

###### train SL networks #######
model_dir='checkpoint/'$model_name
python main.py --mode='pretrain' --model_dir=$model_dir \
				       --batch_size=$batch_size --embed_size=$dim --hidden_size=$dim \
				       --oracle_dst=$oracle_dst --attn_prev_bs=$prev_bs \
				       --epoch=30 --no_improve_epoch=10 \
				       train_size=200
