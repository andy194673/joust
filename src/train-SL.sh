#!/bin/bash

model_name="full_train_SL-2"
batch_size=100; dim=150

# dst option
oracle_dst=false; prev_bs=true
#oracle_dst=true; prev_bs=false # uncomment this line for oracle belief state setup

# remove this
tmp='/home/bht26/rds/hpc-work/self-play'

###### train SL networks #######
model_dir='checkpoint/pretrain/'$model_name
log='log/pretrain/'$model_name'.log'
python main.py --mode='pretrain' --model_dir=$model_dir \
				       --batch_size=$batch_size --embed_size=$dim --hidden_size=$dim \
				       --oracle_dst=$oracle_dst --attn_prev_bs=$prev_bs \
				       --data_path=$tmp/data/MultiWOZ/self-play-fix2/delex.json \
				       --ontology_goal_path=$tmp/data/MultiWOZ/ontology_goal.json \
				       --train_path=$tmp/data/MultiWOZ/self-play-fix2/train_dials_pct100.json \
				       --valid_path=$tmp/data/MultiWOZ/self-play-fix2/val_dials.json \
				       --test_path=$tmp/data/MultiWOZ/self-play-fix2/test_dials.json \
				       --word2count_path=$tmp/data/MultiWOZ/self-play-fix2/word2count.json \
				       --act2count_path=$tmp/data/MultiWOZ/self-play-fix2/act2count.json \
				       --epoch=30 --no_improve_epoch=10 #> $log