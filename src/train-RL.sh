#!/bin/bash

sl_model_name=$1
sl_model_dir='checkpoint/'$sl_model_name
rl_model_name=$2
rl_model_dir='checkpoint/'$rl_model_name

# copy pre-trained model to avoid overwrite
mkdir -p $rl_model_dir
cp $sl_model_dir'/epoch-best.pt' $rl_model_dir

# dst option
oracle_dst=false; prev_bs=true
#oracle_dst=true; prev_bs=false # uncomment this line for oracle belief state setup

# running command
python main.py --mode='rl' --model_dir=$rl_model_dir \
               --embed_size=150 --hidden_size=150 \
               --oracle_dst=$oracle_dst --attn_prev_bs=$prev_bs \
               --dropout=0 --epoch=80 --no_improve_epoch=10 \
			         --entity_provide_reward=0 --no_entity_provide_reward=-5 \
			         --no_repeat_ask_reward=0 --repeat_ask_reward=-1 \
			         --no_miss_answer_reward=2.5 --miss_answer_reward=-5 \
			         --usr_no_repeat_info_reward=1 --usr_repeat_info_reward=-1 \
			         --usr_no_repeat_ask_reward=1 --usr_repeat_ask_reward=-1 \
			         --usr_no_miss_answer_reward=1 --usr_miss_answer_reward=-1 \
