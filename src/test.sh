#!/bin/bash


checkpoint=$1

# best rl networks
model_dir='/home/bht26/rds/hpc-work/self-play/checkpoint/rl/5430' # provide this one
#model_dir='/home/bht26/rds/hpc-work/self-play/checkpoint/rl/5457'

######## test SL networks #######
# results of corpus interaction
corpus_word='result/pretrain/word/'$model_name'.json'
corpus_act='result/pretrain/act/'$model_name'.json'
corpus_dst='result/pretrain/dst/'$model_name'.json'
# results of agents' interaction
usr_word='result_usr/pretrain/word/'$model_name'.json'
usr_act='result_usr/pretrain/act/'$model_name'.json'
usr_dst='result_usr/pretrain/dst/'$model_name'.json'
#res='log/pretrain/'$model_name'.res'
python main.py --mode='test' --model_dir=$model_dir \
					     --embed_size=$dim --hidden_size=$dim \
				    	 --oracle_dst=$oracle_dst --attn_prev_bs=$prev_bs \
				    	 --corpus_word_result=$corpus_word --corpus_act_result=$corpus_act --corpus_dst_result=$corpus_dst \
					     --usr_word_result=$usr_word --usr_act_result=$usr_act --usr_dst_result=$usr_dst