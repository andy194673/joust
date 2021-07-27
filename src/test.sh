#!/bin/bash

checkpoint=$1

# dst option
oracle_dst=false; prev_bs=true
#oracle_dst=true; prev_bs=false # uncomment this line for oracle belief state setup

# create output folder
for folder_type in 'corpus_interact_output' 'user_interact_output'; do
  for result_type in 'dst' 'policy' 'nlg'; do
    mkdir -p $folder_type/$result_type
	done
done

# results of corpus interaction
corpus_dst='corpus_interact_output/dst/'$model_name'.json'
corpus_act='corpus_interact_output/policy/'$model_name'.json'
corpus_word='corpus_interact_output/nlg/'$model_name'.json'

# results of agent-agent interaction
usr_dst='user_interact_output/dst/'$model_name'.json'
usr_act='user_interact_output/policy/'$model_name'.json'
usr_word='user_interact_output/nlg/'$model_name'.json'

python main.py --mode='test' --model_dir=$model_dir \
					     --embed_size=150 --hidden_size=150 \
				    	 --oracle_dst=$oracle_dst --attn_prev_bs=$prev_bs \
				    	 --corpus_word_result=$corpus_word --corpus_act_result=$corpus_act --corpus_dst_result=$corpus_dst \
					     --usr_word_result=$usr_word --usr_act_result=$usr_act --usr_dst_result=$usr_dst