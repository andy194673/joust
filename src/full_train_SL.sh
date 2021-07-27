model_name="full_train_SL"
batch_size=100; dim=150

# dst option
oracle_dst=false; prev_bs=true
#oracle_dst=true; prev_bs=false # use this for oracle dst setup

###### train SL networks #######
model_dir='checkpoint/pretrain/'$model_name
log='log/pretrain/'$model_name'.log'
python main.py --mode='pretrain' --model_dir=$model_dir \
				       --batch_size=$batch_size --embed_size=$dim --hidden_size=$dim \
				       --oracle_dst=$oracle_dst --attn_prev_bs=$prev_bs \
				       --epoch=30 --no_improve_epoch=10 > $log
exit


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