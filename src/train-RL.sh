model_name="full_train_RL"
dim=150

# dst
oracle_dst=false
prev_bs=true

# sys reward
ent_pos=0; ent_neg=-5
ask_pos=0; ask_neg=-1
miss_pos=2.5; miss_neg=-5
dom_pos=0
dom_neg=0

# usr reward
usr_epoch=5
update_usr=true
usr_info_pos=1
usr_info_neg=-1
usr_ask_pos=0
usr_ask_neg=-1
usr_ans_pos=1
usr_ans_neg=-1

###### pretrain #######
model_dir_pretrain='checkpoint/pretrain/sota-pretrain_pct-'$pct'_oracleDST-'$oracle_dst'_prevBS-'$prev_bs'_share-'$share_dial_rnn'_seed'$seed
pretrain_train='data/MultiWOZ/self-play-fix2/train_dials_pct'$pct'.json'
pretrain_val='data/MultiWOZ/self-play-fix2/val_dials.json'
pretrain_test='data/MultiWOZ/self-play-fix2/test_dials.json'


######### rl ########
model_dir_rl='checkpoint/rl/'$model_name
rm -r $model_dir_rl
mkdir -p $model_dir_rl
cp $model_dir_pretrain'/epoch-best.pt' $model_dir_rl
log='log/rl/'$model_name'.log'
python3 main-sl4.py --mode='rl' --model_dir=$model_dir_rl \
			--embed_size=$dim --hidden_size=$dim \
			--oracle_dst=$oracle_dst --attn_prev_bs=$prev_bs \
			--dropout=0 --epoch=80 --no_improve_epoch=10 \
			--batch_size=50 \
			--rl_iterate=$iterate \
			--entity_provide_reward=$ent_pos --no_entity_provide_reward=$ent_neg \
			--no_repeat_ask_reward=$ask_pos --repeat_ask_reward=$ask_neg \
			--no_miss_answer_reward=$miss_pos --miss_answer_reward=$miss_neg \
            --correct_domain_reward=$dom_pos --wrong_domain_reward=$dom_neg \
            --rl_update_usr=$update_usr --rl_usr_epoch=$usr_epoch \
			--usr_no_repeat_info_reward=$usr_info_pos --usr_repeat_info_reward=$usr_info_neg \
			--usr_no_repeat_ask_reward=$usr_ask_pos --usr_repeat_ask_reward=$usr_ask_neg \
			--usr_no_miss_answer_reward=$usr_ans_pos --usr_miss_answer_reward=$usr_ans_neg \
			--train_path=$pretrain_train --valid_path=$pretrain_val --test_path=$pretrain_test > $log

######## test rl #######
corpus_word='result/rl/word/'$model_name'.json'
corpus_act='result/rl/act/'$model_name'.json'
corpus_dst='result/rl/dst/'$model_name'.json'
usr_word='result_usr/rl/word/'$model_name'.json'
usr_act='result_usr/rl/act/'$model_name'.json'
usr_dst='result_usr/rl/dst/'$model_name'.json'
rm $corpus_word $corpus_act $corpus_dst $usr_word $usr_act $usr_dst
res='log/rl/'$model_name'.res'
python3 main-sl4.py --mode='test' --model_dir=$model_dir_rl \
					--embed_size=$dim --hidden_size=$dim \
					--oracle_dst=$oracle_dst --attn_prev_bs=$prev_bs \
					--entity_provide_reward=$ent_pos --no_entity_provide_reward=$ent_neg \
					--no_repeat_ask_reward=$ask_pos --repeat_ask_reward=$ask_neg \
					--no_miss_answer_reward=$miss_pos --miss_answer_reward=$miss_neg \
		            --correct_domain_reward=$dom_pos --wrong_domain_reward=$dom_neg \
        		    --rl_update_usr=$update_usr --rl_usr_epoch=$usr_epoch \
					--usr_no_repeat_info_reward=$usr_info_pos --usr_repeat_info_reward=$usr_info_neg \
					--usr_no_repeat_ask_reward=$usr_ask_pos --usr_repeat_ask_reward=$usr_ask_neg \
					--usr_no_miss_answer_reward=$usr_ans_pos --usr_miss_answer_reward=$usr_ans_neg \
					--corpus_word_result=$corpus_word --corpus_act_result=$corpus_act --corpus_dst_result=$corpus_dst \
					--usr_word_result=$usr_word --usr_act_result=$usr_act --usr_dst_result=$usr_dst \
					--train_path=$pretrain_train --valid_path=$pretrain_val --test_path=$pretrain_test > $res