import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def validate_args(args):
    assert args.mode in ['pretrain', 'finetune', 'rl', 'test']

    assert args.goal_state_change in ['none', 'smooth', 'finish', 'both']
    assert args.usr_act_type in ['oracle', 'gen']
    assert args.sys_act_type in ['oracle', 'gen', 'no_use']
    assert args.ft_method in ['naive', 'ewc', 'gem']
    if args.mode == 'finetune' and args.ft_method == 'ewc':
        assert args.fisher_sample > 0
        assert args.src_train_path != ''
        assert args.src_valid_path != ''
        assert args.src_test_path != ''
    assert args.max_dial_len % 2 == 0
    # dst
    assert args.dst_pred_type in ['nlu', 'bs']
    if args.dst_pred_type == 'nlu':
        assert args.max_slot_dec_len == 10
    else:  # bs
        assert args.max_slot_dec_len == 22

    if args.mode in ['finetune', 'rl']:
        model_name = args.model_dir + '/epoch-{}.pt'.format(str(args.load_epoch))
        assert os.path.exists(model_name)

    if args.mode == 'rl':
        assert args.rl_max_dial_len % 2 == 0
        assert args.dropout == 0  # turn on dropout will ruin generation quality during rl
        assert args.usr_act_type == 'gen' and args.sys_act_type == 'gen'
        #		assert args.rl_update in ['iterate', 'weighted_sum']
        #		if args.rl_update == 'weighted_sum':
        #			assert args.rl_iterate_ratio == 1

        assert args.entity_provide_reward >= 0 and args.no_entity_provide_reward <= 0
        assert args.no_repeat_ask_reward >= 0 and args.repeat_ask_reward <= 0
        assert args.no_miss_answer_reward >= 0 and args.miss_answer_reward <= 0
        assert args.correct_domain_reward >= 0 and args.wrong_domain_reward <= 0
        #		assert args.usr_correct_transit_reward >= 0 and args.usr_wrong_transit_reward <= 0
        #		assert args.usr_follow_goal_reward >= 0 and args.usr_not_follow_goal_reward <= 0
        #		assert args.usr_no_miss_answer_reward >= 0 and args.usr_miss_answer_reward <= 0
        assert args.usr_no_repeat_info_reward >= 0 and args.usr_repeat_info_reward <= 0
        assert args.usr_no_repeat_ask_reward >= 0 and args.usr_repeat_ask_reward <= 0
        assert args.usr_no_miss_answer_reward >= 0 and args.usr_miss_answer_reward <= 0
        assert args.reward_type in ['turn_reward', 'dialogue_reward']

    #	if args.mode == 'test':
    #		assert args.corpus_word_result != ''
    #		assert args.corpus_act_result != ''
    #		assert args.usr_word_result != ''
    #		assert args.usr_act_result != ''
    #		assert args.full_usr_word_result != ''
    #		assert args.full_usr_act_result != ''


def get_config():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--mode', type=str, required=True, help='running mode: pretrain, finetune, rl or test')

    # data path
    parser.add_argument('--data_path', type=str, default='data/process_data/delex.json')
    parser.add_argument('--ontology_goal_path', type=str, default='data/raw_data/ontology_goal.json')
    parser.add_argument('--train_path', type=str, default='data/process_data/train_dials.json')
    parser.add_argument('--valid_path', type=str, default='data/process_data/val_dials.json')
    parser.add_argument('--test_path', type=str, default='data/process_data/test_dials.json')
    parser.add_argument('--word2count_path', type=str, default='data/process_data/word2count.json')
    parser.add_argument('--act2count_path', type=str, default='data/process_data/act2count.json')

    # training config
    parser.add_argument('--shuffle', type=str2bool, default=True, help='whether to shuffle the data')
    parser.add_argument('--batch_size', type=int, default=150, help='number of dialogue turns in a batch')
    parser.add_argument('--eval_batch_size', type=int, default=100, help='number of dialogue turns in a batch during evaluation')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--no_improve_epoch', type=int, default=5, help='used for early stop')
    parser.add_argument('--seed', type=int, default=1122)
    parser.add_argument('--grad_clip', type=float, default=5)
    parser.add_argument('--model_dir', type=str, default='checkpoint/')

    # output file path
    parser.add_argument('--corpus_dst_result', type=str, default='')
    parser.add_argument('--corpus_word_result', type=str, default='')
    parser.add_argument('--corpus_act_result', type=str, default='')
    parser.add_argument('--usr_dst_result', type=str, default='')
    parser.add_argument('--usr_word_result', type=str, default='')
    parser.add_argument('--usr_act_result', type=str, default='')
    # parser.add_argument('--full_usr_dst_result', type=str, default='')
    # parser.add_argument('--full_usr_word_result', type=str, default='')
    # parser.add_argument('--full_usr_act_result', type=str, default='')
    parser.add_argument('--print_sample', type=int, default=1000, help='the number of printed dialogues')

    # model hyper-parameter
    parser.add_argument('--load_epoch', type=str, default='best')
    parser.add_argument('--train_size', type=int, default=10000, help='max number of train dialogues used')
    parser.add_argument('--vocab_size', type=int, default=1000, help='top frequent words as vocab')
    parser.add_argument('--embed_size', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=300)
    parser.add_argument('--db_size', type=int, default=12)
    parser.add_argument('--bs_size', type=int, default=37)
    parser.add_argument('--gs_size', type=int, default=65)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--num_layers', type=int, default=1, help='number of LSTM layers')
    parser.add_argument('--share_dial_rnn', type=str2bool, default=True, help='whether to share the dialogue level LSTM between two agents')
    parser.add_argument('--oracle_dst', type=str2bool, default=True, help='true for oracle belief state, otherwise predicted belief state')
    parser.add_argument('--usr_act_type', type=str, default='gen', help='how to use user act during testing: generated or oracle')
    parser.add_argument('--sys_act_type', type=str, default='gen', help='how to use sys act during testing: generated, oracle or not use')
    parser.add_argument('--goal_state_change', type=str, default='both', help='which goal state type to use: static, dynamic') # TODO

    # decoding
    parser.add_argument('--max_word_dec_len', type=int, default=50, help='max decoding length for word sequence')
    parser.add_argument('--max_act_dec_len', type=int, default=30, help='max decoding length for act seq') # 19 is max act len in the corpus
    parser.add_argument('--max_dial_len', type=int, default=30, help='max dialogue length during interaction, one utterance counts one, an even number')

    # user/agent interaction RL
    parser.add_argument('--update', type=str, default='joint', help='optimise user simulator and agent jointly or iteratively') # TODO
    parser.add_argument('--rl_lr', type=float, default=0.0001)
    parser.add_argument('--rl_max_dial_len', type=int, default=30, help='max dialogue during RL interaction')
    parser.add_argument('--rl_batch_size', type=int, default=10, help='number of dialogue in a batch during rl training')
    parser.add_argument('--rl_eval_batch_size', type=int, default=100, help='number of dialogue in a batch during interaction')
    parser.add_argument('--rl_dial_one_epoch', type=int, default=100, help='number of dialogues used before evaluation in rl')
    # parser.add_argument('--rl_iterate_ratio', type=int, default=1, help='ratio between rl update and sl update') # TODO
    parser.add_argument('--rl_iterate', type=str2bool, default=False, help='whether iterate between rl and sl')

    # dialogue agent reward
    parser.add_argument('--entity_provide_reward', type=float, default=0,
                        help='positive reward to encourage entity provision')
    parser.add_argument('--no_entity_provide_reward', type=float, default=0,
                        help='negative reward to penalise no entity provision')
    parser.add_argument('--repeat_ask_reward', type=float, default=0,
                        help='negative reward to penalise for repetitively reqesting')
    parser.add_argument('--no_repeat_ask_reward', type=float, default=0,
                        help='positive reward, the opposite to repeat_ask_reward')
    parser.add_argument('--miss_answer_reward', type=float, default=0,
                        help='negative reward to penalise for missing answer')
    parser.add_argument('--no_miss_answer_reward', type=float, default=0,
                        help='positive reward, the opposite to miss_answer_reward')

    # rewards about domain consistancy
    # TODO: check if used
    # parser.add_argument('--correct_domain_reward', type=float, default=0,
    #                     help='(+)encourage sys to be domain consistant to usr in a turn')
    # parser.add_argument('--wrong_domain_reward', type=float, default=0,
    #                     help='(-)punish sys being not domain consistant')

    # user simulator reward
    parser.add_argument('--rl_update_usr', type=str2bool, default=True, help='whether to update usr during rl')
    parser.add_argument('--rl_usr_epoch', type=int, default=5, help='update usr in how many epochs')
    parser.add_argument('--usr_repeat_info_reward', type=float, default=0,
                        help='negative reward to penalise repetitive informs')
    parser.add_argument('--usr_no_repeat_info_reward', type=float, default=0,
                        help='positive reward, the opposite to usr_repeat_info_reward')
    parser.add_argument('--usr_repeat_ask_reward', type=float, default=0,
                        help='negative reward to penalise repetitive requests')
    parser.add_argument('--usr_no_repeat_ask_reward', type=float, default=0,
                        help='positive reward, the opposite to usr_repeat_ask_reward')
    parser.add_argument('--usr_miss_answer_reward', type=float, default=0,
                        help='negative reward to penalise missing answers')
    parser.add_argument('--usr_no_miss_answer_reward', type=float, default=0,
                        help='positive reward, the opposite to usr_miss_answer_reward')

    # dialogue state tracking (DST)
    # TODO: check path
    parser.add_argument('--max_slot_dec_len', type=int, default=22, help='max decoding length for slot seq')
    parser.add_argument('--dst_vocab_size', type=int, default=2500, help='top frequent words as vocab for dst input')
    parser.add_argument('--dst_slot_list', type=str, default='data/dst/slot_list.json')
    parser.add_argument('--slot2value', type=str, default='data/dst/slot2value.json')
    parser.add_argument('--dst2count_path', type=str, default='data/dst/dst_word2count_sysLex.json')
    parser.add_argument('--dst_train_path', type=str, default='data/dst/train_dials.json')
    parser.add_argument('--dst_valid_path', type=str, default='data/dst/dev_dials.json')
    parser.add_argument('--dst_test_path', type=str, default='data/dst/test_dials.json')
    parser.add_argument('--fix_wrong_domain', type=str2bool, default=True, help='whether to fix wrong dst label by domain label')

    # tune
    parser.add_argument('--dst_hst_len', type=int, default=100, help='take how many tuns in dialogue history as input')
    parser.add_argument('--dst_pred_type', type=str, default='bs',
                        help='predict turn-level understanding or whole belief state')
    parser.add_argument('--separate_value_list', type=str2bool, default=False,
                        help='whether to use separate value list for each slot')
    parser.add_argument('--value_mask', type=str2bool, default=True, help='whether to use value mask on value logits')
    parser.add_argument('--remove_dontcare', type=str2bool, default=False,
                        help='whether to remove dontcare in the value list')
    parser.add_argument('--attn_prev_bs', type=str2bool, default=True,
                        help='whether to attend over belief state at previous turn.')



    parser.add_argument('--reward_type', type=str, default='turn_reward')

    # fine tune method
    parser.add_argument('--ft_method', type=str, default='naive') # finetune method
    parser.add_argument('--fisher_sample', type=int, default=0,
                        help='# of dialogues from source domain used for calculating fisher matrix')
    parser.add_argument('--ewc_lambda', type=float, default=0)
    parser.add_argument('--src_train_path', type=str, default='')
    parser.add_argument('--src_valid_path', type=str, default='')
    parser.add_argument('--src_test_path', type=str, default='')

    args = parser.parse_args()
    validate_args(args)
    print(args)
    return args