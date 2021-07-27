import sys, time
import random
from tqdm import tqdm
import numpy as np
import torch

from utils.argument import get_config
from data_loader import DataLoader
from networks.model import Model
from evaluate import MultiWozEvaluator
from utils.util_dst import dict2list
from utils.util_writeout import write_sample, collect_dial, collect_dial_interact


def compute_fisher_matric(config, dataset):
	'''Compute fisher matric for EWC adaptation method'''
	model = Model(config, dataset)
	model = model.cuda()
	print('Load model in computing fisher')
	model.loadModel(config.model_dir, config.load_epoch)

	print('Computing Fisher Matrix')
	model.train()
	fisher, optpar = {}, {}
	for n, p in model.named_parameters():
		optpar[n] = torch.Tensor(p.cpu().data).cuda()
		p.data.zero_()
		fisher[n] = torch.Tensor(p.cpu().data).cuda()

	src_dataset = DataLoader(config, load_src=True)
	n_examples = 0
	while True:
		# get batch
		batch_list = src_dataset.next_batch_list('train')
		if batch_list == None:
			break

		for turn_idx, batch in enumerate(batch_list):
			n_examples += 1
			if turn_idx != 0:
				if config.share_dial_rnn: assert batch['init_dial_rnn'] == None
				if not config.share_dial_rnn: assert batch['init_dial_rnn']['usr'] == None and batch['init_dial_rnn']['sys'] == None
				batch['init_dial_rnn'] = decode_batch['init_dial_rnn']

			decode_batch = model(batch, turn_idx=turn_idx, mode='teacher_force')
			loss, update_loss = model.get_loss(batch)
			update_loss.backward(retain_graph=True)

			for n, p in model.named_parameters():
				if p.grad is not None:
					fisher[n].data += p.grad.data ** 2
			model.optimizer.zero_grad()
			print('Done collect {} examples in fisher matrix'.format(n_examples), file=sys.stderr)

	for name_f, _ in fisher.items():
		fisher[name_f] /= n_examples

	print('Done collect {} examples in fisher matrix'.format(n_examples))
	return fisher, optpar


def runBatchDialogue(batch_list, LOSS, dType, mode, decode_all, grad_list):
	'''Train both agents using a batch of dialogues'''
	def token2tensor(sent_batch, vocab):
		idx_batch = []
		for sent in sent_batch:
			assert isinstance(sent, str) or isinstance(sent, list)
			if isinstance(sent, str):
				sent += ' <EOS>'
				idx = [vocab[token] for token in sent.split()]
			if isinstance(sent, list):
				sent.append('<EOS>')
				idx = [vocab[token] for token in sent]
			idx_batch.append(idx)
		seq_len = dataset.pad_seq(idx_batch, vocab)
		return torch.tensor(idx_batch).long().cuda(), torch.tensor(seq_len).long().cuda()

	for turn_idx, batch in enumerate(batch_list):
		# previous ground-truth or prediction of act sequence
		if mode == 'gen' and config.sys_act_type == 'gen' and turn_idx != 0:
			batch['prev_act_idx']['usr'], batch['sent_len']['prev_act_usr'] = token2tensor(decode_batch['act_usr'], dataset.act_vocab)
			batch['prev_act_idx']['sys'], batch['sent_len']['prev_act_sys'] = token2tensor(decode_batch['act_sys'], dataset.act_vocab)

		# use previous generated word seq during evaluation
		if mode == 'gen' and turn_idx != 0:
			batch['word_idx']['ctx_usr'], batch['sent_len']['ctx_usr'] = token2tensor(decode_batch['word_sys'], dataset.vocab)

		# take init dialogue-level rnn from previous turn
		if turn_idx != 0:
			if config.share_dial_rnn: assert batch['init_dial_rnn'] == None
			if not config.share_dial_rnn: assert batch['init_dial_rnn']['usr'] == None and batch['init_dial_rnn']['sys'] == None
			batch['init_dial_rnn'] = decode_batch['init_dial_rnn']

		# dst, use predicted belief state during inference
		if not config.oracle_dst and mode == 'gen' and turn_idx != 0:
			slot_pred = [] # a list of list
			value_pred = [] # a list of list
			for bs in decode_batch['bs_pred']:
				bs = dict2list(bs) # list of slot value pair
				slots = [sv.split('=')[0] for sv in bs] # list of slot
				values = [sv.split('=')[1] for sv in bs] # list of value
				slot_pred.append( slots )
				value_pred.append( values )
			batch['dst_idx']['prev_bs_slot'], batch['sent_len']['prev_bs_slot'] = token2tensor(slot_pred, dataset.slot_vocab)
			batch['dst_idx']['prev_bs_value'], batch['sent_len']['prev_bs_value'] = token2tensor(value_pred, dataset.value_vocab['all'])

		# forward & update
		if mode == 'teacher_force':
			decode_batch = model(batch, turn_idx=turn_idx, mode='teacher_force')
			loss, update_loss = model.get_loss(batch)

			# update
			if dType == 'train':
				if config.mode == 'finetune' and config.ft_method == 'ewc':
					grad_norm = model.update_ewc(update_loss, config.ewc_lambda)
				else:
					grad_norm = model.update(update_loss, 'sl')
				grad_list.append(grad_norm)

			# collect loss
			if LOSS != None:
				for k, _ in loss.items():
					LOSS[k] += loss[k]
				LOSS['count'] += 1

		else: # generation mode
			decode_batch = model(batch, turn_idx=turn_idx, mode='gen')
			collect_dial(decode_all, decode_batch, 'usr', batch, turn_idx, dataset, config)
			collect_dial(decode_all, decode_batch, 'sys', batch, turn_idx, dataset, config)

	# check correct number of generated turns within a dialogue
	if mode == 'gen':
		for dial_name, dial_len in zip(batch_list[0]['dial_name'], batch_list[0]['dial_len']):
			for side in ['usr', 'sys']:
				for key in ['word', 'act']:
					assert dial_len == len(decode_all[dial_name][side]['ref_{}'.format(key)])
					assert dial_len == len(decode_all[dial_name][side]['gen_{}'.format(key)])
				if side == 'sys':
					key = 'bs'
					assert dial_len == len(decode_all[dial_name][side]['gen_{}'.format(key)])


def runRLOneEpoch(epoch_idx):
	'''Train both agents one epoch using reinforcement learning'''
	t0 = time.time()
	update_count = 0 # count of rl update
	while True:
		if (update_count * config.rl_batch_size) >= config.rl_dial_one_epoch:
			print('Done RL one epoch | Time: {:.1f}'.format(time.time()-t0))
			print('Done RL one epoch | Time: {:.1f}'.format(time.time()-t0), file=sys.stderr)
			break

		# run rl
		RL_LOSS = []
		for i in range(config.rl_iterate_ratio):
			# sample goals for interaction
			dial_name_batch = dataset.next_rl_batch() # a list of dial name
			if len(dial_name_batch) != config.rl_batch_size:
				dataset.init_rl()
				dial_name_batch = dataset.next_rl_batch()

			gen_dial_batch = model.interact(beam_search=False, dial_name_batch=dial_name_batch)

			# check maximum act seq of each dialogue
			max_act_len_batch = model.check_max_gen_act_seq(gen_dial_batch)

			if config.reward_type == 'turn_reward':
				avg_sys_r, avg_usr_r = model.get_turn_reward(gen_dial_batch)
			else:
				avg_sys_r, avg_usr_r = model.get_success_reward(gen_dial_batch, evaluator)

			rl_loss = model.get_rl_loss(gen_dial_batch, 'sys')
			grad_norm = model.update(rl_loss, 'rl_sys')

			if config.rl_update_usr and epoch_idx < config.rl_usr_epoch:
				rl_usr_loss = model.get_rl_loss(gen_dial_batch, 'usr')
				grad_usr_norm = model.update(rl_usr_loss, 'rl_usr')
			else:
				rl_usr_loss, grad_usr_norm = 0, 0

			update_count += 1
			gpu = torch.cuda.max_memory_allocated() // 1000000
			print('idx: {}, loss sys: {:.3f} usr: {:.3f} | avg reward sys: {:.3f} usr {:.3f} | grad sys: {:.2f} usr: {:.2f} | gpu: {} | max_act_len: {} -> avg: {}'.format(update_count, rl_loss, rl_usr_loss, avg_sys_r, avg_usr_r, grad_norm, grad_usr_norm, gpu, max_act_len_batch, np.mean(max_act_len_batch)))

			# trace generated dialogues
			# for gen_dial in gen_dial_batch:
			# 	for i, (act_usr, act_sys, word_usr, word_sys) in \
			# 			enumerate(zip(gen_dial['act_usr'], gen_dial['act_sys'], gen_dial['word_usr'], gen_dial['word_sys'])):
			# 		print('At side turn: {}'.format(i))
			# 		print('USR: {} ({})'.format(word_usr, act_usr))
			# 		print('SYS: {} ({})'.format(word_sys, act_sys))
			del gen_dial_batch
			del avg_sys_r, avg_usr_r
			del rl_loss, rl_usr_loss
			del grad_norm, grad_usr_norm
			torch.cuda.empty_cache()

		# run sl for one batch if iterate between sl and rl
		if config.rl_iterate:
			batch_list = dataset.next_batch_list('train')
			grad_list = []
			if batch_list == None:
				dataset.init()
				batch_list = dataset.next_batch_list('train')
			runBatchDialogue(batch_list, None, 'train', 'teacher_force', None, grad_list) # update by sl
			print('sl grad: {:.2f}'.format(np.mean(grad_list)))
			print('sl grad: {:.2f}'.format(np.mean(grad_list)), file=sys.stderr)
			del batch_list

		if update_count == 1:
			t1 = time.time()-t0
			print('update once: {:.1f}, estimate time rl one epoch: {:.1f}'.format(t1, config.rl_dial_one_epoch/config.rl_batch_size*t1))


def runOneEpoch(dType, epoch_idx, mode, beam_search=False):
	'''Train both agents one epoch using supervised learning'''
	t0 = time.time()
	LOSS = {
		'word_usr': 0, 'word_sys': 0,
		'act_usr': 0, 'act_sys': 0,
		'dst_slot': 0, 'dst_value': 0,
		'count': 0}
	n = 0
	grad_list, decode_all = [], dict()
	data_len = len(dataset.data[dType])
	n_batch = data_len // config.batch_size if dType == 'train' else data_len // config.eval_batch_size
	for _ in tqdm(range(n_batch)):
	# while True:
		# get a batch of dialogues
		batch_list = dataset.next_batch_list(dType)
		if batch_list == None:
			break

		runBatchDialogue(batch_list, LOSS, dType, mode, decode_all, grad_list)
		n += 1
		if n == 1 and epoch_idx == 0 and dType == 'train':
			print('{} dialogues takes {:.1f} sec, estimated time for an epoch: {:.1f}'
				  .format(config.batch_size, time.time()-t0, len(dataset.data[dType])/config.batch_size*(time.time()-t0) ), file=sys.stderr)
		print("batch list idx:", n, file=sys.stderr)

	if mode == 'teacher_force':
		n = LOSS['count']
		grad_norm = np.mean(grad_list) if len(grad_list) > 0 else 0
		print('{} Loss Epoch: {} | Word usr: {:.3f}, sys: {:.3f} | Act usr: {:.3f}, sys: {:.3f} | Dst slot: {:.3f}, value: {:.3f} | grad: {:.2f} | time: {:.1f}'.format(dType, epoch_idx, LOSS['word_usr']/n, LOSS['word_sys']/n, LOSS['act_usr']/n, LOSS['act_sys']/n, LOSS['dst_slot']/n, LOSS['dst_value']/n, grad_norm, time.time()-t0))

		total_loss = 0
		for k, v in LOSS.items():
			total_loss += v
		return total_loss/n

	else:
		# calcuate success, match and bleu
		print('# of decoded dialogus: {}'.format(len(decode_all)), file=sys.stderr)
		success, match, record = evaluator.context_to_response_eval(decode_all, dType)
		reqt_acc, reqt_total, reqt_record = evaluator.calculate_reqt_acc(decode_all, mode='fix_corpus')
		reward = evaluator.calculate_eval_reward(decode_all, model, mode='fix_corpus')

		bleu_usr, bleu_sys = evaluator.calculateBLEU(decode_all)
		score = 0.5*(success+match)+bleu_sys
		if not config.oracle_dst:
			joint_acc, sv_acc, slot_acc = evaluator.eval_dst(decode_all)
		else: # oracle dst
			joint_acc, sv_acc, slot_acc = 1, 1, 1
		print('{} Eval Epoch: {} | Score: {:.1f} | Success: {:.1f}, Match: {:.1f} | BLEU usr: {:.1f} sys: {:.1f} | DST joint_acc: {:.2f}%, sv_acc: {:.2f}%, slot_acc: {:.2f}% | reqt: {:.2f} ({}) | sys reward: {:.2f} {:.2f} {:.2f} {:.2f} | usr reward: {:.2f} {:.2f} {:.2f} | time: {:.0f}'.format(dType, epoch_idx, score, success, match, bleu_usr, bleu_sys, joint_acc*100, sv_acc*100, slot_acc*100, reqt_acc, reqt_total, reward['ent'], reward['ask'], reward['miss'], reward['dom'], reward['re_info'], reward['re_ask'], reward['miss_ans'], time.time()-t0))

		# write output sample
		if dType == 'test' and config.mode == 'test':
			res = {'success': success, 'match': match, 'bleu_sys': bleu_sys, 'bleu_usr': bleu_usr, 'score': score,
				   'reqt_acc': reqt_acc, 'reqt_total': reqt_total,
				   'dst_joint_acc': joint_acc*100, 'dst_sv_acc': sv_acc*100, 'dst_slot_acc': slot_acc*100}
			write_sample(config, decode_all, 'word', epoch_idx, config.corpus_word_result, record, reqt_record, res, reward)
			write_sample(config, decode_all, 'act', epoch_idx, config.corpus_act_result, record, reqt_record, res, reward)
			if not config.oracle_dst:
				write_sample(config, decode_all, 'dst', epoch_idx, config.corpus_dst_result, record, reqt_record, res, reward)
		return success, match, bleu_sys, score


def trainIter(config, dataset, CT):
	'''Training over epochs'''
	if config.mode == 'finetune' and config.ft_method == 'ewc':
		fisher, optpar = compute_fisher_matric(config, dataset)

	# test before finetune or rl
	if config.mode in ['finetune', 'rl']:
		print('Load model')
		CT.loadModel(config.model_dir, config.load_epoch)
		if config.mode == 'finetune' and config.ft_method == 'ewc':
			CT.fisher = fisher
			CT.optpar = optpar
		print('Test before doing finetune or rl')
		with torch.no_grad():
			test(config, dataset, CT) # check performance before doing finetune or rl
	print('-------------------------------------------------------------------------')

	best_score = -100
	no_improve_count = 0
	for epoch_idx in range(config.epoch):
		# train
		dataset.init()
		CT.train()
		if config.mode in ['pretrain', 'finetune']:
			_ = runOneEpoch('train', epoch_idx, 'teacher_force')
		else: # rl
			runRLOneEpoch(epoch_idx)

		# evaluate
		CT.eval()
		with torch.no_grad():
			if config.mode in ['pretrain', 'finetune']:
				dataset.init() # reset data pointer
				loss = runOneEpoch('valid', epoch_idx, 'teacher_force')

			if config.mode in ['pretrain', 'finetune']:
				success, match, bleu, score_usr = test_with_usr_simulator(config, dataset, CT, 'valid', tag='usr')

			elif config.mode == 'rl':
				dataset.init()
				success, match, bleu, score_auto = runOneEpoch('valid', epoch_idx, 'gen')

			if config.mode == 'rl': # pick the best model based on automatic evaluation on dev during interaction
				score = score_auto
			else: # pretrain, finetune, pick the best model based on interaction result during supervised learning
				score = score_usr

		# save model
		if score > best_score:
			dataset.init()
			runOneEpoch('test', epoch_idx, 'gen')
			best_score = score
			no_improve_count = 0
			print('Best score on validation!')
			CT.saveModel('best')
		else:
			no_improve_count += 1
		print('----------------------------------------------------------------------------')

		# early stop
		if no_improve_count > config.no_improve_epoch:
			print('Early stop!')
			break
	print('Done Training!')


def test(config, dataset, CT):
	'''Test the dialogue system against fixed test corpus'''
	# load checkpoint
	CT.loadModel(config.model_dir, config.load_epoch)

	# evaluate
	CT.eval()
	with torch.no_grad():
		# NOTE: uncomment here for generation on valid set
		# dataset.init()
		# runOneEpoch('valid', config.load_epoch, 'gen')

		# test with fixed corpus
		dataset.init()
		runOneEpoch('test', config.load_epoch, 'gen')

		# test with the trained user simulator
		test_with_usr_simulator(config, dataset, CT, 'valid', act_result=config.usr_act_result,
								word_result=config.usr_word_result, dst_result=config.usr_dst_result, tag='usr')


def test_with_usr_simulator(config, dataset, CT, dType, act_result=None, word_result=None, dst_result=None, scan_examples=False, tag=None):
	'''Test the dialogue agent against the user simulator'''
	beam_search = False
	# eval mode
	CT.eval() # turn off dropout

	# feed goals from corpus
	dial_name_all = [dial['dial_name'] for dial in dataset.data[dType]]

	dial_name_batch = []
	decode_all = {}
	t0 = time.time()
	p = 0
	while True:
		if p >= len(dial_name_all):
			break

		dial_name_batch = dial_name_all[p: min(p+config.rl_eval_batch_size, len(dial_name_all))]
		p += config.rl_eval_batch_size

		with torch.no_grad():
			gen_dial_batch = CT.interact(beam_search=beam_search, dial_name_batch=dial_name_batch)
		if p == config.rl_eval_batch_size:
			print('Finish 1 batch: {:.1f}'.format(time.time()-t0), file=sys.stderr)

		# trace generated dialogues
		if scan_examples:
			for gen_dial in gen_dial_batch:
				print('dial_name:', gen_dial['dial_name'])
				for i, (act_usr, act_sys, word_usr, word_sys) in \
						enumerate(zip(gen_dial['act_usr'], gen_dial['act_sys'], gen_dial['word_usr'], gen_dial['word_sys'])):
					print('At side turn: {}'.format(i))
					print('USR: {} ({})'.format(word_usr, act_usr))
					print('SYS: {} ({})'.format(word_sys, act_sys))
				input('press...')

		# form dummy batch
		batch = {}
		dial_len = [len(gen_dial['word_usr']) for gen_dial in gen_dial_batch]
		total_turns = sum(dial_len)
		batch['dial_len'] = torch.tensor(dial_len).long().cuda()
		batch['dial_name'] = [x for x in dial_name_batch]
		batch['ref'] = {'act': {}, 'word': {}}
		batch['ref']['act']['usr'] = batch['ref']['act']['sys'] = ['None' for _ in range(total_turns)] # because no ref in interaction
		batch['ref']['word']['usr'] = batch['ref']['word']['sys'] = ['None' for _ in range(total_turns)]
		batch['full_bs'] = ['None' for _ in range(total_turns)]

		decode_batch = {} # word_{side}, act_{side}, bs
		for key in ['bs', 'act_usr', 'act_sys', 'word_usr', 'word_sys', 'lex_word_usr', 'lex_word_sys', 'bs_pred']:
			decode_batch[key] = []
			for gen_dial in gen_dial_batch:
				decode_batch[key].extend(gen_dial[key])

		collect_dial_interact(decode_all, decode_batch, 'usr', batch, dataset, config)
		collect_dial_interact(decode_all, decode_batch, 'sys', batch, dataset, config)

	# evaluate generated dialogues
	success, match, record = evaluator.context_to_response_eval(decode_all, dType)
	reqt_acc, reqt_total, reqt_record = evaluator.calculate_reqt_acc(decode_all, mode='interaction')
	reward = evaluator.calculate_eval_reward(decode_all, CT, mode='interaction')

	bleu_usr = bleu_sys = 0
	score = 0.5*(success+match)+bleu_sys

	# like bleu, no reference for dst during interaction
	joint_acc, sv_acc, slot_acc = 0,0,0
	epoch_idx = 'usr' if tag == 'usr' else 'full_usr'
	print('{} Eval Epoch: {} | Score: {:.1f} | Success: {:.1f}, Match: {:.1f} | BLEU usr: {:.1f} sys: {:.1f} | DST joint_acc: {:.2f}%, sv_acc: {:.2f}%, slot_acc: {:.2f}% | reqt: {:.2f} ({}) | sys reward: {:.2f} {:.2f} {:.2f} {:.2f} | usr reward: {:.2f} {:.2f} {:.2f} | time: {:.0f}'.format(dType, epoch_idx, score, success, match, bleu_usr, bleu_sys, joint_acc*100, sv_acc*100, slot_acc*100, reqt_acc, reqt_total, reward['ent'], reward['ask'], reward['miss'], reward['dom'], reward['re_info'], reward['re_ask'], reward['miss_ans'], time.time()-t0))

	# write samples
	if config.mode == 'test' and act_result != None and word_result != None and dst_result != None:
		res = {'success': success, 'match': match, 'bleu_sys': bleu_sys, 'bleu_usr': bleu_usr, 'score': score,
			   'reqt_acc': reqt_acc, 'reqt_total': reqt_total, 'dst_joint_acc': joint_acc*100, 'dst_sv_acc': sv_acc*100, 'dst_slot_acc': slot_acc*100}
		write_sample(config, decode_all, 'word', epoch_idx, word_result, record, reqt_record, res, reward)
		write_sample(config, decode_all, 'act', epoch_idx, act_result, record, reqt_record, res, reward)
		if not config.oracle_dst:
			write_sample(config, decode_all, 'dst', epoch_idx, dst_result, record, reqt_record, res, reward)
	return success, match, bleu_sys, score


def set_seed(args):
	"""for reproduction"""
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.enabled = False
	torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
	# load config
	config = get_config()

	# set seed for reproduction
	set_seed(config)

	# load data
	dataset = DataLoader(config)
	evaluator = MultiWozEvaluator(dataset, config)

	# construct models in corpus training
	model = Model(config, dataset)
	model = model.cuda()
	
	# run training / testing
	if config.mode in ['pretrain', 'finetune', 'rl']:
		trainIter(config, dataset, model)
	else: # test
		test(config, dataset, model)