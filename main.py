from utils.argument import get_config
from data_loader import DataLoader

#from nn.sl4 import CorpusTraining
from nn.sl4_2 import CorpusTraining # including succ reward
from evaluate import MultiWozEvaluator
import time
import torch
import os
import json
import sys
import random
from tqdm import tqdm
import numpy as np
from utils.util_dst import dict2list
from nn.dst import DST


def write_sample(decode_all, src, epoch_idx, sample_file, record, reqt_record, res, reward):
	def two_digits(x):
		if x < 10:
			return '0'+str(x)
		else:
			return str(x)

	epoch_idx = 'Epoch-{}'.format(epoch_idx)
	out_f = sample_file
	if os.path.exists(out_f):
		with open(out_f) as f:
			sample = json.load(f)
	else:
		sample = {}

	# pairwise ref and decoded str in order
	sample[epoch_idx] = {}
	for dial_idx, dial_name in enumerate(sorted(decode_all.keys())):
		if dial_idx > config.print_sample:
			break

		dial = decode_all[dial_name]
		sample[epoch_idx][dial_name] = {}

		if src == 'dst':
			for i, (pred_nlu, pred_bs, ref_bs) in enumerate(zip(dial['sys']['pred_nlu'], dial['sys']['pred_bs'], dial['sys']['ref_bs'])):
				idx_sys = two_digits(2*i+1)
				sample[epoch_idx][dial_name]['{}-nlu({})'.format(idx_sys, 'gen')] = ' | '.join(pred_nlu)
				sample[epoch_idx][dial_name]['{}-bs({})'.format(idx_sys, 'gen')] = ' | '.join(pred_bs)
				sample[epoch_idx][dial_name]['{}-bs({})'.format(idx_sys, 'ref')] = ' | '.join(ref_bs)
				pred_bs, ref_bs = set(pred_bs), set(ref_bs)
				match_bs = pred_bs & ref_bs
				sample[epoch_idx][dial_name]['{}-{}'.format(idx_sys, 'miss')] = ' | '.join(sorted(list(ref_bs-match_bs)))
				sample[epoch_idx][dial_name]['{}-{}'.format(idx_sys, 'redt')] = ' | '.join(sorted(list(pred_bs-match_bs)))
				sample[epoch_idx][dial_name]['{}-{}'.format(idx_sys, 'MATCH')] = 1 if pred_bs==ref_bs else 0
			continue

		for i, (ref_usr, gen_usr, ref_sys, gen_sys) in enumerate(zip(dial['usr']['ref_'+src], dial['usr']['gen_'+src], dial['sys']['ref_'+src], dial['sys']['gen_'+src])):
			idx_usr = two_digits(2*i)
			idx_sys = two_digits(2*i+1)
			sample[epoch_idx][dial_name]['{}-usr({})'.format(idx_usr, 'gen')] = '{}'.format(gen_usr)
			sample[epoch_idx][dial_name]['{}-usr({})'.format(idx_usr, 'ref')] = '{}'.format(ref_usr)
			sample[epoch_idx][dial_name]['{}-sys({})'.format(idx_sys, 'gen')] = '{}'.format(gen_sys)
			sample[epoch_idx][dial_name]['{}-sys({})'.format(idx_sys, 'ref')] = '{}'.format(ref_sys)

		for metric, value in record[dial_name].items(): # metric=success or match
			sample[epoch_idx][dial_name][metric] = value

		sample[epoch_idx][dial_name]['--miss_reqt--'] = reqt_record[dial_name]

	sample['result'] = res
	sample['reward'] = reward
		
	with open(out_f, 'w') as f:
		json.dump(sample, f, indent=2, sort_keys=True)
	print('Done writing out', file=sys.stderr)


def runBatchDialogue(batch_list, LOSS, dType, mode, decode_all, grad_list):
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

	UPDATE_LOSS = 0
	for turn_idx, batch in enumerate(batch_list):
#		# prev GT or prediction during evalution
		# use previous generated act seq during evaluation
		if mode == 'gen' and config.sys_act_type == 'gen' and turn_idx != 0:
#			batch['prev_act_idx']['usr'], batch['sent_len']['prev_act_usr'] = token2tensor(decode_batch['act_usr'], 'act')
#			batch['prev_act_idx']['sys'], batch['sent_len']['prev_act_sys'] = token2tensor(decode_batch['act_sys'], 'act')
			batch['prev_act_idx']['usr'], batch['sent_len']['prev_act_usr'] = token2tensor(decode_batch['act_usr'], dataset.act_vocab)
			batch['prev_act_idx']['sys'], batch['sent_len']['prev_act_sys'] = token2tensor(decode_batch['act_sys'], dataset.act_vocab)

		# use previous generated word seq during evaluation
		if mode == 'gen' and turn_idx != 0:
#			batch['word_idx']['ctx_usr'], batch['sent_len']['ctx_usr'] = token2tensor(decode_batch['word_sys'], 'word')
			batch['word_idx']['ctx_usr'], batch['sent_len']['ctx_usr'] = token2tensor(decode_batch['word_sys'], dataset.vocab)

		# take init dial rnn from previous turn
		if turn_idx != 0:
			if config.share_dial_rnn: assert batch['init_dial_rnn'] == None
			if not config.share_dial_rnn: assert batch['init_dial_rnn']['usr'] == None and batch['init_dial_rnn']['sys'] == None
			batch['init_dial_rnn'] = decode_batch['init_dial_rnn']

		# dst, use predicted belief state during inference
		if not config.oracle_dst and mode == 'gen' and turn_idx != 0:
			slot_pred = [] # a list of list
			value_pred = [] # a list of list
#			for bs in bs_pred:
			for bs in decode_batch['bs_pred']:
				bs = dict2list(bs) # list of slot value pair
				slots = [sv.split('=')[0] for sv in bs] # list of slot
				values = [sv.split('=')[1] for sv in bs] # list of value
				slot_pred.append( slots )
				value_pred.append( values )
			batch['dst_idx']['prev_bs_slot'], batch['sent_len']['prev_bs_slot'] = token2tensor(slot_pred, dataset.slot_vocab)
			batch['dst_idx']['prev_bs_value'], batch['sent_len']['prev_bs_value'] = token2tensor(value_pred, dataset.value_vocab['all'])
			# batch['dst_idx']['dst_ctx'], batch['dst_idx']['dst_ctx_len'] =

		# forward & update
		if mode == 'teacher_force':
#			t = time.time()
			decode_batch = CT(batch, turn_idx=turn_idx, mode='teacher_force')
			loss, update_loss = CT.get_loss(batch)
#			dst(batch, turn_idx=turn_idx, mode='teacher_force')
#			loss, update_loss = dst.get_loss(batch)

			# update
			if dType == 'train':
				if config.mode == 'finetune' and config.ft_method == 'ewc':
					grad_norm = CT.update_ewc(update_loss, config.ewc_lambda)
				else:
					grad_norm = CT.update(update_loss, 'sl')
				grad_list.append(grad_norm)

			# collect loss
			if LOSS != None:
				for k, _ in loss.items():
					LOSS[k] += loss[k]
				LOSS['count'] += 1

		else: # generation mode
			decode_batch = CT(batch, turn_idx=turn_idx, mode='gen')
#			bs_pred, nlu_pred = dst(batch, turn_idx=turn_idx, mode='gen')
#			decode_batch = {'bs_pred': bs_pred, 'nlu_pred': nlu_pred}
			collect_dial(decode_all, decode_batch, 'usr', batch, turn_idx)
			collect_dial(decode_all, decode_batch, 'sys', batch, turn_idx)

			# check dst output at time
#			if mode == 'gen':
#				for b_idx in range(5):
#					bs = dict2list(bs_pred[b_idx])
#					print('In dialogue {} turn {}:'.format(batch['dial_name'][b_idx], turn_idx), file=sys.stderr)
#					print('ref:', dict2list(batch['ref']['dst'][b_idx]), file=sys.stderr)
#					print('gen:', bs, file=sys.stderr)
#					input('press...')

	# check correct number of generated turns within a dialogue
	if mode == 'gen':
		for dial_name, dial_len in zip(batch_list[0]['dial_name'], batch_list[0]['dial_len']):
			for side in ['usr', 'sys']:
				for key in ['word', 'act']:
					assert dial_len == len(decode_all[dial_name][side]['ref_{}'.format(key)])
					assert dial_len == len(decode_all[dial_name][side]['gen_{}'.format(key)])
				if side == 'sys':
					key = 'bs'
#					assert dial_len == len(decode_all[dial_name][side]['ref_{}'.format(key)])
					assert dial_len == len(decode_all[dial_name][side]['gen_{}'.format(key)])


def runOneEpoch(dType, epoch_idx, mode, beam_search=False):
	t0 = time.time()
	LOSS = {'word_usr': 0, 'word_sys': 0, \
			'act_usr': 0, 'act_sys': 0, \
			'dst_slot': 0, 'dst_value': 0, \
			'count': 0}
	n = 0
	grad_list = []
	decode_all = {}
	while True:
		# get batch
		batch_list = dataset.next_batch_list(dType)
		if batch_list == None:
			break

		runBatchDialogue(batch_list, LOSS, dType, mode, decode_all, grad_list)
#		del batch_list
#		torch.cuda.empty_cache()
		
		n += 1
		if n == 1 and epoch_idx == 0 and dType == 'train':
			print('{} dialogues takes {:.1f} sec, estimated time for an epoch: {:.1f}'.format(config.batch_size, time.time()-t0, \
					len(dataset.data[dType])/config.batch_size*(time.time()-t0) ), file=sys.stderr)
		print("batch list idx:", n, file=sys.stderr)

	if mode == 'teacher_force':
		n = LOSS['count']
		grad_norm = np.mean(grad_list) if len(grad_list) > 0 else 0
		print('{} Loss Epoch: {} | Word usr: {:.3f}, sys: {:.3f} | Act usr: {:.3f}, sys: {:.3f} | Dst slot: {:.3f}, value: {:.3f} | grad: {:.2f} | time: {:.1f}'.format(dType, epoch_idx, LOSS['word_usr']/n, LOSS['word_sys']/n, LOSS['act_usr']/n, LOSS['act_sys']/n, LOSS['dst_slot']/n, LOSS['dst_value']/n, grad_norm, time.time()-t0))
		print('{} Loss Epoch: {} | Word usr: {:.3f}, sys: {:.3f} | Act usr: {:.3f}, sys: {:.3f} | Dst slot: {:.3f}, value: {:.3f} | grad: {:.2f} | time: {:.1f}'.format(dType, epoch_idx, LOSS['word_usr']/n, LOSS['word_sys']/n, LOSS['act_usr']/n, LOSS['act_sys']/n, LOSS['dst_slot']/n, LOSS['dst_value']/n, grad_norm, time.time()-t0), file=sys.stderr)

		total_loss = 0
		for k, v in LOSS.items():
			total_loss += v
		return total_loss/n
	else:
		# calcuate success, match and bleu
		print('# of decoded dialogus: {}'.format(len(decode_all)), file=sys.stderr)
		success, match, record = evaluator.context_to_response_eval(decode_all, dType)
		reqt_acc, reqt_total, reqt_record = evaluator.calculate_reqt_acc(decode_all, mode='fix_corpus')
		reward = evaluator.calculate_eval_reward(decode_all, CT, mode='fix_corpus')

		bleu_usr, bleu_sys = evaluator.calculateBLEU(decode_all)
		score = 0.5*(success+match)+bleu_sys
		if not config.oracle_dst:
			joint_acc, sv_acc, slot_acc = evaluator.eval_dst(decode_all)
		else: # oracle dst
			joint_acc, sv_acc, slot_acc = 1, 1, 1

		print('{} Eval Epoch: {} | Score: {:.1f} | Success: {:.1f}, Match: {:.1f} | BLEU usr: {:.1f} sys: {:.1f} | DST joint_acc: {:.2f}%, sv_acc: {:.2f}%, slot_acc: {:.2f}% | reqt: {:.2f} ({}) | sys reward: {:.2f} {:.2f} {:.2f} {:.2f} | usr reward: {:.2f} {:.2f} {:.2f} | time: {:.0f}'.format(dType, epoch_idx, score, success, match, bleu_usr, bleu_sys, joint_acc*100, sv_acc*100, slot_acc*100, reqt_acc, reqt_total, reward['ent'], reward['ask'], reward['miss'], reward['dom'], reward['re_info'], reward['re_ask'], reward['miss_ans'], time.time()-t0))
		print('{} Eval Epoch: {} | Score: {:.1f} | Success: {:.1f}, Match: {:.1f} | BLEU usr: {:.1f} sys: {:.1f} | DST joint_acc: {:.2f}%, sv_acc: {:.2f}%, slot_acc: {:.2f}% | reqt: {:.2f} ({}) | sys reward: {:.2f} {:.2f} {:.2f} {:.2f} | usr reward: {:.2f} {:.2f} {:.2f} | time: {:.0f}'.format(dType, epoch_idx, score, success, match, bleu_usr, bleu_sys, joint_acc*100, sv_acc*100, slot_acc*100, reqt_acc, reqt_total, reward['ent'], reward['ask'], reward['miss'], reward['dom'], reward['re_info'], reward['re_ask'], reward['miss_ans'], time.time()-t0), file=sys.stderr)

		# write output sample
		if dType == 'test' and config.mode == 'test':
			res = {'success': success, 'match': match, 'bleu_sys': bleu_sys, 'bleu_usr': bleu_usr, 'score': score, \
				'reqt_acc': reqt_acc, 'reqt_total': reqt_total, 'dst_joint_acc': joint_acc*100, 'dst_sv_acc': sv_acc*100, 'dst_slot_acc': slot_acc*100}
			write_sample(decode_all, 'word', epoch_idx, config.corpus_word_result, record, reqt_record, res, reward)
			write_sample(decode_all, 'act', epoch_idx, config.corpus_act_result, record, reqt_record, res, reward)
			if not config.oracle_dst:
				write_sample(decode_all, 'dst', epoch_idx, config.corpus_dst_result, record, reqt_record, res, reward)
#			res = {'dst_joint_acc': joint_acc, 'dst_sv_acc': sv_acc, 'dst_slot_acc': slot_acc}
#			write_sample(decode_all, 'dst', epoch_idx, config.corpus_dst_result, None, None, res, None)

		return success, match, bleu_sys, score
#		return joint_acc, sv_acc, slot_acc

def collect_dial(decode_all, decode_batch, side, batch, turn_idx):
	for batch_idx, dial_name in enumerate(batch['dial_name']):
		if not batch['valid_turn'][batch_idx]:
			continue

		if dial_name not in decode_all:
#			decode_all[dial_name] = {'usr': {}, 'sys': {}}
			decode_all[dial_name] = {'usr': {}, 'sys': {}, 'goal': dataset.all_data[dial_name]['goal']}

		if turn_idx == 0:
			for k in ['ref_word', 'ref_act', 'gen_word', 'gen_act']:
				decode_all[dial_name][side][k] = []
			if side == 'sys':
				decode_all[dial_name][side]['gen_bs'] = []
				if not config.oracle_dst: # collect bs prediction
					decode_all[dial_name][side]['pred_bs'] = []
					decode_all[dial_name][side]['pred_nlu'] = []
					decode_all[dial_name][side]['ref_bs'] = []

		# reference
		decode_all[dial_name][side]['ref_word'].append( batch['ref']['word'][side][batch_idx] )
		decode_all[dial_name][side]['ref_act'].append( batch['ref']['act'][side][batch_idx] )

		# decode
		decode_all[dial_name][side]['gen_word'].append( decode_batch['word_{}'.format(side)][batch_idx] )
		if (side == 'usr' and config.usr_act_type == 'gen') or (side == 'sys' and config.sys_act_type == 'gen'): # gen act
			decode_all[dial_name][side]['gen_act'].append( decode_batch['act_{}'.format(side)][batch_idx] )
		else: # oracle act
			decode_all[dial_name][side]['gen_act'].append( batch['ref']['act'][side][batch_idx] )

		if side == 'sys':
			decode_all[dial_name][side]['gen_bs'].append( decode_batch['full_bs'][batch_idx] )
			if not config.oracle_dst: # collect bs prediction
				decode_all[dial_name][side]['pred_bs'].append( dict2list(decode_batch['bs_pred'][batch_idx]) ) # dict
				decode_all[dial_name][side]['pred_nlu'].append( dict2list(decode_batch['nlu_pred'][batch_idx]) ) # dict
				decode_all[dial_name][side]['ref_bs'].append( dict2list(batch['ref']['dst']['bs'][batch_idx]) )


def collect_dial_interact(decode_all, decode_batch, side, batch):
	'''
	collect decoded word, act seq and bs 
	'''
	dial_len, dial_name = batch['dial_len'], batch['dial_name']
	assert len(decode_batch['word_{}'.format(side)]) == torch.sum(dial_len).item()
	if (side == 'usr' and config.usr_act_type == 'gen') or (side == 'sys' and config.sys_act_type == 'gen'):
		assert len(decode_batch['act_{}'.format(side)]) == torch.sum(dial_len).item() # batch_size
	assert len(dial_len) == len(dial_name)

	for dial_idx, (_len, _name) in enumerate(zip(dial_len, dial_name)):
		start = torch.sum(dial_len[:dial_idx])
		if _name not in decode_all:
#			decode_all[_name] = {}
			decode_all[_name] = {'goal': dataset.all_data[_name]['goal']}

		decode_all[_name][side] = {}
		decode_all[_name][side]['dial_len'] = _len
#		decode_all[_name][side]['ref_word'] = batch['ref']['word'][side][start: start+_len] # none
		decode_all[_name][side]['ref_word'] = decode_batch['lex_word_{}'.format(side)][start: start+_len] # put lex word here
		decode_all[_name][side]['ref_act'] = batch['ref']['act'][side][start: start+_len]

		decode_all[_name][side]['gen_word'] = decode_batch['word_{}'.format(side)][start: start+_len]

		if (side == 'usr' and config.usr_act_type == 'gen') or (side == 'sys' and config.sys_act_type == 'gen'):
			decode_all[_name][side]['gen_act'] = decode_batch['act_{}'.format(side)][start: start+_len]
		else: # usr=oracle_act or sys=oracle_act/no_use
			decode_all[_name][side]['gen_act'] = decode_all[_name][side]['ref_act']

#		decode_all[_name][side]['gen_word'] = decode_all[_name][side]['ref_word']
#		decode_all[_name][side]['gen_act'] = decode_all[_name][side]['ref_act']

		if side == 'sys':
#			decode_all[_name][side]['ref_bs'] = batch['full_bs'][start: start+_len] # list of dict
			decode_all[_name][side]['gen_bs'] = decode_batch['bs'][start: start+_len]
			if not config.oracle_dst: # collect bs prediction
				decode_all[_name][side]['pred_bs'] = [ dict2list(bs_dict) for bs_dict in decode_batch['bs_pred'][start: start+_len] ]
				decode_all[_name][side]['pred_nlu'] = [ dict2list(bs_dict) for bs_dict in decode_batch['bs_pred'][start: start+_len] ]
				decode_all[_name][side]['ref_bs'] = [ dict2list(bs_dict) for bs_dict in decode_batch['bs_pred'][start: start+_len] ]


def compute_fisher_matric(config, dataset):
	model = CorpusTraining(config, dataset)
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
#			grad_norm = nn.utils.clip_grad_norm_(self.parameters(), self.config.grad_clip)

			for n, p in model.named_parameters():
				if p.grad is not None:
					fisher[n].data += p.grad.data ** 2
			model.optimizer.zero_grad()
			print('Done collect {} examples in fisher matrix'.format(n_examples), file=sys.stderr)

	for name_f, _ in fisher.items():
#		fisher[name_f] /= config.fisher_sample
		fisher[name_f] /= n_examples

	print('Done collect {} examples in fisher matrix'.format(n_examples))
	print('Done collect {} examples in fisher matrix'.format(n_examples), file=sys.stderr)
#	CT.fisher = fisher
#	CT.optpar = optpar
#	sys.exit(1)
	return fisher, optpar
	

def trainIter(config, dataset, CT):

	if config.mode == 'finetune' and config.ft_method == 'ewc':
		fisher, optpar = compute_fisher_matric(config, dataset)

	# test before finetune or rl
	if config.mode in ['finetune', 'rl']:
		print('Load model')
		CT.loadModel(config.model_dir, config.load_epoch)
		if config.mode == 'finetune' and config.ft_method == 'ewc':
			CT.fisher = fisher
			CT.optpar = optpar
		# BACK
		print('Test before doing finetune or rl')
		print('Test before doing finetune or rl', file=sys.stderr)
		with torch.no_grad():
			test(config, dataset, CT) # check performance before doing finetune or rl

	# # test with full usr if necessary
	# if config.full_usr_dir != '':
	# 	full_CT = CorpusTraining(config, dataset)
	# 	full_CT = full_CT.cuda()
	# 	full_CT.loadModel(config.full_usr_dir, 'best')
	print('-------------------------------------------------------------------------')
	print('-------------------------------------------------------------------------', file=sys.stderr)

	best_score = -100
	no_improve_count = 0
	for epoch_idx in range(config.epoch):
		# train
		dataset.init()
		CT.train()
#		dst.train()
		if config.mode in ['pretrain', 'finetune']:
			_ = runOneEpoch('train', epoch_idx, 'teacher_force')
		else: # rl
			runRLOneEpoch(epoch_idx)
#		print('Done one epoch', file=sys.stderr)
#		input('press...')

		# evaluate
		CT.eval()
#		dst.eval()
		with torch.no_grad():
			if config.mode in ['pretrain', 'finetune']:
				dataset.init() # reset data pointer
				loss = runOneEpoch('valid', epoch_idx, 'teacher_force')

#			# TEST
#			dataset.init()
#			runOneEpoch('test', epoch_idx, 'gen')

			if config.mode in ['pretrain', 'finetune']: # INTERACT with simulator trained together
				success, match, bleu, score_usr = test_with_usr_simulator(config, dataset, CT, 'valid', tag='usr')
			elif config.mode == 'rl': # VALID
				dataset.init()
				success, match, bleu, score_auto = runOneEpoch('valid', epoch_idx, 'gen')

			# INTERACT with pretrained simulator
			# if config.full_usr_dir != '':
			# 	full_CT.sys = CT.sys # use the real sys instead of sys trained with all data
			# 	test_with_usr_simulator(config, dataset, full_CT, 'valid', tag='full_usr')

#			score = score_auto + score_usr
			if config.mode == 'rl': # pick best model based on automatic evaluation on dev during interaction
				score = score_auto
			else: # pretrain, finetune, pick best model based on interaction result during supervised learning
				score = score_usr

		# save model
		if score > best_score:
			# TEST
			dataset.init()
			runOneEpoch('test', epoch_idx, 'gen')

#			if config.mode == 'rl':
#				success, match, bleu, score_usr = test_with_usr_simulator(config, dataset, CT, 'valid', tag='usr')

			best_score = score
			no_improve_count = 0
			print('Best score on validation!')
			print('Best score on validation!', file=sys.stderr)
#			CT.saveModel(str(epoch_idx))
			CT.saveModel('best')
		else:
			no_improve_count += 1
#		CT.saveModel('latest')
		print('----------------------------------------------------------------------------')
		print('----------------------------------------------------------------------------', file=sys.stderr)

		# early stop
		if no_improve_count > config.no_improve_epoch:
			print('Early stop!')
			print('Early stop!', file=sys.stderr)
			break

	print('Done Training!')
	print('Done Training!', file=sys.stderr)


def test(config, dataset, CT):
	# load checkpoint
	CT.loadModel(config.model_dir, config.load_epoch)

	# evaluate
	CT.eval()
	with torch.no_grad():
		# dataset.init()
		# runOneEpoch('valid', config.load_epoch, 'gen')

		# BACK
		dataset.init()
		runOneEpoch('test', config.load_epoch, 'gen')

		# test with usr trained with same data amount
		test_with_usr_simulator(config, dataset, CT, 'valid', act_result=config.usr_act_result, word_result=config.usr_word_result, \
									dst_result=config.usr_dst_result, tag='usr')
#
#		# test with full usr if necessary
#		if config.full_usr_dir != '':
#			full_CT = CorpusTraining(config, dataset)
#			full_CT = full_CT.cuda()
#			full_CT.loadModel(config.full_usr_dir, 'best')
#			full_CT.sys = CT.sys # use the real sys instead of sys trained with all data
#			test_with_usr_simulator(config, dataset, full_CT, 'valid', act_result=config.full_usr_act_result, \
#				word_result=config.full_usr_word_result, dst_result=config.full_usr_dst_result, tag='full_usr')


def runRLOneEpoch(epoch_idx):
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

			gen_dial_batch = CT.interact(beam_search=False, dial_name_batch=dial_name_batch)

			# check maximum act seq of each dialogue
			max_act_len_batch = CT.check_max_gen_act_seq(gen_dial_batch)

			if config.reward_type == 'turn_reward':
				avg_sys_r, avg_usr_r = CT.get_reward(gen_dial_batch)
			else:
				avg_sys_r, avg_usr_r = CT.get_success_reward(gen_dial_batch, evaluator)

			rl_loss = CT.get_rl_loss(gen_dial_batch, 'sys')
			grad_norm = CT.update(rl_loss, 'rl_sys')

			if config.rl_update_usr and epoch_idx < config.rl_usr_epoch:
				rl_usr_loss = CT.get_rl_loss(gen_dial_batch, 'usr')
				grad_usr_norm = CT.update(rl_usr_loss, 'rl_usr')
			else:
				rl_usr_loss, grad_usr_norm = 0, 0

			update_count += 1
			gpu = torch.cuda.max_memory_allocated() // 1000000 
			print('idx: {}, loss sys: {:.3f} usr: {:.3f} | avg reward sys: {:.3f} usr {:.3f} | grad sys: {:.2f} usr: {:.2f} | gpu: {} | max_act_len: {} -> avg: {}'.format(update_count, rl_loss, rl_usr_loss, avg_sys_r, avg_usr_r, grad_norm, grad_usr_norm, gpu, max_act_len_batch, np.mean(max_act_len_batch)))
			print('idx: {}, loss sys: {:.3f} usr: {:.3f} | avg reward sys: {:.3f} usr {:.3f} | grad sys: {:.2f} usr: {:.2f} | gpu: {} | max_act_len: {} -> avg: {}'.format(update_count, rl_loss, rl_usr_loss, avg_sys_r, avg_usr_r, grad_norm, grad_usr_norm, gpu, max_act_len_batch, np.mean(max_act_len_batch)), file=sys.stderr)

			# trace generated dialogues
#			for gen_dial in gen_dial_batch:
#				for i, (act_usr, act_sys, word_usr, word_sys) in \
#						enumerate(zip(gen_dial['act_usr'], gen_dial['act_sys'], gen_dial['word_usr'], gen_dial['word_sys'])):
#					print('At side turn: {}'.format(i))
#					print('USR: {} ({})'.format(word_usr, act_usr))
#					print('SYS: {} ({})'.format(word_sys, act_sys))
			del gen_dial_batch
			del avg_sys_r, avg_usr_r
			del rl_loss, rl_usr_loss
			del grad_norm, grad_usr_norm
			torch.cuda.empty_cache()

		if config.rl_iterate: # run sl for one batch
			batch_list = dataset.next_batch_list('train')
			grad_list = []
			if batch_list == None:
				dataset.init()
				batch_list = dataset.next_batch_list('train')
			runBatchDialogue(batch_list, None, 'train', 'teacher_force', None, grad_list) # update by sl
			print('sl grad: {:.2f}'.format(np.mean(grad_list)))
			print('sl grad: {:.2f}'.format(np.mean(grad_list)), file=sys.stderr)
			del batch_list

#		input('after sl update, press...')
#		torch.cuda.empty_cache()
#		decode_batch = CT(batch, mode='teacher_force')
#		_, sl_loss = CT.get_loss(batch)

#		# update
#		if config.rl_update == 'iterate':
#			for rl_loss in RL_LOSS:
#				CT.update(rl_loss, 'rl')
#				update_count += 1
##			CT.update(sl_loss, 'sl')
#
#		else: # weighted sum
#			raise NotImplementedError # since now sl update is inside the runBatchDialogue func
#			assert (RL_LOSS) == 1
#			rl_loss = RL_LOSS[0]
#			total_loss = config.rl_loss_weight * rl_loss + (1-config.rl_loss_weight) * sl_loss
#			CT.update(total_loss, 'rl')
#			update_count += 1

		if update_count == 1:
			t1 = time.time()-t0
			print('update once: {:.1f}, estimate time rl one epoch: {:.1f}'.format(t1, config.rl_dial_one_epoch/config.rl_batch_size*t1))
			print('update once: {:.1f}, estimate time rl one epoch: {:.1f}'.format(t1, config.rl_dial_one_epoch/config.rl_batch_size*t1), file=sys.stderr)
			


def test_with_usr_simulator(config, dataset, CT, dType, act_result=None, word_result=None, dst_result=None, scan_examples=False, tag=None):
	beam_search = False
	# load checkpoint
#	if load_model:
#		CT.loadModel(config.model_dir, config.load_epoch)

	# eval mode
	CT.eval() # turn off dropout

	# feed goals in train/dev/test
#	if scan_examples:
#		dial_name_all = ['MUL0001.json', 'MUL0201.json', 'MUL0401.json', 'MUL0018.json', 'MUL0869.json'] # examples
#	else:
	dial_name_all = [dial['dial_name'] for dial in dataset.data[dType]]

	dial_name_batch = []
	decode_all = {}
	t0 = time.time()
#	for dial_idx, dial_name in enumerate(dial_name_all):
#		dial_name_batch.append(dial_name)
	p = 0
	while True:
		if p >= len(dial_name_all):
			break

#		dial_name_batch = dial_name_all[p: min(p+config.rl_batch_size, len(dial_name_all))]
#		p += config.rl_batch_size
		dial_name_batch = dial_name_all[p: min(p+config.rl_eval_batch_size, len(dial_name_all))]
		p += config.rl_eval_batch_size

		with torch.no_grad():
			gen_dial_batch = CT.interact(beam_search=beam_search, dial_name_batch=dial_name_batch)
#		if p == config.rl_batch_size:
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
#		for key in ['bs', 'act_usr', 'act_sys', 'word_usr', 'word_sys']:
		for key in ['bs', 'act_usr', 'act_sys', 'word_usr', 'word_sys', 'lex_word_usr', 'lex_word_sys', 'bs_pred']:
			decode_batch[key] = []
			for gen_dial in gen_dial_batch:
				decode_batch[key].extend(gen_dial[key])

		collect_dial_interact(decode_all, decode_batch, 'usr', batch)
		collect_dial_interact(decode_all, decode_batch, 'sys', batch)

#		# init for next batch
#		dial_name_batch = []

	# evaluate generated dialogues
	success, match, record = evaluator.context_to_response_eval(decode_all, dType)
	reqt_acc, reqt_total, reqt_record = evaluator.calculate_reqt_acc(decode_all, mode='interaction')
	reward = evaluator.calculate_eval_reward(decode_all, CT, mode='interaction')

	bleu_usr = bleu_sys = 0
	score = 0.5*(success+match)+bleu_sys

	# like bleu, no reference for dst during interaction
	joint_acc, sv_acc, slot_acc = 0,0,0
#	epoch_idx = 'rl'
	epoch_idx = 'usr' if tag == 'usr' else 'full_usr'
	print('{} Eval Epoch: {} | Score: {:.1f} | Success: {:.1f}, Match: {:.1f} | BLEU usr: {:.1f} sys: {:.1f} | DST joint_acc: {:.2f}%, sv_acc: {:.2f}%, slot_acc: {:.2f}% | reqt: {:.2f} ({}) | sys reward: {:.2f} {:.2f} {:.2f} {:.2f} | usr reward: {:.2f} {:.2f} {:.2f} | time: {:.0f}'.format(dType, epoch_idx, score, success, match, bleu_usr, bleu_sys, joint_acc*100, sv_acc*100, slot_acc*100, reqt_acc, reqt_total, reward['ent'], reward['ask'], reward['miss'], reward['dom'], reward['re_info'], reward['re_ask'], reward['miss_ans'], time.time()-t0))
	print('{} Eval Epoch: {} | Score: {:.1f} | Success: {:.1f}, Match: {:.1f} | BLEU usr: {:.1f} sys: {:.1f} | DST joint_acc: {:.2f}%, sv_acc: {:.2f}%, slot_acc: {:.2f}% | reqt: {:.2f} ({}) | sys reward: {:.2f} {:.2f} {:.2f} {:.2f} | usr reward: {:.2f} {:.2f} {:.2f} | time: {:.0f}'.format(dType, epoch_idx, score, success, match, bleu_usr, bleu_sys, joint_acc*100, sv_acc*100, slot_acc*100, reqt_acc, reqt_total, reward['ent'], reward['ask'], reward['miss'], reward['dom'], reward['re_info'], reward['re_ask'], reward['miss_ans'], time.time()-t0), file=sys.stderr)

	# write samples
#	if act_sample_file != None and word_sample_file != None:
	if config.mode == 'test' and act_result != None and word_result != None and dst_result != None:
		res = {'success': success, 'match': match, 'bleu_sys': bleu_sys, 'bleu_usr': bleu_usr, 'score': score, \
				'reqt_acc': reqt_acc, 'reqt_total': reqt_total, 'dst_joint_acc': joint_acc*100, 'dst_sv_acc': sv_acc*100, 'dst_slot_acc': slot_acc*100}
		write_sample(decode_all, 'word', epoch_idx, word_result, record, reqt_record, res, reward)
		write_sample(decode_all, 'act', epoch_idx, act_result, record, reqt_record, res, reward)
		if not config.oracle_dst:
			write_sample(decode_all, 'dst', epoch_idx, dst_result, record, reqt_record, res, reward)

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
	CT = CorpusTraining(config, dataset)
	CT = CT.cuda()
	
	# start training / testing
	if config.mode in ['pretrain', 'finetune', 'rl']:
		trainIter(config, dataset, CT)
#		trainIter(config, dataset, dst)
	elif config.mode == 'test':
		test(config, dataset, CT)
#		test(config, dataset, dst)
	else: # test with usr simulator
		for dType in ['test']:
			test_with_usr_simulator(config, dataset, CT, dType, act_sample_file=config.act_sample_file, word_sample_file=config.word_sample_file, scan_examples=True)

