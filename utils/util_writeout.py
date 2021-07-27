import os, sys
import json


def write_sample(decode_all, src, epoch_idx, sample_file, record, reqt_record, res, reward):
	'''Method to dump model generation'''
	def two_digits(x):
		if x < 10:
			return '0' + str(x)
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
				idx_sys = two_digits(2 * i + 1)
				sample[epoch_idx][dial_name]['{}-nlu({})'.format(idx_sys, 'gen')] = ' | '.join(pred_nlu)
				sample[epoch_idx][dial_name]['{}-bs({})'.format(idx_sys, 'gen')] = ' | '.join(pred_bs)
				sample[epoch_idx][dial_name]['{}-bs({})'.format(idx_sys, 'ref')] = ' | '.join(ref_bs)
				pred_bs, ref_bs = set(pred_bs), set(ref_bs)
				match_bs = pred_bs & ref_bs
				sample[epoch_idx][dial_name]['{}-{}'.format(idx_sys, 'miss')] = ' | '.join(sorted(list(ref_bs - match_bs)))
				sample[epoch_idx][dial_name]['{}-{}'.format(idx_sys, 'redt')] = ' | '.join(sorted(list(pred_bs - match_bs)))
				sample[epoch_idx][dial_name]['{}-{}'.format(idx_sys, 'MATCH')] = 1 if pred_bs == ref_bs else 0
			continue

		for i, (ref_usr, gen_usr, ref_sys, gen_sys) in enumerate(zip(dial['usr']['ref_' + src], dial['usr']['gen_' + src], dial['sys']['ref_' + src],dial['sys']['gen_' + src])):
			idx_usr = two_digits(2 * i)
			idx_sys = two_digits(2 * i + 1)
			sample[epoch_idx][dial_name]['{}-usr({})'.format(idx_usr, 'gen')] = '{}'.format(gen_usr)
			sample[epoch_idx][dial_name]['{}-usr({})'.format(idx_usr, 'ref')] = '{}'.format(ref_usr)
			sample[epoch_idx][dial_name]['{}-sys({})'.format(idx_sys, 'gen')] = '{}'.format(gen_sys)
			sample[epoch_idx][dial_name]['{}-sys({})'.format(idx_sys, 'ref')] = '{}'.format(ref_sys)

		for metric, value in record[dial_name].items():  # metric=success or match
			sample[epoch_idx][dial_name][metric] = value

		sample[epoch_idx][dial_name]['--miss_reqt--'] = reqt_record[dial_name]

	sample['result'] = res
	sample['reward'] = reward

	with open(out_f, 'w') as f:
		json.dump(sample, f, indent=2, sort_keys=True)
	print('Done writing out model generation!')


def collect_dial(decode_all, decode_batch, side, batch, turn_idx):
	'''Method to collect model generation in inference (interact with fixed test corpus)'''
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
	'''Method to collect model generation in interaction between two agents'''
	dial_len, dial_name = batch['dial_len'], batch['dial_name']
	assert len(decode_batch['word_{}'.format(side)]) == torch.sum(dial_len).item()
	if (side == 'usr' and config.usr_act_type == 'gen') or (side == 'sys' and config.sys_act_type == 'gen'):
		assert len(decode_batch['act_{}'.format(side)]) == torch.sum(dial_len).item() # batch_size
	assert len(dial_len) == len(dial_name)

	for dial_idx, (_len, _name) in enumerate(zip(dial_len, dial_name)):
		start = torch.sum(dial_len[:dial_idx])
		if _name not in decode_all:
			decode_all[_name] = {'goal': dataset.all_data[_name]['goal']}

		decode_all[_name][side] = {}
		decode_all[_name][side]['dial_len'] = _len
		decode_all[_name][side]['ref_word'] = decode_batch['lex_word_{}'.format(side)][start: start+_len] # put lex word here
		decode_all[_name][side]['ref_act'] = batch['ref']['act'][side][start: start+_len]
		decode_all[_name][side]['gen_word'] = decode_batch['word_{}'.format(side)][start: start+_len]

		if (side == 'usr' and config.usr_act_type == 'gen') or (side == 'sys' and config.sys_act_type == 'gen'):
			decode_all[_name][side]['gen_act'] = decode_batch['act_{}'.format(side)][start: start+_len]
		else: # usr=oracle_act or sys=oracle_act/no_use
			decode_all[_name][side]['gen_act'] = decode_all[_name][side]['ref_act']

		if side == 'sys':
			decode_all[_name][side]['gen_bs'] = decode_batch['bs'][start: start+_len]
			if not config.oracle_dst: # collect bs prediction
				decode_all[_name][side]['pred_bs'] = [ dict2list(bs_dict) for bs_dict in decode_batch['bs_pred'][start: start+_len] ]
				decode_all[_name][side]['pred_nlu'] = [ dict2list(bs_dict) for bs_dict in decode_batch['bs_pred'][start: start+_len] ]
				decode_all[_name][side]['ref_bs'] = [ dict2list(bs_dict) for bs_dict in decode_batch['bs_pred'][start: start+_len] ]