import os, sys
import copy
import numpy as np
import torch
import torch.nn as nn

from networks.agent import Agent
from networks.encoder import RNN
from utils.check_turn_info import decide_turn_domain, check_turn_type, form_key_slot, get_turn_act_slot
from utils.criterion import NLLEntropyValid
from utils.util_dst import dict2list
from data_preprocess.create_delex_data import get_summary_bstate, addDBPointer


class Model(nn.Module):
	def __init__(self, config, dataset):
		super(Model, self).__init__()
		self.config = config
		self.dataset = dataset

		self.usr = Agent(config, dataset, 'usr')
		self.sys = Agent(config, dataset, 'sys')

		if config.share_dial_rnn:
			self.dial_rnn = RNN(config.hidden_size, config.hidden_size, dropout=config.dropout, bidirectional=False)

		self.set_optimizer()
		self.keySlot = form_key_slot()

		self.all_rewards = [] # reward container


	def forward(self, batch, turn_idx=None, mode='teacher_force', beam_search=False):
		'''Both agents takes utterances from corpus and perform one generation, used for supervised learning'''
		# unpack data, usr
		ctx_usr = batch['word_idx']['ctx_usr'] # sys output at previous turn, (total_turn, sent_len)
		out_usr = batch['word_idx']['out_usr'] # (total_turn, sent_len)
		act_usr = batch['act_idx']['usr'] # act for current turn, (total_turn, act_len)
		prev_act_usr = batch['prev_act_idx']['usr'] # act for previous turn, (total_turn, act_len)
		ctx_usr_len = batch['sent_len']['ctx_usr'] # (total_turn, )
		out_usr_len = batch['sent_len']['out_usr'] # (total_turn, )
		act_usr_len = batch['sent_len']['act_usr'] # (total_turn, )
		prev_act_usr_len = batch['sent_len']['prev_act_usr'] # (total_turn, )
		gs = batch['gs'] # (total_turn, goal size)

		# unpack data, sys
		ctx_sys = batch['word_idx']['ctx_sys'] # usr ouput at current turn
		out_sys = batch['word_idx']['out_sys']
		act_sys = batch['act_idx']['sys']
		prev_act_sys = batch['prev_act_idx']['sys']
		ctx_sys_len = batch['sent_len']['ctx_sys']
		out_sys_len = batch['sent_len']['out_sys']
		act_sys_len = batch['sent_len']['act_sys'] # (total_turn, )
		prev_act_sys_len = batch['sent_len']['prev_act_sys'] # (total_turn, )
		bs = batch['bs'] # belief state summary, a vector
		db = batch['db']

		# init dial rnn
		init_dial_rnn = batch['init_dial_rnn'] # tuple of (L, B, dir*H)
		self.batch_size = ctx_usr.size(0) # == total_turn
		add_usr = gs # goal state for usr

		# returns
		self.logits, decode = dict(), dict()

		# encode input utterance
		# ctx_out: (B, sent_len, dir*H/2) & ctx_emb: (layer, B, dir*H/2)
		ctx_out_usr, (ctx_emb_usr, _) = self.usr.encode_ctx(ctx_usr, ctx_usr_len)
		ctx_out_sys, (ctx_emb_sys, _) = self.sys.encode_ctx(ctx_sys, ctx_sys_len)

		# run dst networks if not using oracle belief state
		if not self.config.oracle_dst:
			bs_pred, nlu_pred = self.sys.dst(batch, turn_idx=turn_idx, mode=mode) # list (len=B) of dict {domain-slot: value}

		# use dst prediction to query db in inference
		if not self.config.oracle_dst and mode == 'gen':
			full_bs_batch = self.bs_pred2metadata(bs_pred)
			bs = torch.tensor([get_summary_bstate(x) for x in full_bs_batch]).float().cuda()
			db = torch.tensor([addDBPointer({'metadata': x}) for x in full_bs_batch]).float().cuda()
			add_sys = torch.cat([bs, db], dim=1) # additional info for sys
			decode['full_bs'] = full_bs_batch
		# use groundtruth bs, db in training
		else:
			add_sys = torch.cat([bs, db], dim=1) # additional info for sys
			decode['full_bs'] = batch['full_bs']

		# encoding previous act
		prev_act_out_usr, _ = self.usr.encode_prev_act(prev_act_usr, prev_act_usr_len)
		prev_act_out_sys, _ = self.sys.encode_prev_act(prev_act_sys, prev_act_sys_len)

		# dialogue-level rnn encoder
		if turn_idx == 0:
			if self.config.share_dial_rnn: assert init_dial_rnn == None
			if not self.config.share_dial_rnn: assert init_dial_rnn['usr'] == None and init_dial_rnn['sys'] == None
		else:
			if self.config.share_dial_rnn: assert init_dial_rnn != None
			if not self.config.share_dial_rnn: assert init_dial_rnn['usr'] != None and init_dial_rnn['sys'] != None
		dial_len = torch.ones(self.batch_size).long().cuda()

		# run one step on usr ctx
		ctx_emb_usr = ctx_emb_usr.permute(1,0,2)
		if self.config.share_dial_rnn:
			dial_emb_usr, init_dial_rnn = self.dial_rnn(ctx_emb_usr, dial_len, init_state=init_dial_rnn)
		else: # run both dial rnn same input
			dial_emb_usr, init_dial_rnn['usr'] = self.usr.dial_rnn(ctx_emb_usr, dial_len, init_state=init_dial_rnn['usr'])
			_, 			  init_dial_rnn['sys'] = self.sys.dial_rnn(ctx_emb_usr, dial_len, init_state=init_dial_rnn['sys'])
		init_usr = (dial_emb_usr.permute(1,0,2), dial_emb_usr.permute(1,0,2))

		# run one step on sys ctx
		ctx_emb_sys = ctx_emb_sys.permute(1,0,2)
		if self.config.share_dial_rnn:
			dial_emb_sys, init_dial_rnn = self.dial_rnn(ctx_emb_sys, dial_len, init_state=init_dial_rnn)
		else: # use sys dial rnn
			_, 			  init_dial_rnn['usr'] = self.usr.dial_rnn(ctx_emb_sys, dial_len, init_state=init_dial_rnn['usr'])
			dial_emb_sys, init_dial_rnn['sys'] = self.sys.dial_rnn(ctx_emb_sys, dial_len, init_state=init_dial_rnn['sys'])
		init_sys = (dial_emb_sys.permute(1,0,2), dial_emb_sys.permute(1,0,2))

		# record dialogue-level encoder state for next step
		decode['init_dial_rnn'] = init_dial_rnn

		# usr policy
		pol_mode = mode
		if mode == 'gen' and self.config.usr_act_type == 'oracle': pol_mode = 'teacher_force' # decide oracle act on usr or not
		enc_out_usr = {'ctx': ctx_out_usr, 'prev_act': prev_act_out_usr}
		enc_mask_usr = {'ctx': self.len2mask(ctx_usr_len), 'prev_act': self.len2mask(prev_act_usr_len)}
		act_out_usr = self.usr.policy(act_usr, act_usr_len, init_usr, add_usr, enc_out_usr, enc_mask_usr, mode=pol_mode)

		# usr generation
		enc_out_usr['act'], enc_mask_usr['act'] = act_out_usr['hiddens'], self.len2mask(act_out_usr['decode_len'])
		dec_out_usr = self.usr.decode(out_usr, out_usr_len, init_usr, add_usr, enc_out_usr, enc_mask_usr, mode=mode)

		# sys policy
		pol_mode = mode
		if mode == 'gen' and self.config.sys_act_type == 'oracle': pol_mode = 'teacher_force' # decide oracle act on sys or not
		enc_out_sys = {'ctx': ctx_out_sys, 'prev_act': prev_act_out_sys}
		enc_mask_sys = {'ctx': self.len2mask(ctx_sys_len), 'prev_act': self.len2mask(prev_act_sys_len)}
		act_out_sys = self.sys.policy(act_sys, act_sys_len, init_sys, add_sys, enc_out_sys, enc_mask_sys, mode=pol_mode)

		# sys generation
		enc_out_sys['act'], enc_mask_sys['act'] = act_out_sys['hiddens'], self.len2mask(act_out_sys['decode_len'])
		dec_out_sys = self.sys.decode(out_sys, out_sys_len, init_sys, add_sys, enc_out_sys, enc_mask_sys, mode=mode)

		if mode == 'teacher_force':
			self.logits['act_usr'] = act_out_usr['logits']
			self.logits['act_sys'] = act_out_sys['logits']
			self.logits['word_usr'] = dec_out_usr['logits']
			self.logits['word_sys'] = dec_out_sys['logits']
			# logits of dst prediction is stored inside self.sys.dst if not oracle dst
		decode['act_usr'] = act_out_usr['decode']
		decode['act_sys'] = act_out_sys['decode']
		decode['word_usr'] = dec_out_usr['decode'] # list (size=B) of str 
		decode['word_sys'] = dec_out_sys['decode']
		if not self.config.oracle_dst:
			decode['bs_pred'] = bs_pred
			decode['nlu_pred'] = nlu_pred
		return decode


	def len2mask(self, length):
		'''
		convert length tensor to mask tensor for further attention use
		Args:
			length: a tensor (B, ), each indicates the length of sentence
		Returns
			mask: a tensor (B, T), padding with -inf
		'''
		max_len = torch.max(length).item()
		B = length.size()[0]
		assert B == self.batch_size
		mask = torch.ones(B, max_len)
		for i, l in enumerate(length):
			mask[i, l:] = float('-inf')
		return mask.cuda()


	def bs_pred2metadata(self, bs_pred):
		'''Convert the prediction by dst networks into metadata format for db query'''
		assert len(bs_pred) == self.batch_size
		full_bs_batch = [self.init_empty_fullBS() for _ in range(self.batch_size)]
		for bs_dict, full_bs in zip(bs_pred, full_bs_batch):
			for domain_slot, value in bs_dict.items():
				domain, slot = domain_slot.split('-')
#				assert slot in full_bs[domain]['semi'] or slot in full_bs[domain]['book']
				if slot in full_bs[domain]['semi']:
					full_bs[domain]['semi'][slot] = value
				elif slot in full_bs[domain]['book']:
					full_bs[domain]['book'][slot] = value
				else:
					print('undefined domain-slot pair', file=sys.stderr)
					sys.exit(1)
		return full_bs_batch


	def init_empty_fullBS(self):
		full_bs = copy.deepcopy(self.dataset.all_data['MUL0001.json']['log'][1]['metadata'])
		for slot in full_bs['restaurant']['semi']:
			full_bs['restaurant']['semi'][slot] = ''
		del full_bs['train']['book']['ticket']
		full_bs['train']['book']['people'] = ''
		return full_bs


	def interact(self, beam_search=False, dial_name_batch=None):
		'''Interact method: two agents interact to generate dialogues given the user goals'''
		self.batch_size = len(dial_name_batch)

		# init necessary info
		ctx_word = ['hi' for _ in range(self.batch_size)] # will pad with eos in the word2tensor function
		goal_batch = self.goal_sampler(dial_name_batch)
		goal_vec_batch = [self.dataset.getGoalVector(goal) for goal in goal_batch]
		full_bs_batch = [self.init_empty_fullBS() for _ in range(self.batch_size)]

		# dst input material
		lex_dial_history = ['' for _ in range(self.batch_size)]
		bs_pred = [dict() for _ in range(self.batch_size)]

		# start interact between pretrained user and system
		# generate one turn per loop, alternate between user and system
		turn_idx = 0
		extra_slot = [0 for _ in range(self.batch_size)] # tell us how much redundant slots that simulated usr said
		# NOTE: n_side_when_done means dialogue ends with this number of turns on one side, -1 means dialogue is not done yet

		gen_dial_batch = [{'bs': [], 'word_usr': [], 'word_sys': [], 'act_usr': [], 'act_sys': [], 'n_side_when_done': -1,
						   'lex_word_usr': [], 'lex_word_sys': [], 'bs_pred': [],
						   'goal_vec': [], 'bs_vec': [],
						   'act_usr_logprob': [], 'act_usr_len': [],
						   'act_sys_logprob': [], 'act_sys_len': []} for _ in range(self.batch_size)]
		booked_domain_batch = [set() for _ in range(self.batch_size)]

		if self.config.share_dial_rnn:
			init_dial_rnn = None
		else:
			init_dial_rnn = {'usr': None, 'sys': None}

		# record goal
		for gen_dial, goal, dial_name in zip(gen_dial_batch, goal_batch, dial_name_batch):
			gen_dial['goal'] = goal
			gen_dial['dial_name'] = dial_name

		while True:
			ctx, ctx_len = self.word2tensor(ctx_word, self.dataset.vocab) # ctx: (B=num_dial, T), ctx_len: (B=num_dial, )

			# get additional info for the speaker
			if turn_idx % 2 == 0: # usr
				speaker, side = self.usr, 'usr'
				ctx_out, (ctx_emb, _) = speaker.encode_ctx(ctx, ctx_len) # ctx_out: (num_dial, sent_len, dir*H/2) & ctx_emb: (layer, num_dial, dir*H/2)
				goal_vec_batch, gs = self.goal_change(turn_idx, goal_vec_batch, gen_dial_batch, booked_domain_batch, full_bs_batch, goal_batch)
				add = gs

				# collect goal state vec
				for gen_dial, goal_vec in zip(gen_dial_batch, goal_vec_batch):
					gen_dial['goal_vec'].append(goal_vec)

			else: # sys
				speaker, side = self.sys, 'sys'
				ctx_out, (ctx_emb, _) = speaker.encode_ctx(ctx, ctx_len)

				if self.config.oracle_dst:
					self.update_oracle_fullBS(full_bs_batch, ctx_word, goal_batch, extra_slot, gen_dial_batch)
				else: # run dst
					batch = self.prepare_dst_input(lex_dial_history, bs_pred)
					bs_pred, nlu_pred = speaker.dst(batch, turn_idx=(turn_idx-1)/2, mode='gen') # list (len=B) of dict {domain-slot: value}
					full_bs_batch = self.bs_pred2metadata(bs_pred)

				bs = torch.tensor([get_summary_bstate(x) for x in full_bs_batch]).float().cuda()
				db = torch.tensor([addDBPointer({'metadata': x}) for x in full_bs_batch]).float().cuda()
				add = torch.cat([bs, db], dim=1) # additional info for sys

				# collect bs vec
				for gen_dial, full_bs in zip(gen_dial_batch, full_bs_batch):
					bs_vec = get_summary_bstate(full_bs)
					gen_dial['bs_vec'].append(bs_vec)

				# record bs along dialogue
				for b_idx, (gen_dial, full_bs) in enumerate(zip(gen_dial_batch, full_bs_batch)):
					gen_dial['bs'].append( copy.deepcopy(full_bs) )
					if not self.config.oracle_dst:
						gen_dial['bs_pred'].append( copy.deepcopy(bs_pred[b_idx]) )

			# prev act
			if turn_idx <= 1: # first turn for usr or sys, no prev act
				assert len(gen_dial_batch[0]['act_{}'.format(side)]) == 0
				prev_act = [ ' '.join(['<EOS>']) for _ in range(self.batch_size) ]
			else:
				prev_act = [ ' '.join(gen_dial['act_{}'.format(side)][-1]) for gen_dial in gen_dial_batch ]
			prev_act, prev_act_len = self.word2tensor(prev_act, self.dataset.act_vocab)
			prev_act_out, _ = speaker.encode_prev_act(prev_act, prev_act_len)

			# shared dial rnn
			ctx_emb = ctx_emb.permute(1,0,2)
			dial_len = torch.ones(self.batch_size).long().cuda()
			if self.config.share_dial_rnn:
				dial_emb, init_dial_rnn = self.dial_rnn(ctx_emb, dial_len, init_state=init_dial_rnn)
				init = (dial_emb.permute(1,0,2), dial_emb.permute(1,0,2))
			else:
				dial_emb_usr, init_dial_rnn['usr'] = self.usr.dial_rnn(ctx_emb, dial_len, init_state=init_dial_rnn['usr'])
				dial_emb_sys, init_dial_rnn['sys'] = self.sys.dial_rnn(ctx_emb, dial_len, init_state=init_dial_rnn['sys'])
				if side == 'usr':
					init = (dial_emb_usr.permute(1,0,2), dial_emb_usr.permute(1,0,2))
				else:
					init = (dial_emb_sys.permute(1,0,2), dial_emb_sys.permute(1,0,2))

			# policy
			act = act_len = out = out_len = None # no target act/word seq during interaction to use
			enc_out = {'ctx': ctx_out, 'prev_act': prev_act_out}
			enc_mask = {'ctx': self.len2mask(ctx_len), 'prev_act': self.len2mask(prev_act_len)}
			act_out = speaker.policy(act, act_len, init, add, enc_out, enc_mask, mode='gen')

			# generation
			enc_out['act'], enc_mask['act'] = act_out['hiddens'], self.len2mask(act_out['decode_len'])
			dec_out = speaker.decode(out, out_len, init, add, enc_out, enc_mask, mode='gen')

			# collect dialogues, logprobs and gen_len for each act_seq
			for batch_idx, (word, act) in enumerate(zip(dec_out['decode'], act_out['decode'])):
				gen_dial_batch[batch_idx]['word_{}'.format(side)].append(word)
				gen_dial_batch[batch_idx]['act_{}'.format(side)].append(act)
				# lexicalise back
				if side == 'usr':
					lex_word = self.usr_lexicalise(word, goal_batch[batch_idx])
				else:
					lex_word = self.sys_lexicalise(word, full_bs_batch[batch_idx])
				gen_dial_batch[batch_idx]['lex_word_{}'.format(side)].append(lex_word) # save lex word for reference
				lex_dial_history[batch_idx] += (' ; ' + lex_word)

			act_logprob = act_out['logprobs'] # (batch_size, act_max_len)
			act_logprob = act_logprob.split(1, dim=0) # a list of (1, act_max_len)
			act_len = act_out['decode_len'].tolist() # a list of valid act_len
			for batch_idx in range(self.batch_size):
				gen_dial_batch[batch_idx]['act_{}_logprob'.format(side)].append(act_logprob[batch_idx].squeeze(0)) # (act_max_len)
				gen_dial_batch[batch_idx]['act_{}_len'.format(side)].append(act_len[batch_idx])

			# dialogue terminates once usr act thank or bye and sys act welcome or bye
			n_done_dial = 0
			if side == 'sys':
				assert len(gen_dial['act_usr']) == len(gen_dial['act_sys']) == len(gen_dial['word_usr']) == len(gen_dial['word_sys']) == len(gen_dial['bs'])
				for gen_dial in gen_dial_batch:
					if gen_dial['n_side_when_done'] != -1: # dialogue finished already
						n_done_dial += 1
					else: # dialogue not done yet
#						if ('act_bye' in gen_dial['act_sys'][-1] or 'act_welcome' in gen_dial['act_sys'][-1]) \
#							and ('act_bye' in gen_dial['act_usr'][-1] or 'act_thank' in gen_dial['act_usr'][-1]): # end of dialogue
						# NOTE: with limited data, usr is hard to train, and might keep repeating information, so use only sys to terminate dialogues
						if ('act_bye' in gen_dial['act_sys'][-1] or 'act_welcome' in gen_dial['act_sys'][-1]): # end of dialogue
							gen_dial['n_side_when_done'] = len(gen_dial['act_sys'])
							n_done_dial += 1

						elif (len(gen_dial['act_usr']) + len(gen_dial['act_sys'])) >= self.config.rl_max_dial_len: # reach dialogue length limit
							gen_dial['n_side_when_done'] = len(gen_dial['act_sys'])
							n_done_dial += 1

			if n_done_dial >= self.batch_size:
				break
			else: # continue interaction
				turn_idx += 1
				ctx_word = dec_out['decode']

		# trim extra turns
		for gen_dial in gen_dial_batch:
			for key in ['bs', 'act_usr', 'act_sys', 'word_usr', 'word_sys', 'act_sys_logprob', 'act_sys_len', 'lex_word_usr', 'lex_word_sys', 'bs_pred', 'goal_vec', 'bs_vec']:
				gen_dial[key] = gen_dial[key][:gen_dial['n_side_when_done']]
		return gen_dial_batch


	def prepare_dst_input(self, lex_dial_history, bs_pred):
		'''Convert lex_dial_history, a list (len=B) of str and full_bs_batch, a list (len=B) of bs_dict into dst input tensor'''
		batch = {'dst_idx': {}, 'sent_len': {}}
		batch['dst_idx']['dst_ctx'], batch['sent_len']['dst_ctx'] = self.word2tensor(lex_dial_history, self.dataset.dstWord_vocab)

		slot_pred = [] # a list of list
		value_pred = [] # a list of list
		for bs in bs_pred:
			bs = dict2list(bs) # list of slot value pair
			slots = [sv.split('=')[0] for sv in bs] # list of slot
			values = [sv.split('=')[1] for sv in bs] # list of value
			slot_pred.append( slots )
			value_pred.append( values )
		batch['dst_idx']['prev_bs_slot'], batch['sent_len']['prev_bs_slot'] = self.word2tensor(slot_pred, self.dataset.slot_vocab)
		batch['dst_idx']['prev_bs_value'], batch['sent_len']['prev_bs_value'] = self.word2tensor(value_pred, self.dataset.value_vocab['all'])
		batch['dst_idx']['curr_nlu_slot'] = batch['dst_idx']['curr_nlu_value'] = None
		batch['sent_len']['curr_nlu_slot'] = batch['sent_len']['curr_nlu_value'] = None
		return batch


	def get_usr_transit_domain_reward(self, gen_dial, print_log=True):
		'''
		a good domain transit for usr needs to meet two conditions:
			1) the system provides an entity in previous domain
			2) the system has not provided an entity for current domain
		'''
		# trace usr turn domain
		domain_prev = 'none'
		done_domain = set()
		reward = []
		for side_idx, (act_usr, act_sys) in enumerate(zip(gen_dial['act_usr'], gen_dial['act_sys'])):

			turn_domain = decide_turn_domain(act_usr, '', domain_prev)
			r = 0 # 0 for non-transit turn
			if side_idx != 0 and turn_domain != domain_prev: # domain transit happens
				if domain_prev in done_domain and turn_domain not in done_domain: # good transit
					r = self.config.usr_correct_transit_reward # (+)
				else:
					r = self.config.usr_wrong_transit_reward # (-)
			if turn_domain not in ['restaurant', 'hotel', 'attraction', 'train', 'taxi']: # dont care transit to other domains
				r = 0
			reward.append(r)

			act = act_usr + ' ' + act_sys # name might be informed by usr itself
			if 'name' in act:
				done_domain.add(turn_domain)
#				for slot in act.split():
#					if 'name' in slot:
#						domain = slot.split('_')[0]
#						done_domain.add(domain)
#						break
			elif 'trainID' in act:
				done_domain.add('train')
			elif 'taxi_type' in act:
				done_domain.add('taxi')
			elif 'act_offerbooked' in act_sys:
				done_domain.add(turn_domain)
			domain_prev = turn_domain

		# trace reward
		if print_log:
			print('usr transit - dial name:', gen_dial['dial_name'])
			for side_idx, (r, usr_act, sys_act) in enumerate(zip(reward, gen_dial['act_usr'], gen_dial['act_sys'])):
#				if r != 0:
				print('{}, usr transit r: {}, | {} -> {}'.format(side_idx, r, usr_act, sys_act))
		return reward


	def get_usr_follow_goal_reward(self, gen_dial, print_log=True):
		goal = gen_dial['goal']
		reward = []
		for side_idx, act_usr in enumerate(gen_dial['act_usr']):
			if side_idx == 0:
				act_sys = ''
			else:
				act_sys = gen_dial['act_sys'][side_idx-1] # check previous sys act

			r = 0
			info_slots_usr = get_turn_act_slot(act_usr, 'inform') # slots informed by usr, e.g., hotel_pricerange
			reqt_slots_sys = get_turn_act_slot(act_sys, 'request') # slots requested by sys, e.g., hotel_pricerange
			for domain_slot in info_slots_usr:
				if 'none' in domain_slot: # none is a special slot in act seq
					continue
				domain, slot = domain_slot.split('_')

				if domain not in ['restaurant', 'hotel', 'attraction', 'train', 'taxi']:
					continue

				# good informed slot if in goal or requested by sys
				if ('info' in goal[domain] and slot in goal[domain]['info']) or \
 						('book' in goal[domain] and slot in goal[domain]['book']) or domain_slot in reqt_slots_sys:
					r += self.config.usr_follow_goal_reward
				else:
					r += self.config.usr_not_follow_goal_reward
			reward.append(r)

		if print_log:
			print('usr goal - dial name:', gen_dial['dial_name'])
			for side_idx, (r, usr_act, sys_act) in enumerate(zip(reward, gen_dial['act_usr'], gen_dial['act_sys'])):
#				if r != 0:
				print('{}, usr goal r: {}, | {} -> {}'.format(side_idx, r, usr_act, sys_act))
		return reward


	def get_usr_repeat_info_reward(self, gen_dial, print_log=True):
		'''Check whether the usr repeatedly informs'''
		reward = []
		informed_slot_set = set()
		for side_idx, act_usr in enumerate(gen_dial['act_usr']):
			# no repeat info in first turn
			if side_idx == 0:
				reward.append(0)
				continue

			r = 0 # assume correct
			act_sys, full_bs = gen_dial['act_sys'][side_idx-1], gen_dial['bs'][side_idx-1]
			if 'act_nooffer' in act_sys: # usr needs to repeat the same slot if nooffer
				reward.append(0)
				continue

			info_slots = get_turn_act_slot(act_usr, 'inform')
			reqt_slots_sys = get_turn_act_slot(act_sys, 'request')
			for domain_slot in info_slots:
				if '_' not in domain_slot: # or 'none' in domain_slot:
					continue
				domain, slot = domain_slot.split('_')
				if domain in ['booking', 'general']: # FIX: these two do not matter
					continue
				if domain_slot not in reqt_slots_sys and domain_slot in informed_slot_set:
					r += self.config.usr_repeat_info_reward # punish more if make more mistakes, accumulate rewards

			if len(info_slots) != 0 and r == 0: # no repeat info
				r = self.config.usr_no_repeat_info_reward
			reward.append(r)

			# collect informed slots
			for slot in info_slots:
				informed_slot_set.add(slot)

		# trace reward
		if print_log:
			print('usr repeat info - dial name:', gen_dial['dial_name'])
			for side_idx, (r, usr_act, sys_act) in enumerate(zip(reward, gen_dial['act_usr'], gen_dial['act_sys'])):
#				if r != 0:
				print('{}, usr repeat info r: {}, | {} -> {}'.format(side_idx, r, usr_act, sys_act))
		return reward


	def get_usr_repeat_ask_reward(self, gen_dial, print_log=True):
		'''Check whether the usr requests what the sys just informed in last turn'''
		reward = []
		for side_idx, (act_usr, word_usr) in enumerate(zip(gen_dial['act_usr'], gen_dial['word_usr'])):
			# no repeat ask in first turn
			if side_idx == 0:
				reward.append(0)
				continue
			r = 0 # assume correct
			act_sys = gen_dial['act_sys'][side_idx-1] # check previous sys act
			reqt_slots = get_turn_act_slot(act_usr, 'request') # slots requested by usr, e.g., hotel_pricerange
			for slot in reqt_slots:
				if slot in act_sys:
					r += self.config.usr_repeat_ask_reward # punish more if make more mistakes, accumulate rewards

			if len(reqt_slots) != 0 and r == 0: # answer all slots
				r = self.config.usr_no_repeat_ask_reward
			reward.append(r)

		# trace reward
		if print_log:
			print('usr repeat ask - dial name:', gen_dial['dial_name'])
			for side_idx, (r, usr_act, sys_act) in enumerate(zip(reward, gen_dial['act_usr'], gen_dial['act_sys'])):
#				if r != 0:
				print('{}, usr repeat ask r: {}, | {} -> {}'.format(side_idx, r, usr_act, sys_act))
		return reward


	def get_usr_miss_answer_reward(self, gen_dial, print_log=True):
		reward = []
		for side_idx, (act_usr, word_usr) in enumerate(zip(gen_dial['act_usr'], gen_dial['word_usr'])):
			# no miss answer in first turn
			if side_idx == 0:
				reward.append(0)
				continue
			act_sys = gen_dial['act_sys'][side_idx-1] # check previous sys act
			reqt_slots = get_turn_act_slot(act_sys, 'request') # slots requested by sys, e.g., hotel_pricerange

			if len(reqt_slots) == 0: # no request by sys
				reward.append(0)
				continue

			if 'care' in word_usr or 'preference' in word_usr or 'booking' in act_sys: # intent 'dont care' cannot be captured by usr act
				reward.append(0)
				continue

			answered = False
			for slot in reqt_slots:
				if slot in act_usr:
					answered = True
					break
			if answered:
				r = self.config.usr_no_miss_answer_reward
			else:
				r = self.config.usr_miss_answer_reward
			reward.append(r)

		# trace reward
		if print_log:
			print('usr miss answer - dial name:', gen_dial['dial_name'])
			for side_idx, (r, usr_act, sys_act) in enumerate(zip(reward, gen_dial['act_usr'], gen_dial['act_sys'])):
#				if r != 0:
				print('{}, usr miss answer r: {}, | {} -> {}'.format(side_idx, r, usr_act, sys_act))
		return reward


	def get_domain_reward(self, gen_dial, print_log=True):
		'''
		Description:
			the system will get a positive reward (+) if the response is in the same domain as usr query, otherwise, negative reward (-)
		Return:
			a list (len=dial_len) of real number
		'''
		# trace usr turn domain
		domain_prev = 'none'
		domain_history_usr = []
		for act_usr in gen_dial['act_usr']:
			turn_domain = decide_turn_domain(act_usr, '', domain_prev)
			domain_history_usr.append(turn_domain)
			domain_prev = turn_domain

		# trace sys turn domain
		domain_prev = 'none'
		domain_history_sys = []
		for act_sys in gen_dial['act_sys']:
			turn_domain = decide_turn_domain('', act_sys, domain_prev)
			domain_history_sys.append(turn_domain)
			domain_prev = turn_domain

		reward = []
		for domain_usr, domain_sys in zip(domain_history_usr, domain_history_sys):
			if domain_sys == domain_usr:
				r = self.config.correct_domain_reward
			else:
				r = self.config.wrong_domain_reward
			if domain_sys == 'general':
				r = 0
			reward.append(r)
				
		# trace reward
		if print_log:
			print('sys domain - dial name:', gen_dial['dial_name'])
			for side_idx, (r, usr_act, sys_act) in enumerate(zip(reward, gen_dial['act_usr'], gen_dial['act_sys'])):
#				if r != 0:
				print('{}, sys domain r: {}, | {} -> {}'.format(side_idx, r, usr_act, sys_act))
		return reward
		

	def get_entity_provide_reward(self, gen_dial, print_log=True):
		'''
		Description:
			get the turn level reward within a dialogue, (+) if entity is provided and (-) if entity is no provided
			this reward is domain-independent, and only affects the turns at info stage (which means book/reqt turns are not affected) 
		Return:
			a list (len=dial_len) of real number
		'''
		# trace turn domain within dialogue
		domain_prev = 'none'
		domain_history = []
		for side_idx, (act_usr, act_sys) in enumerate(zip(gen_dial['act_usr'], gen_dial['act_sys'])):
			turn_domain = decide_turn_domain(act_usr, act_sys, domain_prev)
			domain_history.append(turn_domain)
			domain_prev = turn_domain
			if turn_domain == 'none': # should not enter
				print(side_idx, act_usr, act_sys, file=sys.stderr)
#				input('press...', file=sys.stderr)
				sys.eixt(1)

		# decide valid turns that influence entity provide
		domain_prev = 'none'
		domain_entityProvided = set(['taxi', 'police', 'hospital', 'general'])
		weight = []
		w = 1.
		for act_usr, act_sys, turn_domain in zip(gen_dial['act_usr'], gen_dial['act_sys'], domain_history):
			if turn_domain != domain_prev: # domain transit
				w = 1.

			if turn_domain in ['taxi', 'police', 'hospital', 'general']: # no need reward for those domain
				weight.append(0)
			else:
				weight.append(w)

			# TODO: re-think this for domain transfer case
			if '_name' in act_sys or '_trainID' in act_sys: # make the turns afterward not affected
				domain_entityProvided.add(turn_domain)
				w = 0
			domain_prev = turn_domain

		# deal with domain without name provide since boundary between info and book/reqt is not decided yet
		for side_idx, (act_usr, act_sys, turn_domain) in enumerate(zip(gen_dial['act_usr'], gen_dial['act_sys'], domain_history)):
			if turn_domain in domain_entityProvided: # done already
				continue
			turn_stage = check_turn_type(act_usr, act_sys, turn_domain, self.keySlot) # info/book/reqt
			if turn_stage in ['book', 'reqt']: # make it not affected
				weight[side_idx] = 0

		reward = []
		for side_idx, (w, turn_domain) in enumerate(zip(weight, domain_history)):
			if turn_domain in domain_entityProvided:
				r = self.config.entity_provide_reward
			else:
				r = self.config.no_entity_provide_reward
			reward.append(w*r)

		# trace reward
		if print_log:
			print('sys provide - dial name:', gen_dial['dial_name'])
			print('name provied:', domain_entityProvided-set(['taxi', 'police', 'hospital', 'general']))
			for side_idx, (w, r, usr_act, sys_act, turn_domain, usr_word, sys_word) in enumerate(zip(weight, reward, gen_dial['act_usr'], gen_dial['act_sys'], domain_history, gen_dial['word_usr'], gen_dial['word_sys'])):
#				if r != 0:
				print('{}, w: {}, sys r: {}, d: {} | {} -> {}'.format(side_idx, w, r, turn_domain, usr_act, sys_act))
		return reward


	def get_repeat_ask_reward(self, gen_dial, print_log=True):
		'''
		Description:
			check if sys requests the informed slots
		Return:
			a list of real value rewards
		'''
		reward = []
		for side_idx, (full_bs, act_usr, act_sys) in enumerate(zip(gen_dial['bs'], gen_dial['act_usr'], gen_dial['act_sys'])):
			r = 0 # assume no repeat ask until fine one
			reqt_slots = get_turn_act_slot(act_sys, 'request')
			for domain_slot in reqt_slots:
				if '_' not in domain_slot:
					continue
				domain, slot = domain_slot.split('_')
				if domain in ['booking', 'general']: # FIX: these two do not matter
					continue
				for constraint in ['semi', 'book']:
					if slot in full_bs[domain][constraint] and full_bs[domain][constraint][slot] != '': # already has value in belief state
#						r = self.config.repeat_ask_reward
						r += self.config.repeat_ask_reward # punish more if make more mistakes, accumulate rewards

			if len(reqt_slots) != 0 and r == 0: # request right slots
				r = self.config.no_repeat_ask_reward
			reward.append(r)

		# trace reward
		if print_log:
			print('sys repeat - dial name:', gen_dial['dial_name'])
			for side_idx, (r, usr_act, sys_act) in enumerate(zip(reward, gen_dial['act_usr'], gen_dial['act_sys'])):
#				if r != 0:
				print('{}, sys ask r: {}, | {} -> {}'.format(side_idx, r, usr_act, sys_act))
		return reward


	def get_miss_answer_reward(self, gen_dial, print_log=True):
		reward = []
		for side_idx, (act_usr, act_sys) in enumerate(zip(gen_dial['act_usr'], gen_dial['act_sys'])):
			r = 0 # assume all requestable slots are answered
			reqt_slots = get_turn_act_slot(act_usr, 'request') # slots requested by usr, e.g., hotel_pricerange
			for slot in reqt_slots:
				if slot not in act_sys:
					r += self.config.miss_answer_reward # punish more if make more mistakes, accumulate rewards

			if len(reqt_slots) != 0 and r == 0: # answer all slots
				r = self.config.no_miss_answer_reward
			reward.append(r)

		# trace reward
		if print_log:
			print('sys answer - dial name:', gen_dial['dial_name'])
			for side_idx, (r, usr_act, sys_act) in enumerate(zip(reward, gen_dial['act_usr'], gen_dial['act_sys'])):
#				if r != 0:
				print('{}, sys ans r: {}, | {} -> {}'.format(side_idx, r, usr_act, sys_act))
		return reward


	def check_max_gen_act_seq(self, gen_dial_batch):
		max_act_len_batch = []
		for gen_dial in gen_dial_batch:
			max_act_len = 0
			for sys_act in gen_dial['act_sys']:
				act_len = len(sys_act.split())
				if act_len > max_act_len:
					max_act_len = act_len
			max_act_len_batch.append(max_act_len)
		return max_act_len_batch


	def get_turn_reward(self, gen_dial_batch):
		'''Obtain turn-level rewards'''
		avg_sys_r, avg_usr_r = 0, 0
		for gen_dial in gen_dial_batch:
			goal = gen_dial['goal']
			domain_in_goal = []
			for domain in goal:
				if 'info' in goal[domain]:
					domain_in_goal.append(domain)

			max_act_len = 0
			print('\nDIALOGUE:', gen_dial['dial_name'], domain_in_goal)
			for side_idx, (goal_vec, bs_vec, usr_act, sys_act, usr_word, sys_word) in enumerate(zip(gen_dial['goal_vec'], \
					gen_dial['bs_vec'], gen_dial['act_usr'], gen_dial['act_sys'], gen_dial['word_usr'], gen_dial['word_sys'])):
				# trace act_len from sys side
#				act_len = len(sys_act.split())
				act_len = max( len(sys_act.split()), len(usr_act.split()) )
				if act_len > max_act_len:
					max_act_len = act_len
				print('side_idx:', side_idx)
				print('goal_vec:', goal_vec)
				print('bs_vec:', bs_vec)
				print('act: {} -> {}'.format(usr_act, sys_act))
				print('usr: {}\nsys: {}\n'.format(usr_word, sys_word))
			print('Effective dialogue: {}'.format(max_act_len < 20))
			print('-----Done dialogue-----')

			# filter out dialogues with unknown weird behavior (e.g., too long act seq)
			n_side = gen_dial['n_side_when_done']
			if max_act_len >= 20:
				provide_reward = [0 for _ in range(n_side)]
				repeat_reward = [0 for _ in range(n_side)]
				answer_reward = [0 for _ in range(n_side)]
				domain_reward = [0 for _ in range(n_side)]
				usr_repeat_info = [0 for _ in range(n_side)]
				usr_repeat_ask  = [0 for _ in range(n_side)]
				usr_miss_answer = [0 for _ in range(n_side)]
			else:
				provide_reward = self.get_entity_provide_reward(gen_dial) # a list of dial len
				repeat_reward = self.get_repeat_ask_reward(gen_dial)
				answer_reward = self.get_miss_answer_reward(gen_dial)
				domain_reward = [0 for _ in range(n_side)]
				usr_repeat_info = self.get_usr_repeat_info_reward(gen_dial) if self.config.rl_update_usr else [0 for _ in range(n_side)]
				usr_repeat_ask  = self.get_usr_repeat_ask_reward(gen_dial) if self.config.rl_update_usr else [0 for _ in range(n_side)]
				usr_miss_answer = self.get_usr_miss_answer_reward(gen_dial) if self.config.rl_update_usr else [0 for _ in range(n_side)]

			# total reward
			sys_reward = [ r1+r2+r3+r4 for r1, r2, r3, r4 in zip(provide_reward, repeat_reward, answer_reward, domain_reward) ]
			usr_reward = [ r1+r2+r3 for r1, r2, r3 in zip(usr_repeat_info, usr_repeat_ask, usr_miss_answer) ]
			gen_dial['sys_reward'], gen_dial['usr_reward'] = sys_reward, usr_reward
			avg_sys_r += np.mean(gen_dial['sys_reward'])
			avg_usr_r += np.mean(gen_dial['usr_reward'])

		avg_sys_r /= self.config.rl_batch_size
		avg_usr_r /= self.config.rl_batch_size
		return avg_sys_r, avg_usr_r


	def get_success_reward(self, gen_dial_batch_ori, evaluator):
		for gen_dial in gen_dial_batch_ori:
			decode_all = {}
			# calculate one dialogue per call
			gen_dial_batch = [gen_dial]
			
			# trace generated dialogues
			if True:
				for gen_dial in gen_dial_batch:
					print('dial_name:', gen_dial['dial_name'])
					for i, (act_usr, act_sys, word_usr, word_sys) in \
							enumerate(zip(gen_dial['act_usr'], gen_dial['act_sys'], gen_dial['word_usr'], gen_dial['word_sys'])):
						print('At side turn: {}'.format(i))
						print('USR: {} ({})'.format(word_usr, act_usr))
						print('SYS: {} ({})'.format(word_sys, act_sys))
#					input('press...')
	
			# form dummy batch
			batch = {}
			dial_len = [len(gen_dial['word_usr']) for gen_dial in gen_dial_batch]
			total_turns = sum(dial_len)
			batch['dial_len'] = torch.tensor(dial_len).long().cuda()
			batch['dial_name'] = [gen_dial['dial_name']]
			batch['ref'] = {'act': {}, 'word': {}}
			batch['ref']['act']['usr'] = batch['ref']['act']['sys'] = ['None' for _ in range(total_turns)] # because no ref in interaction
			batch['ref']['word']['usr'] = batch['ref']['word']['sys'] = ['None' for _ in range(total_turns)]
			batch['full_bs'] = ['None' for _ in range(total_turns)]

			decode_batch = {} # word_{side}, act_{side}, bs
			for key in ['bs', 'act_usr', 'act_sys', 'word_usr', 'word_sys', 'lex_word_usr', 'lex_word_sys', 'bs_pred']:
				decode_batch[key] = []
				for gen_dial in gen_dial_batch:
					decode_batch[key].extend(gen_dial[key])
	
			collect_dial_interact(decode_all, decode_batch, 'usr', batch, self.config, self.dataset)
			collect_dial_interact(decode_all, decode_batch, 'sys', batch, self.config, self.dataset)

			success, match, record = evaluator.context_to_response_eval(decode_all, 'test')
			if success > 1: # normalize to 1
				success /= 100

			# calcuate reward for each turn using success rate at dialogue level
			'''we use the same implementation as in https://github.com/snakeztc/NeuralDialog-LaRL'''
			reward = success
			self.all_rewards.append(reward)
			r = (reward - np.mean(self.all_rewards)) / max(1e-4, np.std(self.all_rewards))
			print('success: {:.2f}, normalised reward: {:.2f}'.format(success, r))
			print('success: {:.2f}, normalised reward: {:.2f}'.format(success, r), file=sys.stderr)
			sys_rewards = []
			for _ in range(dial_len[0]):
				sys_rewards.insert(0, r)
				r = r * 0.99 # set gamma to 0.99
			gen_dial['sys_reward'] = sys_rewards
			if self.config.rl_update_usr:
				gen_dial['usr_reward'] = sys_rewards
			else:
				gen_dial['usr_reward'] = [0 for _ in range(dial_len[0])]

		avg_sys_r = np.mean(self.all_rewards)
		avg_usr_r = avg_sys_r if self.config.rl_update_usr else 0
		return avg_sys_r, avg_usr_r



	def get_rl_loss(self, gen_dial_batch, side):
		assert side in ['sys', 'usr']
		rl_loss = 0
		n_token = 0
		for batch_idx in range(self.config.rl_batch_size):
			gen_dial = gen_dial_batch[batch_idx]
			n_side = gen_dial['n_side_when_done']
			reward = gen_dial['{}_reward'.format(side)]
			for side_idx in range(n_side):
#				act_len = gen_dial['act_sys_len'][side_idx] # int
#				act_logprob = gen_dial['act_sys_logprob'][side_idx] # (act_max_len)
				act_len = gen_dial['act_{}_len'.format(side)][side_idx] # int
				act_logprob = gen_dial['act_{}_logprob'.format(side)][side_idx] # (act_max_len)
				rl_loss += (reward[side_idx] * -1*torch.sum(act_logprob[:act_len]))
				n_token += act_len
		rl_loss /= n_token
		return rl_loss


	def update_oracle_fullBS(self, full_bs_batch, ctx_word, goal_batch, extra_slot, gen_dial_batch):
		'''
		update the full_bs based on the user input at previous turn by key word detection of delexicalised words
		this is a compromised way to dst since we don't have oracle belief state or a trained dst during interaction
		when using oracle belief state
		Args:
			full_bs_batch: a list (len=batch_size=num_dial) of dict
			ctx_word: a list of str
			goal_batch: a list of goal
			extra_slot: count of slots that usrs specified but not in goal in previous dialogue
		'''
		for batch_idx, (bs, sent, goal) in enumerate(zip(full_bs_batch, ctx_word, goal_batch)):
			for domain in self.dataset.all_domains:
				for constraint in ['semi', 'book']:
					for slot in bs[domain][constraint]:
						if slot == 'booked':
							continue
						if '[{}_{}]'.format(domain, slot) in sent: # check in NL
							if 'info' in goal[domain] and slot in goal[domain]['info']: # not every goal has info
								bs[domain]['semi'][slot] = goal[domain]['info'][slot]
							elif 'book' in goal[domain] and slot in goal[domain]['book']: # not every goal has book
								bs[domain]['book'][slot] = goal[domain]['book'][slot]
							else: # imperfect simulated usr might say something not in goal, trace the errors
								extra_slot[batch_idx] += 1
								bs[domain][constraint][slot] = 'dont care'

			# fix parking, internet in hotel domain that cannot be detected by natural language
			usr_act = gen_dial_batch[batch_idx]['act_usr'][-1] # the current usr act seq
			for slot in ['parking', 'internet']:
				if 'hotel_'+slot in usr_act:
					bs['hotel']['semi'][slot] = 'dont care'

			# fix (taxi domain) belief state discrepancy between during pretrain and during interaction due to imperfect delex
			# e.g., usr:'leave from hotel' as 'taxi_departure'
			for slot in ['departure', 'destination']:
				if 'taxi_'+slot in usr_act:
					bs['taxi']['semi'][slot] = 'dont care'


	def usr_lexicalise(self, delex_sent, goal):
		lex_sent = []
		for word in delex_sent.split():
			if word[0] == '[' and word[-1] == ']': # delex term
				domain, slot = word[1:-1].split('_')
				if domain == 'value': # terms like 'value-count'
					value = self.beyond_goal_or_belief(domain, slot, delex_sent)
				elif 'info' in goal[domain] and slot in goal[domain]['info']: # not every goal has info
					value = goal[domain]['info'][slot]
				elif 'book' in goal[domain] and slot in goal[domain]['book']: # not every goal has book
					value = goal[domain]['book'][slot]
				else: # imperfect simulated usr might say something not in goal, trace the errors
					value = self.beyond_goal_or_belief(domain, slot, delex_sent)
				lex_sent.append(str(value))
			else: # normal word
				lex_sent.append(word)
		return ' '.join(lex_sent)


	def sys_lexicalise(self, delex_sent, full_bs):
		domain2ent = addDBPointer({'metadata': full_bs}, return_ent=True) # check real db entity
		lex_sent = []
		for word in delex_sent.split():
			if word[0] == '[' and word[-1] == ']': # delex term
				domain, slot = word[1:-1].split('_')
				if domain == 'value': # terms like 'value-count'
					value = self.beyond_goal_or_belief(domain, slot, delex_sent)
				# slot in belief state, e.g., area, pricerange
				elif slot in full_bs[domain]['semi'] and full_bs[domain]['semi'][slot] not in ['', 'not mentioned', 'dontcare', 'dont care']:
					value = full_bs[domain]['semi'][slot]
				elif slot in full_bs[domain]['book'] and full_bs[domain]['book'][slot] not in ['', 'not mentioned', 'dontcare', 'dont care']:
					value = full_bs[domain]['book'][slot]
				# slot in db, e.g., phone, postcode
				elif domain in domain2ent and len(domain2ent[domain]) > 0 and slot in domain2ent[domain][0]:
					value = domain2ent[domain][0][slot]
				else:
					value = self.beyond_goal_or_belief(domain, slot, delex_sent)
				lex_sent.append(str(value))
			else: # normal word
				lex_sent.append(word)
		return ' '.join(lex_sent)


	def beyond_goal_or_belief(self, domain, slot, delex_sent):
		'''Return a value for the slot that cannot be traced'''
		if domain == 'value':
			if ' -s ' in delex_sent:
				return '2'
			else:
				return '1'
		if slot == 'address': return 'Regent Street'

		if slot == 'area': return 'north'

		if slot == 'name':
			if domain == 'attraction': return 'cambridge museum of technology'
			if domain == 'restaurant': return 'alexander bed and breakfast'
			if domain == 'hotel': return 'acorn guest house'

		if slot == 'phone': return '01223323737'

		if slot == 'postcode': return 'cb21ab'

		if slot == 'type':
			if domain == 'attraction': return 'college'
			if domain == 'hotel': return 'hotel'
			if domain == 'taxi': return 'bmw'

		if slot == 'pricerange': return 'cheap'

		if slot in ['people', 'stars', 'stay']: return '2'

		if slot == 'reference': return 'fztwszhh'

		if slot == 'day': return 'sunday'

		if slot == 'food': return 'british'

		if slot == 'time': return '13:00'

		if slot == 'arriveBy': return '21:00'

		if slot == 'leaveAt': return '13:00'

		if slot == 'departure':
			if domain == 'train': 'cambridge'
			if domain == 'taxi': 'acorn guest house'

		if slot == 'destination':
			if domain == 'train': 'london liverpool street'
			if domain == 'taxi': 'alexander bed and breakfast'

		if slot == 'price': return '10'

		if slot == 'duration': return '50'


	def word2tensor(self, ctx_word, vocab):
		'''Convert a list of decoded sent (str/list) into padded tensor'''
		ctx = [self.dataset.parseSent(sent, vocab) for sent in ctx_word] # pad eos inside parseSent func
		ctx_len = [len(sent) for sent in ctx]

		# pad
		max_len = max(ctx_len)
		for sent in ctx: assert vocab['<PAD>'] not in sent # check if in-place PAD before
		for sent in ctx: sent.extend( [vocab['<PAD>'] for _ in range(max_len-len(sent))] )
		return torch.tensor(ctx).cuda(), torch.tensor(ctx_len).cuda()


	def goal_change(self, turn_idx, goal_vec_batch, gen_dial_batch, booked_domain_batch, full_bs_batch, goal_batch):
		if self.config.goal_state_change == 'none' or turn_idx == 0:
			return goal_vec_batch, torch.tensor(goal_vec_batch).float().cuda()
		else:
			new_goal_vec_batch = []
			for batch_idx in range(self.batch_size):
				goal = goal_batch[batch_idx]
				gen_dial = gen_dial_batch[batch_idx]
				old_goal_vec = goal_vec_batch[batch_idx]
				full_bs_prev = full_bs_batch[batch_idx]
				booked_domain = booked_domain_batch[batch_idx]

				new_goal_vec = self.dataset.changeGoalState(old_goal_vec, goal, full_bs_prev, \
					gen_dial['act_usr'][-1], gen_dial['word_usr'][-1], gen_dial['act_sys'][-1], gen_dial['word_sys'][-1], booked_domain)
				new_goal_vec_batch.append(new_goal_vec)
			return new_goal_vec_batch, torch.tensor(new_goal_vec_batch).float().cuda()


	def goal_sampler(self, dial_name_batch):
		if dial_name_batch is not None: # obtain goals from corpus
			goal_batch = []
			for dial_name in dial_name_batch:
				goal_batch.append(self.dataset.all_data[dial_name]['goal'])
		else:
			NotImplementedError
		return goal_batch


	def print_full_bs(self, full_bs):
		for domain in self.dataset.all_domains:
			for constraint in ['semi', 'book']:
				for slot in full_bs[domain][constraint]:
					if slot == 'booked':
						continue
					value = full_bs[domain][constraint][slot]
					if value != '':
						print('\t{}-{} => {}'.format(domain, slot, value))


	def get_loss(self, batch):
		loss = {}
		# reconstruction loss of word sequence
		loss_word_usr = NLLEntropyValid(self.logits['word_usr'], batch['word_idx']['out_usr'], batch['valid_turn'], ignore_idx=self.dataset.vocab['<PAD>'])
		loss_word_sys = NLLEntropyValid(self.logits['word_sys'], batch['word_idx']['out_sys'], batch['valid_turn'], ignore_idx=self.dataset.vocab['<PAD>'])
		loss['word_usr'] = loss_word_usr.item()
		loss['word_sys'] = loss_word_sys.item()

		loss_act_usr = NLLEntropyValid(self.logits['act_usr'], batch['act_idx']['usr'], batch['valid_turn'], ignore_idx=self.dataset.act_vocab['<PAD>'])
		loss_act_sys = NLLEntropyValid(self.logits['act_sys'], batch['act_idx']['sys'], batch['valid_turn'], ignore_idx=self.dataset.act_vocab['<PAD>'])
		loss['act_usr'] = loss_act_usr.item()
		loss['act_sys'] = loss_act_sys.item()
		update_loss = loss_word_usr + loss_word_sys + loss_act_usr + loss_act_sys

		# add dst loss if necessary
		if not self.config.oracle_dst:
			dst_loss, dst_update_loss = self.sys.dst.get_loss(batch)
			update_loss += dst_update_loss
			loss['dst_slot'] = dst_loss['slot']
			loss['dst_value'] = dst_loss['value']
		return loss, update_loss


	def update_ewc(self, update_loss, lambda_ewc):
		ori_loss = update_loss.item()
		for i, (name, p) in enumerate(self.named_parameters()):
			if p.grad is not None:
				l = lambda_ewc * self.fisher[name].cuda() * (p - self.optpar[name].cuda()).pow(2)
				update_loss += l.sum()
		grad_norm = self.update(update_loss, 'sl')
		return grad_norm


	def update(self, update_loss, train_mode):
		assert train_mode in ['sl', 'rl_sys', 'rl_usr']
		if self.config.update == 'joint':
			update_loss.backward(retain_graph=True)
			grad_norm = nn.utils.clip_grad_norm_(self.parameters(), self.config.grad_clip)
			if train_mode == 'sl':
				self.optimizer.step()
				self.optimizer.zero_grad()
			elif train_mode == 'rl_sys':
				self.rl_sys_optimizer.step()
				self.rl_sys_optimizer.zero_grad()
			else:
				self.rl_usr_optimizer.step()
				self.rl_usr_optimizer.zero_grad()
			return grad_norm

		elif self.config.update == 'iterative':
			raise NotImplementedError


	def set_optimizer(self):
		if self.config.optimizer == 'adam':
			self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.config.lr)
			if not self.config.share_dial_rnn:
				self.rl_sys_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.sys.parameters()), lr=self.config.rl_lr)
				self.rl_usr_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.usr.parameters()), lr=self.config.rl_lr)
			else:
				self.rl_sys_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, \
					list(self.sys.parameters()) + list(self.dial_rnn.parameters())  ), lr=self.config.rl_lr)
				self.rl_usr_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, \
					list(self.usr.parameters()) + list(self.dial_rnn.parameters())  ), lr=self.config.rl_lr)
		else:
			raise NotImplementedError


	def saveModel(self, epoch):
		if not os.path.exists(self.config.model_dir):
			os.makedirs(self.config.model_dir)
		torch.save(self.state_dict(), self.config.model_dir + '/epoch-{}.pt'.format(str(epoch)))


	def loadModel(self, model_dir, epoch):
		model_name = model_dir + '/epoch-{}.pt'.format(str(epoch))
		self.load_state_dict(torch.load(model_name))


def collect_dial_interact(decode_all, decode_batch, side, batch, config, dataset):
	'''Collect decoded word, act seq and bs'''
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
