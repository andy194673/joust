import sys
import json
import random
import torch

from utils.util_dst import iter_dst_file, dict2list
from utils.check_turn_info import decide_turn_domain


class DataLoader():
	def __init__(self, config, load_src=False):
		self.load_src = load_src
		self.config = config
		self.all_domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'police', 'hospital']

		self.build_vocab()
		self.getGoalSlotList()

		# collect data
		self.all_data = json.load(open(config.data_path))
#		self.all_dialName = sorted(list(self.all_data.keys()))

		# dst data
		self.process_dst() # DST

		# delex data
		self.data = {'train': [], 'valid': [], 'test': []}
		if load_src:
			assert config.ft_method == 'ewc'
			self.parseData(config.src_train_path, 'train')
			self.parseData(config.src_valid_path, 'valid')
			self.parseData(config.src_test_path, 'test')
			self.init()
			self.data['train'] = self.data['train'][:config.fisher_sample]
		else: # normal case
			self.parseData(config.train_path, 'train')
			self.parseData(config.valid_path, 'valid')
			self.parseData(config.test_path, 'test')

		self.init()
		self.data['train'] = self.data['train'][:config.train_size]
#		self.data['train'] = self.data['train'][:100] # verify
#		self.data['valid'] = self.data['valid'][:300] # BACK
#		self.data['test'] = self.data['test'][:100]
		print('# of dialogues, train: {}, valid: {}, test: {}'.format(len(self.data['train']), len(self.data['valid']), len(self.data['test'])))
		print('# of dialogues, train: {}, valid: {}, test: {}'.format(len(self.data['train']), len(self.data['valid']), len(self.data['test'])), file=sys.stderr)


		# goal of train and valid data for interaction
		train_dial_name = [dialogue['dial_name'] for dialogue in self.data['train']]
		valid_dial_name = [dialogue['dial_name'] for dialogue in self.data['valid']]
		self.rl_dial_name = train_dial_name + valid_dial_name
		self.init_rl()
		print('# of goals for rl interaction (train + valid): {}'.format(len(self.rl_dial_name)))
		print('# of goals for rl interaction: {} (train + valid)'.format(len(self.rl_dial_name)), file=sys.stderr)

		print('Done data loading!', file=sys.stderr)


	def build_vocab(self):
		# vocab for delex output
		self.vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3} # word2id
		with open(self.config.word2count_path) as f:
			word2count = json.load(f)
		for i in range(self.config.vocab_size):
			w = word2count[i][0]
			self.vocab[w] = len(self.vocab)
		self.idx2word = {idx: w for w, idx in self.vocab.items()}

		# vocab for act output
		self.act_vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3} # word2id
		with open(self.config.act2count_path) as f:
			act2count = json.load(f)
		for i in range( min(self.config.vocab_size, len(act2count)) ):
			w = act2count[i][0]
			self.act_vocab[w] = len(self.act_vocab)
		self.idx2act = {idx: w for w, idx in self.act_vocab.items()}

		# vocab for dst slot
		self.slot_vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3} # word2id
		with open(self.config.dst_slot_list) as f:
			dst_slot_list = json.load(f)
		for i in range(len(dst_slot_list)):
			w = dst_slot_list[i]
			self.slot_vocab[w] = len(self.slot_vocab)
		self.idx2slot = {idx: w for w, idx in self.slot_vocab.items()}
		print('# of slot: {}, {}'.format(len(self.slot_vocab), len(self.idx2slot)))

#		return # DST
		# NOTE: prepare slot-specific value list
		with open(self.config.slot2value) as f:
			slot2value = json.load(f)
		self.value_vocab = {}
		self.idx2value = {}
		for slot, value_list in slot2value.items():
			self.value_vocab[slot] = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3} # word2id
			for value in value_list:
				self.value_vocab[slot][value] = len(self.value_vocab[slot])
			self.idx2value[slot] = {idx: w for w, idx in self.value_vocab[slot].items()}
#			print('# of value: {}, {} in slot {}'.format(len(self.value_vocab[slot]), len(self.idx2value[slot]), slot))

		# vocab for dst word
		self.dstWord_vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3} # word2id
		with open(self.config.dst2count_path) as f:
			dst2count = json.load(f)
		for i in range(self.config.dst_vocab_size):
			w = dst2count[i][0]
			self.dstWord_vocab[w] = len(self.dstWord_vocab)
		# NOTE: add words in value into dstWord_vocab to avoid those words in utterances being replaced by <unk>
		for value_idx in range(len(self.value_vocab['all'])):
			value = self.idx2value['all'][value_idx]
			for w in value.split():
				if w not in self.dstWord_vocab:
					self.dstWord_vocab[w] = len(self.dstWord_vocab)
#					print('add {} into dst vocab from value list'.format(w))
		self.idx2dstWord = {idx: w for w, idx in self.dstWord_vocab.items()}
		print('# of lex word: {}, {}'.format(len(self.dstWord_vocab), len(self.idx2dstWord)))


	def pad_seq(self, seqs, vocab):
#		assert src_type in ['word', 'act']
		seq_len = [len(seq) for seq in seqs]
		max_len = max(seq_len)
#		if src_type == 'word':
#			for seq in seqs: seq.extend( [self.vocab['<PAD>'] for _ in range(max_len-len(seq))] )
#		else: # act
#			for seq in seqs: seq.extend( [self.act_vocab['<PAD>'] for _ in range(max_len-len(seq))] )
		for seq in seqs: seq.extend( [vocab['<PAD>'] for _ in range(max_len-len(seq))] )
		return seq_len


	def next_batch_list(self, dType):
		'''
		Return:
			a list (len=max_dial_turn), each element is a batch that considers the same idx turns from dialogues
		'''

		if self.p >= len(self.data[dType]):
			return None

		if dType == 'train':
			dial_idx_range = range(self.p, min(self.p + self.config.batch_size, len(self.data[dType])))
			self.p += self.config.batch_size
		else: # valid/test
			dial_idx_range = range(self.p, min(self.p + self.config.eval_batch_size, len(self.data[dType])))
			self.p += self.config.eval_batch_size
			
		self.batch_size = len(dial_idx_range)

		# check max turns within considered dialogues
		dial_len = [self.data[dType][dial_idx]['dial_len'] for dial_idx in dial_idx_range]
#		max_dial_len = min(max(dial_len), self.config.max_dial_len // 2) # for two sides
		if dType == 'train': # set max dial len to save training time
			max_dial_len = min(max(dial_len), self.config.max_dial_len // 2) # for two sides
		else:
			max_dial_len = max(dial_len)

		batch_list = []
		for turn_idx in range(max_dial_len):
			# prepare a batch
			ctx_usr, ctx_sys, out_usr, out_sys = [], [], [], []
			act_usr, act_sys = [], []
			gs, bs, db = [], [], []
			# dst
			dst_ctx, dst_prev_bs_slot, dst_prev_bs_value, dst_curr_nlu_slot, dst_curr_nlu_value = [], [], [], [], []
			ref = { 'word': {'usr': [], 'sys': []}, 'act': {'usr': [], 'sys': []}, 'dst': {'bs': [], 'nlu_slot': []} } # reference string of word and act tokens / dst list
			full_bs, dial_name = [], []
			valid_turn = []
			for dial_idx in dial_idx_range:
				dial = self.data[dType][dial_idx]
#				if turn_idx >= dial['dial_len']:
#					continue

				# check if turn is valid for calculating loss
				valid = True if turn_idx < dial['dial_len'] else False

				# out word
				out_usr.append( list(dial['word_idx_usr'][turn_idx]) if valid else [self.vocab['<EOS>']] )
				out_sys.append( list(dial['word_idx_sys'][turn_idx]) if valid else [self.vocab['<EOS>']] )

				# ctx word
				if turn_idx == 0:
					s = [ self.vocab['hi'], self.vocab['<EOS>'] ] # dummy sentence
					ctx_usr.append(s)
				else:
					ctx_usr.append( list(dial['word_idx_sys'][turn_idx-1]) if valid else [self.vocab['<EOS>']] )

				# act
				act_usr.append( list(dial['act_idx_usr'][turn_idx]) if valid else [self.act_vocab['<EOS>']] )
				act_sys.append( list(dial['act_idx_sys'][turn_idx]) if valid else [self.act_vocab['<EOS>']] )

				# dst DST
				dst_ctx.append( list(dial['dst_input_utt'][turn_idx]) if valid else [self.dstWord_vocab['<EOS>']] )
				dst_prev_bs_slot.append( list(dial['dst_prev_bs_slot'][turn_idx]) if valid else [self.slot_vocab['<EOS>']] )
				dst_prev_bs_value.append( list(dial['dst_prev_bs_value'][turn_idx]) if valid else [self.value_vocab['all']['<EOS>']] )
				dst_curr_nlu_slot.append( list(dial['dst_curr_nlu_slot'][turn_idx]) if valid else [self.slot_vocab['<EOS>']] )
				dst_curr_nlu_value.append( list(dial['dst_curr_nlu_value'][turn_idx]) if valid else [self.value_vocab['all']['<EOS>']] )

				# add info
				gs.append( dial['gs'][turn_idx] if valid else [0]*self.config.gs_size )
				db.append( dial['db'][turn_idx] if valid else [0]*self.config.db_size )
				bs.append( dial['bs'][turn_idx] if valid else [0]*self.config.bs_size )
				full_bs.append( dial['full_bs'][turn_idx] if valid else {})

				# ref
				ref['word']['usr'].append( dial['word_ref_usr'][turn_idx] if valid else 'invalid turn' )
				ref['word']['sys'].append( dial['word_ref_sys'][turn_idx] if valid else 'invalid turn' )
				ref['act']['usr'].append( dial['act_ref_usr'][turn_idx] if valid else 'invalid turn' )
				ref['act']['sys'].append( dial['act_ref_sys'][turn_idx] if valid else 'invalid turn' )
				ref['dst']['bs'].append( dial['dst_ref_bs'][turn_idx] if valid else {} ) # a dict, DST
				ref['dst']['nlu_slot'].append( dial['dst_curr_nlu_slot_token'][turn_idx] if valid else [] ) # a list

				dial_name.append(dial['dial_name'])
				valid_turn.append(valid)

			# pad seq after collecting of a batch
			out_usr_len = self.pad_seq(out_usr, self.vocab)
			out_sys_len = self.pad_seq(out_sys, self.vocab)
			ctx_usr_len = self.pad_seq(ctx_usr, self.vocab)
			ctx_sys, ctx_sys_len = out_usr, out_usr_len
			act_usr_len = self.pad_seq(act_usr, self.act_vocab)
			act_sys_len = self.pad_seq(act_sys, self.act_vocab)
			dst_ctx_len = self.pad_seq(dst_ctx, self.dstWord_vocab) # DST
			dst_prev_bs_slot_len = self.pad_seq(dst_prev_bs_slot, self.slot_vocab)
			dst_prev_bs_value_len = self.pad_seq(dst_prev_bs_value, self.value_vocab['all'])
			dst_curr_nlu_slot_len = self.pad_seq(dst_curr_nlu_slot, self.slot_vocab)
			dst_curr_nlu_value_len = self.pad_seq(dst_curr_nlu_value, self.value_vocab['all'])
			assert dst_prev_bs_slot_len == dst_prev_bs_value_len
			assert dst_curr_nlu_slot_len == dst_curr_nlu_value_len

			# add a tensor of batch into output list
			batch = {'word_idx': {}, 'sent_len': {}, 'act_idx': {}, 'dst_idx': {}}
			# word id, (batch_size, max_len)
			batch['word_idx']['out_usr'] = torch.tensor(out_usr).long().cuda() # (B, T)
			batch['word_idx']['out_sys'] = torch.tensor(out_sys).long().cuda()
			batch['word_idx']['ctx_usr'] = torch.tensor(ctx_usr).long().cuda()
			batch['word_idx']['ctx_sys'] = torch.tensor(ctx_sys).long().cuda()
			batch['act_idx']['usr'] = torch.tensor(act_usr).long().cuda()
			batch['act_idx']['sys'] = torch.tensor(act_sys).long().cuda()
			batch['dst_idx']['dst_ctx'] = torch.tensor(dst_ctx).long().cuda() # DST
			batch['dst_idx']['prev_bs_slot'] = torch.tensor(dst_prev_bs_slot).long().cuda()
			batch['dst_idx']['prev_bs_value'] = torch.tensor(dst_prev_bs_value).long().cuda()
			batch['dst_idx']['curr_nlu_slot'] = torch.tensor(dst_curr_nlu_slot).long().cuda()
			batch['dst_idx']['curr_nlu_value'] = torch.tensor(dst_curr_nlu_value).long().cuda()
			# sent len, (total_turn, )
			batch['sent_len']['out_usr'] = torch.tensor(out_usr_len).long().cuda() # (B, )
			batch['sent_len']['out_sys'] = torch.tensor(out_sys_len).long().cuda()
			batch['sent_len']['ctx_usr'] = torch.tensor(ctx_usr_len).long().cuda()
			batch['sent_len']['ctx_sys'] = torch.tensor(ctx_sys_len).long().cuda()
			batch['sent_len']['act_usr'] = torch.tensor(act_usr_len).long().cuda()
			batch['sent_len']['act_sys'] = torch.tensor(act_sys_len).long().cuda()
			batch['sent_len']['dst_ctx'] = torch.tensor(dst_ctx_len).long().cuda() # DST
			batch['sent_len']['prev_bs_slot'] = torch.tensor(dst_prev_bs_slot_len).long().cuda()
			batch['sent_len']['prev_bs_value'] = torch.tensor(dst_prev_bs_value_len).long().cuda()
			batch['sent_len']['curr_nlu_slot'] = torch.tensor(dst_curr_nlu_slot_len).long().cuda()
			batch['sent_len']['curr_nlu_value'] = torch.tensor(dst_curr_nlu_value_len).long().cuda()
			# add info
			batch['gs'] = torch.tensor(gs).float().cuda() # (B, feat_size)
			batch['bs'] = torch.tensor(bs).float().cuda() # (B, feat_size)
			batch['db'] = torch.tensor(db).float().cuda() # (B, feat_size)
			batch['full_bs'] = full_bs
			# refernece string
			batch['ref'] = ref
			batch['dial_name'] = dial_name
#			batch['dial_len'] = torch.tensor(dialLen).long().cuda()
			batch['dial_len'] = dial_len
			batch['valid_turn'] = valid_turn # list of binary
			if self.config.share_dial_rnn:
				batch['init_dial_rnn'] = None
			else:
				batch['init_dial_rnn'] = {'usr': None, 'sys': None}

			batch_list.append(batch)

		# add prev act idx
		for turn_idx, batch in enumerate(batch_list) :
			batch['prev_act_idx'] = {}
			if turn_idx == 0: # empty prev act for first turn
				batch['prev_act_idx']['usr'] = self.act_vocab['<EOS>']*torch.ones(self.batch_size, 1).long().cuda()
				batch['prev_act_idx']['sys'] = self.act_vocab['<EOS>']*torch.ones(self.batch_size, 1).long().cuda()
				batch['sent_len']['prev_act_usr'] = torch.ones(self.batch_size).long().cuda()
				batch['sent_len']['prev_act_sys'] = torch.ones(self.batch_size).long().cuda()
			else:
				batch['prev_act_idx']['usr'] = batch_list[turn_idx-1]['act_idx']['usr']
				batch['prev_act_idx']['sys'] = batch_list[turn_idx-1]['act_idx']['sys']
				batch['sent_len']['prev_act_usr'] = batch_list[turn_idx-1]['sent_len']['act_usr']
				batch['sent_len']['prev_act_sys'] = batch_list[turn_idx-1]['sent_len']['act_sys']

#		# print data, verify correctness!!
#		def seqTensor2token(tensor, idx2token, key, actual_len):
#			seq = tensor.tolist()
#			seq = ' | '.join([idx2token[idx] for idx in seq])
#			print('{}({}): {}'.format(key, actual_len.item(), seq))

#		for dial_idx in dial_idx_range:
#			dial = self.data[dType][dial_idx]
#			print('dial_name: {}, dial_len: {}'.format(dial['dial_name'], dial['dial_len']))

#		for turn_idx, batch in enumerate(batch_list):
#			for b in range(self.batch_size):
#				print('B_idx: {}, {}, turn {}, valid or not: {}'.format(b, batch['dial_name'][b], turn_idx, batch['valid_turn'][b]))
#				seqTensor2token(batch['word_idx']['ctx_usr'][b, :], self.idx2word, 'ctx_usr', batch['sent_len']['ctx_usr'][b])
#				seqTensor2token(batch['word_idx']['out_usr'][b, :], self.idx2word, 'out_usr', batch['sent_len']['out_usr'][b])
#				seqTensor2token(batch['word_idx']['ctx_sys'][b, :], self.idx2word, 'ctx_sys', batch['sent_len']['ctx_sys'][b])
#				seqTensor2token(batch['word_idx']['out_sys'][b, :], self.idx2word, 'out_sys', batch['sent_len']['out_sys'][b])
#				seqTensor2token(batch['prev_act_idx']['usr'][b, :], self.idx2act, 'prev_act_usr', batch['sent_len']['prev_act_usr'][b])
#				seqTensor2token(batch['prev_act_idx']['sys'][b, :], self.idx2act, 'prev_act_sys', batch['sent_len']['prev_act_sys'][b])
#				seqTensor2token(batch['act_idx']['usr'][b, :], self.idx2act, 'act_usr', batch['sent_len']['act_usr'][b])
#				seqTensor2token(batch['act_idx']['sys'][b, :], self.idx2act, 'act_sys', batch['sent_len']['act_sys'][b])
#				seqTensor2token(batch['dst_idx']['dst_ctx'][b, :], self.idx2dstWord, 'dst_ctx', batch['sent_len']['dst_ctx'][b])
#				seqTensor2token(batch['dst_idx']['prev_bs_slot'][b, :], self.idx2slot, 'prev_bs_slot', batch['sent_len']['prev_bs_slot'][b])
#				seqTensor2token(batch['dst_idx']['curr_nlu_slot'][b, :], self.idx2slot, 'curr_nlu_slot', batch['sent_len']['curr_nlu_slot'][b])
##				seqTensor2token(batch['dst_idx']['curr_nlu_value'][b, :], self.idx2value, 'curr_nlu_value', batch['sent_len']['curr_nlu_value'][b])
#				seqTensor2token(batch['dst_idx']['curr_nlu_value'][b, :], self.idx2value['all'], 'curr_nlu_value', batch['sent_len']['curr_nlu_value'][b])
#				print('bs_ref:', dict2list(batch['ref']['dst'][b]))
#				print('')
#			input('press...')
#			print('--------------------------------------------------------------------------------------')
#		input('done batch list...')

		return batch_list


	def init(self):
		self.p = 0 # data pointer
		if self.config.shuffle:
			random.shuffle(self.data['train'])


	def init_rl(self):
		self.rl_p = 0 # data pointer for rl
#		random.shuffle(self.all_dialName)
		random.shuffle(self.rl_dial_name)


	def parseData(self, path, dType):
		with open(path) as f:
			data = json.load(f)

		not_in_dst = []
		for dial_count, dial_name in enumerate(sorted(data.keys())):
#			print(dial_name)
#			input('press...')
#			if dial_count < 300:
#				continue

			# filter out dialogues without dst label, DST
			if dial_name not in self.dst_data:
#				print('filter out {} without dst label'.format(dial_name)) # would filter out dialogues with hospital/police domain
				not_in_dst.append(dial_name)
				continue

			dial = data[dial_name]
			dst_dial = self.dst_data[dial_name] # DST
			assert len(dial['sys']) == len(dial['db']) == len(dial['usr']) == len(dial['bs']) == len(dial['usr_act']) == len(dial['sys_act'])
			dial_len = len(dial['sys']) # dial len on one side
			assert dial_len == len(dst_dial['input_utt']) # DST

			# get the initial goal vector
			goal = self.all_data[dial_name]['goal']
			goal_vec = self.getGoalVector(goal)
#			print(dial_name)
#			for slot_idx, slot in enumerate(self.goalSlotList):
#				print(slot, goal_vec[slot_idx])
#			sys.exit(1)

			dialogue = {'word_idx_usr': [], 'word_idx_sys': [], 'word_ref_usr': [], 'word_ref_sys': [], \
						'act_idx_usr': [], 'act_idx_sys': [], 'act_ref_usr': [], 'act_ref_sys': [], \
						'bs': dial['bs'], 'db': dial['db'], 'gs': [], \
						'full_bs': [], \
						# DST
						'dst_ref_bs': dst_dial['curr_bs'], 'dst_input_utt': [], \
						'dst_prev_bs_slot': [], 'dst_prev_bs_value': [], \
						'dst_curr_nlu_slot': [], 'dst_curr_nlu_value': [], \
						'dst_curr_nlu_slot_token': [], \
						'dial_len': dial_len, 'dial_name': dial_name}
#						'dial_len': dial_len, 'dial_name': dial_name}

			domain_prev = 'none'
			booked_domain = set()
			for i in range(dial_len):
#				print('side_idx:', i)
				assert len(dial['bs'][i]) == self.config.bs_size
				assert len(dial['db'][i]) == self.config.db_size
				word_idx_usr = self.parseSent(dial['usr'][i], self.vocab)
				word_idx_sys = self.parseSent(dial['sys'][i], self.vocab)

#				act_idx_usr = self.parseSent(dial['usr_act'][i], self.act_vocab)
#				act_idx_sys = self.parseSent(dial['sys_act'][i], self.act_vocab)
#				assert self.act_vocab['<UNK>'] not in act_idx_usr
#				assert self.act_vocab['<UNK>'] not in act_idx_sys

				# recover turn_domain to replace booking token in act seq
				usr_act, sys_act = dial['usr_act'][i], dial['sys_act'][i]
				turn_domain = decide_turn_domain(usr_act, sys_act, domain_prev)
				if 'booking' in usr_act and turn_domain in ['restaurant', 'hotel', 'train']:
					usr_act = usr_act.replace('booking', turn_domain)
				if 'booking' in sys_act and turn_domain in ['restaurant', 'hotel', 'train']:
					sys_act = sys_act.replace('booking', turn_domain)
				act_idx_usr = self.parseSent(usr_act, self.act_vocab)
				act_idx_sys = self.parseSent(sys_act, self.act_vocab)
#				if self.act_vocab['<UNK>'] in act_idx_usr:
#					print(usr_act, file=sys.stderr)
#					input('')
#				if self.act_vocab['<UNK>'] in act_idx_sys:
#					print(sys_act, file=sys.stderr)
#					input('')
				domain_prev = turn_domain
				
				dialogue['word_idx_usr'].append(word_idx_usr)
				dialogue['word_idx_sys'].append(word_idx_sys)
				dialogue['word_ref_usr'].append(dial['usr'][i])
				dialogue['word_ref_sys'].append(dial['sys'][i])
				dialogue['act_idx_usr'].append(act_idx_usr)
				dialogue['act_idx_sys'].append(act_idx_sys)
				dialogue['act_ref_usr'].append(dial['usr_act'][i])
				dialogue['act_ref_sys'].append(dial['sys_act'][i])
				dialogue['full_bs'].append( self.all_data[dial_name]['log'][2*i+1]['metadata'] ) # oracle full bs

				# dst DST
				history = ' ; '.join( dst_dial['history'][i][-self.config.dst_hst_len: ] )
				input_utt_idx = self.parseSent(history, self.dstWord_vocab)
				dialogue['dst_input_utt'].append(input_utt_idx)

				prev_bs_slot = [ x.split('=')[0] for x in dict2list(dst_dial['prev_bs'][i]) ] # take domain-slot
				prev_bs_slot_idx = self.parseSent(prev_bs_slot, self.slot_vocab)
				dialogue['dst_prev_bs_slot'].append(prev_bs_slot_idx)

				prev_bs_value = [ x.split('=')[1] for x in dict2list(dst_dial['prev_bs'][i]) ] # take value
				prev_bs_value_idx = self.parseSent(prev_bs_value, self.value_vocab['all'])
				dialogue['dst_prev_bs_value'].append(prev_bs_value_idx)

				if self.config.dst_pred_type == 'nlu':
					curr_nlu_slot = [ x.split('=')[0] for x in dict2list(dst_dial['curr_nlu'][i]) ]
					curr_nlu_slot_idx = self.parseSent(curr_nlu_slot, self.slot_vocab)
					curr_nlu_value = [ x.split('=')[1] for x in dict2list(dst_dial['curr_nlu'][i]) ]
					curr_nlu_value_idx = self.parseSent(curr_nlu_value, self.value_vocab['all'])
				else: # bs
					curr_nlu_slot = [ x.split('=')[0] for x in dict2list(dst_dial['curr_bs'][i]) ]
					curr_nlu_slot_idx = self.parseSent(curr_nlu_slot, self.slot_vocab)
					curr_nlu_value = [ x.split('=')[1] for x in dict2list(dst_dial['curr_bs'][i]) ]
					curr_nlu_value_idx = self.parseSent(curr_nlu_value, self.value_vocab['all'])
				dialogue['dst_curr_nlu_slot'].append(curr_nlu_slot_idx)
				dialogue['dst_curr_nlu_slot_token'].append(curr_nlu_slot) # a list of slot token w/o eos
				dialogue['dst_curr_nlu_value'].append(curr_nlu_value_idx)

#				print(dst_dial['prev_bs'][i], '\n', prev_bs_slot)
#				print(dst_dial['curr_nlu'][i], '\n', curr_nlu_slot, '\n', curr_nlu_value)
#				input('press...')

				# prepare goal state
				if self.config.goal_state_change != 'none' and i != 0: # dynamic goal state
#					goal_vec = self.changeGoalState(goal_vec, dial_name, i, \
					full_bs_prev = self.all_data[dial_name]['log'][2*(i-1)+1]['metadata']
					goal_vec = self.changeGoalState(goal_vec, goal, full_bs_prev, \
						dial['usr_act'][i-1], dial['usr'][i-1], dial['sys_act'][i-1], dial['sys'][i-1], booked_domain)

				assert len(goal_vec) == self.config.gs_size
				dialogue['gs'].append(goal_vec)

				# switch to scan the process of goal slots being changed
#				self.printActiveGoalSlot(goal_vec)
#				input('press...')

			self.data[dType].append(dialogue)

#			if sum(goal_vec) > 0:
#				print(dial_count, dial_name, '> 0')
#				self.printActiveGoalSlot(goal_vec)
#				input('press...')

		print('filter out {} dialogues without dst label'.format(len(not_in_dst)))
			

	def printActiveGoalSlot(self, goal_vec):
		for idx, token in enumerate(self.goalSlotList):
			if goal_vec[idx] == 1:
				print(token, end=' | ')
		print('count:', sum(goal_vec))


	def findSlotAct(self, act_seq, slot):
		'''
		return the act that the slot belongs to in the act_seq
		e.g., act_seq: 'act_inform restaurant_choice act_request restaurant_food', slot: 'restaurant_food' => return 'act_request'
		'''
		act_tokens = act_seq.split()
		slot_idx = act_tokens.index(slot)
		act_idx = -1
		for idx in range(slot_idx, -1, -1): # scan back from slot index
			if 'act_' in act_tokens[idx]:
				act_idx = idx
				break
#		assert act_idx != -1 # there must be an act for a slot
		return act_tokens[act_idx]


	def changeGoalState(self, old_goal_vec, goal, full_bs_prev, usr_act_prev, usr_word_prev, sys_act_prev, sys_word_prev, booked_domain):
		'''

		'''
		# record booked/offerbook domain
		for domain in self.all_domains:
			if 'act_offerbooked {}'.format(domain) in sys_act_prev or \
				'act_offerbook {}'.format(domain) in sys_act_prev:
				booked_domain.add(domain)

		new_goal_vec = list(old_goal_vec) # to avoid in-place operation
		for idx, token in enumerate(self.goalSlotList):
			if new_goal_vec[idx] == 0: # this slot is already turned off 
				continue
			domain, slot_type, slot = token.split('_')

			# turn off some missed info slots when dialogue flow is at booking stage
			if self.config.goal_state_change in ['finish', 'both']:
				if slot_type == 'info' and domain in booked_domain:
					new_goal_vec[idx] = 0
##					print('\tturn off by book finish ->', token) # switch
					continue

			if slot_type == 'info':
				if self.config.goal_state_change in ['smooth', 'both']:
					domain_slot = '{}_{}'.format(domain, slot) # e.g., restaurant_pricerange
					if domain_slot in usr_act_prev or domain_slot in usr_word_prev: # check if the slot is informed by usr at last turn
						new_goal_vec[idx] = 0
##						print('\tturn off by info smooth ->', token)
						continue

				if self.config.goal_state_change in ['finish', 'both']:
					if domain == 'train': # train_id
						name = 'id'
					elif domain == 'taxi': # taxi_type
						name = 'type'
					else:
						name = 'name'
					domain_name = '{}_{}'.format(domain, name) # e.g., restaurant_name
#					turn_idx = 2*(side_idx-1)+1 # turn_idx on sys side at previous turn
#					domain_bs = self.all_data[dial_name]['log'][turn_idx]['metadata'][domain]
					domain_bs = full_bs_prev[domain]
#					domain_goal = self.all_data[dial_name]['goal'][domain]
					domain_goal = goal[domain]

					# check if sys provided entity and dst is correct
					if domain_name in sys_word_prev and self.evaluate_domainBS_scanBS(domain_bs, domain_goal, domain) == 1:
						new_goal_vec[idx] = 0
##						print('\tturn off by info finish ->', token)

			elif slot_type == 'book':
				if self.config.goal_state_change in ['smooth', 'both']:
					domain_slot = '{}_{}'.format(domain, slot) # e.g., hotel_people
					if domain_slot in usr_act_prev or domain_slot in usr_word_prev: # check if the slot is informed by usr at last turn
						new_goal_vec[idx] = 0
##						print('\tturn off by book smooth ->', token)
						continue

				if self.config.goal_state_change in ['finish', 'both']:
#					if 'act_offerbooked' in sys_act_prev: # check if sys informed booking is finished at last turn
					if 'act_offerbooked {}'.format(domain) in sys_act_prev:
						new_goal_vec[idx] = 0
##						print('\tturn off by book finish ->', token)

			elif slot_type == 'reqt':
#				if self.config.goal_state_change in ['smooth', 'both']:
				domain_slot = '{}_{}'.format(domain, slot) # e.g., hotel_postcode
				# check if reqt slot is answered by sys at last turn, make sure it's along with answer type action
				if domain_slot in sys_word_prev or \
					( domain_slot in sys_act_prev and self.findSlotAct(sys_act_prev, domain_slot) in ['act_inform', 'act_recommend', 'act_offerbooked', 'act_offerbook'] ):
					new_goal_vec[idx] = 0
##					print('\tturn off by reqt ->', token)
				if domain_slot == 'train_trainID' and 'train_id' in sys_word_prev: # the only slot where delex term differs than slot token
					new_goal_vec[idx] = 0

			else:
				print('Unknown slot type, should not happen')
				sys.exit(1)

		return new_goal_vec
	

	def getGoalSlotList(self):
		'''
		obtain a list of all possible slots in goal ontology, such as 'attraction_info_area', 'train_info_departure'
		'''
		onto = json.load(open(self.config.ontology_goal_path))
		self.goal_ontology = onto
		goalSlotList = []
		for domain in self.all_domains:
			for slot_type in ['info', 'book']:
				if not onto[domain][slot_type]:
					continue
				for slot in sorted(onto[domain][slot_type].keys()):
					if 'valid' in slot:
						continue
					goalSlotList.append('{}_{}_{}'.format(domain, slot_type, self._unify_goalSlot(slot)))
#					goalSlotList.append('{}_{}_{}'.format(domain, slot_type, unify_slot(slot, domain)))
			for slot in sorted(onto[domain]['reqt']):
				if 'valid' in slot:
					continue
				goalSlotList.append('{}_{}_{}'.format(domain, 'reqt', self._unify_goalSlot(slot)))
#				goalSlotList.append('{}_{}_{}'.format(domain, 'reqt', unify_slot(slot, domain)))
				
		self.goalSlotList = goalSlotList
#		for x in goalSlotList:
#			print(x)
#		print(len(goalSlotList))
#		sys.exit(1)
		
	def _unify_goalSlot(self, slot):
		'''
		it is necessary to call this func when dealing with goal/dst slot
		'''
		if slot == 'entrance fee':
			return 'fee'
		elif slot == 'car type':
			return 'type'
		return slot


	def getGoalVector(self, goal):
		'''
		goal_state is composed of domains lists, each domain list is composed of info slots, book slots and reqt slots
		e.g., restaurant(info_list + book_list + reqt_list) + hotel(info_list + book_list + reqt_list) + ...
		'''
		goalVector = [0 for _ in range(len(self.goalSlotList))]
		for domain in self.all_domains:
			if not goal[domain]:
				continue
			for slot_type in goal[domain]:
				if slot_type not in ['info', 'book', 'reqt']:
					continue
				if slot_type == 'reqt':
					assert isinstance(goal[domain][slot_type], list)
					for slot in goal[domain][slot_type]:
						if 'valid' in slot:
							continue
						slot_idx = self.goalSlotList.index('{}_{}_{}'.format(domain, slot_type, self._unify_goalSlot(slot)))
#						slot_idx = self.goalSlotList.index('{}_{}_{}'.format(domain, slot_type, unify_slot(slot, domain)))
						goalVector[slot_idx] = 1
				else: # info/book
					assert isinstance(goal[domain][slot_type], dict)
					for slot in goal[domain][slot_type].keys():
						if 'valid' in slot:
							continue
						if goal[domain][slot_type][slot] in ['', 'dontcare']: # NOTE: no need for dontcare slot in goal state
							continue
						slot_idx = self.goalSlotList.index('{}_{}_{}'.format(domain, slot_type, self._unify_goalSlot(slot)))
#						slot_idx = self.goalSlotList.index('{}_{}_{}'.format(domain, slot_type, unify_slot(slot, domain)))
						goalVector[slot_idx] = 1
		return goalVector


	def parseSent(self, sent, vocab):
		assert isinstance(sent, str) or isinstance(sent, list)
		if isinstance(sent, str):
			sent = sent.split()

		sent_idx = []
#		for tok in sent.split():
		for tok in sent:
			tok_idx = vocab[tok] if tok in vocab else vocab['<UNK>']
			sent_idx.append(tok_idx)
#			sent_idx.append(tok) # verify
		sent_idx.append(vocab['<EOS>']) # add eos at the end of sentence
#		sent_idx.append('<EOS>') # verify
		return sent_idx


	def next_rl_batch(self):
#		dial_names = self.all_dialName[self.rl_p: self.rl_p +self.config.rl_batch_size]
		dial_names = self.rl_dial_name[self.rl_p: self.rl_p +self.config.rl_batch_size]
		self.rl_p += self.config.rl_batch_size
		return dial_names


	def process_dst(self):
		with open(self.config.dst_slot_list) as f:
			dst_slot_list = json.load(f)

		dst_cont = {} # data container
		dst_files = {'train': self.config.dst_train_path, 'valid': self.config.dst_valid_path, 'test': self.config.dst_test_path}
		if self.load_src:
			delex_files = {'train': self.config.src_train_path, 'valid': self.config.src_valid_path, 'test': self.config.src_test_path}
		else:
			delex_files = {'train': self.config.train_path, 'valid': self.config.valid_path, 'test': self.config.test_path}
		for data_type in ['train', 'valid', 'test']:
			dst_f, delex_f = dst_files[data_type], delex_files[data_type]
			with open(dst_f) as f1, open(delex_f) as f2:
				dst_data, delex_data = json.load(f1), json.load(f2)
#				iter_dst_file(dst_cont, dst_data, delex_data, dst_slot_list)
				iter_dst_file(dst_cont, dst_data, delex_data, dst_slot_list, \
								remove_dontcare=self.config.remove_dontcare, fix_wrong_domain=self.config.fix_wrong_domain)
		self.dst_data = dst_cont
		print('# of dialogues in dst data:', len(self.dst_data))


	def evaluate_domainBS_scanBS(self, domain_bs, domain_goal, domain):
		'''Check if detected info in the given bs is correct according to the goal (check only informable slot)'''
		for slot, value in domain_bs['semi'].items():
			if value in ['dont care', "don't care", 'dontcare', "do n't care"]:
				continue
			if slot not in domain_goal['info']:  # might have 'not_mentioned' slot in bs but not in goal
				continue
			if domain_goal['info'][slot] in ['dont care', "don't care", 'dontcare', "do n't care"]:
				continue
			if slot == 'type' and domain == 'hotel':
				continue
			if value != domain_goal['info'][slot]:
				return 0
		return 1