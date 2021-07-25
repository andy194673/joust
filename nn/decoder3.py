import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
#from torch.autograd import Variable
from nn.attention import Attn
import copy
import time
import sys
import random

class Decoder(nn.Module):
#	def __init__(self, config, input_size, vocab, idx2word, peep_size, enc_size, output_type):
	def __init__(self, config, vocab, idx2word, add_size, output_type, attn_src):
		super(Decoder, self).__init__()
		self.config = config
		vocab_size = len(vocab)
		self.hidden_size = config.hidden_size
		self.output_size = vocab_size

		self.dropout = nn.Dropout(p=config.dropout)
		self.embed = nn.Embedding(vocab_size, config.embed_size)

		rnn_input_size = self.config.embed_size + add_size
		self.attn_src = attn_src
		self.attn = nn.ModuleDict()

		for src in attn_src:
			rnn_input_size += config.hidden_size
			self.attn[src] = Attn(config.hidden_size, self.hidden_size)

		rnn_output_size = self.hidden_size
		self.rnn = nn.LSTM(rnn_input_size, rnn_output_size, num_layers=config.num_layers, \
						dropout=config.dropout, bidirectional=False, batch_first=True)
		self.hidden2output = nn.Linear(rnn_output_size, self.output_size)

		self.word2idx = vocab
		self.idx2word = idx2word
		assert output_type in ['act', 'word', 'dst_slot']
		if output_type == 'act':
			self.dec_len = config.max_act_dec_len
#		else: # word
		elif output_type == 'word':
			self.dec_len = config.max_word_dec_len
		else: # dst_slot
			self.dec_len = config.max_slot_dec_len
		self.output_type = output_type

		self.rnn_output_size = rnn_output_size

	def _step(self, t, input_emb, prev_state, add_var, enc_output, enc_mask, CTX_VEC):
#		if add_var != None:
		if isinstance(add_var, torch.Tensor):
			step_input = [input_emb, add_var]
		else:
			step_input = [input_emb]
		attn_query = prev_state[0].permute(1, 0, 2) # (L=1, B, H) -> (B, L=1, H)
		for src in self.attn_src:
#			print('output: {}, SRC: {}'.format(self.output_type, src), file=sys.stderr)
#			print('query: {}, enc: {}'.format(attn_query.size(), enc_output[src].size()), file=sys.stderr)
			attn_dist, ctx_vec = self.attn[src](attn_query, enc_output[src], enc_mask[src]) # (B, T) & (B, H)
			step_input.append(ctx_vec)

			if CTX_VEC != None:
				CTX_VEC[src][:, t, :] = ctx_vec

		step_input = torch.cat(step_input, dim=1) # (B, total_feat)

		output, state = self.rnn(step_input.unsqueeze(1), prev_state) # (B, 1, H) & tuple of (L, B, H)
		# NOTE: dropout is not applied in LSTM pytorch module at the last layer. need to manually apply one
		output = self.dropout(output.squeeze(1))

		# SOTA model
		output = self.hidden2output(output) # (B, V)
#		output = torch.cat([output, add_var], dim=1) # (B, H + feat_size)
#		output = self.hidden2output(output)
		return output, state

#	def forward(self, input_var, input_len, init_state, add_var, enc_output, enc_mask, mode='teacher_force', sample=False, beam_search=False):
	def forward(self, input_var, input_len, init_state, add_var, enc_output, enc_mask, mode='teacher_force', sample=False, beam_search=False, return_ctx_vec=False):
		'''
		Args:
			input_var: (B, T)
			input_len: (B,)
			init_state: tuple of (L, B, H)
			add_var: (B, feat), additional input to each time step 
			enc_output: a dict of encoded source, tensor (B, T, H), used for attention
			enc_mask: a dict of tensor (B, T) used for attention
		Return:
			output_prob: (B, T, V)
		'''
		if beam_search:
			output = self.beam_search(input_var, init_state, add_var, enc_output, enc_mask, beam_size=self.config.beam_size)
			output['decode'] = [sents[0] for sents in output['decode']] # take 1-best
			output['decode_len'] = [lens[0] for lens in output['decode_len']]
		else:
			output = self.greedy(input_var, input_len, init_state, add_var, enc_output, enc_mask, mode=mode, sample=sample, return_ctx_vec=return_ctx_vec)
		return output


	def greedy(self, input_var, input_len, init_state, add_var, enc_output, enc_mask, mode='teacher_force', sample=False, return_ctx_vec=False):
		self.batch_size = init_state[0].size(1)
		max_len = self.dec_len if mode == 'gen' else input_var.size(1)

		# NOTE: make sure to check legitimate arguments, such as mode cannot be 'teach_force'
		assert mode == 'teacher_force' or mode == 'gen'
		self.mode = mode
#		print(mode)

		assert isinstance(init_state, tuple)
		for src in self.attn_src:
			assert src in enc_output
			assert src in enc_mask

		go_idx = torch.tensor([self.word2idx['<SOS>'] for b in range(self.batch_size)]).long().cuda() # (B, )
		input_emb = self.dropout(self.embed(go_idx)) #.unsqueeze(1) # (B, E)

		# output container
		logits = torch.zeros(self.batch_size, max_len, self.output_size).cuda()
		logprobs = torch.zeros(self.batch_size, max_len).cuda()
#		hiddens = torch.zeros(self.batch_size, max_len, self.hidden_size).cuda() # if mode == 'gen' else None # (B, T, H)
		hiddens = torch.zeros(self.batch_size, max_len, self.rnn_output_size).cuda() # if mode == 'gen' else None # (B, T, H)
		sample_wordIdx = torch.zeros(self.batch_size, max_len).long().cuda() if mode == 'gen' else None
		sentences = [[] for b in range(self.batch_size)] if mode == 'gen' else None
		finish_flag = [0 for _ in range(self.batch_size)] if mode == 'gen' else None
		CTX_VEC = { src: torch.zeros(self.batch_size, max_len, self.hidden_size).cuda() for src in self.attn_src} if return_ctx_vec else None

		for t in range(max_len):
#			print('At time {}, input: {}, state: {}'.format(t, input_emb.size(), init_state[0].size()))
#			input('Decoding step {}'.format(t))
			output, state = self._step(t, input_emb, init_state, add_var, enc_output, enc_mask, CTX_VEC) # (B, V)

			# record hidden states
			assert isinstance(state, tuple)
			hiddens[:, t, :] = state[0].squeeze(0)

			logits[:, t, :] = output
			if mode == 'gen': # only sample when gen for speedup training
				self.logits2words(output, sentences, sample_wordIdx, logprobs, t, finish_flag, sample) # collect ouput word at each time step
			if mode == 'gen' and sum(finish_flag) == self.batch_size: # break if all sentences finish
				break

			if mode == 'teacher_force':
				idx = input_var[:, t]
			else:
				value, idx = torch.max(output, dim=1) # (B, )
			input_emb = self.dropout(self.embed(idx)) #.unsqueeze(1) # (B, E)
			init_state = state

		if mode == 'gen':
			sentences_len = [len(sent) for sent in sentences] # sentence length w/i eos
			sentences = [' '.join(sent[:-1]) for sent in sentences] # remove eos and convert to string
			# pad 0 in sample_wordIdx for generating samples
			for b, sent_len in enumerate(sentences_len):
#				print('b idx:', b, file=sys.stderr)
#				print(sent_len, sentences[b], file=sys.stderr)
#				print(sample_wordIdx[b], file=sys.stderr)
				if sent_len < max_len:
					assert sample_wordIdx[b, sent_len-1] == self.word2idx['<EOS>']
				sample_wordIdx[b, sent_len:] = 0
				logprobs[b, sent_len:] = 0

			# keep hiddens in a valid size since the longest sent might not be max_dec_len
			hiddens = hiddens[:, :max(sentences_len), :]
			if return_ctx_vec:
				for src in self.attn_src:
					CTX_VEC[src] = CTX_VEC[src][:, :max(sentences_len), :]

#		return logits, sentences
		if mode == 'gen':
			output = {'logits': logits, 'logprobs': logprobs, 'sample_wordIdx': sample_wordIdx, \
						'decode': sentences, 'decode_len': torch.tensor(sentences_len), 'hiddens': hiddens, 'ctx_vec': CTX_VEC, 'mode': mode}
		else:
			output = {'logits': logits, 'logprobs': logprobs, 'sample_wordIdx': None, \
						'decode': sentences, 'decode_len': input_len, 'hiddens': hiddens, 'ctx_vec': CTX_VEC, 'mode': mode}
		return output
	

	def logits2words(self, logits, sentences, sample_wordIdx, logprobs, t, finish_flag, sample):
		'''
		logits: (B, V)
		sample_wordIdx, logprobs: (B, T)
		'''
		if sample:
			T = 2 # temperature, > 0 to encourage explore
			probs = torch.softmax(logits/T, dim=1)
			cat = Categorical(probs)
			idx = cat.sample() # (B, )
			value = torch.gather(torch.softmax(logits, dim=1), dim=1, index=idx.unsqueeze(1)) # (B, )
			value = value.squeeze(1)
		else:
			value, idx = torch.max(torch.softmax(logits, dim=1), dim=1) # (B, )
		
		sample_wordIdx[:, t] = idx
		logprobs[:, t] = torch.log(value)
		for b_idx, (sentence, i) in enumerate(zip(sentences, idx)):
#			if i == self.word2idx['<EOS>'] or i == self.word2idx['<PAD>']:
			if len(sentence) > 0 and sentence[-1] == '<EOS>':
				finish_flag[b_idx] = 1
				continue
			sentence.append(self.idx2word[i.item()])


	def beam_search(self, input_var, init_state, latent_var, enc_output, enc_mask, beam_size=10):
		'''
		The speed up version of beam search
		'''
		t0 = time.time()
		self.mode = 'gen'
		t_input, t_cand, t_ff, t_sort, t_copy = 0, 0, 0, 0, 0
		t_cons, t_copy, t_his, t_log, t_state, t_pool, t_idx, t_beam = 0,0,0,0,0,0,0,0
		t_beam_size, t_left_size = 0, 0
		t_input1, t_input2 = 0, 0
#		print('using beam search')
		class Beam(object):
			def __init__(self, src_beam, state):
				if src_beam is not None:
					self.history = list(src_beam.history)
					self.logprob = src_beam.logprob.clone()
				else:
					self.history = ['<SOS>']
					self.logprob = torch.tensor(0).float()
					self.state = state # (L, 1, H)
					
#			def __init__(self, h, c):
#				self.history = ['<SOS>']
##				self.logprob = 0 # store a score instead of a list to save time during sorting
#				self.logprob = torch.tensor(0).float()
#				self.state = (h, c) # (L, 1, H)
#
#			def copy(self, beam_src):
#				self.history = []
#				self.history += beam_src.history
##				self.logprob = beam_src.logprob
##				if beam_src.logprob == 0:
##					self.logprob = 0
##				else:
#				self.logprob = beam_src.logprob.clone()
##				self.state = (beam_src.state[0].clone(), beam_src.state[1].clone()) # waste time
#				self.state = None

		assert isinstance(init_state, tuple)

#		batch_size = input_var.size(0)
		batch_size = init_state[0].size(1)
		alpha = 0.7 # length normalization coefficient
#		global_pool = [ [ Beam(init_state[0][:, b, :].unsqueeze(1), init_state[1][:, b, :].unsqueeze(1)) ] \
#							for b in range(batch_size)]
		global_pool = [ [ Beam(None, (init_state[0][:, b, :].unsqueeze(1), init_state[1][:, b, :].unsqueeze(1))) ] \
							for b in range(batch_size)]

		for step_t in range(self.dec_len):
#			print('step:', step_t)
			cand_pool = [ [] for _ in range(batch_size) ]
			logprob_pool =  [ [] for _ in range(batch_size) ] # list for only logprob of beams in cand_pool

			# put finished beam (ending with eos) directly to candidate pool and remove them from the input to next ff
			if step_t > 0:
				trim_pool = []
				for batch_idx in range(batch_size):
					_pool = []
					for beam_idx in range(beam_size):
						beam = global_pool[batch_idx][beam_idx]
						if beam.history[-1] == '<EOS>':
							cand_pool[batch_idx].append(beam)
							logprob_pool[batch_idx].append(beam.logprob)
						else:
							_pool.append(beam)
					trim_pool.append(_pool)
			else:
				trim_pool = global_pool

			num_leftBeam = [len(beams) for beams in trim_pool] # number of left beams in each example
			new_batch_size = sum(num_leftBeam)
			if new_batch_size == 0:
				break

			# run a rnn step
#			t = time.time()
			trim_pool = [beam for beams in trim_pool for beam in beams] # flatten trim pool for creating a batch
			input_idx = torch.tensor([self.word2idx[ trim_pool[l].history[-1] ] \
							for l in range(new_batch_size)]).long().cuda() # (B, )
			input_emb = self.dropout(self.embed(input_idx)).unsqueeze(1) # (B, 1, E)

			init_h = torch.cat([ trim_pool[l].state[0] for l in range(new_batch_size) ], dim=1) # (L, B, H)
			init_c = torch.cat([ trim_pool[l].state[1] for l in range(new_batch_size) ], dim=1)
			init_state = (init_h, init_c)
#			t_input1 += time.time()-t
#			tt = time.time()
			enc_output2, enc_mask2 = [], []
			latent_var2 = []
			for batch_idx, left_size in enumerate(num_leftBeam):
				for beam_idx in range(left_size):
#					enc_output2.append(enc_output[batch_idx].clone().unsqueeze(0)) # (1, T, H)
#					enc_mask2.append(enc_mask[batch_idx].clone().unsqueeze(0)) # (1, T)
					enc_output2.append(enc_output[batch_idx].unsqueeze(0)) # (1, T, H)
					enc_mask2.append(enc_mask[batch_idx].unsqueeze(0)) # (1, T)
					latent_var2.append(latent_var[batch_idx].unsqueeze(0)) # (1, H)
			enc_output2 = torch.cat(enc_output2, dim=0)
			enc_mask2 = torch.cat(enc_mask2, dim=0)
			latent_var2 = torch.cat(latent_var2, dim=0)
#			t_input2 += time.time()-tt
#			t_input += time.time()-t

			# get top k
#			t = time.time()
			output, state = self._step(input_emb, init_state, latent_var2, enc_output2, enc_mask2) # (B, V), (L, B, H)
			top_value, top_index = torch.topk(torch.softmax(output, dim=1), k=beam_size, dim=1) # (B, K)
#			t_ff += time.time()-t
			state = (state[0].unsqueeze(1), state[1].unsqueeze(1)) # batch unsqueeze to speed up

#			top_value = top_value.cpu().numpy() # cannot use numpy during training as it breaks graph
			top_value = top_value.cpu()#.numpy() # use cpu is faster for this samll tensor (B, beam_size)
			top_value = torch.log(top_value) # take batch log to speed up
			top_index = top_index.cpu().numpy()
			# prepare candidate beams for sorting
#			t = time.time()
			for batch_idx, left_size in enumerate(num_leftBeam):
				start_idx = sum(num_leftBeam[:batch_idx])
				for beam_idx in range(left_size):
					beam = trim_pool[start_idx+beam_idx]
					# take shared logprob/state for candidate beams to speed up
					beam_logprob = top_value[start_idx+beam_idx]
					beam_state = (state[0][:, :, start_idx+beam_idx, :], state[1][:, :, start_idx+beam_idx, :])
					for cand_idx in range(beam_size):
#						t1 = time.time()
#						cand_beam = Beam(None, None)
						cand_beam = Beam(beam, None)
#						t_cons += time.time()-t1 # 0.015/0.083 sec
#
#						t1 = time.time()
#						cand_beam.copy(beam)
#						t_copy += time.time()-t1 # 0.01/0.083 sec

						# NOTE: by avoiding using item() with numpy() first, it reduces half of decoding time
#						t1 = time.time()
#						cand_beam.history.append( self.idx2word[top_index[start_idx+beam_idx][cand_idx].item()] )
						cand_beam.history.append( self.idx2word[top_index[start_idx+beam_idx][cand_idx]] )
#						t_his += time.time()-t1 # 0.007/0.081 sec 

						# NOTE: be careful when updating logprob, DO NOT change logprob at previous step
#						t1 = time.time()
#						cand_beam.logprob += torch.log(top_value[start_idx+beam_idx][cand_idx]).item()
#						cand_beam.logprob += np.log(top_value[start_idx+beam_idx][cand_idx])
#						cand_beam.logprob += torch.log(top_value[start_idx+beam_idx][cand_idx])
#						cand_beam.logprob += top_value[start_idx+beam_idx][cand_idx]
						cand_beam.logprob += beam_logprob[cand_idx]
#						t_log += time.time()-t1 # 0.02/0.083 sec

#						if batch_idx == 0 and beam_idx == 0 and step_t == 1:
#							print('{}+{}'.format(cand_beam.logprob, torch.log(top_value[start_idx+beam_idx][cand_idx])))

#						t1 = time.time()
#						cand_beam.state = (state[0][:, start_idx+beam_idx, :].unsqueeze(1), \
#											state[1][:, start_idx+beam_idx, :].unsqueeze(1))
#						cand_beam.state = (state[0][:, :, start_idx+beam_idx, :], \
#											state[1][:, :, start_idx+beam_idx, :])
						cand_beam.state = beam_state
#						t_state += time.time()-t1 # 0.024/0.083 sec

#						t1 = time.time()
						cand_pool[batch_idx].append(cand_beam)
						logprob_pool[batch_idx].append(cand_beam.logprob)
#						t_pool += time.time()-t1 # 0.006/0.083 sec
#			t_cand += time.time()-t

			# sort candidate beams for each example AND update best beam so far
#			t = time.time()
			global_pool = []
			for batch_idx in range(batch_size):
				if len(cand_pool[batch_idx]) == beam_size:
					global_pool.append(cand_pool[batch_idx])
					continue
#				cand_pool[batch_idx] = sorted(cand_pool[batch_idx], \
#						key=lambda x: x.logprob/pow(len(x.history)-1, alpha), reverse=True)
##						key=lambda x: x.logprob.item()/pow(len(x.history)-1, alpha), reverse=True)
##						key=lambda x: x.logprob.cpu().numpy()/pow(len(x.history)-1, alpha), reverse=True)
#				global_pool.append(cand_pool[batch_idx][:beam_size])

				norm = torch.tensor([pow(len(beam.history)-1, alpha) for beam in cand_pool[batch_idx]]).float()
				_, indexes = torch.sort(torch.tensor(logprob_pool[batch_idx]).float()/norm,descending=True)
				cand_pool[batch_idx] = [cand_pool[batch_idx][idx] for idx in indexes.numpy()]
				global_pool.append(cand_pool[batch_idx][:beam_size])
#				indexes = np.argsort(torch.tensor(logprob_pool[batch_idx]).float().numpy())
#				cand_pool[batch_idx] = [ cand_pool[batch_idx][idx] for idx in indexes]
#			t_sort += time.time()-t

#		print('finish a batch: {:.3f}'.format(time.time()-t0))
#		print('input: {:.3f}, ff: {:.3f}, cand: {:.3f}, sort: {:.3f}'.format(t_input, t_ff, t_cand, t_sort))
#		print('in cand: {:.3f}'.format(t_cons+ t_copy+ t_his+ t_log+ t_state+ t_pool))
#		print('in cand: {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}'.format(t_cons, t_copy, t_his, t_log, t_state, t_pool))
#		print('input1: {:.3f}, input2: {:.3f}'.format(t_input1, t_input2))
#		sys.exit(1)

#		sentences_batch, sentences_len_batch, logprobs_batch = [], [], []
		sentences_batch, sentences_len_batch, logprobs_batch = [], [], torch.zeros(batch_size, beam_size)
		for batch_idx in range(batch_size):
			sentences, sentences_len, logprobs = [], [], []
			for beam_idx in range(beam_size):
				sentence_len = len(global_pool[batch_idx][beam_idx].history)-1 # miuns 1 for <SOS>
#				sentence = ' '.join(global_pool[batch_idx][beam_idx].history).replace('<SOS> ', '').replace(' <EOS>', '')
				global_pool[batch_idx][beam_idx].history.remove('<SOS>')
				if sentence_len < self.dec_len: global_pool[batch_idx][beam_idx].history.remove('<EOS>')
				sentence = ' '.join(global_pool[batch_idx][beam_idx].history)
				sentences.append(sentence)
				sentences_len.append(sentence_len)
#				logprobs.append(global_pool[batch_idx][beam_idx].logprob / sentence_len)
				logprobs_batch[batch_idx][beam_idx] = global_pool[batch_idx][beam_idx].logprob / sentence_len
			sentences_batch.append(sentences)
			sentences_len_batch.append(sentences_len)
#			logprobs_batch.append(logprobs)

		# NOTE: DO NOT create new tensor as it does not have grad_fn
#		logprobs_batch = torch.tensor(logprobs_batch).float().cuda()
		logprobs_batch = logprobs_batch.cuda()
		return {'decode': sentences_batch, 'logprobs': logprobs_batch, 'decode_len': sentences_len_batch, 'mode': 'gen'}
#			output = {'logits': logits, 'logprobs': logprobs, 'sample_wordIdx': sample_wordIdx, \
#						'decode': sentences, 'decode_len': torch.tensor(sentences_len), 'hiddens': hiddens, 'mode': mode}
#		hiddens = torch.zeros(self.batch_size, max_len, self.hidden_size).cuda() # if mode == 'gen' else None # (B, T, H)
