import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from networks.attention import Attention


class Decoder(nn.Module):
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
			self.attn[src] = Attention(config.hidden_size, self.hidden_size)

		rnn_output_size = self.hidden_size
		self.rnn = nn.LSTM(rnn_input_size, rnn_output_size, num_layers=config.num_layers,
						   dropout=config.dropout, bidirectional=False, batch_first=True)
		self.hidden2output = nn.Linear(rnn_output_size, self.output_size)

		self.word2idx = vocab
		self.idx2word = idx2word
		assert output_type in ['act', 'word', 'dst_slot']
		if output_type == 'act':
			self.dec_len = config.max_act_dec_len
		elif output_type == 'word':
			self.dec_len = config.max_word_dec_len
		else: # dst_slot
			self.dec_len = config.max_slot_dec_len
		self.output_type = output_type
		self.rnn_output_size = rnn_output_size


	def _step(self, t, input_emb, prev_state, add_var, enc_output, enc_mask, CTX_VEC):
		if isinstance(add_var, torch.Tensor):
			step_input = [input_emb, add_var]
		else:
			step_input = [input_emb]
		attn_query = prev_state[0].permute(1, 0, 2) # (L=1, B, H) -> (B, L=1, H)
		for src in self.attn_src:
			attn_dist, ctx_vec = self.attn[src](attn_query, enc_output[src], enc_mask[src]) # (B, T) & (B, H)
			step_input.append(ctx_vec)

			if CTX_VEC != None:
				CTX_VEC[src][:, t, :] = ctx_vec

		step_input = torch.cat(step_input, dim=1) # (B, total_feat)

		output, state = self.rnn(step_input.unsqueeze(1), prev_state) # (B, 1, H) & tuple of (L, B, H)
		# NOTE: dropout is not applied in LSTM pytorch module at the last layer. need to manually apply one
		output = self.dropout(output.squeeze(1))
		output = self.hidden2output(output) # (B, V)
		return output, state


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
		return self.greedy(input_var, input_len, init_state, add_var, enc_output, enc_mask, mode=mode, sample=sample, return_ctx_vec=return_ctx_vec)


	def greedy(self, input_var, input_len, init_state, add_var, enc_output, enc_mask, mode='teacher_force', sample=False, return_ctx_vec=False):
		self.batch_size = init_state[0].size(1)
		max_len = self.dec_len if mode == 'gen' else input_var.size(1)

		# NOTE: make sure to check legitimate arguments, such as mode cannot be 'teach_force'
		assert mode == 'teacher_force' or mode == 'gen'
		self.mode = mode

		assert isinstance(init_state, tuple)
		for src in self.attn_src:
			assert src in enc_output
			assert src in enc_mask

		go_idx = torch.tensor([self.word2idx['<SOS>'] for b in range(self.batch_size)]).long().cuda() # (B, )
		input_emb = self.dropout(self.embed(go_idx)) #.unsqueeze(1) # (B, E)

		# output container
		logits = torch.zeros(self.batch_size, max_len, self.output_size).cuda()
		logprobs = torch.zeros(self.batch_size, max_len).cuda()
		hiddens = torch.zeros(self.batch_size, max_len, self.rnn_output_size).cuda() # if mode == 'gen' else None # (B, T, H)
		sample_wordIdx = torch.zeros(self.batch_size, max_len).long().cuda() if mode == 'gen' else None
		sentences = [[] for b in range(self.batch_size)] if mode == 'gen' else None
		finish_flag = [0 for _ in range(self.batch_size)] if mode == 'gen' else None
		CTX_VEC = {src: torch.zeros(self.batch_size, max_len, self.hidden_size).cuda() for src in self.attn_src} if return_ctx_vec else None

		for t in range(max_len):
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
			input_emb = self.dropout(self.embed(idx)) # (B, E)
			init_state = state

		if mode == 'gen':
			sentences_len = [len(sent) for sent in sentences] # sentence length w/i eos
			sentences = [' '.join(sent[:-1]) for sent in sentences] # remove eos and convert to string
			# pad 0 in sample_wordIdx for generating samples
			for b, sent_len in enumerate(sentences_len):
				if sent_len < max_len:
					assert sample_wordIdx[b, sent_len-1] == self.word2idx['<EOS>']
				sample_wordIdx[b, sent_len:] = 0
				logprobs[b, sent_len:] = 0

			# keep hiddens in a valid size since the longest sent might not be max_dec_len
			hiddens = hiddens[:, :max(sentences_len), :]
			if return_ctx_vec:
				for src in self.attn_src:
					CTX_VEC[src] = CTX_VEC[src][:, :max(sentences_len), :]

		if mode == 'gen':
			output = {'logits': logits, 'logprobs': logprobs, 'sample_wordIdx': sample_wordIdx,
					  'decode': sentences, 'decode_len': torch.tensor(sentences_len), 'hiddens': hiddens, 'ctx_vec': CTX_VEC, 'mode': mode}
		else:
			output = {'logits': logits, 'logprobs': logprobs, 'sample_wordIdx': None,
					  'decode': sentences, 'decode_len': input_len, 'hiddens': hiddens, 'ctx_vec': CTX_VEC, 'mode': mode}
		return output
	

	def logits2words(self, logits, sentences, sample_wordIdx, logprobs, t, finish_flag, sample):
		'''
		Args:
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
			if len(sentence) > 0 and sentence[-1] == '<EOS>':
				finish_flag[b_idx] = 1
				continue
			sentence.append(self.idx2word[i.item()])