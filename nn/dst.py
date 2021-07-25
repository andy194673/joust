import os
import sys
import torch
import torch.nn as nn
from nn.encoder import SentEncoder, RNN
from nn.decoder3 import Decoder
from utils.criterion import NLLEntropyValid
from utils.util_dst import dict2list

class DST(nn.Module):
	def __init__(self, config, dataset):
		super(DST, self).__init__()
		self.config = config
		self.dataset = dataset
		V_word = len(dataset.dstWord_vocab)
		V_slot = len(dataset.slot_vocab)
		V_value = len(dataset.value_vocab['all'])
		E, H = config.embed_size, config.hidden_size
		D = config.dropout

		# components
		assert H % 2 == 0
		self.encode_ctx = SentEncoder(V_word, E, int(H/2), dropout=D)
		add_size = 0
		if config.attn_prev_bs:
			self.encode_prev_bs = SentEncoder(V_slot, E, int(H/2), dropout=D, input2_size=V_value)
			self.decode_slot = Decoder(config, dataset.slot_vocab, dataset.idx2slot, add_size, 'dst_slot', ['dst_ctx', 'prev_bs'])
		else:
			self.decode_slot = Decoder(config, dataset.slot_vocab, dataset.idx2slot, add_size, 'dst_slot', ['dst_ctx'])
		self.decode_value = nn.Linear(3*H, V_value)

		if config.value_mask:
			self.build_value_mask()

		self.set_optimizer()


	def build_value_mask(self):
		print('build value mask', file=sys.stderr)
		self.mask = {}
		value_len = len(self.dataset.value_vocab['all'])
		for slot in self.dataset.slot_vocab:
			if slot in ['<PAD>', '<SOS>', '<EOS>', '<UNK>']: # skip invalid slot
				continue
			mask = [0] * value_len
			for value_idx in range(value_len):
				value = self.dataset.idx2value['all'][value_idx]
				if value not in self.dataset.value_vocab[slot]:
					mask[value_idx] = float('-inf')
			self.mask[slot] = torch.tensor(mask).float().cuda()


	def forward(self, batch, turn_idx=None, mode='teacher_force', beam_search=False):
		# get tensor
		dst_ctx = batch['dst_idx']['dst_ctx'] # (B, T)
		prev_bs_slot = batch['dst_idx']['prev_bs_slot'] # NOTE: ground truth during training, own prediction during inference
		prev_bs_value = batch['dst_idx']['prev_bs_value']
		curr_nlu_slot = batch['dst_idx']['curr_nlu_slot']
		curr_nlu_value = batch['dst_idx']['curr_nlu_value']

		dst_ctx_len = batch['sent_len']['dst_ctx']
		prev_bs_slot_len = batch['sent_len']['prev_bs_slot']
		prev_bs_value_len= batch['sent_len']['prev_bs_value']
		curr_nlu_slot_len = batch['sent_len']['curr_nlu_slot']
		curr_nlu_value_len = batch['sent_len']['curr_nlu_value']
		self.batch_size = dst_ctx.size(0)

		if turn_idx == 0:
			self.init_bs_pred(self.batch_size)

		# encode both input utt and prev bs slot
		# ctx_out: (B, sent_len, dir*H/2) & ctx_emb: (layer, B, dir*H/2)
		ctx_out, ctx_state = self.encode_ctx(dst_ctx, dst_ctx_len)
		if self.config.attn_prev_bs:
			prev_bs_out, (prev_bs_emb, _) = self.encode_prev_bs(prev_bs_slot, prev_bs_slot_len, input2_var=prev_bs_value, input2_len=prev_bs_value_len)

		# decode slot
		init = ctx_state
		if self.config.attn_prev_bs:
			enc_out = {'dst_ctx': ctx_out, 'prev_bs': prev_bs_out}
			enc_mask = {'dst_ctx': self.len2mask(dst_ctx_len), 'prev_bs': self.len2mask(prev_bs_slot_len)} # same as prev_bs_value_len
		else:
			enc_out = {'dst_ctx': ctx_out}
			enc_mask = {'dst_ctx': self.len2mask(dst_ctx_len)}
		slot_out = self.decode_slot(curr_nlu_slot, curr_nlu_slot_len, init, None, enc_out, enc_mask, mode=mode, return_ctx_vec=True)

		# decode value
		slot_hiddens = slot_out['hiddens'] # (B, T, H)
		slot_ctx_vec = slot_out['ctx_vec']['dst_ctx'] # (B, T, H)
		T = slot_hiddens.size(1)
		ctx_emb = ctx_state[0].permute(1, 0, 2).repeat(1, T, 1) # a tuple of (1, B, H) -> (B, T, H)
		value_feat = torch.cat([slot_hiddens, slot_ctx_vec, ctx_emb], dim=2) # (B, T, 3H)

		if mode == 'teacher_force':
			slot_token = batch['ref']['dst']['nlu_slot']
		else:
			slot_token = [slot_str.split() for slot_str in slot_out['decode']] # convert a list of str into a list of list, no <eos>

		if self.config.separate_value_list:
			raise NotImplementedError
		else:
			value_logits = self.decode_value(value_feat) # (B, T, value_len)
			if self.config.value_mask:
				for b_idx in range(self.batch_size):
					for slot_idx, slot in enumerate(slot_token[b_idx]):
						if mode == 'teacher_force':
							value_idx = curr_nlu_value[b_idx, slot_idx]
							assert self.mask[slot][value_idx].item() != float('-inf') # assert target not being masked
						value_logits[b_idx, slot_idx, :] = value_logits[b_idx, slot_idx, :] + self.mask[slot]

		# return
		self.logits, decode = {}, {}
		if mode == 'teacher_force':
			self.logits['slot'] = slot_out['logits']
			self.logits['value'] = value_logits
			return None, None
#			return None, None, None, None

		if mode == 'gen':
#			slot_token = [slot_str.split() for slot_str in slot_out['decode']] # convert a list of str into a list of list, no <eos>
			slot_len = slot_out['decode_len'].tolist() # w/i <eos>
#			value_token = self.logits2token(value_logits, slot_len, mode)
			value_token, value_logprobs = self.logits2token(value_logits, slot_len, mode)
			nlu_pred = self.update_bs(slot_token, value_token, slot_len)
			return self.bs_pred, nlu_pred # both are a list of dict

#			logprobs = {'slot': slot_out['logprobs'], 'value': value_logprobs} # (B, T=max_slot_dec_len)
#			tokens = {'slot': slot_token, 'value': value_token} # a list (len=B) of list with decoded slot/value tokens (no <eos>)
#			return self.bs_pred, nlu_pred, logprobs, tokens



	def init_bs_pred(self, batch_size):
		'''
		bs_pred: a list (len=B) of bs dict such as {'hotel-area': 'north', 'hotel-internet': 'yes'}
		'''
		self.bs_pred = [ {} for _ in range(batch_size) ]


	def update_bs(self, slot_token, value_token, slot_len):
		nlu_pred = []
		for b_idx, bs_dict in enumerate(self.bs_pred):
			nlu_dict = {}
			slot_list = slot_token[b_idx]
			value_list = value_token[b_idx]
#			print('In batch:{}, # of slot: {}, value: {}, should be: {}'.format(b_idx, len(slot_list), len(value_list), slot_len[b_idx]-1), file=sys.stderr)
			assert len(slot_list) == len(value_list) == (slot_len[b_idx]-1)
			for slot, value in zip(slot_list, value_list):
				if value in ['<PAD>', '<SOS>', '<EOS>', '<UNK>']: # remove invalid value prediction:
					continue
				bs_dict[slot] = value
				nlu_dict[slot] = value
			nlu_pred.append(nlu_dict)
		return nlu_pred


	def logits2token(self, value_out, slot_len, mode):
		'''
		get the decoded value token given the logits of value
		Args:
			value_out: (B, T, value_len)
			slot_len: (B, )
		Return:
			sentences: a list (len=B) of list where each list is a value seq
			logprobs: (B, T)
		'''
		sentences = [[] for b in range(self.batch_size)] if mode == 'gen' else None
		if mode != 'gen':
			return sentences

		B = value_out.size(0)
		T = value_out.size(1)
		logprobs = torch.zeros(B, T).cuda()
		for t in range(T):
			logits = value_out[:, t, :] # (B, V)
			value, idx = torch.max(torch.softmax(logits, dim=1), dim=1) # (B, )

			logprobs[:, t] = torch.log(value)

			for b_idx, (sentence, i) in enumerate(zip(sentences, idx)):
#				if len(sentence) > 0 and sentence[-1] == '<EOS>':
#					continue
				if t < (slot_len[b_idx]-1):
#					sentence.append(self.dataset.idx2value[i.item()])
					sentence.append(self.dataset.idx2value['all'][i.item()])
#		# remove eos
#		for b, sent in enumerate(sentences):
##			assert sent[-1] == '<EOS>'
#			sentences[b] = sent[:-1]
#		return sentences
		for b_idx, sent_len in enumerate(slot_len):
			logprobs[b_idx, sent_len:] = 0
		return sentences, logprobs
		
			

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
#		print('B:', B, self.batch_size, file=sys.stderr)
		assert B == self.batch_size
		mask = torch.ones(B, max_len)
		for i, l in enumerate(length):
			mask[i, l:] = float('-inf')
		return mask.cuda()


	def get_loss(self, batch):
		loss = {}
		# reconstruction loss of slot and value
		loss_slot = NLLEntropyValid(self.logits['slot'], batch['dst_idx']['curr_nlu_slot'], batch['valid_turn'], ignore_idx=self.dataset.slot_vocab['<PAD>'])
		loss_value = NLLEntropyValid(self.logits['value'], batch['dst_idx']['curr_nlu_value'], batch['valid_turn'], ignore_idx=self.dataset.value_vocab['all']['<PAD>'])
		loss['slot'] = loss_slot.item()
		loss['value'] = loss_value.item()
		update_loss = loss_slot + loss_value
		return loss, update_loss


	def update(self, update_loss, train_mode):
		update_loss.backward()
#		update_loss.backward(retain_graph=True)
		grad_norm = nn.utils.clip_grad_norm_(self.parameters(), self.config.grad_clip)
		self.optimizer.step()
		self.optimizer.zero_grad()
		return grad_norm


	def set_optimizer(self):
		self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.config.lr)


	def saveModel(self, epoch):
		if not os.path.exists(self.config.model_dir):
			os.makedirs(self.config.model_dir)
#		torch.save(self.state_dict(), self.config.model_dir + '/epoch-{}.pt'.format(str(epoch)))
		torch.save(self.state_dict(), self.config.model_dir + '/epoch-{}.pt'.format('best'))


	def loadModel(self, model_dir, epoch):
#		model_name = self.config.model_dir + '/epoch-{}.pt'.format(str(epoch))
		model_name = model_dir + '/epoch-{}.pt'.format(str(epoch))
		self.load_state_dict(torch.load(model_name))

