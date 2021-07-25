import torch
import torch.nn as nn
from nn.encoder import SentEncoder, RNN
from nn.decoder3 import Decoder
from nn.dst import DST

class Model(nn.Module):
	'''
	General End2end model for user simulator and dialogue agent
	Note: hidden size for encoder is half of H due to bi-directional
	'''
	def __init__(self, config, dataset, side):
		super(Model, self).__init__()
		assert side in ['usr', 'sys']
		V = len(dataset.vocab)
		act_V = len(dataset.act_vocab)
		E, H = config.embed_size, config.hidden_size
		D = config.dropout
		bs_size, db_size, gs_size = config.bs_size, config.db_size, config.gs_size

		if side == 'usr':
			add_size = gs_size
		else:
			add_size = bs_size + db_size

		# components
		assert H % 2 == 0
		self.encode_ctx = SentEncoder(V, E, int(H/2), dropout=D)
		self.encode_prev_act = SentEncoder(act_V, E, int(H/2), dropout=D)

		if side == 'sys' and not config.oracle_dst:
			self.dst = DST(config, dataset)

		self.policy = Decoder(config, dataset.act_vocab, dataset.idx2act, add_size, 'act', ['ctx', 'prev_act'])

		# NOTE: make nlg only takes act seq as input so that word seq and act seq are consistent
		# otherwise, we cannot really observe the effect caused by our rewards which is designed at act level
		self.decode = Decoder(config, dataset.vocab, dataset.idx2word, add_size, 'word', ['act'])

		if not config.share_dial_rnn:
			self.dial_rnn = RNN(config.hidden_size, config.hidden_size, dropout=config.dropout, bidirectional=False)

#		# print model parameters
#		for name, param in self.named_parameters():
#			if param.requires_grad:
#				print(name)
##    			print(name, param.data)
#		sys.exit(1)


	def forward(self):
		raise NotImplementedError
