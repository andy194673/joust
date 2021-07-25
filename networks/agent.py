import torch.nn as nn
from networks.encoder import SentEncoder, RNN
from networks.decoder import Decoder
from networks.dst import DST

class Agent(nn.Module):
	'''Class of user simulator (usr) / dialogue system (sys)'''
	def __init__(self, config, dataset, side):
		super(Agent, self).__init__()
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

		assert H % 2 == 0
		self.encode_ctx = SentEncoder(V, E, int(H/2), dropout=D)
		self.encode_prev_act = SentEncoder(act_V, E, int(H/2), dropout=D)

		if side == 'sys' and not config.oracle_dst:
			self.dst = DST(config, dataset)
		self.policy = Decoder(config, dataset.act_vocab, dataset.idx2act, add_size, 'act', ['ctx', 'prev_act'])
		self.decode = Decoder(config, dataset.vocab, dataset.idx2word, add_size, 'word', ['act'])

		if not config.share_dial_rnn:
			self.dial_rnn = RNN(config.hidden_size, config.hidden_size, dropout=config.dropout, bidirectional=False)


	def forward(self):
		raise NotImplementedError
