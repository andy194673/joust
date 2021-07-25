import sys
import math

import torch
import torch.nn as nn
from torch.autograd import Variable
#import torch.nn.functional as F

class Attn(nn.Module):
	def __init__(self, context_size, query_size, method='cat'):
		super(Attn, self).__init__()
		self.method = method
		hidden_size = query_size
		self.H = hidden_size

		if self.method == 'cat':
			self.attn = nn.Linear(context_size + query_size, hidden_size)
			self.v = nn.Linear(hidden_size, 1, bias=False)
#			self.v = nn.Parameter(torch.rand(hidden_size)) # (hidden_size, )
#			stdv = 1. / math.sqrt(self.v.size(0))
#			self.v.data.normal_(mean=0, std=stdv)

#		elif self.method == 'dot':
#			self.attn = nn.Linear(self.hidden_size, hidden_size)

		else:
			print('Wrong type of attention mechanism', file=sys.stderr)
			sys.exit(1)


	def forward(self, query, context, mask, return_energy=False):
		'''
		Args:
			query: decoder hidden state (B, 1, H)
			context: encoder hidden states (B, T, H')
			mask: encoder len (B, T)
		Returns:
			context_vec: context vector (B, H')
			attention dist (B, T)
		'''
		self.B, self.T, _ = context.size()
#		hidden = hidden.unsqueeze(1).repeat(1, max_len, 1) # (B, T, H)
		query = query.expand(self.B, self.T, self.H)
		attn_dist = self.score(query, context, mask) # (B, T)
		context_vec = torch.bmm(attn_dist.unsqueeze(1), context) # (B, 1, H)
		return attn_dist, context_vec.squeeze(1)


	def score(self, query, context, mask):
		if self.method == 'cat':
			score = torch.tanh(self.attn(torch.cat([query, context], 2))) # (B, T, feat_size) => (B, T, H)
			score = self.v(score) # (B, T, 1)
			score = (score.squeeze(2) + mask).view(self.B, self.T) # (B, T)
#			v = self.v.view(1, self.hidden_size, 1).repeat(batch_size, 1, 1) # (B, H, 1)
#			energy = torch.bmm(energy, v) # (B, T, 1)

#		elif self.method == 'dot':
#			hidden = F.tanh(self.attn(hidden)) # (B, T, H)
#			energy = encoder_outputs * hidden # (B, T, H)
#			energy = torch.sum(energy, dim=2).unsqueeze(2) # (B, T) => (B, T, 1)
			
#		return F.softmax(energy, dim=1)
		return torch.softmax(score, dim=1)
