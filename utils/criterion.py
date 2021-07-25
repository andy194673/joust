import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

def NLLEntropy(logits, target, ignore_idx=None):
	'''
	Args:
		logits: (B, T, V)
		target: (B, T)
	'''
#	print(logits.size())
#	print(target.size())
	B, T, V = logits.size()
	logProb = F.log_softmax(logits.view(B*T, V), dim=1) 
	loss = F.nll_loss(logProb, target.view(-1), reduction='mean', ignore_index=ignore_idx)
	return loss


#def NLLEntropyValid(logits, target, valid_indicator, ignore_idx=None):
def NLLEntropyValid(logits, target, valid_indicator, ignore_idx=None, weight=None):
	'''
	Args:
		logits: (B, T, V)
		target: (B, T)
		valid_indicator: list (len=B) of binary
	'''
#	print(logits.size())
#	print(target.size())
	B, T, V = logits.size()
	new_logits, new_target = [], []

	# only consider valid turns within a batch
	new_B = 0
	for i, valid in enumerate(valid_indicator):
		if valid:
			new_B += 1
			new_logits.append(logits[i, :, :].unsqueeze(0))
			new_target.append(target[i, :].unsqueeze(0))
	new_logits = torch.cat(new_logits, dim=0) # (B', T, V)
	new_target = torch.cat(new_target, dim=0) # (B', T)

	logProb = F.log_softmax(new_logits.view(new_B*T, V), dim=1) 
	loss = F.nll_loss(logProb, new_target.view(-1), reduction='mean', ignore_index=ignore_idx, weight=weight)
	return loss



class NormKLLoss(_Loss):
	def __init__(self, unit_average=False):
		super(NormKLLoss, self).__init__()
		self.unit_average = unit_average

	def forward(self, recog_mu, recog_logvar, prior_mu, prior_logvar):
		'''
		Args:
			recog_mu, recog_lovar, prior_mu, prior_logvar: (B, Z)
		Return:
			kl_loss: (B,)
		'''
		loss = 1.0 + (recog_logvar - prior_logvar)
		loss -= torch.div(torch.pow(prior_mu - recog_mu, 2), torch.exp(prior_logvar))
		loss -= torch.div(torch.exp(recog_logvar), torch.exp(prior_logvar))
		if self.unit_average:
			kl_loss = -0.5 * torch.mean(loss, dim=1)
		else:
			kl_loss = -0.5 * torch.sum(loss, dim=1)
		return torch.mean(kl_loss)

	def merge(self, kl1, kl2, source):
		'''
		Merge two kl losses kl(p|q), kl(q|p) based on its source of sampling 
		Args:
			kl1, kl2: (B,)
			source: a list (len=B)
		Return:
			average kl loss
		'''
		batch_size = kl1.size(0)
		assert batch_size == len(source)
		kl = torch.zeros(batch_size).cuda()
		for idx, s in enumerate(source):
			if s == 0:
				kl[idx] = kl1[idx]
			elif s == 1:
				kl[idx] = kl2[idx]
			else:
				raise ValueError('Unknown source of sampling')
		return torch.mean(kl)
