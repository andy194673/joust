'''
	The evaluation script is adapted from the following two files:
	1. https://github.com/budzianowski/multiwoz/blob/master/evaluate.py
	2. https://github.com/thu-spmi/damd-multiwoz/blob/master/eval.py
'''

import math, logging, copy, json
import numpy as np
from collections import Counter, OrderedDict
from nltk.util import ngrams
from data_preprocess.db_ops import MultiWozDB
from utils.check_turn_info import get_turn_act_slot, decide_turn_domain


class BLEUScorer(object):
	def __init__(self):
		pass


	def score(self, hypothesis, corpus):
		# containers
		count = [0, 0, 0, 0]
		clip_count = [0, 0, 0, 0]
		r = 0
		c = 0
		weights = [0.25, 0.25, 0.25, 0.25]

		# accumulate ngram statistics
		for hyps, refs in zip(hypothesis, corpus):
			hyps = [hyp.split() for hyp in hyps]
			refs = [ref.split() for ref in refs]
			for hyp in hyps:

				for i in range(4):
					# accumulate ngram counts
					hypcnts = Counter(ngrams(hyp, i + 1))
					cnt = sum(hypcnts.values())
					count[i] += cnt

					# compute clipped counts
					max_counts = {}
					for ref in refs:
						refcnts = Counter(ngrams(ref, i + 1))
						for ng in hypcnts:
							max_counts[ng] = max(max_counts.get(ng, 0), refcnts[ng])
					clipcnt = dict((ng, min(count, max_counts[ng])) \
								   for ng, count in hypcnts.items())
					clip_count[i] += sum(clipcnt.values())

				# accumulate r & c
				bestmatch = [1000, 1000]
				for ref in refs:
					if bestmatch[0] == 0: break
					diff = abs(len(ref) - len(hyp))
					if diff < bestmatch[0]:
						bestmatch[0] = diff
						bestmatch[1] = len(ref)
				r += bestmatch[1]
				c += len(hyp)

		# computing bleu score
		p0 = 1e-7
		bp = 1 if c > r else math.exp(1 - float(r) / float(c))
		p_ns = [float(clip_count[i]) / float(count[i] + p0) + p0 \
				for i in range(4)]
		s = math.fsum(w * math.log(p_n) \
					  for w, p_n in zip(weights, p_ns) if p_n)
		bleu = bp * math.exp(s)
		return bleu * 100


class MultiWozEvaluator(object):
	def __init__(self, dataset, config):
		self.db = MultiWozDB()
		self.all_domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'police', 'hospital']
		self.dataset = dataset
		self.config = config

		self.bleu_scorer = BLEUScorer()

		with open(config.dst_slot_list) as f:
			self.all_info_slot = json.load(f)

		# only evaluate these slots for dialog success
		self.requestables = ['phone', 'address', 'postcode', 'reference', 'id']


	def calculate_eval_reward(self, decode_all, CT, mode='fix_corpus'):
		'''Get per dialogue reward for evaluation'''
		assert mode in ['fix_corpus', 'interaction']
		if mode == 'fix_corpus':
			src = 'ref'
		else: # interaction
			src = 'gen' # since no ref during interaction

		ENT_R, ASK_R, MISS_R, DOM_R = 0, 0, 0, 0 # sys reward
		USR_REPEAT_INFO, USR_REPEAT_ASK, USR_MISS_ANSWER = 0, 0, 0
		for dial_name, dial in decode_all.items():
			gen_dial = {}
			gen_dial['dial_name'] = dial_name
			gen_dial['goal'] = dial['goal']
			gen_dial['bs'] = dial['sys']['gen_bs']
			gen_dial['act_usr'] = dial['usr']['{}_act'.format(src)]
			gen_dial['word_usr'] = dial['usr']['{}_word'.format(src)]
			gen_dial['act_sys'] = dial['sys']['gen_act']
			gen_dial['word_sys'] = dial['sys']['gen_word']

			ent_r = CT.get_entity_provide_reward(gen_dial, print_log=False)
			ask_r = CT.get_repeat_ask_reward(gen_dial, print_log=False)
			miss_r = CT.get_miss_answer_reward(gen_dial, print_log=False)
			dom_r = 0

			usr_repeat_info = CT.get_usr_repeat_info_reward(gen_dial, print_log=False) if self.config.rl_update_usr else 0
			usr_repeat_ask  = CT.get_usr_repeat_ask_reward(gen_dial, print_log=False) if self.config.rl_update_usr else 0
			usr_miss_answer = CT.get_usr_miss_answer_reward(gen_dial, print_log=False) if self.config.rl_update_usr else 0

			ENT_R += np.sum(ent_r)
			ASK_R += np.sum(ask_r)
			MISS_R += np.sum(miss_r)
			DOM_R += np.sum(dom_r)

			USR_REPEAT_INFO += np.sum(usr_repeat_info)
			USR_REPEAT_ASK  += np.sum(usr_repeat_ask)
			USR_MISS_ANSWER += np.sum(usr_miss_answer)

		# get per dialogue reward
		n = len(decode_all)
		R = (ENT_R + ASK_R + MISS_R + DOM_R)/n
		ENT_R /= n
		ASK_R /= n
		MISS_R /= n
		DOM_R /= n

		R_USR = (USR_REPEAT_INFO + USR_REPEAT_ASK + USR_MISS_ANSWER)/n
		USR_REPEAT_INFO /= n
		USR_REPEAT_ASK  /= n
		USR_MISS_ANSWER /= n

		return {'total': R, 'ent': ENT_R, 'ask': ASK_R, 'miss': MISS_R, 'dom': DOM_R, \
				'usr': R_USR, 're_info': USR_REPEAT_INFO, 're_ask': USR_REPEAT_ASK, 'miss_ans': USR_MISS_ANSWER}


	def calculate_reqt_acc(self, decode_all, mode='fix_corpus'):
		'''
		Check if sys answers slots requested by usr
		key difference between this function and success rate:
			1) this function considers all slots instead of predefined slots in self.requestables
			2) this function checks answers at turn level instead of dialogue level, which leads to more an accurate measurement
			3) in order to consider all requested slots, we check answers at act level instead of word level
		'''
		assert mode in ['fix_corpus', 'interaction']
		if mode == 'fix_corpus':
			src = 'ref_act'
		else: # interaction
			src = 'gen_act' # since no ref_act during interaction

		reqt_slot_count = {'total': 0, 'ans': 0}
		record = {}
		for dial_name, dial in decode_all.items():
			domain_prev = 'none'
			record[dial_name] = set()
			for side_idx, (act_usr, act_sys) in enumerate(zip(dial['usr'][src], dial['sys'][src])):
				turn_domain = decide_turn_domain(act_usr, act_sys, domain_prev)
				reqt_slots = get_turn_act_slot(act_usr, 'request')
				for reqt_slot in reqt_slots:
					# filter out wrong labelled slots for better evaluation
					domain, _ = reqt_slot.split('_')
					if domain != turn_domain:
						continue

					reqt_slot_count['total'] += 1
					if reqt_slot in dial['sys']['gen_act'][side_idx]:
						reqt_slot_count['ans'] += 1
					else: # not answered
						record[dial_name].add((side_idx, reqt_slot))
				domain_prev = turn_domain

			record[dial_name] = tuple(record[dial_name])

		if reqt_slot_count['total'] == 0:
			acc = 0
		else:
			acc =  reqt_slot_count['ans'] / reqt_slot_count['total']
		return acc, reqt_slot_count['total'], record


	def pack_dial(self, data):
		dials = {}
		for turn in data:
			dial_id = turn['dial_id']
			if dial_id not in dials:
				dials[dial_id] = []
			dials[dial_id].append(turn)
		return dials


	def run_metrics(self, data):
		if 'all' in cfg.exp_domains:
			metric_results = []
			metric_result = self._get_metric_results(data)
			metric_results.append(metric_result)

			if cfg.eval_per_domain:
				# all domain experiments, sub domain evaluation
				domains = [d+'_single' for d in ontology.all_domains]
				domains = domains + ['restaurant_train', 'restaurant_hotel','restaurant_attraction', 'hotel_train', 'hotel_attraction',
									 'attraction_train', 'restaurant_hotel_taxi', 'restaurant_attraction_taxi', 'hotel_attraction_taxi', ]
				for domain in domains:
					file_list = self.domain_files.get(domain, [])
					if not file_list:
						print('No sub domain [%s]'%domain)
					metric_result = self._get_metric_results(data, domain, file_list)
					if metric_result:
						metric_results.append(metric_result)

		else:
			# sub domain experiments
			metric_results = []
			for domain, file_list in self.domain_files.items():
				if domain not in cfg.exp_domains:
					continue
				metric_result = self._get_metric_results(data, domain, file_list)
				if metric_result:
					metric_results.append(metric_result)
		return metric_results


	def validation_metric(self, data):
		bleu = self.bleu_metric(data)
		accu_single_dom, accu_multi_dom, multi_dom_num = self.domain_eval(data)
		success, match, req_offer_counts, dial_num = \
			self.context_to_response_eval(data, same_eval_as_cambridge=cfg.same_eval_as_cambridge)
		return bleu, success, match


	def _get_metric_results(self, data, domain='all', file_list=None):
		metric_result = {'domain': domain}
		bleu = self.bleu_metric(data, file_list)
		if cfg.bspn_mode == 'bspn' or cfg.enable_dst:
			jg, slot_f1, slot_acc, slot_cnt, slot_corr = self.dialog_state_tracking_eval(data, file_list)
			jg_nn, sf1_nn, sac_nn, _, _ = self.dialog_state_tracking_eval(data, file_list, no_name=True, no_book=False)
			jg_nb, sf1_nb, sac_nb, _, _ = self.dialog_state_tracking_eval(data, file_list, no_name=False, no_book=True)
			jg_nnnb, sf1_nnnb, sac_nnnb, _, _ = self.dialog_state_tracking_eval(data, file_list, no_name=True, no_book=True)
			metric_result.update({'joint_goal':jg, 'slot_acc': slot_acc, 'slot_f1':slot_f1})
		if cfg.bspn_mode == 'bsdx':
			jg_, slot_f1_, slot_acc_, slot_cnt, slot_corr = self.dialog_state_tracking_eval(data, file_list, bspn_mode='bsdx')
			jg_nn_, sf1_nn_, sac_nn_,  _, _ = self.dialog_state_tracking_eval(data, file_list, bspn_mode='bsdx', no_name=True, no_book=False)
			metric_result.update({'joint_goal_delex':jg_, 'slot_acc_delex': slot_acc_, 'slot_f1_delex':slot_f1_})

		info_slots_acc = {}
		for slot in slot_cnt:
			correct = slot_corr.get(slot, 0)
			info_slots_acc[slot] = correct / slot_cnt[slot] * 100
		info_slots_acc = OrderedDict(sorted(info_slots_acc.items(), key = lambda x: x[1]))

		act_f1 = self.aspn_eval(data, file_list)
		avg_act_num, avg_diverse_score = self.multi_act_eval(data, file_list)
		accu_single_dom, accu_multi_dom, multi_dom_num = self.domain_eval(data, file_list)

		success, match, req_offer_counts, dial_num = \
			self.context_to_response_eval(data, file_list, same_eval_as_cambridge=cfg.same_eval_as_cambridge)
		req_slots_acc = {}
		for req in self.requestables:
			acc = req_offer_counts[req+'_offer']/(req_offer_counts[req+'_total'] + 1e-10)
			req_slots_acc[req] = acc * 100
		req_slots_acc = OrderedDict(sorted(req_slots_acc.items(), key = lambda x: x[1]))

		if dial_num:
			metric_result.update({'act_f1':act_f1,'success':success, 'match':match, 'bleu': bleu,
										'req_slots_acc':req_slots_acc, 'info_slots_acc': info_slots_acc,'dial_num': dial_num,
										'accu_single_dom': accu_single_dom, 'accu_multi_dom': accu_multi_dom,
										'avg_act_num': avg_act_num, 'avg_diverse_score': avg_diverse_score})
			if domain == 'all':
				logging.info('-------------------------- All DOMAINS --------------------------')
			else:
				logging.info('-------------------------- %s (# %d) -------------------------- '%(domain.upper(), dial_num))
			if cfg.bspn_mode == 'bspn' or cfg.enable_dst:
				logging.info('[DST] joint goal:%2.1f  slot acc: %2.1f  slot f1: %2.1f  act f1: %2.1f'%(jg, slot_acc, slot_f1, act_f1))
				logging.info('[DST] [not eval name slots] joint goal:%2.1f  slot acc: %2.1f  slot f1: %2.1f'%(jg_nn, sac_nn, sf1_nn))
				logging.info('[DST] [not eval book slots] joint goal:%2.1f  slot acc: %2.1f  slot f1: %2.1f'%(jg_nb, sac_nb, sf1_nb))
				logging.info('[DST] [not eval name & book slots] joint goal:%2.1f  slot acc: %2.1f  slot f1: %2.1f'%(jg_nnnb, sac_nnnb, sf1_nnnb))
			if cfg.bspn_mode == 'bsdx':
				logging.info('[BDX] joint goal:%2.1f  slot acc: %2.1f  slot f1: %2.1f  act f1: %2.1f'%(jg_, slot_acc_, slot_f1_, act_f1))
				logging.info('[BDX] [not eval name slots] joint goal:%2.1f  slot acc: %2.1f  slot f1: %2.1f'%(jg_nn_, sac_nn_, sf1_nn_))
			logging.info('[CTR] match: %2.1f  success: %2.1f  bleu: %2.1f'%(match, success, bleu))
			logging.info('[CTR] ' + '; '.join(['%s: %2.1f' %(req,acc) for req, acc in req_slots_acc.items()]))
			logging.info('[DOM] accuracy: single %2.1f / multi: %2.1f (%d)'%(accu_single_dom, accu_multi_dom, multi_dom_num))
			if self.reader.multi_acts_record is not None:
				logging.info('[MA] avg acts num %2.1f  avg slots num: %2.1f '%(avg_act_num, avg_diverse_score))
			return metric_result
		else:
			return None


	def bleu_metric(self, data, eval_dial_list=None):
		gen, truth = [],[]
		for row in data:
			if eval_dial_list and row['dial_id'] +'.json' not in eval_dial_list:
				continue
			gen.append(row['resp_gen'])
			truth.append(row['resp'])
		wrap_generated = [[_] for _ in gen]
		wrap_truth = [[_] for _ in truth]
		if gen and truth:
			sc = self.bleu_scorer.score(zip(wrap_generated, wrap_truth))
		else:
			sc = 0.0
		return sc


	def value_similar(self, a,b):
		return True if a==b else False
		# the value equal condition used in "Sequicity" is too loose
		if a in b or b in a or a.split()[0] == b.split()[0] or a.split()[-1] == b.split()[-1]:
			return True
		return False


	def _bspn_to_dict(self, bspn, no_name=False, no_book=False, bspn_mode = 'bspn'):
		constraint_dict = self.reader.bspan_to_constraint_dict(bspn, bspn_mode = bspn_mode)
		constraint_dict_flat = {}
		for domain, cons in constraint_dict.items():
			for s,v in cons.items():
				key = domain+'-'+s
				if no_name and s == 'name':
					continue
				if no_book:
					if s in ['people', 'stay'] or key in ['hotel-day', 'restaurant-day','restaurant-time'] :
						continue
				constraint_dict_flat[key] = v
		return constraint_dict_flat


	def _constraint_compare(self, truth_cons, gen_cons, slot_appear_num=None, slot_correct_num=None):
		tp,fp,fn = 0,0,0
		false_slot = []
		for slot in gen_cons:
			v_gen = gen_cons[slot]
			if slot in truth_cons and self.value_similar(v_gen, truth_cons[slot]):  #v_truth = truth_cons[slot]
				tp += 1
				if slot_correct_num is not None:
					slot_correct_num[slot] = 1 if not slot_correct_num.get(slot) else slot_correct_num.get(slot)+1
			else:
				fp += 1
				false_slot.append(slot)
		for slot in truth_cons:
			v_truth = truth_cons[slot]
			if slot_appear_num is not None:
				slot_appear_num[slot] = 1 if not slot_appear_num.get(slot) else slot_appear_num.get(slot)+1
			if slot not in gen_cons or not self.value_similar(v_truth, gen_cons[slot]):
				fn += 1
				false_slot.append(slot)
		acc = len(self.all_info_slot) - fp - fn
		return tp,fp,fn, acc, list(set(false_slot))


	def domain_eval(self, data, eval_dial_list = None):
		dials = self.pack_dial(data)
		corr_single, total_single, corr_multi, total_multi = 0, 0, 0, 0

		dial_num = 0
		for dial_id in dials:
			if eval_dial_list and dial_id+'.json' not in eval_dial_list:
				continue
			dial_num += 1
			dial = dials[dial_id]
			wrong_pred = []

			prev_constraint_dict = {}
			prev_turn_domain = ['general']

			for turn_num, turn in enumerate(dial):
				if turn_num == 0:
					continue
				true_domains = self.reader.dspan_to_domain(turn['dspn'])
				if cfg.enable_dspn:
					pred_domains = self.reader.dspan_to_domain(turn['dspn_gen'])
				else:
					turn_dom_bs = []
					if cfg.enable_bspn and not cfg.use_true_bspn_for_ctr_eval and \
						(cfg.bspn_mode == 'bspn' or cfg.enable_dst):
						constraint_dict = self.reader.bspan_to_constraint_dict(turn['bspn_gen'])
					else:
						constraint_dict = self.reader.bspan_to_constraint_dict(turn['bspn'])
					for domain in constraint_dict:
						if domain not in prev_constraint_dict:
							turn_dom_bs.append(domain)
						elif prev_constraint_dict[domain] != constraint_dict[domain]:
							turn_dom_bs.append(domain)
					aspn = 'aspn' if not cfg.enable_aspn else 'aspn_gen'
					turn_dom_da = []
					for a in turn[aspn].split():
						if a[1:-1] in ontology.all_domains + ['general']:
							turn_dom_da.append(a[1:-1])

					# get turn domain
					turn_domain = turn_dom_bs
					for dom in turn_dom_da:
						if dom != 'booking' and dom not in turn_domain:
							turn_domain.append(dom)
					if not turn_domain:
						turn_domain = prev_turn_domain
					if len(turn_domain) == 2 and 'general' in turn_domain:
						turn_domain.remove('general')
					if len(turn_domain) == 2:
						if len(prev_turn_domain) == 1 and prev_turn_domain[0] == turn_domain[1]:
							turn_domain = turn_domain[::-1]
					prev_turn_domain = copy.deepcopy(turn_domain)
					prev_constraint_dict = copy.deepcopy(constraint_dict)

					turn['dspn_gen'] = ' '.join(['['+d+']' for d in turn_domain])
					pred_domains = {}
					for d in turn_domain:
						pred_domains['['+d+']'] = 1

				if len(true_domains) == 1:
					total_single += 1
					if pred_domains == true_domains:
						corr_single += 1
					else:
						wrong_pred.append(str(turn['turn_num']))
						turn['wrong_domain'] = 'x'
				else:
					total_multi += 1
					if pred_domains == true_domains:
						corr_multi += 1
					else:
						wrong_pred.append(str(turn['turn_num']))
						turn['wrong_domain'] = 'x'

			# dialog inform metric record
			dial[0]['wrong_domain'] = ' '.join(wrong_pred)
		accu_single = corr_single / (total_single + 1e-10)
		accu_multi = corr_multi / (total_multi + 1e-10)
		return accu_singe * 100, accu_multi * 100, total_multi


	def dialog_state_tracking_eval(self, data, eval_dial_list = None, bspn_mode='bspn', no_name=False, no_book=False):
		dials = self.pack_dial(data)
		total_turn, joint_match, total_tp, total_fp, total_fn, total_acc = 0, 0, 0, 0, 0, 0
		slot_appear_num, slot_correct_num = {}, {}
		dial_num = 0
		for dial_id in dials:
			if eval_dial_list and dial_id +'.json' not in eval_dial_list:
				continue
			dial_num += 1
			dial = dials[dial_id]
			missed_jg_turn_id = []
			for turn_num,turn in enumerate(dial):
				if turn_num == 0:
					continue
				gen_cons = self._bspn_to_dict(turn[bspn_mode+'_gen'], no_name=no_name,
																  no_book=no_book, bspn_mode=bspn_mode)
				truth_cons = self._bspn_to_dict(turn[bspn_mode], no_name=no_name,
																   no_book=no_book, bspn_mode=bspn_mode)

				if truth_cons == gen_cons:
					joint_match += 1
				else:
					missed_jg_turn_id.append(str(turn['turn_num']))

				if eval_dial_list is None:
					tp,fp,fn, acc, false_slots = self._constraint_compare(truth_cons, gen_cons,
																							  slot_appear_num, slot_correct_num)
				else:
					tp,fp,fn, acc, false_slots = self._constraint_compare(truth_cons, gen_cons,)

				total_tp += tp
				total_fp += fp
				total_fn += fn
				total_acc += acc
				total_turn += 1
				if not no_name and not no_book:
					turn['wrong_inform'] = '; '.join(false_slots)   # turn inform metric record

			# dialog inform metric record
			if not no_name and not no_book:
				dial[0]['wrong_inform'] = ' '.join(missed_jg_turn_id)

		precision = total_tp / (total_tp + total_fp + 1e-10)
		recall = total_tp / (total_tp + total_fn + 1e-10)
		f1 = 2 * precision * recall / (precision + recall + 1e-10) * 100
		accuracy = total_acc / (total_turn * len(self.all_info_slot) + 1e-10) * 100
		joint_goal = joint_match / (total_turn+1e-10) * 100
		return joint_goal, f1, accuracy, slot_appear_num, slot_correct_num


	def _compute_acc(self, gold, pred):
		# joint acc
		if pred == gold:
			joint_acc = 1
		else:
			joint_acc = 0

		# individual acc
		miss_gold = 0 
		miss_slot = []
		for g in gold:
			if g not in pred:
				miss_gold += 1
#				miss_slot.append(g.rsplit("-", 1)[0])
				miss_slot.append(g.split('=')[0])
		wrong_pred = 0 
		for p in pred:
#			if p not in gold and p.rsplit("-", 1)[0] not in miss_slot:
			if p not in gold and p.split('=')[0] not in miss_slot:
				wrong_pred += 1
		slot_acc = (len(self.all_info_slot) - miss_gold - wrong_pred) / len(self.all_info_slot)
		return slot_acc, joint_acc


	def eval_dst(self, decode_all):
		SV_ACC, SLOT_ACC, JOINT_ACC, n_turns = 0, 0, 0, 0
		for dial_name, dial in decode_all.items():
			assert len(dial['sys']['pred_bs']) == len(dial['sys']['ref_bs'])
			for t, (pred_bs, ref_bs) in enumerate(zip(dial['sys']['pred_bs'], dial['sys']['ref_bs'])):
				pred_bs, ref_bs = set(pred_bs), set(ref_bs)
				sv_acc, joint_acc = self._compute_acc(ref_bs, pred_bs)
				SV_ACC += sv_acc
				JOINT_ACC += joint_acc

				# only consider slot
				pred_bs = set([sv.split('=')[0] for sv in pred_bs])
				ref_bs = set([sv.split('=')[0] for sv in ref_bs])
				slot_acc, _ = self._compute_acc(ref_bs, pred_bs)
				SLOT_ACC += slot_acc
				n_turns += 1
		return JOINT_ACC/n_turns, SV_ACC/n_turns, SLOT_ACC/n_turns
				

	def aspn_eval(self, data, eval_dial_list = None):

		def _get_tp_fp_fn(label_list, pred_list):
			tp = len([t for t in pred_list if t in label_list])
			fp = max(0, len(pred_list) - tp)
			fn = max(0, len(label_list) - tp)
			return tp, fp, fn

		dials = self.pack_dial(data)
		total_tp, total_fp, total_fn = 0, 0, 0

		dial_num = 0
		for dial_id in dials:
			if eval_dial_list and dial_id+'.json' not in eval_dial_list:
				continue
			dial_num += 1
			dial = dials[dial_id]
			wrong_act = []
			for turn_num, turn in enumerate(dial):
				if turn_num == 0:
					continue
				if cfg.same_eval_act_f1_as_hdsa:
					pred_acts, true_acts = {}, {}
					for t in turn['aspn_gen']:
						pred_acts[t] = 1
					for t in  turn['aspn']:
						true_acts[t] = 1
					tp, fp, fn = _get_tp_fp_fn(true_acts, pred_acts)
				else:
					pred_acts = self.reader.aspan_to_act_list(turn['aspn_gen'])
					true_acts = self.reader.aspan_to_act_list(turn['aspn'])
					tp, fp, fn = _get_tp_fp_fn(true_acts, pred_acts)
				if fp + fn !=0:
					wrong_act.append(str(turn['turn_num']))
					turn['wrong_act'] = 'x'

				total_tp += tp
				total_fp += fp
				total_fn += fn

			dial[0]['wrong_act'] = ' '.join(wrong_act)
		precision = total_tp / (total_tp + total_fp + 1e-10)
		recall = total_tp / (total_tp + total_fn + 1e-10)
		f1 = 2 * precision * recall / (precision + recall + 1e-10)
		return f1 * 100


	def multi_act_eval(self, data, eval_dial_list = None):
		dials = self.pack_dial(data)
		total_act_num, total_slot_num = 0, 0

		dial_num = 0
		turn_count = 0
		for dial_id in dials:
			if eval_dial_list and dial_id+'.json' not in eval_dial_list:
				continue
			dial_num += 1
			dial = dials[dial_id]
			for turn_num, turn in enumerate(dial):
				if turn_num == 0:
					continue
				target = turn['multi_act_gen'] if self.reader.multi_acts_record is not None else turn['aspn_gen']


				# diversity
				act_collect, slot_collect = {}, {}
				act_type_collect = {}
				slot_score = 0
				for act_str in target.split(' | '):
					pred_acts = self.reader.aspan_to_act_list(act_str)
					act_type = ''
					for act in pred_acts:
						d,a,s = act.split('-')
						if d + '-' + a not in act_collect:
							act_collect[d + '-' + a] = {s:1}
							slot_score += 1
							act_type += d + '-' + a + ';'
						elif s not in act_collect:
							act_collect[d + '-' + a][s] = 1
							slot_score += 1
						slot_collect[s] = 1
					act_type_collect[act_type] = 1
				total_act_num += len(act_collect)
				total_slot_num += len(slot_collect)
				turn_count += 1

		total_act_num = total_act_num/(float(turn_count) + 1e-10)
		total_slot_num = total_slot_num/(float(turn_count) + 1e-10)
		return total_act_num, total_slot_num


	def context_to_response_eval(self, dialogues_gen, dType):
		assert dType in ['valid', 'test', 'train']
		dial_num, successes, matches = 0, 0, 0
		record = {}
		for dial_name, dial in dialogues_gen.items():
			reqs = {}
			goal = {}
			record[dial_name] = {}
			for domain in self.all_domains:
				if self.dataset.all_data[dial_name]['goal'][domain]: # if domain has goal
					goal = self._parseGoal(goal, self.dataset.all_data[dial_name]['goal'], domain)

			for domain in goal.keys():
				reqs[domain] = goal[domain]['requestable']

			success, match, stats = self._evaluateGeneratedDialogue(dial['sys'], goal, reqs, dType)
			record[dial_name]['--Success--'] = success
			record[dial_name]['--Match--'] = match
			for domain in stats:
				_match, _success, _happen = stats[domain]
				stats[domain] = '{}-{}-{}'.format(_match, _success, _happen)
			record[dial_name]['--stats--'] = stats

			successes += success
			matches += match
			dial_num += 1

		succ_rate = successes/( float(dial_num) + 1e-10) * 100
		match_rate = matches/(float(dial_num) + 1e-10) * 100
		return succ_rate, match_rate, record


	def _evaluateGeneratedDialogue(self, dialog, goal, real_requestables, dType, soft_acc=False):
		"""Evaluates the dialogue created by the networks.
			First we load the user goal of the dialogue, then for each turn
			generated by the system we look for key-words.
			For the Inform rate we look whether the entity was proposed.
			For the Success rate we look for requestables slots

		Args:
			dialog: the evaluated dialogue, a list of turns, each turn is a dict with 'word' and 'bs' as keys
		"""

		assert len(dialog['gen_bs']) == len(dialog['gen_act']) == len(dialog['gen_word'])
		dial_len = len(dialog['gen_bs'])

		# for computing corpus success
		requestables = self.requestables

		# CHECK IF MATCH HAPPENED
		provided_requestables = {}
		venue_offered = {}
		domains_in_goal = []

		for domain in goal.keys():
			venue_offered[domain] = []
			provided_requestables[domain] = []
			domains_in_goal.append(domain)

		for t in range(dial_len):
##			if t == 0:
##				continue
##			sent_t = turn['resp_gen']
#			sent_t = turn['word']
			sent_t = dialog['gen_word'][t]
			for domain in goal.keys():
				# for computing success
##				if same_eval_as_cambridge:
##						# [restaurant_name], [hotel_name] instead of [value_name]
##						if cfg.use_true_domain_for_ctr_eval:
##							dom_pred = [d[1:-1] for d in turn['dspn'].split()]
##						else:
##							dom_pred = [d[1:-1] for d in turn['dspn_gen'].split()]
##						# else:
##						#	 raise NotImplementedError('Just use true domain label')
##						if domain not in dom_pred:  # fail
##							continue
##				if '[value_name]' in sent_t or '[value_id]' in sent_t:
				if '[' + domain + '_name]' in sent_t or '_id' in sent_t:
					if domain in ['restaurant', 'hotel', 'attraction', 'train']:
						# HERE YOU CAN PUT YOUR BELIEF STATE ESTIMATION

##						if cfg.enable_bspn and not cfg.use_true_bspn_for_ctr_eval and \
##							(cfg.bspn_mode == 'bspn' or cfg.enable_dst):
##							bspn = turn['bspn_gen']
##						else:
##							bspn = turn['bspn']
##						# bspn = turn['bspn']
##
##						constraint_dict = self.reader.bspan_to_constraint_dict(bspn)
##						if constraint_dict.get(domain):
##							venues = self.reader.db.queryJsons(domain, constraint_dict[domain], return_name=True)
##						else:
##							venues = []

						# retrieved entities by bs at the turn, andy
#						venues = self.db.queryJsons(domain, turn['bs'][domain], return_name=True)
						venues = self.db.queryJsons(domain, dialog['gen_bs'][t][domain]['semi'], return_name=True)

						# if venue has changed
						if len(venue_offered[domain]) == 0 and venues:
							# venue_offered[domain] = random.sample(venues, 1)
							venue_offered[domain] = venues
##							bspans[domain] = constraint_dict[domain]
						else:
							# flag = False
							# for ven in venues:
							#	 if venue_offered[domain][0] == ven:
							#		 flag = True
							#		 break
							# if not flag and venues:
							flag = False
							for ven in venues:
								if  ven not in venue_offered[domain]:
								# if ven not in venue_offered[domain]:
									flag = True
									break
							# if flag and venues:
							if flag and venues:  # sometimes there are no results so sample won't work
								# print venues
								# venue_offered[domain] = random.sample(venues, 1)
								venue_offered[domain] = venues
##								bspans[domain] = constraint_dict[domain]
					else:  # not limited so we can provide one
##						venue_offered[domain] = '[value_name]'
						venue_offered[domain] = '[' + domain + '_name]'

				# ATTENTION: assumption here - we didn't provide phone or address twice! etc
				for requestable in requestables:
					if requestable == 'reference':
##						if '[value_reference]' in sent_t:
						if '_reference' in sent_t:
##							if 'booked' in turn['pointer'] or 'ok' in turn['pointer']:  # if pointer was allowing for that?
##								provided_requestables[domain].append('reference')
							provided_requestables[domain].append('reference')
					else:
##						if '[value_' + requestable + ']' in sent_t:
						if domain + '_' + requestable + ']' in sent_t:
							provided_requestables[domain].append(requestable)

		# if name was given in the task
		for domain in goal.keys():
			# if name was provided for the user, the match is being done automatically
			if 'name' in goal[domain]['informable']:
				venue_offered[domain] = '[' + domain + '_name]'

			# special domains - entity does not need to be provided
			if domain in ['taxi', 'police', 'hospital']:
				venue_offered[domain] = '[' + domain + '_name]'

			if domain == 'train':
				if not venue_offered[domain] and 'id' not in goal[domain]['requestable']:
					venue_offered[domain] = '[' + domain + '_name]'

		"""
		Given all inform and requestable slots
		we go through each domain from the user goal
		and check whether right entity was provided and
		all requestable slots were given to the user.
		The dialogue is successful if that's the case for all domains.
		"""
		# HARD EVAL
		stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0, 0],
				 'taxi': [0, 0, 0],
				 'hospital': [0, 0, 0], 'police': [0, 0, 0]}

		match = 0
		success = 0
		# MATCH
		for domain in goal.keys():
			match_stat = 0
			if domain in ['restaurant', 'hotel', 'attraction', 'train']:
##				goal_venues = self.reader.db.queryJsons(domain, goal[domain]['informable'], return_name=True)
				goal_venues = self.db.queryJsons(domain, goal[domain]['informable'], return_name=True)
#				if dType in ['valid', 'test']: assert len(goal_venues) > 0 # andy
				if type(venue_offered[domain]) is str and '_name' in venue_offered[domain]:
					match += 1
					match_stat = 1
				elif len(venue_offered[domain]) > 0 and len(set(venue_offered[domain])& set(goal_venues))>0:
					match += 1
					match_stat = 1
			else:
				if '_name]' in venue_offered[domain]:
					match += 1
					match_stat = 1

			stats[domain][0] = match_stat
			stats[domain][2] = 1

		if soft_acc:
			match = float(match)/len(goal.keys())
		else:
			if match == len(goal.keys()):
				match = 1.0
			else:
				match = 0.0

		# SUCCESS
		for domain in domains_in_goal:
			success_stat = 0
			domain_success = 0
			if len(real_requestables[domain]) == 0:
				success += 1
				success_stat = 1
				stats[domain][1] = success_stat
				continue
			# if values in sentences are super set of requestables
			# for request in set(provided_requestables[domain]):
			#	 if request in real_requestables[domain]:
			#		 domain_success += 1
			for request in real_requestables[domain]:
				if request in provided_requestables[domain]:
					domain_success += 1

			if domain_success == len(real_requestables[domain]):
				success += 1
				success_stat = 1

			stats[domain][1] = success_stat

		# final eval
		if soft_acc:
			success = float(success)/len(real_requestables)
		else:
			if success >= len(real_requestables):
				success = 1
			else:
				success = 0

		if match == 0 and success == 1:
			success = 0
		return success, match, stats


	def _parseGoal(self, goal, true_goal, domain):
		"""Parses user goal into dictionary format."""
		goal[domain] = {}
		goal[domain] = {'informable': {}, 'requestable': [], 'booking': []}
		if 'info' in true_goal[domain]:
			if domain == 'train':
				# we consider dialogues only where train had to be booked!
				if 'book' in true_goal[domain]:
					goal[domain]['requestable'].append('reference')
				if 'reqt' in true_goal[domain]:
					if 'trainID' in true_goal[domain]['reqt']:
						goal[domain]['requestable'].append('id')
			else:
				if 'reqt' in true_goal[domain]:
					for s in true_goal[domain]['reqt']:  # addtional requests:
						if s in ['phone', 'address', 'postcode', 'reference', 'id']:
							# ones that can be easily delexicalized
							goal[domain]['requestable'].append(s)
				if 'book' in true_goal[domain]:
					goal[domain]['requestable'].append("reference")

			goal[domain]["informable"] = true_goal[domain]['info']

			if 'book' in true_goal[domain]:
				goal[domain]["booking"] = true_goal[domain]['book']
		return goal


	def calculateBLEU(self, decode_all):
		ref = {'usr': [], 'sys': []}
		gen = {'usr': [], 'sys': []}
		for dial_name in sorted(decode_all.keys()):
			for side in ['usr', 'sys']:
				for turn_ref in decode_all[dial_name][side]['ref_word']:
					ref[side].append( [turn_ref] ) # only one ref
				for turn_gen in decode_all[dial_name][side]['gen_word']:
					gen[side].append( [turn_gen] ) # only one candidate

		# bleu for usr and sys
		bleu_usr = self.bleu_scorer.score(gen['usr'], ref['usr'])
		bleu_sys = self.bleu_scorer.score(gen['sys'], ref['sys'])
		return bleu_usr, bleu_sys
