'''
	Utility functions for DST
'''

import os
import sys

cwd = os.getcwd()
sys.path.insert(0, cwd+'/utils/')
from fix_label import fix_general_label_error

EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]

def convert_slot(slot):
	if 'book' in slot:
		return slot.replace('book ', '')
	if slot == 'arriveby':
		return 'arriveBy'
	if slot == 'leaveat':
		return 'leaveAt'
	return slot


def allign_dict_slot(bs_dict):
	new_bs_dict = {}
	for domain_slot, value in bs_dict.items():
		domain, slot = domain_slot.split('-')
		new_slot = convert_slot(slot)
		new_domain_slot = '{}-{}'.format(domain, new_slot)
		new_bs_dict[new_domain_slot] = value
	return new_bs_dict


def fix_wrong_domain_label(turn_belief_dict, domains, dial_idx, turn_idx):
	remove_keys = []
	for domain_slot in turn_belief_dict:
		domain = domain_slot.split('-')[0]
		if domain not in domains:
			remove_keys.append(domain_slot)

	for key in remove_keys:
		del turn_belief_dict[key]
	# turn on to trace changed turn
#	if len(remove_keys) > 0:
#		print('fix wrong domain on:', dial_idx, turn_idx)
#		input('press...')
	return turn_belief_dict


def get_slot_information(ontology):
	ontology_domains = dict([(k, v) for k, v in ontology.items() if k.split("-")[0] in EXPERIMENT_DOMAINS])
	SLOTS = [k.replace(" ","").lower() if ("book" not in k) else k.lower() for k in ontology_domains.keys()]
	SLOTS = [domain_slot.split('-')[0]+'-'+convert_slot(domain_slot.split('-')[1]) for domain_slot in SLOTS]
	return sorted(SLOTS)


def get_nlu_label(curr_bs_dict, last_bs_dict):
	nlu_dict = {}
	for domain_slot, value in curr_bs_dict.items():
		if domain_slot in last_bs_dict:
			continue
		nlu_dict[domain_slot] = value
	return nlu_dict


def dict2list(bs_dict):
	'''
	convert bs dict in dst files into a list with token domain-slot=value, such as 'hotel-area=north'
	'''
	l = [s+'='+v for s, v in bs_dict.items()]
	return sorted(l)


def remove_dontcare_value(bs_dict):
	'''Remove slot with 'dontcare' value'''
	slots = []
	for slot, value in bs_dict.items():
		if value == 'dontcare':
			slots.append(slot)
	for slot in slots:
		del bs_dict[slot]
	return bs_dict


def iter_dst_file(dst_cont, dst_data, delex_data, ALL_SLOTS, word2count=None, slot2value=None,
				  display=False, remove_dontcare=False, fix_wrong_domain=True):
	for dial_dict in dst_data:
		domains = dial_dict["domains"]
		if 'hospital' in domains or 'police' in domains:
			continue

		dial_name = dial_dict['dialogue_idx']
		if dial_name not in delex_data: # filter out dialogue without delex label
#			print('filter out {} without delex version'.format(dial_name))
			continue

		# make sure dialogue length are same between two data files
		delex_dial = delex_data[dial_name]
		assert len(dial_dict["dialogue"]) == len(delex_dial['sys'])

		# start process dialogue
		if display: print('Dialogue:', dial_name)
		assert dial_name not in dst_cont
		dst_cont[dial_name] = {'input_utt': [], 'prev_bs': [], 'curr_bs': [], 'curr_nlu': [], 'history': []}
		last_bs_dict = {}
		history = []
		for turn_idx, turn in enumerate(dial_dict["dialogue"]):
			lex_usr = turn['transcript'].strip()
			lex_sys = turn['system_transcript'].strip()
			delex_sys = delex_dial['sys'][turn_idx-1] if turn_idx != 0 else ''
			turn_uttr = lex_sys + ' ; ' + lex_usr # BACK
			history.append(turn_uttr)
			if display: print('lex sys:', lex_sys)
			if display: print('dex sys:', delex_sys)
			if display: print('lex usr:', lex_usr)
			if display: print('---------------------------------------')

			if word2count != None:
				for word in turn_uttr.split():
					if word not in word2count:
						word2count[word] = 0
					word2count[word] += 1

			bs_dict = turn["belief_state"]
			bs_dict = fix_general_label_error(bs_dict, False, ALL_SLOTS) # dict bs {'domain-slot': value}, such as {'hotel-area': north}

			if remove_dontcare:
				bs_dict	= remove_dontcare_value(bs_dict) # NOTE: treat dontcare as not_mentioned since its the same for db query

			if fix_wrong_domain:
				bs_dict = fix_wrong_domain_label(bs_dict, dial_dict['domains'], dial_name, turn_idx) # NOTE: fix wrong domain label

			bs_dict = allign_dict_slot(bs_dict) # NOTE: allign slot name with delex word
			nlu_dict = get_nlu_label(bs_dict, last_bs_dict)

			if display: print('prev bs:', dict2list(last_bs_dict))
			if display: print('curr bs:', dict2list(bs_dict))
			if display: print('turn nlu:', dict2list(nlu_dict))
			if display: print('---------------------------------------')
			if display: input('press...')

			if slot2value != None:
				for domain_slot, value in bs_dict.items():
					assert domain_slot in ALL_SLOTS # verify slot name consistency
#					if value not in value2count:
#						value2count[value] = 0
#					value2count[value] += 1
					if domain_slot not in slot2value:
						slot2value[domain_slot] = set()
					slot2value[domain_slot].add(value)
					slot2value['all'].add(value)

			# collect info
			dst_cont[dial_name]['input_utt'].append(turn_uttr) # str
			dst_cont[dial_name]['history'].append( list(history) ) # list of turn_uttr
			dst_cont[dial_name]['prev_bs'].append( last_bs_dict ) # dict of {domain-slot: value}, such as {'hotel-area': north'}
			dst_cont[dial_name]['curr_bs'].append( bs_dict )
			dst_cont[dial_name]['curr_nlu'].append( nlu_dict )
			last_bs_dict = bs_dict