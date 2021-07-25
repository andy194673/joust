import os
import sys
import json
import operator

cwd = os.getcwd()
sys.path.insert(0, cwd+'/utils/')
from fix_label import fix_general_label_error
'''
build 1) slot list, 2) value list and 3) word2count in dst files
slot name are alligned with delex words
words are corrected by fix_label* function
'''

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
	'''
	remove slot with 'dontcare' value
	'''
	slots = []
	for slot, value in bs_dict.items():
		if value == 'dontcare':
			slots.append(slot)
	for slot in slots:
		del bs_dict[slot]
	return bs_dict

#def iter_dst_file(dst_cont, dst_data, delex_data, ALL_SLOTS, word2count=None, value2count=None, display=False, \
def iter_dst_file(dst_cont, dst_data, delex_data, ALL_SLOTS, word2count=None, slot2value=None, display=False, \
					remove_dontcare=False, fix_wrong_domain=True):
	for dial_dict in dst_data:
		domains = dial_dict["domains"]
#		for domain in dial_dict["domains"]:
#			if domain not in EXPERIMENT_DOMAINS:
#				continue
#		if len(domains) != 1 or 'restaurant' not in domains: # BACK
#			continue

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
#			turn_uttr = turn["system_transcript"] + " ; " + turn["transcript"]
			lex_usr = turn['transcript'].strip()
			lex_sys = turn['system_transcript'].strip()
			delex_sys = delex_dial['sys'][turn_idx-1] if turn_idx != 0 else ''
#			turn_uttr = delex_sys + ' ; ' + lex_usr # NOTE: use delex on sys side for rl interaction
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

#			if value2count != None:
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
#			dst_cont[dial_name]['prev_bs'].append( dict2list(last_bs_dict) ) # list of domain-slot=value, such as 'hotel-area=north'
#			dst_cont[dial_name]['curr_bs'].append( dict2list(bs_dict) )
#			dst_cont[dial_name]['curr_nlu'].append( dict2list(nlu_dict) )
			dst_cont[dial_name]['prev_bs'].append( last_bs_dict ) # dict of {domain-slot: value}, such as {'hotel-area': north'}
			dst_cont[dial_name]['curr_bs'].append( bs_dict )
			dst_cont[dial_name]['curr_nlu'].append( nlu_dict )
			last_bs_dict = bs_dict

		# check dialogue history
#		for i, x in enumerate(dst_cont[dial_name]['history']):
#			print('turn:', i, file=sys.stderr)
#			print(x, file=sys.stderr)
#		input('press...')
			

if __name__ == '__main__':
	#file_train = 'data/MultiWOZ/dst/train_dials.json'
	#file_dev = 'data/MultiWOZ/dst/dev_dials.json'
	#file_test = 'data/MultiWOZ/dst/test_dials.json'
	
	# build slot list
	ontology = json.load(open("data/MultiWOZ/dst/ontology.json", 'r'))
	ALL_SLOTS = get_slot_information(ontology)
	#print(ALL_SLOTS)
	#with open('data/MultiWOZ/dst/slot_list.json', 'w') as f:
	#	json.dump(ALL_SLOTS, f, indent=4)

	dst_cont = {} # data container
	word2count = {}
#	value_set = set()
#	value2count = {}
	slot2value = {'all': set()}
	for data_type in ['train', 'dev', 'test']:
		print('Read', data_type)
#		dst_file = 'data/MultiWOZ/dst/{}_dials.json'.format(data_type)
		dst_file = 'data/MultiWOZ/dst/{}2.1_dials.json'.format(data_type)
		delex_file = 'data/MultiWOZ/self-play-fix2/{}_dials.json'.format(data_type if data_type!='dev' else 'val')
		with open(dst_file) as f1, open(delex_file) as f2:
			dst_data = json.load(f1)
			delex_data = json.load(f2)
#			iter_dst_file(dst_cont, dst_data, delex_data, ALL_SLOTS, word2count=word2count, value2count=value2count, display=True)
#			iter_dst_file(dst_cont, dst_data, delex_data, ALL_SLOTS, word2count=word2count, value2count=value2count, display=False)
			iter_dst_file(dst_cont, dst_data, delex_data, ALL_SLOTS, word2count=word2count, slot2value=slot2value, display=False)

	# dump value list
#	value_list = sorted(list(value_set))
#	value2count = sorted(value2count.items(), key=operator.itemgetter(1), reverse=True)
#	with open('data/MultiWOZ/dst/value2count.json', 'w') as f:
#	with open('data/MultiWOZ/dst/value2count_sysLex.json', 'w') as f:
##		json.dump(value_list, f, indent=4)
#		json.dump(value2count, f, indent=4)

	for slot, value_list in slot2value.items():
		slot2value[slot] = sorted(list(value_list))
#	with open('data/MultiWOZ/dst/slot2value.json', 'w') as f:
	with open('data/MultiWOZ/dst/slot2value_2.1.json', 'w') as f:
		json.dump(slot2value, f, indent=4, sort_keys=True)

	# dump word2count
	word2count = sorted(word2count.items(), key=operator.itemgetter(1), reverse=True)
#	with open('data/MultiWOZ/dst/dst_word2count.json', 'w') as f:
#	with open('data/MultiWOZ/dst/dst_word2count_sysLex.json', 'w') as f:
	with open('data/MultiWOZ/dst/dst_word2count_sysLex_2.1.json', 'w') as f:
		json.dump(word2count, f, indent=4)

	print('Done!')
