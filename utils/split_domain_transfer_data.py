import os
import sys
import json
import random

random.seed(2)

def randomPickKey(dic, size):
	dic = sorted(list(dic.items()))
	random.shuffle(dic)
	dic = dict(dic[:size])
	return dic

print('Loading data')
train_dial = json.load(open('data/process_data/train_dials.json'))
val_dial = json.load(open('data/process_data/val_dials.json'))
test_dial = json.load(open('data/process_data/test_dials.json'))
all_data = json.load(open('data/raw_data/annotated_user_da_with_span_full_patchName.json'))

domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi']
data = {domain: {'involved': {'train': {}, 'val': {}, 'test': {}},
				'not_involved': {'train': {}, 'val': {}, 'test': {}}} for domain in domains}

print('Parsing dialogues')
not_used = []
police_hospital = []
for dial_name, dial in all_data.items():
	dial_name = dial_name + '.json'

	if dial_name not in train_dial and dial_name not in val_dial and dial_name not in test_dial:
		not_used.append(dial_name)

	goal = dial['goal']
	dial_domain = set([])
	for key in goal:
		if key in ['topic', 'message']:
			continue
		if goal[key]:
			dial_domain.add(key)

	# exclude dialogues with only police or hospital
	if 'police' in dial_domain: dial_domain.remove('police')
	if 'hospital' in dial_domain: dial_domain.remove('hospital')
	if len(dial_domain) == 0:
		police_hospital.append(dial_name)
		continue

	for domain in domains:
		if domain in dial_domain: # involved in this dialogue
			if dial_name in train_dial:
				data[domain]['involved']['train'][dial_name] = train_dial[dial_name]
			elif dial_name in val_dial:
				data[domain]['involved']['val'][dial_name] = val_dial[dial_name]
			elif dial_name in test_dial:
				data[domain]['involved']['test'][dial_name] = test_dial[dial_name]
		else:
			if dial_name in train_dial:
				data[domain]['not_involved']['train'][dial_name] = train_dial[dial_name]
			elif dial_name in val_dial:
				data[domain]['not_involved']['val'][dial_name] = val_dial[dial_name]
			elif dial_name in test_dial:
				data[domain]['not_involved']['test'][dial_name] = test_dial[dial_name]

print('not used dials: {}'.format(len(not_used)))
print('police or hospital involved dials: {}'.format(len(police_hospital)))
print('--------------------------------------------')

# use only 5k dialogues for base model
for domain in domains:
	# augment base dataset with its val and test
	data[domain]['not_involved']['train'].update(data[domain]['not_involved']['val'])
	data[domain]['not_involved']['train'].update(data[domain]['not_involved']['test'])
	print('Original not_involved (include its val and test) in {} => {}'.format(domain, len(data[domain]['not_involved']['train'])))
	data[domain]['not_involved']['train'] = randomPickKey(data[domain]['not_involved']['train'], 5000)
print('--------------------------------------------')

# size_list = [100, 300, 500, 1000, 'All']
size_list = [300]
for domain in domains:
	if domain in ['police', 'hospital']:
		continue
	print(domain)
	os.makedirs('data/domain_transfer/{}'.format(domain))
	os.makedirs('data/MultiWOZ/domain_transfer/{}/{}'.format(domain, 'not_involved'))
	for size in size_list:
		os.makedirs('data/domain_transfer/{}/{}_size{}'.format(domain, 'involved', size))

	for dType in ['train', 'val', 'test']:
		inv, not_inv = len(data[domain]['involved'][dType]), len(data[domain]['not_involved'][dType])
		print('\t{} | involved: {} | not_involved: {} | total: {}'.format(dType, inv, not_inv, inv+not_inv))

		# write out not_involved
		with open('data/domain_transfer/{}/{}/{}_dials.json'.format(domain, 'source', dType), 'w') as f:
			json.dump(data[domain]['not_involved'][dType], f, indent=4, sort_keys=True)

		# write out involved
		for size in size_list:
			if dType == 'train':
				data_reduce = randomPickKey(data[domain]['involved'][dType], size if size != 'All' else 10000)
				with open('data/domain_transfer/{}/{}_size{}/{}_dials.json'.format(domain, 'target', size, dType), 'w') as f:
					json.dump(data_reduce, f, indent=4, sort_keys=True)
			else:
				with open('data/domain_transfer/{}/{}_size{}/{}_dials.json'.format(domain, 'target', size, dType), 'w') as f:
					json.dump(data[domain]['involved'][dType], f, indent=4, sort_keys=True)
	print('Done', domain)
