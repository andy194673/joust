import os
import sys
import json
import random

def key2str(key):
	return '_'.join(sorted(list(key)))

def str2key(s):
	return tuple(sorted(s.split('_')))

def print_multi_stat(multi):
	for idx, dial_domain in enumerate(sorted(multi.keys())):
		dials = multi[dial_domain]
		dial_domain = '_'.join(sorted(list(dial_domain)))
		print('{}: {} -> train: {}, val: {}, test: {}'.format(idx, dial_domain, len(dials['train']), len(dials['val']), len(dials['test'])))

def randomPickKey(dic, size):
	dic = list(dic.items())
	random.shuffle(dic)
	dic = dict(dic[:size])
	return dic

def write(multi, data_size, single):
	write_dir = 'data/MultiWOZ/single_to_multi/'
	all_multi = {'train': {}, 'val': {}, 'test': {}}
	for idx, dial_domain in enumerate(sorted(multi.keys())):
		if 'train' not in dial_domain and 'taxi' not in dial_domain:
			continue

		# random select training examples
		dials = multi[dial_domain]['train']
		dials = randomPickKey(dials, data_size)

		# add single to selected multi dialogues
		if single:
			for domain in dial_domain:
				single_dials = randomPickKey(single[domain]['train'], 200) # add 200 single domain dialogues into multi domain dialogue
				dials.update(single_dials)

		# write out
		target_dir = '{}/{}_size{}'.format(write_dir, key2str(dial_domain), 'All' if data_size == 10000 else data_size)
		if single: target_dir += '_single'
		print('write {} -> train: {} | val: {} | test: {}'.format(target_dir, len(dials), len(multi[dial_domain]['val']), len(multi[dial_domain]['test'])))
		os.makedirs(target_dir)

		with open('{}/{}_dials.json'.format(target_dir, 'train'), 'w') as f:
			json.dump(dials, f, indent=4, sort_keys=True)
			all_multi['train'].update(dials)

		# write out fixed val/test examples
		for dType in ['val', 'test']:
			with open('{}/{}_dials.json'.format(target_dir, dType), 'w') as f:
				json.dump(multi[dial_domain][dType], f, indent=4, sort_keys=True)
			all_multi[dType].update(multi[dial_domain][dType])

	# write out all adapt
	target_dir = '{}/{}_size{}'.format(write_dir, 'all_multi', 'All' if data_size == 10000 else data_size)
	if single: target_dir += '_single'
	print('write {} -> train: {} | val: {} | test: {}'.format(target_dir, len(all_multi['train']), len(all_multi['val']), len(all_multi['test'])))
	os.makedirs(target_dir)

	for dType in ['train', 'val', 'test']:
		with open('{}/{}_dials.json'.format(target_dir, dType), 'w') as f:
			json.dump(all_multi[dType], f, indent=4, sort_keys=True)

# --------------------------------------------------------------------------------------
# original split
#train_dial = json.load(open('data/MultiWOZ/self-play-fix/train_dials.json'))
#val_dial = json.load(open('data/MultiWOZ/self-play-fix/val_dials.json'))
#test_dial = json.load(open('data/MultiWOZ/self-play-fix/test_dials.json'))
train_dial = json.load(open('data/MultiWOZ/self-play-fix2/train_dials.json'))
val_dial = json.load(open('data/MultiWOZ/self-play-fix2/val_dials.json'))
test_dial = json.load(open('data/MultiWOZ/self-play-fix2/test_dials.json'))

all_data = json.load(open('data/MultiWOZ/annotated_user_da_with_span_full_patchName.json'))
print('done loading')

domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'police', 'hospital']
#single = {domain: [] for domain in domains}
#single = {domain: {'train': [], 'val': [], 'test': []} for domain in domains}
single = {domain: {'train': {}, 'val': {}, 'test': {}} for domain in domains}
multi = {}
not_used = []

for dial_name, dial in all_data.items():
	dial_name = dial_name + '.json'
	goal = dial['goal']

	# get domains involved
	dial_domain = set([])
	for key in goal:
		if key in ['topic', 'message']:
			continue
		if goal[key]:
			dial_domain.add(key)

	if len(dial_domain) == 1:
		domain = dial_domain.pop()
		if dial_name in train_dial:
			single[domain]['train'][dial_name] = train_dial[dial_name]
		elif dial_name in val_dial:
			single[domain]['val'][dial_name] = val_dial[dial_name]
		elif dial_name in test_dial:
			single[domain]['test'][dial_name] = test_dial[dial_name]
		else:
#			print('error in single', dial_name)
#			sys.exit(1)
			not_used.append(dial_name)
	else:
#		dial_domain = tuple(dial_domain)
		dial_domain = tuple(sorted(dial_domain))
		if dial_domain not in multi:
#			multi[dial_domain] = {'train': [], 'val': [], 'test': []}
			multi[dial_domain] = {'train': {}, 'val': {}, 'test': {}}

#		multi[dial_domain].append(dial_name)
		if dial_name in train_dial:
			multi[dial_domain]['train'][dial_name] = train_dial[dial_name]
		elif dial_name in val_dial:
			multi[dial_domain]['val'][dial_name] = val_dial[dial_name]
		elif dial_name in test_dial:
			multi[dial_domain]['test'][dial_name] = test_dial[dial_name]
		else:
#			print('error in multi', dial_name)
#			sys.exit(1)
			not_used.append(dial_name)

print('not used dials: {}'.format(len(not_used)))
print('--------------------------------------------')
# done, start write out and analysis

write_out_dir = 'data/MultiWOZ/single_to_multi'
c, C = {'train': 0, 'val': 0, 'test': 0}, {'train': 0, 'val': 0, 'test': 0}
base = {'train': {}, 'val': {}, 'test': {}}
print('Single domain dialogues')
for idx, (domain, dials) in enumerate(single.items()):
	if domain in ['police', 'hospital']:
		continue
	print('{}: {} -> train: {}, val: {}, test: {}'.format(idx, domain, len(dials['train']), len(dials['val']), len(dials['test'])))

	for dType in ['train', 'val', 'test']:
		c[dType] += len(dials[dType])
		base[dType].update(dials[dType])

print('single -> train: {}, val: {}, test: {}'.format(c['train'], c['val'], c['test']))
print('--------------------------------------------')

# augment base dataset with its val and test
base['train'].update(base['val'])
base['train'].update(base['test'])
# write out single
os.makedirs('{}/all_single'.format(write_out_dir), exist_ok=True)
for dType in ['train', 'val', 'test']:
	with open('{}/all_single/{}_dials.json'.format(write_out_dir, dType), 'w') as f:
		json.dump(base[dType], f, indent=4, sort_keys=True)
#sys.exit(1)


print('Original multi-domain data distribution without merging and augmentation with single dialogus')
print_multi_stat(multi)
print('--------------------------------------------')
#sys.exit(1)

# merge w/i w/o taxi domains
for idx, (dial_domain, dials) in enumerate(multi.items()):
	if 'train' in dial_domain or 'taxi' in dial_domain:
		continue
	if 'taxi' not in dial_domain:
		dial_domain = key2str(dial_domain)
		dial_domain_taxi = dial_domain + '_taxi'
		for dType in ['train', 'val', 'test']:
			multi[str2key(dial_domain_taxi)][dType].update(multi[str2key(dial_domain)][dType])
print('after merging taxi into non-taxi')
print_multi_stat(multi)
print('--------------------------------------------')
#sys.exit(1)

# write out all comb
for data_size in [10000, 500, 300, 100]:
	print('{} multi'.format(data_size))
	write(multi, data_size, {})
#	print_multi_stat(multi)

	print('{} multi + single'.format(data_size))
	write(multi, data_size, single)
#	print_multi_stat(multi)
#	input('press...')


##### OLD CODE #####
## NOTE: ADD SINGLE TO MULTI, back
#multi_add_single = {}
#for domains in multi:
#	dial2meta = multi[domains]['train']
#	multi_add_single[domains] = {}
#	multi_add_single[domains].update(dial2meta)
#	print('before add single, {} -> # {} dials, {} dials'.format(domains, len(dial2meta), len(multi_add_single[domains])))
#	for domain in domains:
#		for idx, (dial_name, meta) in enumerate(single[domain]['train'].items()):
#			if idx == 200:
#				break
##			assert dial_name not in dials
##			dial2meta[dial_name] = meta
#			assert dial_name not in multi_add_single[domains]
#			multi_add_single[domains][dial_name] = meta
#	print('after add single, {} -> # {} dials, {} dials'.format(domains, len(dial2meta), len(multi_add_single[domains])))
#print('--------------------------------------------')
##sys.exit(1)

#overlap_domains = ['attraction_hotel', 'hotel_restaurant', 'attraction_restaurant'] # w/i or w/o taxi
#overlap = {'{}___taxi'.format(d): {'train': {}, 'val': {}, 'test': {}} for d in overlap_domains}
#print('Multi domain dialogues')
#for idx, (domains, dials) in enumerate(multi.items()):
#	domains = '_'.join(sorted(list(domains)))
#	print('{}: {} -> train: {}, val: {}, test: {}'.format(idx, domains, len(dials['train']), len(dials['val']), len(dials['test'])))
##	os.makedirs('{}/{}'.format(write_out_dir, domains), exist_ok=True)
#	os.makedirs('{}/{}_2'.format(write_out_dir, domains), exist_ok=True) # back
#	for dType in ['train', 'val', 'test']:
#		C[dType] += len(dials[dType])
#
#		# write out multi
##		with open('{}/{}/{}_dials.json'.format(write_out_dir, domains, dType), 'w') as f:
#		with open('{}/{}_2/{}_dials.json'.format(write_out_dir, domains, dType), 'w') as f: # back
#			json.dump(dials[dType], f, indent=4, sort_keys=True)
#
#	if domains.replace('_taxi', '') in overlap_domains:
#		for dType in ['train', 'val', 'test']:
#			overlap['{}___taxi'.format(domains.replace('_taxi', ''))][dType].update(dials[dType])
#
#print('multi -> train: {}, val: {}, test: {}'.format(C['train'], C['val'], C['test']))
#print('--------------------------------------------')
#for idx, (domains, dials) in enumerate(overlap.items()):
#	print('{}: {} -> train: {}, val: {}, test: {}'.format(idx, domains, len(dials['train']), len(dials['val']), len(dials['test'])))
##	os.makedirs('{}/{}'.format(write_out_dir, domains), exist_ok=True)
#	os.makedirs('{}/{}_2'.format(write_out_dir, domains), exist_ok=True) # back
#
#	# write out __taxi
#	for dType in ['train', 'val', 'test']:
##		with open('{}/{}/{}_dials.json'.format(write_out_dir, domains, dType), 'w') as f:
#		with open('{}/{}_2/{}_dials.json'.format(write_out_dir, domains, dType), 'w') as f: # back
#			json.dump(dials[dType], f, indent=4, sort_keys=True)
#	
