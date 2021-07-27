import os
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
	write_dir = 'data/single_to_multi/'
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


print('Loading data...')
train_dial = json.load(open('data/process_data/train_dials.json'))
val_dial = json.load(open('data/process_data/val_dials.json'))
test_dial = json.load(open('data/process_data/test_dials.json'))
all_data = json.load(open('data/raw_data/annotated_user_da_with_span_full_patchName.json'))

domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'police', 'hospital']
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

write_out_dir = 'data/single_to_multi'
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

print('Original multi-domain data distribution without merging and augmentation with single dialogus')
print_multi_stat(multi)
print('--------------------------------------------')

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

# write out all comb
# for data_size in [10000, 500, 300, 100]:
for data_size in [100]:
	print('{} multi'.format(data_size))
	write(multi, data_size, {})

	print('{} multi + single'.format(data_size))
	write(multi, data_size, single)
