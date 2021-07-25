def check_turn_act(act_seq, act, domain, slot):
	'''
	decide if a slot with act exists in a act_seq
	e.g., slot=hotel_people, act=inform exists in act_seq='act_inform hotel_people act_reqt hotel_stars'
	however, slot=hotel_people, act=reqt doest not exist in this act_seq because wrong act
	'''
	# turn into correct form
	slot = '{}_{}'.format(domain, slot)
	act = 'act_{}'.format(act)

	# first check if slot and act exists
	if slot not in act_seq:
		return False
	if act not in act_seq:
		return False

	# check if its with correct act, if the slot belongs to that act, then there is no act token in interval seq
	act_seq = act_seq.split()
	act_idx = act_seq.index(act)
	slot_idx = act_seq.index(slot)
	if slot_idx < act_idx: # slot appears before act
		return False

	interval_seq = ' '.join(act_seq[act_idx+1: slot_idx])
	if 'act_' in interval_seq: # exist other act in interval
		return False
	else:
		return True


def get_turn_act_slot(act_seq, act):
	'''return slots that are with specified act if exist'''
	slots = []
	act = 'act_{}'.format(act)
	if act not in act_seq:
		return slots

	act_seq = act_seq.split()
	seq_len = len(act_seq)
	act_idx = act_seq.index(act)

	# scan act seq to obtain slots
	for idx in range(act_idx+1, seq_len, 1):
		token = act_seq[idx]
		if 'act_' in token: # encounter next act
			break
		slots.append(token)
	return slots


def decide_turn_domain(usr_act, sys_act, domain_prev):
	'''
	decide turn-level domain by the majority of domain slot in (generated) usr_act and sys_act
	if cannot decide, follow the domain in previous turn
	'''
	assert isinstance(usr_act, str)
	assert isinstance(sys_act, str)
	domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'police', 'hospital', 'general']
	count = [0 for _ in range(len(domains))]
	act = usr_act + ' ' + sys_act
	for slot in act.split():
		if slot == 'act_reqmore':
			break
		domain = slot.split('_')[0]
		if domain not in domains:
			continue
		count[domains.index(domain)] += 1

	max_count = max(count)
#	assert count.count(max_count) == 1
#		return 'unsure'
#		print('############ cannot decide ##############')

	if count.count(max_count) != 1: # no majority domain
		if 'taxi' in act:
			return 'taxi'
		if len(act.split()) > 0 and act.split()[0] == 'act_reqmore':
			return 'general'
		if domain_prev == 'none':
			return domains[count.index(max_count)]
		else:
			return domain_prev
	else:
		domain = domains[count.index(max_count)]
		return domain


def check_turn_type(act_usr, act_sys, turn_domain, keySlot):
	'''
	decide turn type within a domain, either info, book, reqt or none (for general domain)
	check by some rules:
		book: if usr informs any booking slot or if sys reqt any booking slot
		reqt: if usr reqt any reqt slot or if sys inform any reqt slot
	'''
	if turn_domain in ['taxi', 'police', 'hospital', 'general']:
		return 'none'

	# check book
	if 'book' in keySlot[turn_domain]: # some domains have no booking stage
		for slot in keySlot[turn_domain]['book']:
#			if checkActSlotInTurn(act_usr, 'inform', turn_domain, slot) or checkActSlotInTurn(act_sys, 'request', turn_domain, slot):
			if check_turn_act(act_usr, 'inform', turn_domain, slot) or 'act_offerbooked' in act_sys:
				return 'book'

	# check reqt
	for slot in keySlot[turn_domain]['reqt']:
		if check_turn_act(act_usr, 'request', turn_domain, slot) or check_turn_act(act_sys, 'inform', turn_domain, slot):
			return 'reqt'

	# else
	return 'info'


def form_key_slot():
	# form key slot, this is used to decide a turn at which stage (info/book/reqt)
	domain_list = ['restaurant', 'hotel', 'attraction', 'train', 'taxi']
	keySlot = {domain: {} for domain in domain_list}
	keySlot['restaurant']['reqt'] = ['address', 'phone', 'postcode']
	keySlot['hotel']['reqt'] = ['address', 'phone', 'postcode']
	keySlot['attraction']['reqt'] = ['address', 'phone', 'postcode', 'fee']
	keySlot['train']['reqt'] = ['duration', 'price']
	keySlot['taxi']['reqt'] = ['type', 'phone']
	keySlot['restaurant']['book'] = ['day', 'people', 'time']
	keySlot['hotel']['book'] = ['day', 'people', 'stay']
	return keySlot