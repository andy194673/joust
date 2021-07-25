# -*- coding: utf-8 -*-
'''
	This script is adapted from multiwoz benchmark repository
	https://github.com/budzianowski/multiwoz/blob/master/create_delex_data.py
'''
import sys
import copy
import json
import os
import re
import shutil
import urllib
from collections import OrderedDict
from io import BytesIO
from zipfile import ZipFile
from tqdm import tqdm

import numpy as np
import operator

sys.path.append(os.getcwd())
from data_preprocess.nlp import normalize
from data_preprocess import dbPointer, delexicalize
from data_preprocess.db_ops import MultiWozDB

DB = MultiWozDB()
np.set_printoptions(precision=3)
np.random.seed(2)

# GLOBAL VARIABLES
DICT_SIZE = 400
MAX_LENGTH = 50


def is_ascii(s):
	return all(ord(c) < 128 for c in s)


def fixDelex(filename, data, data2, idx, idx_acts):
	"""Given system dialogue acts fix automatic delexicalization."""
	try:
		turn = data2[filename.strip('.json')][str(idx_acts)]
	except:
		return data

	text = data['log'][idx]['text']
	if not isinstance(turn, str):
		for k, act in turn.items():
			if 'Attraction' in k:
				if 'restaurant_' in data['log'][idx]['text']:
					data['log'][idx]['text'] = data['log'][idx]['text'].replace("restaurant", "attraction")
				if 'hotel_' in data['log'][idx]['text']:
					data['log'][idx]['text'] = data['log'][idx]['text'].replace("hotel", "attraction")
				if 'restaurant_' in text or 'hotel_' in text:
					raise ValueError('weird problem')
			if 'Hotel' in k:
				if 'attraction_' in data['log'][idx]['text']:
					data['log'][idx]['text'] = data['log'][idx]['text'].replace("attraction", "hotel")
				if 'restaurant_' in data['log'][idx]['text']:
					data['log'][idx]['text'] = data['log'][idx]['text'].replace("restaurant", "hotel")
				if 'restaurant_' in text or 'attraction_' in text:
					raise ValueError('weird problem')
			if 'Restaurant' in k:
				if 'attraction_' in data['log'][idx]['text']:
					data['log'][idx]['text'] = data['log'][idx]['text'].replace("attraction", "restaurant")
				if 'hotel_' in data['log'][idx]['text']:
					data['log'][idx]['text'] = data['log'][idx]['text'].replace("hotel", "restaurant")
				if 'hotel_' in text or 'attraction_' in text:
					raise ValueError('weird problem')
	return data


def delexicaliseReferenceNumber(sent, turn):
	"""Based on the belief state, we can find reference number that
	during data gathering was created randomly."""
	domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital']  # , 'police']
	if turn['metadata']:
		for domain in domains:
			if turn['metadata'][domain]['book']['booked']:
				for slot in turn['metadata'][domain]['book']['booked'][0]:
					if slot == 'reference':
						val = '[' + domain + '_' + slot + ']'
					else:
						val = '[' + domain + '_' + slot + ']'
					key = normalize(turn['metadata'][domain]['book']['booked'][0][slot])
					sent = (' ' + sent + ' ').replace(' ' + key + ' ', ' ' + val + ' ')

					# try reference with hashtag
					key = normalize("#" + turn['metadata'][domain]['book']['booked'][0][slot])
					sent = (' ' + sent + ' ').replace(' ' + key + ' ', ' ' + val + ' ')

					# try reference with ref#
					key = normalize("ref#" + turn['metadata'][domain]['book']['booked'][0][slot])
					sent = (' ' + sent + ' ').replace(' ' + key + ' ', ' ' + val + ' ')
	return sent


def addBookingPointer(task, turn, pointer_vector):
	"""Add information about availability of the booking option."""
	# Booking pointer
	rest_vec = np.array([1, 0])
	if task['goal']['restaurant']:
		if turn['metadata']['restaurant'].has_key("book"):
			if turn['metadata']['restaurant']['book'].has_key("booked"):
				if turn['metadata']['restaurant']['book']["booked"]:
					if "reference" in turn['metadata']['restaurant']['book']["booked"][0]:
						rest_vec = np.array([0, 1])

	hotel_vec = np.array([1, 0])
	if task['goal']['hotel']:
		if turn['metadata']['hotel'].has_key("book"):
			if turn['metadata']['hotel']['book'].has_key("booked"):
				if turn['metadata']['hotel']['book']["booked"]:
					if "reference" in turn['metadata']['hotel']['book']["booked"][0]:
						hotel_vec = np.array([0, 1])

	train_vec = np.array([1, 0])
	if task['goal']['train']:
		if turn['metadata']['train'].has_key("book"):
			if turn['metadata']['train']['book'].has_key("booked"):
				if turn['metadata']['train']['book']["booked"]:
					if "reference" in turn['metadata']['train']['book']["booked"][0]:
						train_vec = np.array([0, 1])

	pointer_vector = np.append(pointer_vector, rest_vec)
	pointer_vector = np.append(pointer_vector, hotel_vec)
	pointer_vector = np.append(pointer_vector, train_vec)
	return pointer_vector


def addDBPointer(turn, return_ent=None):
	"""Create database pointer for all related domains."""
	domains = ['restaurant', 'hotel', 'attraction', 'train']
	pointer_vector = np.zeros(3 * len(domains)) # only use 3 classes: 0, 1 and >2
	domain2ents = {}
	for domain in domains:
		ents = DB.queryJsons(domain, turn['metadata'][domain]['semi'])
		domain2ents[domain] = ents
		pointer_vector = dbPointer.oneHotVector(len(ents), domain, pointer_vector)
	if return_ent == None:
		return pointer_vector
	else:
		return domain2ents


def get_summary_bstate(bstate):
	"""Based on the mturk annotations we form multi-domain belief state"""
	domains = ['taxi', 'restaurant', 'hospital', 'hotel', 'attraction', 'train', 'police']
	summary_bstate = []

	###### bs can only have information which is available during usr/sys interaction #####
	for domain in domains:
		domain_active = False
		booking = []
		if domain != 'train': # since dst slots in train domain is varying (some dialogues have people, some have ticket)
			for slot in sorted(bstate[domain]['book'].keys()):
				if slot == 'booked':
					continue # we don't have the info about whether booking is successful during the interaction
				else:
					if bstate[domain]['book'][slot] != "": # in data.json, this is either is empty or real value, no other options
						booking.append(1)
					else:
						booking.append(0)
		else: # train domain
			if 'people' in bstate[domain]['book'].keys() and bstate[domain]['book']['people'] != "":
				booking.append(1)
			else:
				booking.append(0)
		summary_bstate += booking

		for slot in sorted(bstate[domain]['semi'].keys()):
			if domain == 'hotel' and slot == 'type': # since this info is not obtained during usr/sys interaction
				continue
			if bstate[domain]['semi'][slot] == 'not mentioned' or bstate[domain]['semi'][slot] == '':
				slot_enc = [0] # treat empty value as not_mentioned for unity between SL data and RL data
			elif bstate[domain]['semi'][slot] == 'dont care' or bstate[domain]['semi'][slot] == 'dontcare' or bstate[domain]['semi'][slot] == "don't care":
				slot_enc = [0] # bs unity between SL and RL
			elif bstate[domain]['semi'][slot]:
				slot_enc = [1]
			else:
				raise ValueError('Unexpected value!')
			if slot_enc != [0]:
				domain_active = True
			summary_bstate += slot_enc

		# quasi domain-tracker
		if domain_active:
			summary_bstate += [1]
		else:
			summary_bstate += [0]
	assert len(summary_bstate) == 37
	return summary_bstate


def analyze_dialogue(dialogue, maxlen):
	"""Cleaning procedure for all kinds of errors in text and annotation."""
	d = dialogue
	# do all the necessary postprocessing
	if len(d['log']) % 2 != 0:
		print('odd # of turns')
		return None  # odd number of turns, wrong dialogue
	d_pp = {}
	d_pp['goal'] = d['goal']  # for now we just copy the goal
	usr_turns = []
	sys_turns = []
	for i in range(len(d['log'])):
		if len(d['log'][i]['text'].split()) > maxlen:
			# print('too long') # only 15 dialogues here
			return None  # too long sentence, wrong dialogue
		if i % 2 == 0:  # usr turn
			if 'db_pointer' not in d['log'][i]:
				print('no db')
				return None  # no db_pointer, probably 2 usr turns in a row, wrong dialogue
			text = d['log'][i]['text']
			if not is_ascii(text):
				print('not ascii')
				return None
			usr_turns.append(d['log'][i])
		else:  # sys turn
			text = d['log'][i]['text']
			if not is_ascii(text):
				print('not ascii')
				return None
			belief_summary = get_summary_bstate(d['log'][i]['metadata'])
			d['log'][i]['belief_summary'] = belief_summary
			sys_turns.append(d['log'][i])
	d_pp['usr_log'] = usr_turns
	d_pp['sys_log'] = sys_turns
	return d_pp


def get_dial(dialogue):
	"""Extract a dialogue from the file"""
	dial = []
	d_orig = analyze_dialogue(dialogue, MAX_LENGTH)  # max turn len is 50 words
	if d_orig is None:
		return None
	usr = [t['text'] for t in d_orig['usr_log']]
	db = [t['db_pointer'] for t in d_orig['usr_log']]
	bs = [t['belief_summary'] for t in d_orig['sys_log']]
	sys = [t['text'] for t in d_orig['sys_log']]
	usr_act = [t['usr_act'] for t in d_orig['usr_log']]
	sys_act = [t['sys_act'] for t in d_orig['sys_log']]

	for u, d, s, b, ua, sa in zip(usr, db, sys, bs, usr_act, sys_act):
		dial.append((u, s, d, b, ua, sa))

	return dial


def createDict(word_freqs):
	words = word_freqs.keys()
	freqs = word_freqs.values()

	sorted_idx = np.argsort(freqs)
	sorted_words = [words[ii] for ii in sorted_idx[::-1]]

	# Extra vocabulary symbols
	_GO = '_GO'
	EOS = '_EOS'
	UNK = '_UNK'
	PAD = '_PAD'
	extra_tokens = [_GO, EOS, UNK, PAD]

	worddict = OrderedDict()
	for ii, ww in enumerate(extra_tokens):
		worddict[ww] = ii
	for ii, ww in enumerate(sorted_words):
		worddict[ww] = ii + len(extra_tokens)

	for key, idx in worddict.items():
		if idx >= DICT_SIZE:
			del worddict[key]
	return worddict


def detectDoamin(da_dict, dial_name, turn_idx, text):
	domain2count = {'restaurant': 0, 'hotel': 0, 'attraction': 0}
	for da, sv_list in da_dict.items():
		domain, act = da.lower().split('-')
		if domain == 'booking':
			return 'Booking'
		if domain == 'train' or domain == 'taxi':
			return 'None'
		if domain not in ['restaurant', 'hotel', 'attraction']:
			continue
		for s, v in sv_list:
			if v == '?':
				continue
			if s not in ['Addr', 'Name', 'Phone', 'Post', 'Ref']:
				continue
			domain2count[domain] += 1

	domain2count = sorted(domain2count.items(), key=operator.itemgetter(1), reverse=True)
	if domain2count[1][1] > 0:
		return 'Multi'
	elif domain2count[0][1] == 0: # cannot decide what domain involved
		return 'None'
	else:
		return domain2count[0][0]


def fixDomainDelex(text, da_dict, dial_name, turn_idx):
	'''
	Since the function fixDelex looks wrong, it cannot be used to fix the delex slots (address, name, phone, postcode, reference)
	in domains ['restaurant, hotel, attraction'] might be potentially wrong.
	This function for fixing this issue.
	'''
	if '[restaurant_' not in text and '[hotel_' not in text and '[attraction_' not in text: # no need to fix
		return text

	new_text = []
	domain = detectDoamin(da_dict, dial_name, turn_idx, text)
	if domain == 'Multi' or domain == 'None' or domain == 'Booking': # only fix when we sure it's single-domain utterance
		return text

	if domain == 'restaurant':
		text = text.replace('[hotel_', '[restaurant_')
		text = text.replace('[attraction_', '[restaurant_')
	elif domain == 'hotel':
		text = text.replace('[restaurant_', '[hotel_')
		text = text.replace('[attraction_', '[hotel_')
	elif domain == 'attraction':
		text = text.replace('[restaurant_', '[attraction_')
		text = text.replace('[hotel_', '[attraction_')
	return text


def fixSlotValuesIndependent(dic):
	'''There are few venue names that cause bad delex, remove them from delex dic'''
	dic.remove(('restaurant 17', '[restaurant_name]'))
	dic.remove(('17', '[restaurant_name]'))
	dic.remove(('ask restaurant', '[restaurant_name]'))
	dic.remove(('ask', '[restaurant_name]'))
	dic.remove(('the place', '[attraction_name]'))
	return dic


def unify_actSlot_to_goalSlot(s, domain):
	'''Unify slots between GOAL/DST annotation and DA annotation'''
	if s == 'Addr':
		return 'address'
	elif s == 'Post':
		return 'postcode'
	elif s == 'Price':
		return 'pricerange'
	elif s == 'Arrive': # arriveBy = Arrive
		return 'arriveBy'
	elif s == 'Car':
		return 'type'
	elif s == 'Depart': # departure = Depart
		return 'departure'
	elif s == 'Dest': # destination = Dest
		return 'destination'
	elif s == 'Leave': # leaveAt = Leave
		return 'leaveAt'
	elif s == 'Time' and domain == 'train': # duration = Time, FIX
		return 'duration'
	elif s == 'Ticket': # price = Ticket, FIX
		return 'price'
	elif s == 'Id':	# trainID (dst) = Id (da)
		return 'trainID'
	elif s == 'Ref':
		return 'reference'
	else:
		return s.lower()


def delexValueToken(text, da_dict, change_daDict=False, old_daDict=None):
	'''
	delexicalise the value which is replaced with [value_*] before
	FIX: if change_daDict is True, use sys da label to improve usr da label and usr delex
	'''
	def addTonewFindDaDict(newFind_daDict, da, slot, value):
		'''
		add original da with its slot value to old usr da_dict
		can only manage to fix missing label on Inform act
		'''
		domain, act = da.split('-') # e.g., Restaurant-NoBook
		act = 'Inform'
		da = '{}-{}'.format(domain, act)

		if da not in newFind_daDict:
			newFind_daDict[da] = []
		newFind_daDict[da].append([slot, value])

	def proc_booking(text, slot, value, newFind_daDict, da):
		# could be restaurant or hotel, decide domain first
		if 'restaurant' in text or 'table' in text:
			domain = 'restaurant'
#		elif 'hotel' in text or 'accommodate' in text or 'stay' in text or 'night' in text:
		elif slot == 'stay' or 'hotel' in text or 'accommodate' in text or 'stay' in text or 'night' in text:
			domain = 'hotel'
		else:
			return text # leave it for the db delex

		if change_daDict and value in text:
			addTonewFindDaDict(newFind_daDict, da, slot, value)

		text = text.replace(value, ' [' + domain + '_' + slot + '] ')
		return text

	time_slot = ['time', 'arriveBy', 'leaveAt']
	# since we use do da delex before db delex, no processed slot before by pawel's script
	ignore_slot = ['choice', 'fee', 'open', 'internet', 'parking', 'ticket', 'trainID'] # no need to delex
	ignore_domain = ['police', 'hospital', 'general']
	ignore_value = ['?', 'none', 'do nt care']
	processed_value = set()

	newFind_daDict = {}
	# first deal with time slots only since order matters
	for da in sorted(da_dict.keys(), reverse=True):
		sv_list = da_dict[da]
		s_list = set()
		domain, act = da.lower().split('-')
		if domain in ignore_domain:
			continue

		# priority slot with time
		for slot, value in sv_list:
			slot = unify_actSlot_to_goalSlot(slot, domain)
			value = normalize(value)
			if value in ignore_value:
				continue
			if value in processed_value:
				continue
			if domain == 'taxi' and value not in text:
				continue
			if slot in time_slot:
				if value not in text and value[0] == '0' and value[1:] in text:
					value = value[1:]
				# turn on the switch to check real stuff
				if value not in text:
					continue

				if change_daDict and value in text:
					addTonewFindDaDict(newFind_daDict, da, slot, value)

				if domain == 'booking':
					text = text.replace(value, ' [restaurant_time] ')
				else:
					text = text.replace(value, ' [' + domain + '_' + slot + '] ')
				processed_value.add(value)

	# the rest slot		
	# sort by domain first, since one value might have multiple labels
	# e.g., a place can be both restaurant_name or taxi_destination, we want taxi_destination is used first
	for da in sorted(da_dict.keys(), reverse=True):
		sv_list = da_dict[da]
		s_list = set()
		domain, act = da.lower().split('-')
		if domain in ignore_domain:
			continue

		for slot, value in sv_list:
			slot = unify_actSlot_to_goalSlot(slot, domain)
			if value in ignore_value:
				continue
#			if slot in processed_slot:
#				continue
			if slot in ignore_slot:
				continue
			if slot in time_slot:
				continue
			if slot == 'type' and domain != 'attraction':
				continue

			# now can delexicalise
			value = normalize(value)
			if value in processed_value:
				continue
			if value not in text and 'same' in text: # co-reference happens
				continue
			if slot == 'area' and value == 'centre' and 'center' in text:
				value = 'center'
			if slot == 'type' and value == 'theatre' and 'theater' in text:
				value = 'theater'
			if domain == 'taxi' and value not in text:
				continue
			# turn on the input switch to check real stuff
			if value not in text:
				continue

			if domain == 'booking': # only restaurant, hotel involved
				text = proc_booking(text, slot, value, newFind_daDict, da)

			else:
				if change_daDict and value in text:
					addTonewFindDaDict(newFind_daDict, da, slot, value)
				text = text.replace(value, ' [' + domain + '_' + slot + '] ')

			processed_value.add(value)

	if change_daDict and newFind_daDict != {}:
		# TODO: start from here, add new da with sv list to old daDict, and scan examples
		for da in newFind_daDict:
			if da not in old_daDict: # new da with new slot values
				old_daDict[da] = newFind_daDict[da]
			else: # only new slots
				old_daDict[da] += newFind_daDict[da]

	return text


def removeRepeatToken(da_list, threshold=3):
	new_da_list = []
	prev_token = None
	for token in da_list:
		if token != prev_token:
			c = 0
		c += 1
		if c <= 3:
			new_da_list.append(token)
		prev_token = token
	return new_da_list


def prepareDaLabel(da_dict, text):
	''' prepare act seq label per turn keep all labels and sort by act first then by slot within a act '''
	def decide_domain(text):
		if 'restaurant' in text or 'table' in text:
			domain = 'restaurant'
		elif 'hotel' in text or 'accommodate' in text or 'stay' in text or 'night' in text:
			domain = 'hotel'
		else:
#			print('cannot decide booking domain:', text)
#			raise ValueError('Domain problem')
			'''
			usually sentences like 'may I ask how many people?' with request as act are impossible to decide domain
			only this case we keep booking as a domain since it does not hurt generation
			however, for other language pattern such as 'booking is successful. here is your [restaurant_reference]'
			we need to have domain info in da label to signify NLG to generate [restaurant_reference] instead of [hotel_reference]
			'''
			domain = 'booking'
		return domain

	act2slots = {}
	for da, sv_list in da_dict.items():
		domain, act = da.lower().split('-')
		if domain == 'booking' and act == 'inform': # FIX: it's offer booking action, so important so that it cannot be messed up with normal inform
			act = 'offerbook' # same as token 'offerbook' in train domain
		if domain == 'booking' and act == 'book': # FIX: make it the same act between restaurant/hotel and train
			act = 'offerbooked' # same as token 'offerbooked' in train domain
		if domain == 'booking':
			domain = decide_domain(text)
		for s, v in sv_list:
			s = unify_actSlot_to_goalSlot(s, domain)
			if act not in act2slots:
				act2slots[act] = []
			act2slots[act].append(domain + '_' + s)

	label_seq = []
	for act in sorted(act2slots.keys()): # sort act first
		label_seq.append('act_' + act)
		label_seq += sorted(act2slots[act]) # sort slot within an act
	label_seq = removeRepeatToken(label_seq)
	return ' '.join(label_seq)


def createDelexData():
	"""Main function of the script - loads delexical dictionary,
	goes through each dialogue and does:
	1) data normalization
	2) delexicalization
	3) addition of database pointer
	4) saves the delexicalized data
	"""

	# create dictionary of delexicalied values that then we will search against, order matters here!
	dic = delexicalize.prepareSlotValuesIndependent()
	dic = fixSlotValuesIndependent(dic)
	delex_data = {}

	print('Loading raw data...')
	fin1 = open('data/raw_data/data.json')
	data = json.load(fin1)
	data2 = json.load(open('data/raw_data/annotated_user_da_with_span_full_patchName.json'))
	improved_usrSent_by_sysAct = 0
	# for dial_count, dialogue_name in enumerate(sorted(data.keys())):
	for dialogue_name in tqdm(sorted(data.keys())):
		dialogue = data[dialogue_name]

		if dialogue_name.replace('.json', '') not in data2: # only 5 dialogues here
			continue

		for idx, turn in enumerate(dialogue['log']):
			# normalization, split and delexicalization of the sentence
			sent = normalize(turn['text']) # *[value_time], [value_price]
			da_dict = data2[dialogue_name.replace('.json', '')]['log'][idx]['dialog_act']
			# perform two steps delex by first da matching and then db matching
			# using da label to delex original [value_*] token into domain-dependent token
			sent = delexValueToken(sent, da_dict)

			if idx % 2 == 0: # usr, since da label on usr side is bad, use da label on sys side to help delex
				da_dict_act = data2[dialogue_name.replace('.json', '')]['log'][idx+1]['dialog_act']
				sent2 = delexValueToken(sent, da_dict_act, change_daDict=True, old_daDict=da_dict)
				if sent != sent2:
					improved_usrSent_by_sysAct += 1

			sent = delexicalize.delexicalise(sent, dic) # [value_place], [value_day], [value_area], [value_food], [value_pricerange]

			# parsing reference number GIVEN belief state
			sent = delexicaliseReferenceNumber(sent, turn)

			# fix delex
			sent = fixDomainDelex(sent, da_dict, dialogue_name, idx)

			# get da seq label
			da_label = prepareDaLabel(da_dict, sent)
			if idx % 2 == 0: # usr
				dialogue['log'][idx]['usr_act'] = da_label
			else: # sys
				dialogue['log'][idx]['sys_act'] = da_label

			# changes to numbers only here
			digitpat = re.compile('\d+')
			sent = re.sub(digitpat, '[value_count]', sent) # *[value_count]

			# delexicalized sentence added to the dialogue
			dialogue['log'][idx]['text'] = sent

			if idx % 2 == 1:  # if it's a system turn
				# add database pointer
				pointer_vector = addDBPointer(turn)
				# add booking pointer
				dialogue['log'][idx - 1]['db_pointer'] = pointer_vector.tolist()

		delex_data[dialogue_name] = dialogue

	print('Done delexicalisation!')
	with open('data/process_data/delex.json', 'w') as outfile:
		json.dump(delex_data, outfile, indent=4, sort_keys=True)
	return delex_data


def divideData(data):
	"""Given test and validation sets, divide the data for three different sets"""

	def collect_freq(utt, token2count):
		for token in utt.split():
			if token not in token2count:
				token2count[token] = 0
			token2count[token] += 1

	testListFile = []
	fin = open('data/raw_data/testListFile.json')
	for line in fin:
		testListFile.append(line[:-1])
	fin.close()

	valListFile = []
	fin = open('data/raw_data/valListFile.json')
	for line in fin:
		valListFile.append(line[:-1])
	fin.close()

	trainListFile = open('data/raw_data/trainListFile.json', 'w')
	test_dials = {}
	val_dials = {}
	train_dials = {}
		
	# dictionaries
	word2count = {}
	act2count = {}
	
	for dial_count, dialogue_name in enumerate(sorted(data.keys())):
		dial = get_dial(data[dialogue_name])
		if dial:
			dialogue = {}
			dialogue['usr'] = []
			dialogue['sys'] = []
			dialogue['db'] = []
			dialogue['bs'] = []
			dialogue['usr_act'] = []
			dialogue['sys_act'] = []
			for turn in dial:
				dialogue['usr'].append(turn[0])
				dialogue['sys'].append(turn[1])
				dialogue['db'].append(turn[2])
				dialogue['bs'].append(turn[3])
				dialogue['usr_act'].append(turn[4])
				dialogue['sys_act'].append(turn[5])

				collect_freq(turn[0], word2count)
				collect_freq(turn[1], word2count)
				collect_freq(turn[4], act2count)
				collect_freq(turn[5], act2count)

			if dialogue_name in testListFile:
				test_dials[dialogue_name] = dialogue
			elif dialogue_name in valListFile:
				val_dials[dialogue_name] = dialogue
			else:
				trainListFile.write(dialogue_name + '\n')
				train_dials[dialogue_name] = dialogue

	with open('data/process_data/val_dials.json', 'w') as f:
		json.dump(val_dials, f, indent=4, sort_keys=True)

	with open('data/process_data/test_dials.json', 'w') as f:
		json.dump(test_dials, f, indent=4, sort_keys=True)

	with open('data/process_data/train_dials.json', 'w') as f:
		json.dump(train_dials, f, indent=4, sort_keys=True)

	# sort the vocab by word frequency and save the file
	word2count = sorted(word2count.items(), key=operator.itemgetter(1), reverse=True)
	act2count = sorted(act2count.items(), key=operator.itemgetter(1), reverse=True)
	with open('data/process_data/word2count.json', 'w') as f:
		json.dump(word2count, f, indent=4)

	with open('data/process_data/act2count.json', 'w') as f:
		json.dump(act2count, f, indent=4)


def main():
	print('Create delexicalized dialogues. Get yourself a coffee, this might take a while.')
	delex_data = createDelexData()
	print('Divide dialogues for separate bits - usr, sys, db, bs, usr-act, sys-act')
	divideData(delex_data)


if __name__ == "__main__":
	main()
