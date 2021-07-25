import json
import sys

def evaluate_domainBS_scanBS(domain_bs, domain_goal, domain):
	'''Check if detected info in the given bs is correct according to the goal (check only informable slot)'''
	for slot, value in domain_bs['semi'].items():
		if value in ['dont care', "don't care", 'dontcare', "do n't care"]:
			continue
		if slot not in domain_goal['info']: # might have 'not_mentioned' slot in bs but not in goal
			continue
		if domain_goal['info'][slot] in ['dont care', "don't care", 'dontcare', "do n't care"]:
			continue
		if slot == 'type' and domain == 'hotel':
			continue
		if value != domain_goal['info'][slot]:
			return 0
	return 1


def evaluate_domainBS_scanGoal(domain_bs, domain_goal, domain):
	for slot, value in domain_goal['info'].items():
		if value in ['dont care', "don't care", 'dontcare', "do n't care"]:
			continue
		if slot == 'type' and domain == 'hotel':
			continue
		if domain_bs['semi'][slot] != value:
			return 0
	return 1