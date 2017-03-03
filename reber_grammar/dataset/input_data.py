#!/usr/bin/env python3
##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2017-01-01
# Purpose: Load dataset or generate it if it does not exist yet.
# License: For legal information see LICENSE in the home directory.
##########################################################################################

##########################################################################################
# Libraries
##########################################################################################

import os
import numpy as np

##########################################################################################
# Settings
##########################################################################################

# Reber grammar
#
states = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
transitions = {
	1: [2, 7],
	2: [3, 4],
	3: [3, 4],
	4: [5, 6],
	5: [12],
	6: [8, 9],
	7: [8, 9],
	8: [8, 9],
	9: [10, 11],
	10: [5, 6],
	11: [12],
}
aliases = {
	1: 'B', 2: 'T', 3: 'S', 4: 'X', 5: 'S', 6: 'X',
	7: 'P', 8: 'T', 9: 'V', 10: 'P', 11: 'V', 12: 'E',
}
encoding = {'B': 0, 'E': 1, 'P': 2, 'S': 3, 'T': 4, 'V': 5, 'X': 6}

# Data dimensions
#
num_train = 10000
num_test = 10000
max_length = 50
num_features = 7

##########################################################################################
# Utilities
##########################################################################################

def make_chain():
	chain = [1]
	while chain[-1] != states[-1]:
		choices = transitions[chain[-1]]
		j = np.random.randint(len(choices))
		chain.append(choices[j])	
	return chain

def valid_chain(chain):
	if len(chain) == 0:
		return False
	if chain[0] != states[0]:
		return False
	for i in range(1, len(chain)):
		if chain[i] not in transitions[chain[i-1]]:
			return False
	return True

def convert_chain(chain):
	sequence = ''
	for value in chain:
		sequence += aliases[value]
	return sequence

##########################################################################################
# Generate/Load dataset
##########################################################################################

# Make directory
#
_path = '/'.join(__file__.split('/')[:-1])
os.makedirs(_path+'/bin', exist_ok=True)

# Training data
#
if not os.path.isfile(_path+'/bin/xs_train.npy') or \
	not os.path.isfile(_path+'/bin/ls_train.npy') or \
	not os.path.isfile(_path+'/bin/ys_train.npy'):
	xs_train = np.zeros((num_train, max_length, num_features))
	ls_train = np.zeros(num_train)
	ys_train = np.zeros(num_train)
	for i in range(num_train):
		chain = make_chain()
		valid = 1.0
		if np.random.rand() >= 0.5:	# Randomly insert a single typo with proability 0.5
			hybrid = chain
			while valid_chain(hybrid):
				chain_ = make_chain()
				j = np.random.randint(len(chain))
				j_ = np.random.randint(len(chain_))
				hybrid = chain[:j]+chain_[j_:]
			chain = hybrid
			valid = 0.0
		sequence = convert_chain(chain)
		for j, symbol in enumerate(sequence):
			k = encoding[sequence[j]]
			xs_train[i,j,k] = 1.0
		ls_train[i] = len(sequence)
		ys_train[i] = valid
	np.save(_path+'/bin/xs_train.npy', xs_train)
	np.save(_path+'/bin/ls_train.npy', ls_train)
	np.save(_path+'/bin/ys_train.npy', ys_train)
else:
	xs_train = np.load(_path+'/bin/xs_train.npy')
	ls_train = np.load(_path+'/bin/ls_train.npy')
	ys_train = np.load(_path+'/bin/ys_train.npy')

# Test data
#
if not os.path.isfile(_path+'/bin/xs_test.npy') or \
	not os.path.isfile(_path+'/bin/ls_test.npy') or \
	not os.path.isfile(_path+'/bin/ys_test.npy'):
	xs_test = np.zeros((num_test, max_length, num_features))
	ls_test = np.zeros(num_test)
	ys_test = np.zeros(num_test)
	for i in range(num_test):
		chain = make_chain()
		valid = 1.0
		if np.random.rand() >= 0.5:	# Randomly insert a single typo with proability 0.5
			hybrid = chain
			while valid_chain(hybrid):
				chain_ = make_chain()
				j = np.random.randint(len(chain))
				j_ = np.random.randint(len(chain))
				hybrid = chain[:j]+chain_[j_:]
			chain = hybrid
			valid = 0.0
		sequence = convert_chain(chain)
		for j, symbol in enumerate(sequence):
			k = encoding[sequence[j]]
			xs_test[i,j,k] = 1.0
		ls_test[i] = len(sequence)
		ys_test[i] = valid
	np.save(_path+'/bin/xs_test.npy', xs_test)
	np.save(_path+'/bin/ls_test.npy', ls_test)
	np.save(_path+'/bin/ys_test.npy', ys_test)
else:
	xs_test = np.load(_path+'/bin/xs_test.npy')
	ls_test = np.load(_path+'/bin/ls_test.npy')
	ys_test = np.load(_path+'/bin/ys_test.npy')



