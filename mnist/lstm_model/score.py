#!/usr/bin/env python3
##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2017-01-01 (This is my new year's resolution)
# Purpose: Score recurrent neural network on test dataset
# License: For legal information see LICENSE in the home directory.
##########################################################################################

##########################################################################################
# Libraries
##########################################################################################

import os

import numpy as np
import tensorflow as tf
import dataplumbing as dp

##########################################################################################
# Settings
##########################################################################################

# Model settings
#
num_features = dp.test.num_features
max_steps = dp.test.max_length
num_cells = 250
num_classes = dp.test.num_classes
activation = tf.nn.tanh
initialization_factor = 3.0

# Test parameters
#
batch_size = dp.test.num_samples

##########################################################################################
# Model
##########################################################################################

# Inputs
#
x = tf.placeholder(tf.float32, [batch_size, max_steps, num_features])	# Features
l = tf.placeholder(tf.int32, [batch_size])	# Sequence length
y = tf.placeholder(tf.float32, [batch_size, num_classes])	# Labels

# Trainable parameters
#
W_ig = tf.Variable(
	tf.random_uniform(
		[num_features+num_cells, num_cells],
		minval=-np.sqrt(6.0/(num_features+2.0*num_cells)),
		maxval=np.sqrt(6.0/(num_features+2.0*num_cells))
	)
)
b_ig = tf.Variable(tf.zeros([num_cells]))

W_fg = tf.Variable(
	tf.random_uniform(
		[num_features+num_cells, num_cells],
		minval=-np.sqrt(6.0/(num_features+2.0*num_cells)),
		maxval=np.sqrt(6.0/(num_features+2.0*num_cells))
	)
)
b_fg = tf.Variable(tf.ones([num_cells]))	# Initial bias of 1

W_og = tf.Variable(
	tf.random_uniform(
		[num_features+num_cells, num_cells],
		minval=-np.sqrt(6.0/(num_features+2.0*num_cells)),
		maxval=np.sqrt(6.0/(num_features+2.0*num_cells))
	)
)
b_og = tf.Variable(tf.zeros([num_cells]))

W_c = tf.Variable(
	tf.random_uniform(
		[num_features+num_cells, num_cells],
		minval=-np.sqrt(6.0/(num_features+2.0*num_cells)),
		maxval=np.sqrt(6.0/(num_features+2.0*num_cells))
	)
)
b_c = tf.Variable(tf.zeros([num_cells]))

W_o = tf.Variable(
	tf.random_uniform(
		[num_cells, num_classes],
		minval=-np.sqrt(6.0/(num_cells+num_classes)),
		maxval=np.sqrt(6.0/(num_cells+num_classes))

	)
)
b_o = tf.Variable(tf.zeros([1]))

# Internal states
#
h = tf.zeros([batch_size, num_cells])
c = tf.zeros([batch_size, num_cells])

# Define model
#
for i in range(max_steps):

	x_step = x[:,i,:]
	xh_join = tf.concat(1, [x_step, h])	# Combine the features and hidden state into one tensor

	ig = tf.sigmoid(tf.matmul(xh_join, W_ig)+b_ig)
	fg = tf.sigmoid(tf.matmul(xh_join, W_fg)+b_fg)
	og = tf.sigmoid(tf.matmul(xh_join, W_og)+b_og)
	c_in = tf.tanh(tf.matmul(xh_join, W_c)+b_c)
	c_out = fg*c+ig*c_in
	h_out = og*tf.tanh(c)

	c = tf.select(tf.greater(l, i), c_out, c)	# Use old states only if the sequence length has not been exceeded
	h = tf.select(tf.greater(l, i), h_out, h)

ly = tf.matmul(h, W_o)+b_o
py = tf.nn.sigmoid(ly)

##########################################################################################
# Analyzer
##########################################################################################

# Cost function
#
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(ly, y))   # Cross-entropy cost function

# Evaluate performance
#
correct = tf.equal(tf.argmax(py, 1), tf.argmax(y, 1))
accuracy = 100.0*tf.reduce_mean(tf.cast(correct, tf.float32))

##########################################################################################
# Score
##########################################################################################

# Operation to initialize session
#
initializer = tf.global_variables_initializer()

# Open session
#
with tf.Session() as session:

	# Initialize variables
	#
	session.run(initializer)

	# Load the trained model
	#
	loader = tf.train.Saver()
	loader.restore(session, 'bin/train.ckpt')

	# Grab the test data
	#
	xs, ls, ys = dp.test.batch(batch_size)
	feed = {x: xs, l: ls, y: ys}

	# Run model
	#
	out = session.run((cost, accuracy), feed_dict=feed)
	print('Dataset:', 'test', 'Cost:', out[0]/np.log(2.0), 'Accuracy:', out[1])

