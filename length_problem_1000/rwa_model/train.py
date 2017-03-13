#!/usr/bin/env python3
##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2017-01-01 (This is my new year's resolution)
# Purpose: Train recurrent neural network
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
num_features = dp.train.num_features
max_steps = dp.train.max_length
num_cells = 250
num_classes = dp.train.num_classes
activation = tf.nn.tanh
initialization_factor = 1.0

# Training parameters
#
num_iterations = 20000
batch_size = 100
learning_rate = 0.001

##########################################################################################
# Model
##########################################################################################

# Inputs
#
x = tf.placeholder(tf.float32, [batch_size, max_steps, num_features])	# Features
l = tf.placeholder(tf.int32, [batch_size])	# Sequence length
y = tf.placeholder(tf.float32, [batch_size])	# Labels

# Trainable parameters
#
s = tf.Variable(tf.random_normal([num_cells], stddev=np.sqrt(initialization_factor)))	# Determines initial state

W_g = tf.Variable(
	tf.random_uniform(
		[num_features+num_cells, num_cells],
		minval=-np.sqrt(6.0*initialization_factor/(num_features+2.0*num_cells)),
		maxval=np.sqrt(6.0*initialization_factor/(num_features+2.0*num_cells))
	)
)
b_g = tf.Variable(tf.zeros([num_cells]))
W_u = tf.Variable(
	tf.random_uniform(
		[num_features, num_cells],
		minval=-np.sqrt(6.0*initialization_factor/(num_features+num_cells)),
		maxval=np.sqrt(6.0*initialization_factor/(num_features+num_cells))
	)
)
b_u = tf.Variable(tf.zeros([num_cells]))
W_a = tf.Variable(
	tf.random_uniform(
		[num_features+num_cells, num_cells],
		minval=-np.sqrt(6.0*initialization_factor/(num_features+2.0*num_cells)),
		maxval=np.sqrt(6.0*initialization_factor/(num_features+2.0*num_cells))
	)
)

W_o = tf.Variable(
	tf.random_uniform(
		[num_cells, num_classes],
		minval=-np.sqrt(6.0*initialization_factor/(num_cells+num_classes)),
		maxval=np.sqrt(6.0*initialization_factor/(num_cells+num_classes))
	)
)
b_o = tf.Variable(tf.zeros([num_classes]))

# Internal states
#
n = tf.zeros([batch_size, num_cells])
d = tf.zeros([batch_size, num_cells])
h = tf.zeros([batch_size, num_cells])
a_max = tf.fill([batch_size, num_cells], -1E38)	# Start off with lowest number possible

# Define model
#
h += activation(tf.expand_dims(s, 0))

for i in range(max_steps):

	x_step = x[:,i,:]
	xh_join = tf.concat(1, [x_step, h])	# Combine the features and hidden state into one tensor

	u = tf.matmul(x_step, W_u)+b_u
	g = tf.matmul(xh_join, W_g)+b_g
	a = tf.matmul(xh_join, W_a)

	z = tf.mul(u, tf.nn.tanh(g))

	a_newmax = tf.maximum(a_max, a)
	exp_diff = tf.exp(a_max-a_newmax)
	exp_scaled = tf.exp(a-a_newmax)

	n = tf.mul(n, exp_diff)+tf.mul(z, exp_scaled)	# Numerically stable update of numerator
	d = tf.mul(d, exp_diff)+exp_scaled	# Numerically stable update of denominator
	h_new = activation(tf.div(n, d))
	a_max = a_newmax

	h = tf.select(tf.greater(l, i), h_new, h)	# Use new hidden state only if the sequence length has not been exceeded

ly = tf.matmul(h, W_o)+b_o
ly_flat = tf.reshape(ly, [batch_size])
py = tf.nn.sigmoid(ly_flat)

##########################################################################################
# Optimizer/Analyzer
##########################################################################################

# Cost function and optimizer
#
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ly_flat, y))	# Cross-entropy cost function
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Evaluate performance
#
correct = tf.equal(tf.round(py), tf.round(y))
accuracy = 100.0*tf.reduce_mean(tf.cast(correct, tf.float32))

##########################################################################################
# Train
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

	# Each training session represents one batch
	#
	for iteration in range(num_iterations):

		# Grab a batch of training data
		#
		xs, ls, ys = dp.train.batch(batch_size)
		feed = {x: xs, l: ls, y: ys}

		# Update parameters
		#
		out = session.run((cost, accuracy, optimizer), feed_dict=feed)
		print('Iteration:', iteration, 'Dataset:', 'train', 'Cost:', out[0]/np.log(2.0), 'Accuracy:', out[1])

		# Periodically run model on test data
		#
		if iteration%100 == 0:

			# Grab a batch of test data
			#
			xs, ls, ys = dp.test.batch(batch_size)
			feed = {x: xs, l: ls, y: ys}

			# Run model
			#
			out = session.run((cost, accuracy), feed_dict=feed)
			print('Iteration:', iteration, 'Dataset:', 'test', 'Cost:', out[0]/np.log(2.0), 'Accuracy:', out[1])

	# Save the trained model
	#
	os.makedirs('bin', exist_ok=True)
	saver = tf.train.Saver()
	saver.save(session, 'bin/train.ckpt')

