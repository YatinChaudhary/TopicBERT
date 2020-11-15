import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
import random
import numpy as np
from collections import Counter
import os, sys, csv

seed = 42
tf_op_seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)

def loss(model, data, session):
	loss = []
	for _, x, seq_lengths in data:
		loss.append(
			session.run([model.loss], feed_dict={
				model.x: x,
				model.seq_lengths: seq_lengths
			})[0]
		)
	return sum(loss) / len(loss)


def gradients(opt, loss, vars, step, max_gradient_norm=None, dont_clip=[], name=""):
	print("Optimizer output for ", name)

	gradients = opt.compute_gradients(loss, vars)
	if max_gradient_norm is not None:
		to_clip = [(g, v) for g, v in gradients if v.name not in dont_clip]
		not_clipped = [(g, v) for g, v in gradients if v.name in dont_clip]
		gradients, variables = zip(*to_clip)
		clipped_gradients, _ = clip_ops.clip_by_global_norm(
			gradients,
			max_gradient_norm
		)
		gradients = list(zip(clipped_gradients, variables)) + not_clipped

	# Add histograms for variables, gradients and gradient norms
	for gradient, variable in gradients:
		if isinstance(gradient, ops.IndexedSlices):
			grad_values = gradient.values
		else:
			grad_values = gradient
		if grad_values is None:
			print('warning: missing gradient: {}'.format(variable.name))
		if grad_values is not None:
			tf.summary.histogram(variable.name, variable)
			tf.summary.histogram(variable.name + '/gradients', grad_values)
			tf.summary.histogram(
				variable.name + '/gradient_norm',
				clip_ops.global_norm([grad_values])
			)

	return opt.apply_gradients(gradients, global_step=step)


def masked_sequence_cross_entropy_loss(
	x,
	seq_lengths,
	logits,
	loss_function=None,
	norm_by_seq_lengths=True,
	name=None
):
	'''
	Compute the cross-entropy loss between all elements in x and logits.
	Masks out the loss for all positions greater than the sequence
	length (as we expect that sequences may be padded).

	Optionally, also either use a different loss function (eg: sampled
	softmax), and/or normalise the loss for each sequence by the
	sequence length.
	'''
	batch_size = tf.shape(x)[0]
	batch_len = tf.shape(x)[1]
	labels = tf.reshape(x, [-1])

	#max_doc_length = tf.reduce_max(seq_lengths)
	mask = tf.less(
		#tf.range(0, max_doc_length, 1),
		tf.range(0, batch_len, 1),
		tf.reshape(seq_lengths, [batch_size, 1])
	)
	mask = tf.reshape(mask, [-1])
	mask = tf.to_float(tf.where(
		mask,
		tf.ones_like(labels, dtype=tf.float32),
		tf.zeros_like(labels, dtype=tf.float32),
		name = name + '_mask'
	))

	if loss_function is None:
		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
			logits=logits,
			labels=labels
		)
	else:
		loss = loss_function(logits, labels)
	
	loss *= mask
	loss = tf.reshape(loss, [batch_size, -1])
	loss = tf.reduce_sum(loss, axis=1, name = name + '_unnormed')
	loss_unnormed = loss
	
	if norm_by_seq_lengths:
		loss_normed = tf.divide(loss, tf.to_float(seq_lengths), name = name + '_normed')
	
	#return tf.reduce_mean(loss, name=name + '_normed'), labels, mask, tf.reduce_mean(loss_unnormed, name=name + '_unnormed')
	return loss_normed, labels, mask, loss_unnormed


def linear(input, output_dim, scope=None, suffix="", stddev=None, V_initializer=None, b_initializer=None):
	const = tf.constant_initializer(0.0)

	with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
		if V_initializer is None:
			V_initializer = tf.glorot_uniform_initializer(seed=tf_op_seed)
			w = tf.get_variable(
				'output_softmax_V',
				[input.get_shape()[1].value, output_dim],
				dtype=tf.float32,
				#initializer=norm,
				initializer=V_initializer,
				trainable=True
			)
		else:
			w = tf.get_variable(
				'output_softmax_V',
				dtype=tf.float32,
				initializer=V_initializer,
				trainable=True
			)

		if b_initializer is None:
			b = tf.get_variable('output_softmax_b', [output_dim], dtype=tf.float32, initializer=const, trainable=True)
		else:
			b = tf.get_variable('output_softmax_b', dtype=tf.float32, initializer=b_initializer, trainable=True)

	logits = tf.nn.xw_plus_b(input, w, b, name='logits' + suffix)

	return logits

def create_multilabel(label_string, num_class):
	label_list = label_string.split(":")
	label_ids = [0]*num_class
	for label in label_list:
		label_ids[int(label)] = 1

	return label_ids


def format_doc(doc):
	new_doc_tokens = []
	counts = Counter(doc.split())
	for index, count in counts.items():
		new_doc_tokens.append(str(index) + ":" + str(count))
	new_doc = " ".join(new_doc_tokens)
	return new_doc

def data_set(data_url, params):
	data = []
	word_count = []
	labels = []
	doc_ids = []
	fin = open(data_url)
	csv_reader = csv.reader(fin, delimiter=",")
	#while True:
	#	line = fin.readline()
	for index, line in enumerate(csv_reader):
		if not line:
			break

		doc_ids.append(int(line[2]))
		if params.multilabel:
			labels.append(create_multilabel(line[0], num_class = params.num_labels))

		else:	
			labels.append(int(line[0]))
		line = format_doc(line[1].strip())
		id_freqs = line.split()
		doc = {}
		count = 0
		#for id_freq in id_freqs[1:]:
		for id_freq in id_freqs:
			items = id_freq.split(':')
			# python starts from 0
			#doc[int(items[0])-1] = int(items[1])
			doc[int(items[0])] = int(items[1])
			count += int(items[1])
		if count > 0:
			data.append(doc)
			word_count.append(count)


	fin.close()
	#return data, word_count, labels
	return data, word_count, labels, doc_ids


def create_batches(data_size, batch_size, shuffle=True):
	batches = []
	ids = list(range(data_size))
	if shuffle:
		random.shuffle(ids)
	for i in range(data_size // batch_size):
		start = i * batch_size
		end = (i + 1) * batch_size
		batches.append(ids[start:end])
	# the batch of which the length is less than batch_size
	rest = data_size % batch_size
	if rest > 0:
		batches.append(ids[-rest:] + [-1] * (batch_size - rest))  # -1 as padding
	return batches


def fetch_data(data, count, idx_batch, vocab_size, params, labels = None):
	batch_size = len(idx_batch)
	data_batch = np.zeros((batch_size, vocab_size))
	if labels:
		if params.multilabel:
			label_batch = [[0]*params.num_labels]*batch_size
		else:
			label_batch = -np.ones(batch_size, dtype = np.int32)
	count_batch = []
	mask = np.zeros(batch_size)
	indices = []
	values = []
	for i, doc_id in enumerate(idx_batch):
		if doc_id != -1:
			if labels:
				label_batch[i] = labels[doc_id]
			for word_id, freq in data[doc_id].items():
				data_batch[i, word_id] = freq
			count_batch.append(count[doc_id])
			mask[i]=1.0
		else:
			count_batch.append(0)

	if labels:
		return data_batch, count_batch, mask, label_batch

	return data_batch, count_batch, mask


def variable_parser(var_list, prefix):
	ret_list = []
	for var in var_list:
		varname = var.name
		varprefix = varname.split('/')[0]
		if varprefix == prefix:
			ret_list.append(var)
	return ret_list


def nvdm_linear(inputs,
		   output_size,
		   no_bias=False,
		   bias_start_zero=False,
		   matrix_start_zero=False,
		   scope=None,
		   get_matrix=False,
		   matrix_initializer=None,
		   bias_initializer=None):
	with tf.variable_scope(scope or 'Linear', reuse=tf.AUTO_REUSE):
		input_size = inputs.get_shape()[1].value

		if matrix_start_zero:
			matrix_initializer = tf.constant_initializer(0)
			matrix = tf.get_variable('Matrix', [input_size, output_size],
									initializer=matrix_initializer)
		else:
			if matrix_initializer is None:
				matrix_initializer = tf.glorot_uniform_initializer(seed=tf_op_seed)
				matrix = tf.get_variable('Matrix', [input_size, output_size],
										initializer=matrix_initializer)
			else:
				matrix = tf.get_variable('Matrix',
										initializer=matrix_initializer)
		
		if bias_start_zero:
			bias_initializer = tf.constant_initializer(0)
			bias_term = tf.get_variable('Bias', [output_size], 
										initializer=bias_initializer)
		else:
			if bias_initializer is None:
				bias_initializer = tf.glorot_uniform_initializer(seed=tf_op_seed)
				bias_term = tf.get_variable('Bias', [output_size], 
											initializer=bias_initializer)
			else:
				bias_term = tf.get_variable('Bias', 
											initializer=bias_initializer)
		
		output = tf.matmul(inputs, matrix)
		if not no_bias:
			output = output + bias_term

		if get_matrix:
			return output, matrix
		else:
			return output


def mlp(inputs, 
		mlp_hidden=[], 
		mlp_nonlinearity=tf.nn.tanh,
		scope=None,
		initializer=[[None, None]]):
	with tf.variable_scope(scope or 'Linear', reuse=tf.AUTO_REUSE):
		mlp_layer = len(mlp_hidden)
		#if not initializer[0][0] is None:
		#	assert(mlp_layer == len(initializer))
		res = inputs
		for l in range(mlp_layer):
			"""
			res = mlp_nonlinearity(nvdm_linear(res, 
											mlp_hidden[l], 
											scope='l'+str(l), 
											matrix_initializer=initializer[l][0],
											bias_initializer=initializer[l][1]))
			"""
			res = mlp_nonlinearity(nvdm_linear(res, 
											mlp_hidden[l], 
											scope='l'+str(l)))
			
		return res


def create_data(dataset_path, train_filename, val_filename, test_filename):
	train_docs = []
	train_labels = []
	with open(dataset_path + "/" + train_filename, "r") as f:
		csv_reader = csv.reader(f, delimiter=",")
		for line in csv_reader:
			train_docs.append(line[0].strip() + " " + format_doc(line[1].strip()))
			train_labels.append(line[0].strip())

	val_docs = []
	val_labels = []
	with open(dataset_path + "/" + val_filename, "r") as f:
		csv_reader = csv.reader(f, delimiter=",")
		for line in csv_reader:
			val_docs.append(line[0].strip() + " " + format_doc(line[1].strip()))
			val_labels.append(line[0].strip())

	test_docs = []
	test_labels = []
	with open(dataset_path + "/" + test_filename, "r") as f:
		csv_reader = csv.reader(f, delimiter=",")
		for line in csv_reader:
			test_docs.append(line[0].strip() + " " + format_doc(line[1].strip()))
			test_labels.append(line[0].strip())

	with open(dataset_path + "/" + "train.feat", "w") as f:
		f.write("\n".join(train_docs))

	with open(dataset_path + "/" + "dev.feat", "w") as f:
		f.write("\n".join(val_docs))

	with open(dataset_path + "/" + "test.feat", "w") as f:
		f.write("\n".join(test_docs))

def read_data(dataset_path, train_filename, val_filename, test_filename):
	train_docs = []
	with open(dataset_path + "/" + train_filename, "r") as f:
		csv_reader = csv.reader(f, delimiter=",")
		for line in csv_reader:
			train_docs.append(line[0].strip() + " " + format_doc(line[1].strip()))

	val_docs = []
	with open(dataset_path + "/" + val_filename, "r") as f:
		csv_reader = csv.reader(f, delimiter=",")
		for line in csv_reader:
			val_docs.append(line[0].strip() + " " + format_doc(line[1].strip()))

	test_docs = []
	with open(dataset_path + "/" + test_filename, "r") as f:
		csv_reader = csv.reader(f, delimiter=",")
		for line in csv_reader:
			test_docs.append(line[0].strip() + " " + format_doc(line[1].strip()))

	return train_docs, val_docs, test_docs