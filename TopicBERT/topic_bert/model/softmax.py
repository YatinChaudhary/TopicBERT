#!/usr/bin/python -u
#encoding=utf8
#Author: yangsaiyong@gmail.com
#Update: 2018.10.17

import tensorflow as tf
import numpy as np

seed = 42
tf_op_seed = 1234
np.random.seed(seed)
tf.set_random_seed(seed)

class FullSoftmax(object):
	def __init__(self, input_dim, output_dim, scope=None, suffix="", 
				stddev=None, V_initializer=None, b_initializer=None,
				concat_V=None, concat_dim=0):
		with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
			self.suffix = suffix

			V_shape = None
			if V_initializer is None:
				V_initializer = tf.glorot_uniform_initializer(seed=tf_op_seed)
				V_shape = [input_dim, output_dim]
			b_shape = None
			if b_initializer is None:
				b_initializer = tf.constant_initializer(0.0, dtype=tf.float32)
				b_shape = [output_dim]
			
			if not concat_V is None:
				V_shape = [input_dim-concat_dim, output_dim]
				softmax_w_additional = tf.get_variable("output_softmax_V_additional", V_shape, \
					initializer=V_initializer)
				self.softmax_w = tf.concat([concat_V, softmax_w_additional], axis=0, name="output_softmax_V")
			else:
				self.softmax_w = tf.get_variable("output_softmax_V", V_shape, \
					initializer=V_initializer)
			self.softmax_b = tf.get_variable("output_softmax_b", b_shape, \
				initializer=b_initializer)

	def loss(self, labels, seq_lengths, inputs, loss_function=None, norm_by_seq_lengths=True, name='loss'):
		'''
		Compute the cross-entropy loss between all elements in x and logits.
		Masks out the loss for all positions greater than the sequence
		length (as we expect that sequences may be padded).

		Optionally, also either use a different loss function (eg: sampled
		softmax), and/or normalise the loss for each sequence by the
		sequence length.
		'''
		batch_size = tf.shape(labels)[0]
		batch_len = tf.shape(labels)[1]
		labels = tf.reshape(labels, [-1])

		mask = tf.less(
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

		logits = tf.nn.xw_plus_b(inputs, self.softmax_w, self.softmax_b)

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
		
		#return loss_normed, labels, mask, loss_unnormed, [tf.reduce_sum(loss_unnormed) / batch_size]
		#return loss_normed, labels, mask, loss_unnormed, [tf.reduce_mean(loss_unnormed)]
		return loss_normed, labels, mask, loss_unnormed, tf.reduce_mean(loss_unnormed)

	def softmax(self, inputs, name='softmax'):
		#logits = tf.matmul(inputs, self.softmax_w) + self.softmax_b
		logits = tf.nn.xw_plus_b(inputs, self.softmax_w, self.softmax_b)
		return tf.nn.softmax(logits, name=name + self.suffix)

	def log_softmax(self, inputs, name='log_softmax'):
		#logits = tf.matmul(inputs, self.softmax_w) + self.softmax_b
		logits = tf.nn.xw_plus_b(inputs, self.softmax_w, self.softmax_b)
		return tf.nn.log_softmax(logits, name=name + self.suffix)

	def logits(self, inputs, name='logits'):
		#logits = tf.matmul(inputs, self.softmax_w) + self.softmax_b
		logits = tf.nn.xw_plus_b(inputs, self.softmax_w, self.softmax_b, name=name + self.suffix)
		return logits

class AdaptiveSoftmax(object):
	def __init__(self, input_dim, cutoff, scope=None, suffix="", project_factor=4, project_dims=None, initializer=None):
		self.cluster_num = len(cutoff) - 1
		if project_dims:
			assert(len(project_dims) == self.cluster_num)
		else:
			project_dims = []
			tail_project_factor = project_factor
			for i in range(self.cluster_num):
				#dim = max(1, input_dim / tail_project_factor)
				dim = max(1, input_dim // tail_project_factor)
				project_dims.append(dim)
				tail_project_factor *= project_factor

		self.cutoff = cutoff
		initializer = tf.glorot_uniform_initializer(seed=tf_op_seed)
		with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
			head_dim = cutoff[0] + self.cluster_num
			self.head_w = tf.get_variable("adaptive_softmax_head_w", [input_dim, head_dim], initializer=initializer)
			
			self.tail_w = []
			for i in range(self.cluster_num):
				project_dim = project_dims[i]
				tail_dim = cutoff[i + 1] - cutoff[i]
				self.tail_w.append([
					tf.get_variable("adaptive_softmax_tail{}_proj_w".format(i+1), [input_dim, project_dim], initializer=initializer),
					tf.get_variable("adaptive_softmax_tail{}_w".format(i+1), [project_dim, tail_dim], initializer=initializer)
				])

	def loss(self, labels, seq_lengths, inputs, loss_function=None, norm_by_seq_lengths=True, name='loss'):
		# Get tail masks and update head labels
		batch_size = tf.shape(labels)[0]
		batch_len = tf.shape(labels)[1]
		labels = tf.reshape(labels, [-1])

		loss_mask = tf.less(
			tf.range(0, batch_len, 1),
			tf.reshape(seq_lengths, [batch_size, 1])
		)
		loss_mask = tf.reshape(loss_mask, [-1])
		loss_mask = tf.to_float(tf.where(
			loss_mask,
			tf.ones_like(labels, dtype=tf.float32),
			tf.zeros_like(labels, dtype=tf.float32),
			name = name + '_mask'
		))

		training_losses = []
		head_labels = labels
		ones = tf.ones([tf.size(labels)], dtype=tf.int32)
		for i in range(self.cluster_num):
			# Non-differentiable
			mask = tf.logical_and(tf.greater_equal(labels, self.cutoff[i]), tf.less(labels, self.cutoff[i + 1]))
			
			# Update head labels
			head_labels = tf.where(mask, ones * (self.cutoff[0] + i), head_labels)

			# Compute tail loss
			tail_inputs = tf.boolean_mask(inputs, mask)
			tail_logits = tf.matmul(tf.matmul(tail_inputs, self.tail_w[i][0]), self.tail_w[i][1])
			tail_labels = tf.boolean_mask(labels - self.cutoff[i], mask)
			tail_loss_mask = tf.boolean_mask(loss_mask, mask)
			tail_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tail_logits, labels=tail_labels)
			#training_losses.append(tail_loss)
			training_losses.append(tail_loss * tail_loss_mask)
			aligned_tail_loss = tf.sparse.SparseTensor(tf.squeeze(tf.where(mask)), tail_loss, [tf.size(labels, out_type=tf.int64)])
			loss = tf.sparse.to_dense(aligned_tail_loss) if i == 0 else \
				loss + tf.sparse.to_dense(aligned_tail_loss)
			loss *= loss_mask

		# Compute head loss
		head_logits = tf.matmul(inputs, self.head_w) # (sample_num, head_size)
		head_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=head_logits, labels=head_labels) #(sample_num)
		head_loss *= loss_mask
		training_losses.append(head_loss)
		loss = tf.add(loss, head_loss, name = name + '_unnormed_flat')
		loss = tf.reshape(loss, [batch_size, -1])
		loss = tf.reduce_sum(loss, axis=1, name = name + '_unnormed')
		loss_unnormed = loss

		if norm_by_seq_lengths:
			loss_normed = tf.divide(loss, tf.to_float(seq_lengths), name = name + '_normed')

		#losses = [tf.reduce_sum(loss) / batch_size for loss in training_losses]
		losses = [tf.reduce_mean(loss) for loss in training_losses]

		#return loss_normed, labels, loss_mask, loss_unnormed, losses
		return loss_normed, labels, loss_mask, loss_unnormed, tf.reduce_sum(losses)
		#return loss_normed, labels, loss_mask, loss_unnormed

	def softmax(self, inputs, name='softmax'):
		head_logits = tf.matmul(inputs, self.head_w)
		head_softmax = tf.nn.softmax(head_logits)
		softmax_list = [head_softmax[:, :self.cutoff[0]]]
		for i in range(self.cluster_num):
			tail_logits = tf.matmul(tf.matmul(inputs, self.tail_w[i][0]), self.tail_w[i][1])
			tail_softmax = tf.nn.softmax(tail_logits)
			index = self.cutoff[0] + i
			softmax_list.append(tail_softmax * head_softmax[:, index:index+1])
		return tf.concat(softmax_list, axis=1, name=name)

	def log_softmax(self, inputs, name='log_softmax'):
		head_logits = tf.matmul(inputs, self.head_w)
		head_logsoftmax = tf.nn.log_softmax(head_logits)
		logsoftmax_list = [head_logsoftmax[:, :self.cutoff[0]]]
		for i in range(self.cluster_num):
			tail_logits = tf.matmul(tf.matmul(inputs, self.tail_w[i][0]), self.tail_w[i][1])
			tail_logsoftmax = tf.nn.log_softmax(tail_logits)
			index = self.cutoff[0] + i
			logsoftmax_list.append(tail_logsoftmax + head_logsoftmax[:, index:index+1])
		return tf.concat(logsoftmax_list, axis=1, name=name)

	def logits(self, inputs, name='logits'):
		head_logits = tf.matmul(inputs, self.head_w)
		logits_list = [head_logits[:, :self.cutoff[0]]]
		for i in range(self.cluster_num):
			tail_logits = tf.matmul(tf.matmul(inputs, self.tail_w[i][0]), self.tail_w[i][1])
			index = self.cutoff[0] + i
			logits_list.append(tail_logits * head_logits[:, index:index+1])
		return tf.concat(logits_list, axis=1, name=name)