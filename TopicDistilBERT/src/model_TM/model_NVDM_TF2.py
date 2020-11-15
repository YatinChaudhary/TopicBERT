from __future__ import print_function

import os, sys, csv
import numpy as np
import tensorflow.compat.v1 as tf1
import math, random
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
import sklearn.metrics.pairwise as pw
from gensim.models import CoherenceModel
from gensim.corpora.dictionary import Dictionary

#tf1.disable_v2_behavior()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

seed = 42
tf_op_seed = 1234
random.seed(seed)
np.random.seed(seed)
#tf1.set_random_seed(seed)


def create_initializer(self, initializer_range=0.02):
	return tf1.truncated_normal_initializer(stddev=initializer_range, seed=tf_op_seed)

def format_doc(doc):
	new_doc_tokens = []
	counts = Counter(doc.split())
	for index, count in counts.items():
		new_doc_tokens.append(str(index) + ":" + str(count))
	new_doc = " ".join(new_doc_tokens)
	return new_doc

def data_set(data_url):
	data = []
	word_count = []
	fin = open(data_url)
	csv_reader = csv.reader(fin, delimiter=",")
	#while True:
	#	line = fin.readline()
	for index, line in enumerate(csv_reader):
		if not line:
			break
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
	return data, word_count

def get_initializers(scope_name, vars_dict):
	matrix_var_name = scope_name + "/Matrix:0"
	bias_var_name = scope_name + "/Bias:0"
	if matrix_var_name in vars_dict:
		matrix_initializer = vars_dict[matrix_var_name]
		print("Matrix initialized for {}".format(scope_name))
	else:
		matrix_initializer = None
	if bias_var_name in vars_dict:
		bias_initializer = vars_dict[bias_var_name]
		print("Bias initialized for {}".format(scope_name))
	else:
		bias_initializer = None
	return matrix_initializer, bias_initializer

class NVDM(object):
	"""
	Neural Variational Document Model -- BOW VAE.
	"""
	def __init__(self, params, non_linearity=tf1.nn.sigmoid):

		self.vocab_size = params.TM_vocab_length
		self.n_hidden = params.hidden_size_TM
		self.n_topic = params.n_topic
		self.n_sample = params.n_sample
		self.learning_rate = params.learning_rate
		self.non_linearity = non_linearity

		#self.x = tf1.placeholder(tf1.float32, [None, self.vocab_size], name='x')
		#self.mask = tf1.placeholder(tf1.float32, [None], name='mask')  # mask paddings
		
		input_size = self.vocab_size

		## pretrained_weights
		if params.TM_pretrained_model_path:
			with tf1.Session() as sess:
				saver_ir = tf1.train.import_meta_graph(os.path.join(params.TM_pretrained_model_path, "model_ppl_nvdm_pretrain", "model_ppl_nvdm_pretrain-1.meta"))
				saver_ir.restore(sess, os.path.join(params.TM_pretrained_model_path, "model_ppl_nvdm_pretrain", "model_ppl_nvdm_pretrain-1"))
				enc_var_names = [var.name for var in tf1.get_collection(tf1.GraphKeys.TRAINABLE_VARIABLES, scope='TM_encoder')]
				enc_var_values = {var_name: sess.run(var_name) for var_name in enc_var_names}
				dec_var_names = [var.name for var in tf1.get_collection(tf1.GraphKeys.TRAINABLE_VARIABLES, scope='TM_decoder')]
				dec_var_values = {var_name: sess.run(var_name) for var_name in dec_var_names}
		#with open("input_matrix.npy", "wb") as f:
		#	np.save(f, enc_var_values["TM_encoder/Linear/l0/Matrix:0"])
		#with open("output_matrix.npy", "wb") as f:
		#	np.save(f, dec_var_values["TM_decoder/projection/Matrix:0"])
		
		## encoder parameters
		self.encoder_params = []
		with tf1.variable_scope('TM_encoder', reuse=tf1.AUTO_REUSE):
			# mlp parameters
			num_mlp_layers = [self.n_hidden]
			with tf1.variable_scope('Linear', reuse=tf1.AUTO_REUSE):
				self.mlp_params = []
				for l, hidden_size in enumerate(num_mlp_layers):
					matrix_initializer, bias_initializer = get_initializers("TM_encoder/Linear/" + "l" + str(l), enc_var_values)
					self.mlp_params.append(self.nvdm_linear_params(input_size, 
																hidden_size,
																scope='l'+str(l),
																matrix_initializer=None,
																bias_initializer=None))
					input_size = hidden_size
					self.encoder_params.extend(self.mlp_params[-1])
			
			# mean parameters
			matrix_initializer, bias_initializer = get_initializers("TM_encoder/mean", enc_var_values)
			self.mean_params = self.nvdm_linear_params(input_size, 
													self.n_topic,
													scope="mean",
													matrix_initializer=matrix_initializer,
													bias_initializer=bias_initializer)
			self.encoder_params.extend(self.mean_params)

			# sigma parameters
			matrix_initializer, bias_initializer = get_initializers("TM_encoder/logsigm", enc_var_values)
			self.logsigm_params = self.nvdm_linear_params(input_size, 
													self.n_topic,
													scope="logsigm",
													bias_start_zero=True,
													matrix_start_zero=True,
													matrix_initializer=matrix_initializer,
													bias_initializer=bias_initializer)
			self.encoder_params.extend(self.logsigm_params)

		## decoder params
		with tf1.variable_scope('TM_decoder', reuse=tf1.AUTO_REUSE):
			matrix_initializer, bias_initializer = get_initializers("TM_decoder/projection", dec_var_values)
			self.decoder_params = self.nvdm_linear_params(self.n_topic, 
													self.vocab_size,
													scope='projection', 
													matrix_initializer=matrix_initializer,
													bias_initializer=bias_initializer)
			self.decoder_params = list(self.decoder_params)
		
		## optimizer
		self.optimizer = tf1.train.AdamOptimizer(learning_rate=self.learning_rate)

	@tf1.function
	def forward(self, input, mask):
		## encoder
		# mlp computation
		enc_vec = input
		for layer_params in self.mlp_params:
			enc_vec = self.non_linearity(tf1.matmul(enc_vec, layer_params[0]) + layer_params[1])
		
		# mean computation
		mean = tf1.matmul(enc_vec, self.mean_params[0]) + self.mean_params[1]

		# sigma computation
		logsigm = tf1.matmul(enc_vec, self.logsigm_params[0]) + self.logsigm_params[1]

		# KLD loss
		kld = -0.5 * tf1.reduce_sum(1 - tf1.square(mean) + 2 * logsigm - tf1.exp(2 * logsigm), 1)
		kld = tf1.multiply(mask, kld, name='kld')  # mask paddings

		## decoder
		input_batch_size = tf1.shape(input)[0]
		if self.n_sample == 1:
			eps = tf1.random_normal((input_batch_size, self.n_topic), mean=0.0, stddev=1.0, seed=seed)
			doc_vec = tf1.add(tf1.multiply(tf1.exp(logsigm), eps), mean, name='doc_hidden')
			logits = tf1.matmul(doc_vec, self.decoder_params[0]) + self.decoder_params[1]
			logits = tf1.nn.log_softmax(logits)
			recons_loss = - tf1.reduce_sum(tf1.multiply(logits, input), 1)
		else:
			eps = tf1.random_normal((self.n_sample*input_batch_size, self.n_topic), mean=0.0, stddev=1.0, seed=seed)
			eps_list = tf1.split(eps, self.n_sample, 0)
			recons_loss_list = []
			doc_vec_list = []
			for i in range(self.n_sample):
				if i > 0: tf1.get_variable_scope().reuse_variables()
				curr_eps = eps_list[i]
				doc_vec = tf1.add(tf1.multiply(tf1.exp(logsigm), curr_eps), mean)
				doc_vec_list.append(doc_vec)
				logits = tf1.matmul(doc_vec, self.decoder_params[0]) + self.decoder_params[1]
				logits = tf1.nn.log_softmax(logits)
				recons_loss_list.append(-tf1.reduce_sum(tf1.multiply(logits, self.x), 1))
			doc_vec = tf1.add_n(doc_vec_list) / self.n_sample
			recons_loss = tf1.add_n(recons_loss_list) / self.n_sample
			
		#self.objective_TM = self.recons_loss + self.kld
		#self.objective_TM = tf1.add(self.recons_loss, self.kld, name='TM_loss_unnormed')
		final_loss = tf1.add(recons_loss, kld, name='TM_loss_unnormed')
		objective_TM = tf1.reduce_mean(final_loss)
		"""
		enc_grads = tf1.gradients(objective_TM, self.enc_vars)
		dec_grads = tf1.gradients(objective_TM, self.dec_vars)

		self.optim_enc = optimizer.apply_gradients(zip(enc_grads, enc_vars))
		self.optim_dec = optimizer.apply_gradients(zip(dec_grads, dec_vars))
		"""
		return doc_vec, objective_TM
	
	def nvdm_linear_params(
					self,
					input_size,
					output_size,
					no_bias=False,
					bias_start_zero=False,
					matrix_start_zero=False,
					scope=None,
					get_matrix=False,
					matrix_initializer=None,
					bias_initializer=None):
		with tf1.variable_scope(scope or 'Linear', reuse=tf1.AUTO_REUSE):

			if matrix_start_zero:
				matrix_initializer = tf1.constant_initializer(0)
				matrix = tf1.get_variable('Matrix', [input_size, output_size],
										initializer=matrix_initializer)
			else:
				if matrix_initializer is None:
					matrix_initializer = tf1.glorot_uniform_initializer(seed=tf_op_seed)
					matrix = tf1.get_variable('Matrix', [input_size, output_size],
											initializer=matrix_initializer)
				else:
					matrix = tf1.get_variable('Matrix',
											initializer=matrix_initializer)
			
			if bias_start_zero:
				bias_initializer = tf1.constant_initializer(0)
				bias = tf1.get_variable('Bias', [output_size], 
											initializer=bias_initializer)
			else:
				if bias_initializer is None:
					bias_initializer = tf1.glorot_uniform_initializer(seed=tf_op_seed)
					bias = tf1.get_variable('Bias', [output_size], 
												initializer=bias_initializer)
				else:
					bias = tf1.get_variable('Bias', 
												initializer=bias_initializer)
			
			return matrix, bias

	def fetch_data(self, data, count, idx_batch):
		batch_size = len(idx_batch)
		data_batch = np.zeros((batch_size, self.vocab_size), dtype=np.float32)
		count_batch = np.zeros(batch_size, dtype=np.int32)
		mask = np.zeros(batch_size, dtype=np.float32)
		indices = []
		values = []
		for i, doc_id in enumerate(idx_batch):
			if doc_id != -1:
				for word_id, freq in data[doc_id].items():
					data_batch[i, word_id] = freq
				count_batch[i] = count[doc_id]
				mask[i]=1.0
		return data_batch, count_batch, mask