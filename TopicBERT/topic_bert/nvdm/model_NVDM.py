"""NVDM Tensorflow implementation by Yishu Miao"""
from __future__ import print_function

import numpy as np
import tensorflow as tf
import math
import os
from nvdm import utils
#import model.utils as utils
#from sklearn.preprocessing import MultiLabelBinarizer
#import sklearn.metrics.pairwise as pw
#from gensim.models import CoherenceModel
#from gensim.corpora.dictionary import Dictionary
#import model.evaluate as eval
#import model.data_lstm as data

#seed = 42
#tf_op_seed = 1234
#np.random.seed(seed)
#tf.set_random_seed(seed)


seed = 42
tf.set_random_seed(seed)
np.random.seed(seed)
tf_op_seed = 42

#learning_rate = 5e-5
#batch_size = 64
#n_hidden = 256

#fixed_topic_params 

#n_topic = 150
#n_sample = 1
#non_linearity = tf.nn.tanh
non_linearity = tf.nn.sigmoid

######


class NVDM(object):
	""" Neural Variational Document Model -- BOW VAE.
	"""
	#def __init__(self, topic_params, prior_embeddings=None, initializer_nvdm=None):
	def __init__(self, topic_params, x, mask , topic_vocab_size,  prior_embeddings=None, initializer_nvdm=None):

		#self.vocab_size = topic_params.TM_vocab_length
		self.vocab_size = topic_vocab_size
		self.n_hidden = topic_params.hidden_size_TM
		self.n_topic =  topic_params.n_topic
		self.n_sample = topic_params.n_sample
		self.non_linearity = non_linearity
		self.learning_rate = topic_params.nvdm_learning_rate
		self.batch_size = topic_params.nvdm_batch_size
		self.x = x
		self.mask = mask 


		#self.x = tf.placeholder(tf.float32, [None, self.vocab_size], name='x')
		#self.mask = tf.placeholder(tf.float32, [None], name='mask')  # mask paddings

		#if topic_params.use_sent_topic_rep:
			#self.x_sent = tf.placeholder(tf.float32, [None, None, self.vocab_size], name='x_sent')

		#if topic_params.use_topic_embedding:
		#	self.x_doc_mask = tf.placeholder(tf.float32, [None, self.vocab_size], name='x_doc_mask')

		#self.input_batch_size = tf.placeholder(tf.int32, (), name='input_batch_size')
		self.input_batch_size = tf.shape(self.x)[0]
		#if topic_params.use_sent_topic_rep:
		#	self.input_batch_size_sent = tf.shape(self.x_sent)[0]
		#	self.input_batch_len_sent = tf.shape(self.x_sent)[1]
		#	self.batch_size_sent = self.input_batch_size_sent * self.input_batch_len_sent

		# encoder
		with tf.variable_scope('TM_encoder', reuse=tf.AUTO_REUSE): 
			self.enc_vec = utils.mlp(self.x, [self.n_hidden], self.non_linearity, initializer=initializer_nvdm[0])
			#self.enc_vec = utils.mlp(self.x, [self.n_hidden, self.n_hidden], self.non_linearity, initializer=initializer_nvdm[0])
			#self.enc_vec = utils.mlp(self.x, [self.n_hidden, self.n_hidden], self.non_linearity)
			self.mean = utils.nvdm_linear(self.enc_vec,
										self.n_topic,
										scope='mean',
										matrix_initializer=initializer_nvdm[1][0],
										bias_initializer=initializer_nvdm[1][1])
			self.logsigm = utils.nvdm_linear(self.enc_vec, 
									self.n_topic, 
									bias_start_zero=True,
									matrix_start_zero=True,
									scope='logsigm',
									matrix_initializer=initializer_nvdm[2][0],
									bias_initializer=initializer_nvdm[2][1])
			self.kld = -0.5 * tf.reduce_sum(1 - tf.square(self.mean) + 2 * self.logsigm - tf.exp(2 * self.logsigm), 1)
			#self.kld = self.mask*self.kld  # mask paddings
			self.kld = tf.multiply(self.mask, self.kld, name='kld')  # mask paddings

			#if topic_params.use_sent_topic_rep:
			#	self.x_sent_reshape = tf.reshape(self.x_sent, [-1, self.vocab_size])
			#	self.enc_vec_sent = utils.mlp(self.x_sent_reshape, [self.n_hidden], self.non_linearity)
			#	#self.enc_vec = utils.mlp(self.x, [self.n_hidden, self.n_hidden], self.non_linearity)
			#	self.mean_sent = utils.nvdm_linear(self.enc_vec_sent, self.n_topic, scope='mean')
			#	self.logsigm_sent = utils.nvdm_linear(self.enc_vec_sent, 
			#							self.n_topic, 
			#							bias_start_zero=True,
			#							matrix_start_zero=True,
			#							scope='logsigm')

			#if topic_params.prior_emb_for_topics:
			#	W_prior = tf.get_variable(
			#		'embeddings_TM_prior',
			#		dtype=tf.float32,
			#		initializer=prior_embeddings,
			#		trainable=False
			#	)
			"""
			W_prior_proj = tf.get_variable(
				'embeddings_TM_prior_proj',
				[prior_embeddings.shape[1], self.n_topic],
				dtype=tf.float32,
				trainable=False
			)
			W_prior = tf.matmul(W_prior, W_prior_proj, name='W_prior_projected')
			"""
				

		
		with tf.variable_scope('TM_decoder', reuse=tf.AUTO_REUSE):
			if self.n_sample == 1:
				eps = tf.random_normal((self.input_batch_size, self.n_topic), mean=0.0, stddev=1.0, seed=seed)
				#doc_vec = tf.mul(tf.exp(self.logsigm), eps) + self.mean

				## Hidden representation to be used in BERT
				self.doc_vec = tf.add(tf.multiply(tf.exp(self.logsigm), eps), self.mean, name='doc_hidden')
				
				self.last_h = self.doc_vec
				logits_projected, self.decoding_matrix = utils.nvdm_linear(self.doc_vec, 
																		self.vocab_size, 
																		scope='projection', 
																		get_matrix=True,
																		matrix_initializer=initializer_nvdm[3][0],
																		bias_initializer=initializer_nvdm[3][1])
				logits = tf.nn.log_softmax(logits_projected)
				self.recons_loss = -tf.reduce_sum(tf.multiply(logits, self.x), 1)


				"""
				if topic_params.use_topic_embedding:
					#self.last_h_topic_emb = utils.nvdm_linear(tf.nn.softmax(self.last_h, axis=1), self.vocab_size, scope='projection')
					#self.top_k = tf.nn.top_k(self.decoding_matrix, k=topic_params.use_k_topic_words, sorted=False)
					topics_masked = tf.multiply(tf.expand_dims(self.x_doc_mask, axis=1), tf.expand_dims(self.decoding_matrix, axis=0), name='topics_masked')
					self.top_k = tf.nn.top_k(topics_masked, k=topic_params.use_k_topic_words, sorted=False)
					if topic_params.prior_emb_for_topics:
						self.top_k_embeddings = tf.nn.embedding_lookup(W_prior, self.top_k.indices)
						self.topic_emb_size = prior_embeddings.shape[1]
						#self.topic_emb_size = prior_embeddings.shape[1] * topic_params.use_k_topics
						#self.topic_emb_size = prior_embeddings.shape[1] + self.n_topic
						#self.topic_emb_size = self.n_topic
						#self.topic_emb_size = self.n_topic * 2
					else:
						self.top_k_embeddings = tf.nn.embedding_lookup(tf.transpose(self.decoding_matrix), self.top_k.indices)
						#self.topic_emb_size = self.n_topic
						self.topic_emb_size = self.n_topic * 2
					#self.top_k_embeddings = tf.multiply(tf.expand_dims(tf.nn.softmax(self.top_k.values, axis=1), axis=2), self.top_k_embeddings)
					#self.temp_1 = tf.expand_dims(tf.nn.softmax(self.top_k.values, axis=2), axis=2)
					#self.topic_embeddings = tf.squeeze(tf.matmul(self.temp_1, self.top_k_embeddings), axis=2, name='topic_embeddings')
					#self.topic_embeddings = tf.reduce_sum(self.top_k_embeddings, axis=1, name='topic_embeddings')
					#self.topic_embeddings = tf.reduce_mean(self.top_k_embeddings, axis=1, name='topic_embeddings')
					self.topic_embeddings = tf.reduce_mean(self.top_k_embeddings, axis=2, name='topic_embeddings')

					if topic_params.use_k_topics > 0:
						# Masking document topic proportion vector
						top_k_h_values, top_k_h_indices = tf.nn.top_k(self.last_h, k=topic_params.use_k_topics, sorted=False, name='top_k_h')
						row_numbers = tf.tile(tf.expand_dims(tf.range(0, self.input_batch_size), 1), [1, topic_params.use_k_topics], name='row_numbers')
						full_indices = tf.concat([tf.expand_dims(row_numbers, -1), tf.expand_dims(top_k_h_indices, -1)], axis=2)
						full_indices = tf.reshape(full_indices, [-1, 2], name='full_indices')
						#mask_updates = tf.ones([self.input_batch_size * topic_params.use_k_topics], dtype=tf.float32, name='mask_updates')
						#new_mask = tf.scatter_nd(full_indices, mask_updates, [self.input_batch_size, self.n_topic], name='new_mask')
						#last_h_softmax = tf.multiply(tf.nn.softmax(self.last_h, axis=1), new_mask, name='last_h_softmax')
						last_h_softmax = tf.scatter_nd(
							full_indices, 
							tf.reshape(tf.nn.softmax(top_k_h_values, axis=1), [-1]), 
							#tf.ones([self.input_batch_size * topic_params.use_k_topics], dtype=tf.float32), 
							[self.input_batch_size, self.n_topic], 
							name='last_h_softmax'
						)
					else:
						last_h_softmax = tf.nn.softmax(self.last_h, axis=1, name='last_h_softmax')
						#last_h_softmax = self.last_h
						
					#self.last_h_topic_emb = tf.matmul(last_h_softmax, self.topic_embeddings, name='last_h_topic_emb')
					self.last_h_topic_emb = tf.squeeze(tf.matmul(tf.expand_dims(last_h_softmax, axis=1), self.topic_embeddings), axis=1, name='last_h_topic_emb')
					#temp = tf.nn.embedding_lookup(self.topic_embeddings, top_k_h_indices)
					#self.last_h_topic_emb = tf.reduce_sum(temp, axis=1, name='last_h_topic_emb')
					#self.last_h_topic_emb = tf.reshape(temp, [self.input_batch_size, self.topic_emb_size], name='last_h_topic_emb')
					#self.last_h_topic_emb = tf.concat([self.last_h_topic_emb, last_h_softmax], axis=1)
					#self.last_h_topic_emb = tf.concat([self.last_h_topic_emb, self.last_h], axis=1)
			"""
			else:
				#eps = tf.random_normal((self.n_sample*self.batch_size, self.n_topic), mean=0.0, stddev=1.0)
				"""
				eps = tf.random_normal((self.n_sample*self.input_batch_size, self.n_topic), mean=0.0, stddev=1.0, seed=seed)
				eps_list = tf.split(eps, self.n_sample, 0)
				recons_loss_list = []
				for i in range(self.n_sample):
					if i > 0: tf.get_variable_scope().reuse_variables()
					curr_eps = eps_list[i]
					doc_vec = tf.multiply(tf.exp(self.logsigm), curr_eps) + self.mean
					logits = tf.nn.log_softmax(utils.nvdm_linear(doc_vec, self.vocab_size, scope='projection'))
					recons_loss_list.append(-tf.reduce_sum(tf.multiply(logits, self.x), 1))
				self.recons_loss = tf.add_n(recons_loss_list) / self.n_sample
				"""
				eps = tf.random_normal((self.n_sample*self.input_batch_size, self.n_topic), mean=0.0, stddev=1.0, seed=seed)
				eps_list = tf.split(eps, self.n_sample, 0)
				recons_loss_list = []
				doc_vec_list = []
				for i in range(self.n_sample):
					if i > 0: tf.get_variable_scope().reuse_variables()
					curr_eps = eps_list[i]
					doc_vec = tf.add(tf.multiply(tf.exp(self.logsigm), curr_eps), self.mean)
					doc_vec_list.append(doc_vec)
					logits = tf.nn.log_softmax(utils.nvdm_linear(doc_vec, self.vocab_size, scope='projection'))
					recons_loss_list.append(-tf.reduce_sum(tf.multiply(logits, self.x), 1))
				self.recons_loss = tf.add_n(recons_loss_list) / self.n_sample
				self.doc_vec = tf.add_n(doc_vec_list) / self.n_sample
				self.last_h = self.doc_vec


			""""
			if topic_params.use_sent_topic_rep:
				if self.n_sample == 1:
					eps_sent = tf.random_normal((self.batch_size_sent, self.n_topic), mean=0.0, stddev=1.0, seed=seed)
					self.last_h_sent = tf.add(tf.multiply(tf.exp(self.logsigm_sent), eps_sent), self.mean_sent, name='sent_hidden')
					self.last_h_sent = tf.reshape(self.last_h_sent, [self.input_batch_size_sent, self.input_batch_len_sent, self.n_topic])

					if topic_params.use_topic_embedding:
						#self.last_h_topic_emb_sent = utils.nvdm_linear(tf.nn.softmax(self.last_h_sent, axis=1), self.vocab_size, scope='projection')
						if topic_params.use_k_topics > 0:
							# Masking sentence topic proportion vector
							top_k_h_sent_values, top_k_h_sent_indices = tf.nn.top_k(self.last_h_sent, k=topic_params.use_k_topics, sorted=False, name='top_k_h_sent')
							row_numbers_sent = tf.tile(tf.expand_dims(tf.range(0, self.batch_size_sent), 1), [1, topic_params.use_k_topics], name='row_numbers_sent')
							full_indices_sent = tf.concat([tf.expand_dims(row_numbers_sent, -1), tf.expand_dims(top_k_h_sent_indices, -1)], axis=2)
							full_indices_sent = tf.reshape(full_indices_sent, [-1, 2], name='full_indices_sent')
							#mask_updates_sent = tf.ones([self.batch_size_sent * topic_params.use_k_topics], dtype=tf.float32, name='mask_updates_sent')
							#new_mask_sent = tf.scatter_nd(full_indices_sent, mask_updates_sent, [self.batch_size_sent, self.n_topic], name='new_mask_sent')
							#last_h_softmax_sent = tf.multiply(tf.nn.softmax(self.last_h_sent, axis=1), new_mask_sent, name='last_h_softmax_sent')
							last_h_softmax_sent = tf.scatter_nd(full_indices_sent, tf.reshape(tf.nn.softmax(top_k_h_sent_values, axis=1), [-1]), [self.batch_size_sent, self.n_topic], name='last_h_softmax_sent')
						else:
							last_h_softmax_sent = tf.nn.softmax(self.last_h_sent, axis=2, name='last_h_softmax_sent')
						
						self.last_h_topic_emb_sent = tf.matmul(last_h_softmax_sent, self.topic_embeddings, name='last_h_topic_emb_sent')
						#self.last_h_topic_emb_sent = tf.concat([self.last_h_topic_emb_sent, self.last_h_sent], axis=2, name='last_h_topic_emb_sent')
						#self.last_h_topic_emb_sent = tf.concat([self.last_h_topic_emb_sent, last_h_softmax_sent], axis=2, name='last_h_topic_emb_sent')
						#self.last_h_topic_emb_sent = tf.reshape(self.last_h_topic_emb_sent, [self.input_batch_size_sent, self.input_batch_len_sent, self.vocab_size])
						
				else:
					print("Error: model_NVDM.py - Decoder")
					sys.exit()
			"""

		#self.objective_TM = self.recons_loss + self.kld
		#self.objective_TM = tf.add(self.recons_loss, self.kld, name='TM_loss_unnormed')
		self.final_loss = tf.add(self.recons_loss, self.kld, name='TM_loss_unnormed')
		self.objective_TM = tf.reduce_mean(self.final_loss)

		"""
		if topic_params.TM_uniqueness_loss:
			## NVDM topic uniqueness loss
			eye = tf.constant(np.eye(self.n_topic), dtype=tf.float32)
			topicnorm = matrix / tf.sqrt(tf.reduce_sum(tf.square(self.decoding_matrix), 1, keepdims=True))
			uniqueness = tf.reduce_max(tf.square(tf.matmul(topicnorm, tf.transpose(topicnorm)) - eye))
			self.objective_TM += topic_params.alpha_uniqueness * uniqueness
		"""

	
		
		optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
		#fullvars = tf.trainable_variables()

		#enc_vars = utils.variable_parser(fullvars, 'TM_encoder')
		enc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='TM_encoder')
		#dec_vars = utils.variable_parser(fullvars, 'TM_decoder')
		dec_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='TM_decoder')
		self.pretrain_saver = tf.train.Saver(enc_vars + dec_vars)

		enc_grads = tf.gradients(self.objective_TM, enc_vars)
		dec_grads = tf.gradients(self.objective_TM, dec_vars)

		self.optim_enc = optimizer.apply_gradients(zip(enc_grads, enc_vars))
		self.optim_dec = optimizer.apply_gradients(zip(dec_grads, dec_vars))
		

	## Pretraining of NVDM-TM
	def pretrain(self, dataset, topic_params, nvdm_datadir , session,
				#training_epochs=1000, alternate_epochs=10):
				#training_epochs=100, alternate_epochs=10):
				training_epochs=20, alternate_epochs=10):
				#training_epochs=1, alternate_epochs=1):

		#log_dir = os.path.join(topic_params.model, 'logs_nvdm_pretrain')
		#model_dir_ir_nvdm = os.path.join(topic_params.model, 'model_ir_nvdm_pretrain')
		#model_dir_ppl_nvdm = os.path.join(topic_params.model, 'model_ppl_nvdm_pretrain')
		log_dir = os.path.join(topic_params.output_dir, 'logs_nvdm_pretrain')
		model_dir_ir_nvdm = os.path.join(topic_params.output_dir, 'model_ir_nvdm_pretrain')
		model_dir_ppl_nvdm = os.path.join(topic_params.output_dir, 'model_ppl_nvdm_pretrain')

		#model_dir_supervised = os.path.join(topic_params.model, 'model_supervised_nvdm_pretrain')

		if not os.path.isdir(log_dir):
			os.mkdir(log_dir)
		if not os.path.isdir(model_dir_ir_nvdm):
			os.mkdir(model_dir_ir_nvdm)
		if not os.path.isdir(model_dir_ppl_nvdm):
			os.mkdir(model_dir_ppl_nvdm)
		#if not os.path.isdir(model_dir_supervised):
		#	os.mkdir(model_dir_supervised)

		#train_url = os.path.join(topic_params.dataset, 'training_nvdm_docs_non_replicated.csv')
		#dev_url = os.path.join(topic_params.dataset, 'validation_nvdm_docs_non_replicated.csv')
		#test_url = os.path.join(topic_params.dataset, 'test_nvdm_docs_non_replicated.csv')

		train_url = os.path.join(nvdm_datadir, 'training_nvdm_docs_non_replicated.csv')
		dev_url = os.path.join(nvdm_datadir, 'validation_nvdm_docs_non_replicated.csv')
		test_url = os.path.join(nvdm_datadir, 'test_nvdm_docs_non_replicated.csv')

		train_set, train_count, train_labels, train_doc_ids = utils.data_set(train_url, topic_params)
		test_set, test_count, test_labels, test_doc_ids = utils.data_set(test_url, topic_params)
		dev_set, dev_count, dev_labels, dev_doc_ids = utils.data_set(dev_url, topic_params)

		dev_batches = utils.create_batches(len(dev_set), self.batch_size, shuffle=False)
		#dev_batches = utils.create_batches(len(dev_set), 512, shuffle=False)
		test_batches = utils.create_batches(len(test_set), self.batch_size, shuffle=False)
		#test_batches = utils.create_batches(len(test_set), 512, shuffle=False)
		
		#training_labels = np.array(
		#	[[y] for y, _ in dataset.rows('training_nvdm_docs_non_replicated', num_epochs=1)]
		#)
		#validation_labels = np.array(
		#	[[y] for y, _ in dataset.rows('validation_nvdm_docs_non_replicated', num_epochs=1)]
		#)
		#test_labels = np.array(
		#	[[y] for y, _ in dataset.rows('test_nvdm_docs_non_replicated', num_epochs=1)]
		#)

		patience = topic_params.nvdm_patience
		patience_count = 0
		best_dev_ppl = np.inf
		best_test_ppl = np.inf
		best_val_nvdm_IR = -1.0
		best_test_nvdm_IR = -1.0

		enc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='TM_encoder')
		dec_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='TM_decoder')
		self.pretrain_saver = tf.train.Saver(enc_vars + dec_vars)

		ppl_model = False
		ir_model = False
		
		for epoch in range(training_epochs):
			epoch_counter = epoch + 1
			train_batches = utils.create_batches(len(train_set), self.batch_size, shuffle=True)
			#train_batches = utils.create_batches(len(train_set), 512, shuffle=True)
			
			#-------------------------------
			# train
			for switch in range(0, 2):
				if switch == 0:
					optim = self.optim_dec
					print_mode = 'updating decoder'
				else:
					optim = self.optim_enc
					print_mode = 'updating encoder'
				for i in range(alternate_epochs):
					print_ppx, print_ppx_perdoc, print_kld = self.run_epoch(
						train_batches, 
						train_set, 
						train_count, 
						topic_params, 
						session,
						optimizer=optim
					)

					print('| Epoch train: {:d} |'.format(epoch_counter), 
						print_mode, '{:d}'.format(i),
						'| Corpus Perplexity: {:.5f}'.format(print_ppx),  # perplexity for all docs
						'| Per doc Perplexity: {:.5f}'.format(print_ppx_perdoc),  # perplexity for per doc
						'| KLD: {:.5}'.format(print_kld))
			
			if epoch_counter >= 1 and epoch_counter % topic_params.nvdm_validation_ppl_freq == 0:
				ppl_model = True
				
				print_ppx, print_ppx_perdoc, print_kld = self.run_epoch(
					dev_batches, 
					dev_set, 
					dev_count, 
					topic_params, 
					session
				)
				if print_ppx_perdoc < best_dev_ppl:
				#if print_ppx_perdoc <= best_dev_ppl:
					best_dev_ppl = print_ppx_perdoc
					print("Saving best model.")
					self.pretrain_saver.save(session, model_dir_ppl_nvdm + '/model_ppl_nvdm_pretrain', global_step=1)
					self.save_to_s3_TM(topic_params)

					patience_count = 0
				else:
					patience_count += 1

				print('| Epoch dev: {:d} |'.format(epoch_counter), 
					'| Corpus Perplexity: {:.9f} |'.format(print_ppx),
					'| Per doc Perplexity: {:.5f} |'.format(print_ppx_perdoc),
					'| KLD: {:.5} |'.format(print_kld),
					'| Best dev PPL: {:.5} |'.format(best_dev_ppl))
				
				with open(log_dir + "/logs_ppl_nvdm_pretrain.txt", "a") as f:
					f.write('| Epoch Val: {:d} || Val Corpus PPL: {:.9f} || Val Per doc PPL: {:.5f} || Best Val PPL: {:.5} || KLD Val: {:.5} |\n'.format(epoch+1, print_ppx, print_ppx_perdoc, best_dev_ppl, print_kld))
		
			if epoch_counter >= 1 and epoch_counter % topic_params.nvdm_validation_ir_freq == 0:
				ir_model = True

				validation_vectors_nvdm = self.hidden_vectors(
					#dataset.batches_nvdm_LM('validation_nvdm_docs_non_replicated', topic_params.nvdm_batch_size, topic_params.TM_vocab_length, num_epochs=1, multilabel=topic_params.multi_label),
					dataset.batches_nvdm_LM('validation_nvdm_docs_non_replicated', topic_params.nvdm_batch_size, self.vocab_size, num_epochs=1, multilabel=topic_params.multilabel),
					topic_params,
					session
				)

				training_vectors_nvdm = self.hidden_vectors(
					#dataset.batches_nvdm_LM('training_nvdm_docs_non_replicated', topic_params.nvdm_batch_size, topic_params.TM_vocab_length, num_epochs=1, multilabel=topic_params.multi_label),
					dataset.batches_nvdm_LM('training_nvdm_docs_non_replicated', topic_params.nvdm_batch_size, self.vocab_size, num_epochs=1, multilabel=topic_params.multilabel),
					topic_params,
					session
				)

				val_nvdm_ir, _ = eval.evaluate(
					training_vectors_nvdm,
					validation_vectors_nvdm,
					training_labels,
					validation_labels,
					recall=[0.02],
					num_classes=topic_params.nvdm_num_classes,
					multi_label=topic_params.multilabel
				)
				val_nvdm_ir = val_nvdm_ir[0]
				
				# Saving model and Early stopping on IR
				if val_nvdm_ir > best_val_nvdm_IR:
					best_val_nvdm_IR = val_nvdm_ir
					print('saving: {}'.format(model_dir_ir_nvdm))
					self.pretrain_saver.save(session, model_dir_ir_nvdm + '/model_ir_nvdm_pretrain', global_step=1)
					self.save_to_s3_TM(topic_params)
				#	patience_count = 0
				#else:
				#	patience_count += 1

				print("Epoch: %i,	Val NVDM IR: %s,	best val NVDM IR: %s\n" %
						(epoch_counter, val_nvdm_ir, best_val_nvdm_IR))

				# logging information
				with open(log_dir + "/logs_ir_nvdm_pretrain.txt", "a") as f:
					f.write("Epoch: %i,	Val NVDM IR: %s,	best val NVDM IR: %s\n" %
							(epoch_counter, val_nvdm_ir, best_val_nvdm_IR))

			if patience_count > patience:
				print("Early stopping.")
				break

		if ppl_model:
			print("Calculating Test PPL.")

			self.pretrain_saver.restore(session, tf.train.latest_checkpoint(model_dir_ppl_nvdm))

			print_ppx, print_ppx_perdoc, print_kld = self.run_epoch(
				test_batches, 
				test_set, 
				test_count, 
				topic_params, 
				session
			)

			print('| Corpus Perplexity: {:.9f}'.format(print_ppx),
					'| Per doc Perplexity: {:.5f}'.format(print_ppx_perdoc),
					'| KLD: {:.5}'.format(print_kld))

			with open(log_dir + "/logs_ppl_nvdm_pretrain.txt", "a") as f:
				f.write('\n\nTest Corpus PPL: {:.9f} || Test Per doc PPL: {:.5f} || KLD Test: {:.5} |\n'.format(print_ppx, print_ppx_perdoc, print_kld))

		if ir_model:
			print("Calculating Test IR.")

			self.pretrain_saver.restore(session, tf.train.latest_checkpoint(model_dir_ir_nvdm))

			test_vectors_nvdm = self.hidden_vectors(
				#dataset.batches_nvdm_LM('test_nvdm_docs_non_replicated', topic_params.nvdm_batch_size, topic_params.TM_vocab_length, num_epochs=1, multilabel=topic_params.multi_label),
				dataset.batches_nvdm_LM('test_nvdm_docs_non_replicated', topic_params.nvdm_batch_size, self.vocab_size, num_epochs=1, multilabel=topic_params.multilabel),
				topic_params,
				session
			)

			test_nvdm_ir, _ = eval.evaluate(
				training_vectors_nvdm,
				test_vectors_nvdm,
				training_labels,
				test_labels,
				recall=[0.02],
				num_classes=topic_params.nvdm_num_classes,
				multi_label=topic_params.multilabel
			)
			test_nvdm_ir = test_nvdm_ir[0]

			print("Epoch: %i,	Test NVDM IR: %s\n" %
					(epoch_counter, test_nvdm_ir))

			# logging information
			with open(log_dir + "/logs_ir_nvdm_pretrain.txt", "a") as f:
				f.write("Epoch: %i,	Test NVDM IR: %s\n" %
						(epoch_counter, test_nvdm_ir))


	def hidden_vectors(self, data, topic_params, session):
		vecs = []
		for y, x, count, mask in data:

			feed_dict = {
				self.x.name: x,
				self.mask.name: mask
				#self.input_batch_size: x.shape[0]
			}
			
			vecs.extend(
				session.run([self.last_h], feed_dict=feed_dict)[0]
			)
		
		return np.array(vecs)


	def topic_dist(self, input_batches, input_set, input_doc_ids , input_count, topic_params, session):
		topic_dist = []
		mask_list = []
		doc_id_list = []
		for idx_batch in input_batches:
			data_batch, count_batch, mask = utils.fetch_data(
			input_set, input_count, idx_batch, self.vocab_size, topic_params)
			input_feed = {self.x.name: data_batch,
							self.mask.name: mask}
							
			doc_vec = session.run([self.doc_vec], input_feed)
			topic_dist.extend(list(doc_vec[0]))
			mask_list.extend(list(mask))

			for idx in idx_batch:
				if idx != -1: 
					doc_id_list.append(input_doc_ids[idx])
				else:
					doc_id_list.append(-1)


		assert len(topic_dist) == len(doc_id_list)
		topic_dist_unique = {}

		for id, dist in zip(doc_id_list, topic_dist):
			if id != -1:
				topic_dist_unique[str(id)] = dist
		
		"""
		topic_dist_unique = []
		for num, m in enumerate(mask_list):
			if m!= 0.0:
				topic_dist_unique.append(topic_dist[num])

		topic_dist_unique = np.asarray(topic_dist_unique)
		"""

		return topic_dist_unique, mask_list

	def save_to_s3_TM(self, topic_params):
		pass


	def run_epoch(self, input_batches, input_set, input_count, topic_params, session, optimizer=None):
		loss_sum = 0.0
		ppx_sum = 0.0
		kld_sum = 0.0
		word_count = 0
		doc_count = 0
		for idx_batch in input_batches:
			data_batch, count_batch, mask = utils.fetch_data(
			input_set, input_count, idx_batch, self.vocab_size, topic_params)
			#import pdb; pdb.set_trace()
			input_feed = {self.x.name: data_batch,
							self.mask.name: mask}#, 
							#self.input_batch_size: data_batch.shape[0]
							#}
			if not optimizer is None:
				_, (loss, kld) = session.run((optimizer, 
											[self.final_loss, self.kld]),
											input_feed)
			else:
				loss, kld = session.run([self.final_loss, self.kld],
											input_feed)
			
			loss_sum += np.sum(loss)
			kld_sum += np.sum(kld) / np.sum(mask) 
			word_count += np.sum(count_batch)
			# to avoid nan error
			count_batch = np.add(count_batch, 1e-12)
			# per document loss
			ppx_sum += np.sum(np.divide(loss, count_batch)) 
			doc_count += np.sum(mask)

		print_ppx = np.exp(loss_sum / word_count)
		print_ppx_perdoc = np.exp(ppx_sum / doc_count)
		print_kld = kld_sum/len(input_batches)

		return print_ppx, print_ppx_perdoc, print_kld


	def run_epoch_v2(self, data, topic_params, session):
		# train_y, train_x, train_count, train_mask = dataset.batches_nvdm_LM(training_data_filename_TM, topic_params.batch_size, topic_params.TM_vocab_length, num_epochs=1, multilabel=topic_params.multi_label)
		# val_y, val_x, val_count, val_mask = dataset.batches_nvdm_LM(validation_data_filename_TM, topic_params.batch_size, topic_params.TM_vocab_length, num_epochs=1, multilabel=topic_params.multi_label)
		# test_y, test_x, test_count, test_mask = dataset.batches_nvdm_LM(test_data_filename_TM, topic_params.batch_size, topic_params.TM_vocab_length, num_epochs=1, multilabel=topic_params.multi_label)
		
		kld_sum = []
		this_nvdm_loss_normed = []
		this_nvdm_loss_unnormed = []
		this_nvdm_words = []
		for nvdm_y, nvdm_x, nvdm_count, nvdm_mask in data:
			nvdm_feed_dict = {
				model.topic_model.x.name: nvdm_x,
				model.topic_model.mask.name: nvdm_mask#,
				#model.topic_model.input_batch_size: nvdm_x.shape[0]
			}
		
			if topic_params.supervised:
				sys.exit()
			else:
				loss, kld = session.run([model.topic_model.final_loss, 
										model.topic_model.kld], 
										feed_dict=nvdm_feed_dict)
				nvdm_count = np.add(nvdm_count, 1e-12)
				this_nvdm_loss_normed.extend(np.divide(loss, nvdm_count))
				this_nvdm_loss_unnormed.extend(loss)
				this_nvdm_words.append(np.sum(nvdm_count))
				kld_sum.append(np.sum(kld) / np.sum(nvdm_mask))

		total_nvdm_nll = np.mean(this_nvdm_loss_unnormed)
		#total_nvdm_ppl = np.exp(np.sum(this_nvdm_loss_unnormed) / np.sum(this_val_nvdm_words))
		total_nvdm_ppl = np.exp(np.mean(this_nvdm_loss_normed))
		print_kld = np.mean(kld_sum)

		return total_nvdm_nll, total_nvdm_ppl, print_kld

	