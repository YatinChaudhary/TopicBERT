import os, sys
import pickle, json
import math, csv
import time, datetime
import logging
import random

import numpy as np
import tensorflow as tf
from transformers import *

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import pairwise as pw
from sklearn.preprocessing import normalize
from sklearn.metrics import precision_recall_fscore_support

from scipy.special import expit
from scipy.spatial.distance import cosine

from src.model_TM.model_NVDM_TF2 import NVDM, create_initializer

seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


def gelu(x):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.

  Returns:
    `x` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf

class Transformer_Model:
	"""Transformer architecture based Language Modeling class"""

	def __init__(self, params, model_params, params_TM, num_labels=None):
		self.model_class		   = params.model_class_unsup
		self.tokenizer_class	   = params.tokenizer_class
		self.pretrained_model_name = params.pretrained_model_name
		self.max_length			   = params.max_length
		self.batch_size			   = params.batch_size
		self.special_token_begin   = params.special_token_begin
		self.special_token_end     = params.special_token_end
		self.seq_start_from_begin  = params.seq_start_from_begin
		self.cls_comb_strategy	   = params.cls_comb_strategy
		self.alpha                 = params.alpha
		self.num_labels			   = num_labels
		#if not num_labels is None:
		#	model_params["num_labels"] = num_labels
		
		self.tokenizer = self.tokenizer_class.from_pretrained(self.pretrained_model_name)
		self.model	   = self.model_class.from_pretrained(self.pretrained_model_name, **model_params)

		self.model_TM = NVDM(params_TM)
		
		self.projection = tf.keras.layers.Dense(
            self.model.config.dim, 
			activation=gelu,
			kernel_initializer=create_initializer(self.model.config.initializer_range), 
			name="projection"
        )

		self.classifier = tf.keras.layers.Dense(
            self.num_labels, 
			kernel_initializer=create_initializer(self.model.config.initializer_range), 
			name="classifier"
        )

		self.trainable_params = self.model.trainable_weights
		self.trainable_params.extend(self.model_TM.encoder_params)
		self.trainable_params.extend(self.model_TM.decoder_params)

		self.copy_to_s3 = params.copy_to_s3
		self.log_dir	= os.path.join(params.output_dir, params.savemodeldir, "log")
		self.model_dir  = os.path.join(params.output_dir, params.savemodeldir, "model")
		self.s3_transfer_dir = os.path.join(params.output_dir, params.savemodeldir)

		if not os.path.isdir(self.log_dir):
			os.makedirs(self.log_dir)
		if not os.path.isdir(self.model_dir):
			os.makedirs(self.model_dir)
		
		self.data_reps = None
	
	def send_to_s3(self):
		pass

	def get_preds(self, logits, multilabel=False):
		if multilabel:
			sigmoid_logits = expit(logits)
			preds = np.zeros_like(logits)
			preds[sigmoid_logits > 0.5] = 1.0
		else:
			preds = np.argmax(logits, axis=1)
		return preds

	def merge_logits(self, logits, doc_mapping_ids, combination_strategy="avg"):
		if combination_strategy == "sum":
			merged_logits = np.array([np.sum([logits[index] for index in doc_splitted_indices], axis=0) 
														for doc_id, doc_splitted_indices in doc_mapping_ids.items()])
		elif combination_strategy == "avg":
			merged_logits = np.array([np.mean([logits[index] for index in doc_splitted_indices], axis=0) 
														for doc_id, doc_splitted_indices in doc_mapping_ids.items()])
		return merged_logits
	
	"""
	def example_to_features(self, input_ids, attention_mask, token_type_ids, seq_lengths, labels=None):
		if not labels is None:
			return {"input_ids": input_ids,
					"attention_mask": attention_mask,
					"token_type_ids": token_type_ids}, seq_lengths, labels
		else:
			return {"input_ids": input_ids,
					"attention_mask": attention_mask,
					"token_type_ids": token_type_ids}, seq_lengths
	"""
	def example_to_features(self, input_ids, attention_mask, seq_lengths, doc_ids, labels=None):
		if not labels is None:
			return {"input_ids": input_ids,
					"attention_mask": attention_mask}, seq_lengths, doc_ids, labels
		else:
			return {"input_ids": input_ids,
					"attention_mask": attention_mask}, seq_lengths, doc_ids
	
	def split_docs_by_max_seq_length(self, doc_word_piece_tokens_list):
		max_seq_length = self.max_length - 2 # to add special tokens
		
		doc_word_piece_tokens_list_splitted = []
		doc_splitted_mapping_ids = {doc_id: [] for doc_id in range(len(doc_word_piece_tokens_list))}
		doc_splitted_mapping_ids_list = []
		doc_splitted_sequence_lengths = []
		for doc_id, doc_word_piece_tokens in enumerate(doc_word_piece_tokens_list):			
			start_index = 0
			end_index = 0
			while True:
				start_index = end_index
				end_index = start_index + max_seq_length
				if end_index >= len(doc_word_piece_tokens):
					break

				while True:
					if doc_word_piece_tokens[end_index].startswith("##"):
						end_index -= 1
					else:
						break
				
				doc_word_piece_tokens_seq_length = [(end_index - start_index) + self.special_token_begin + self.special_token_end, self.seq_start_from_begin]
				doc_word_piece_tokens_segment = doc_word_piece_tokens[start_index:end_index]
				
				doc_word_piece_tokens_list_splitted.append(doc_word_piece_tokens_segment)
				doc_splitted_sequence_lengths.append(doc_word_piece_tokens_seq_length)
				doc_splitted_mapping_ids[doc_id].append(len(doc_word_piece_tokens_list_splitted) - 1)
				doc_splitted_mapping_ids_list.append(doc_id)
			
			if start_index != 0:
				doc_word_piece_tokens_seq_length = [(len(doc_word_piece_tokens) - start_index) + self.special_token_end, 0]
				start_index = - max_seq_length
				while True:
					if doc_word_piece_tokens[start_index].startswith("##"):
						start_index += 1
					else:
						break
			else:
				doc_word_piece_tokens_seq_length = [len(doc_word_piece_tokens) + self.special_token_begin + self.special_token_end, self.seq_start_from_begin]
			doc_word_piece_tokens_segment = doc_word_piece_tokens[start_index:]
			
			doc_word_piece_tokens_list_splitted.append(doc_word_piece_tokens_segment)
			doc_splitted_sequence_lengths.append(doc_word_piece_tokens_seq_length)
			doc_splitted_mapping_ids[doc_id].append(len(doc_word_piece_tokens_list_splitted) - 1)
			doc_splitted_mapping_ids_list.append(doc_id)

		return doc_word_piece_tokens_list_splitted, doc_splitted_sequence_lengths, doc_splitted_mapping_ids, doc_splitted_mapping_ids_list

	def process_docs(self, doc_word_piece_tokens_list):
		"""
		Split documents above max sequence length + Add special tokens + Tokenize docs_list
		
		Args:
			doc_word_piece_tokens_list: list of document-wise word piece tokens

		Returns:
			input_ids: mapped ids input for corresponding input documents in docs_list
			attention_mask: attention mask for corresponding input
		"""
		## Fast batch wise encoding
		#processed_batch_dict = self.tokenizer.batch_encode_plus(docs_list, max_length=self.max_length, pad_to_max_length=True)
		#return processed_batch_dict["input_ids"], processed_batch_dict["attention_mask"]
		
		input_ids, attention_mask, token_type_ids = [], [], []
		for doc_word_piece_tokens in doc_word_piece_tokens_list:
			outputs = self.tokenizer.encode_plus(doc_word_piece_tokens, max_length=self.max_length, add_special_tokens=True, pad_to_max_length=True)
			input_ids.append(outputs["input_ids"])
			attention_mask.append(outputs["attention_mask"])
			#token_type_ids.append(outputs["token_type_ids"])
		
		if isinstance(input_ids[0][0], int):
			input_ids = tf.constant(np.stack(input_ids, axis=0).astype(np.int32))
		else:
			input_ids = tf.constant(np.stack(input_ids, axis=0).astype(np.float32))
		
		if isinstance(attention_mask[0][0], int):
			attention_mask = tf.constant(np.stack(attention_mask, axis=0).astype(np.int32))
		else:
			attention_mask = tf.constant(np.stack(attention_mask, axis=0).astype(np.float32))

		#if isinstance(token_type_ids[0][0], int):
		#	token_type_ids = tf.constant(np.stack(token_type_ids, axis=0).astype(np.int32))
		#else:
		#	token_type_ids = tf.constant(np.stack(token_type_ids, axis=0).astype(np.float32))
		
		#return input_ids, attention_mask, token_type_ids
		return input_ids, attention_mask

	def prepare_data(self, docs_list, labels=None, shuffle=False, batch_size=None):
		# Splitting documents in "max_seq_length" size chunks
		doc_word_piece_tokens_list = [self.tokenizer.tokenize(doc) for doc in docs_list]
		doc_word_piece_tokens_list_splitted, \
		doc_splitted_sequence_lengths, \
		doc_splitted_mapping_ids, \
		doc_splitted_mapping_ids_list = \
			self.split_docs_by_max_seq_length(doc_word_piece_tokens_list)
		
		if batch_size is None:
			batch_size = self.batch_size
			
		# Data preparation
		input_ids, \
		attention_mask = self.process_docs(doc_word_piece_tokens_list_splitted)
		#token_type_ids = self.process_docs(doc_word_piece_tokens_list_splitted)
		
		if not labels is None:
			dataset = tf.data.Dataset.from_tensor_slices(
				#(input_ids, attention_mask, token_type_ids, tf.constant(doc_splitted_sequence_lengths), labels[doc_splitted_mapping_ids_list]))
				(input_ids, attention_mask, tf.constant(doc_splitted_sequence_lengths), doc_splitted_mapping_ids_list, labels[doc_splitted_mapping_ids_list]))
		else:
			dataset = tf.data.Dataset.from_tensor_slices(
				#(input_ids, attention_mask, token_type_ids, tf.constant(doc_splitted_sequence_lengths)))
				(input_ids, attention_mask, tf.constant(doc_splitted_sequence_lengths), doc_splitted_mapping_ids_list))
		
		dataset = dataset.map(self.example_to_features).batch(batch_size, drop_remainder=False)
		if shuffle:
			dataset = dataset.shuffle(100)
		
		return dataset, doc_splitted_mapping_ids

	def fit(self, train_data_comb, val_data_comb, test_data_comb, training_epochs, shuffle=False, multilabel=False):
		train_data, train_data_TM = train_data_comb
		val_data, val_data_TM = val_data_comb
		test_data, test_data_TM = test_data_comb

		training_labels, training_docs = train_data
		validation_labels, validation_docs = val_data
		test_labels, test_docs = test_data

		train_dataset, \
		train_doc_splitted_mapping_ids = self.prepare_data( 
			training_docs, 
			labels=training_labels,
			shuffle=shuffle
		)

		val_dataset, \
		val_doc_splitted_mapping_ids = self.prepare_data(
			validation_docs, 
			labels=validation_labels,
			shuffle=False
		)

		test_dataset, \
		test_doc_splitted_mapping_ids = self.prepare_data(
			test_docs, 
			labels=test_labels,
			shuffle=False
		)

		if multilabel:
			loss_fn = tf.nn.sigmoid_cross_entropy_with_logits
		else:
			loss_fn = tf.nn.sparse_softmax_cross_entropy_with_logits
		
		# TRAINING
		# Iterate over epochs.
		logging.info("Num GPUs Available: {}".format(len(tf.config.experimental.list_physical_devices('GPU'))))

		loss_metric = tf.keras.metrics.Mean()
		optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
		
		best_val_f1 = -1.0
		best_test_f1 = -1.0
		for epoch in range(training_epochs):
			logging.info('Training epoch {}'.format(epoch))
			train_loss = self.run_train_epoch(train_dataset, train_data_TM, loss_fn=loss_fn, loss_metric=loss_metric, optimizer=optimizer)

			logging.info('Validation epoch {}'.format(epoch))
			val_logits, val_loss = self.run_eval_epoch(val_dataset, val_data_TM, loss_fn=loss_fn, loss_metric=loss_metric)
			val_logits = self.merge_logits(val_logits, val_doc_splitted_mapping_ids)
			val_preds = self.get_preds(val_logits, multilabel=multilabel)
			val_pre, val_rec, val_f1, _ = precision_recall_fscore_support(validation_labels, val_preds, average="macro")

			with open(os.path.join(self.log_dir, "training_summary.txt"), "a") as f:
				f.write('Training:      epoch = {},    loss = {}\n'.format(epoch, train_loss))
				f.write('Validation:    epoch = {},    loss = {},    Prec = {},    Rec = {},    F1 = {},    best F1 = {}\n\n'.format(epoch, val_loss, val_pre, val_rec, val_f1, best_val_f1))
			
			if val_f1 > best_val_f1:
				best_val_f1 = val_f1
				self.model.save_weights(os.path.join(self.model_dir, "best_checkpoint.ckpt"))

				logging.info('Test epoch {}'.format(epoch))
				test_logits, test_loss = self.run_eval_epoch(test_dataset, test_data_TM, loss_fn=loss_fn, loss_metric=loss_metric)
				test_logits = self.merge_logits(test_logits, test_doc_splitted_mapping_ids)
				test_preds = self.get_preds(test_logits, multilabel=multilabel)
				test_pre, test_rec, test_f1, _ = precision_recall_fscore_support(test_labels, test_preds, average="macro")

				if test_f1 > best_test_f1:
					best_test_f1 = test_f1

				with open(os.path.join(self.log_dir, "training_summary.txt"), "a") as f:
					f.write('Test:    epoch = {},    loss = {},    Prec = {},    Rec = {},    F1 = {},    best F1 = {}\n\n'.format(epoch, test_loss, test_pre, test_rec, test_f1, best_test_f1))

	def run_train_epoch(self, dataset, dataset_TM, loss_fn=None, loss_metric=None, optimizer=None):
		# Iterate over the batches of the training dataset.
		TM_train_set, TM_train_count = dataset_TM
		all_loss = []
		all_times = []
		epoch_start_time = time.time()
		for step, (x_batch_train, seq_len_batch_train, x_doc_ids, y_batch_train) in enumerate(dataset):
			x_batch_train_TM, \
			count_batch_train_TM, \
			mask_batch_train_TM = self.model_TM.fetch_data(TM_train_set, TM_train_count, x_doc_ids)
			
			x_batch_train_TM = tf.constant(x_batch_train_TM)
			mask_batch_train_TM = tf.constant(mask_batch_train_TM)
			count_batch_train_TM = tf.constant(count_batch_train_TM)
			
			start_time = time.time()
			with tf.GradientTape() as tape:
				# get CLS vector
				outputs = self.model(x_batch_train, training=True)
				o_CLS = outputs[0][:,0,:]
				# get h_TM vector
				h_TM, loss_TM = self.model_TM.forward(x_batch_train_TM, mask_batch_train_TM)
				# concatenation + projection
				h_concat = tf.concat([o_CLS, h_TM], axis=-1)
				h_p = self.projection(h_concat)
				# classification
				logits = self.classifier(h_p)
				# loss
				loss = tf.reduce_mean(loss_fn(y_batch_train, logits))
				joint_loss = self.alpha * loss + (1 - self.alpha) * loss_TM
				all_loss.append(joint_loss)
			grads = tape.gradient(joint_loss, self.trainable_params)
			optimizer.apply_gradients(zip(grads, self.trainable_params))
			end_time = time.time()
			all_times.append(end_time - start_time)
			
			if step and (step % 100 == 0):
				#logging.info('step {}: batch loss = {}'.format(step, loss))
				logging.info('step {}: batch loss = {}'.format(step, all_loss[-1]))
				logging.info('step {}: batch time = {},    avg batch time = {}'.format(step, all_times[-1], np.mean(all_times)))
		epoch_end_time = time.time()
		logging.info('Training data mean loss = {},    epoch time = {}'.format(np.mean(all_loss), (epoch_end_time - epoch_start_time)))
		return np.mean(all_loss)

	def run_eval_epoch(self, dataset, dataset_TM, loss_fn=None, loss_metric=None):
		# Iterate over the batches of the evaluation dataset.
		TM_eval_set, TM_eval_count = dataset_TM
		all_loss = []
		all_logits = []
		for step, (x_batch_eval, seq_len_batch_eval, x_doc_ids, y_batch_eval) in enumerate(dataset):
			x_batch_eval_TM, \
			count_batch_eval_TM, \
			mask_batch_eval_TM = self.model_TM.fetch_data(TM_eval_set, TM_eval_count, x_doc_ids)
			
			x_batch_eval_TM = tf.constant(x_batch_eval_TM)
			mask_batch_eval_TM = tf.constant(mask_batch_eval_TM)
			count_batch_eval_TM = tf.constant(count_batch_eval_TM)

			# get CLS vector
			outputs = self.model(x_batch_eval, training=False)
			o_CLS = outputs[0][:,0,:]
			# get h_TM vector
			h_TM, loss_TM = self.model_TM.forward(x_batch_eval_TM, mask_batch_eval_TM)
			# concatenation + projection
			h_concat = tf.concat([o_CLS, h_TM], axis=-1)
			h_p = self.projection(h_concat)
			# classification
			logits = self.classifier(h_p)
			# loss
			loss = tf.reduce_mean(loss_fn(y_batch_eval, logits))
			joint_loss = self.alpha * loss + (1 - self.alpha) * loss_TM
			all_loss.append(joint_loss.numpy())
			all_logits.append(logits.numpy())
		logging.info('Validation data mean loss = {}'.format(np.mean(all_loss)))
		return np.concatenate(all_logits, axis=0), np.mean(all_loss)

	def predict(self, docs_list, multilabel=False):
		dataset, \
		doc_splitted_mapping_ids = self.prepare_data( 
			docs_list, 
			labels=None,
			shuffle=False
		)
		
		# Iterate over the batches of the dataset.
		logits = []
		for step, (x_batch, seq_length_batch) in enumerate(dataset):
			outputs_batch = self.model(x_batch, training=False)
			logits_batch = outputs_batch[0]
			logits.append(logits_batch)
		logits = np.concatenate(logits, axis=0)
		merged_logits = self.merge_logits(logits, doc_splitted_mapping_ids)
		predictions = self.get_preds(merged_logits, multilabel=multilabel)
		return predictions