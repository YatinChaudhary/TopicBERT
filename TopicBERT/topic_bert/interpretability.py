import os
import sys
import modeling
import optimization
import tensorflow as tf
#from nvdm import model_NVDM as model_TM
#from nvdm import model_GSM as model_TM
from nvdm import data_lstm
from nvdm import utils
import logging
from collections import Counter
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _linear as linear
import numpy as np 
import collections
import tokenization
import json
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import pickle
import random
import time

from tfdeterminism import patch
patch()
seed = 42

tf.set_random_seed(seed)
np.random.seed(seed)
tf_op_seed = 42

flags = tf.flags
FLAGS = flags.FLAGS



flags.DEFINE_string(
	"bert_config_file", "./cased_L-12_H-768_A-12/bert_config.json",
	"The config json file corresponding to the pre-trained BERT model."
	"This specifies the model architecture.")

flags.DEFINE_string(
	"bert_vocab_file", "./cased_L-12_H-768_A-12/vocab.txt",
	"the vocabulary file of the bert model")

flags.DEFINE_string(
	"data_dir", None,
	"Data directory where training/validation/test tfrecord files are stored")

flags.DEFINE_bool(
	"propD", False,
	"whether given dataset is propaganda detection shared task")

flags.DEFINE_float(
	"prob_thres", 0.5,
	"threshhold applied for prediction if propaganda detection task")

flags.DEFINE_string(
		"output_dir", None,
		"The output directory where the model checkpoints will be written.")

flags.DEFINE_integer(
		"max_seq_length", 128,
		"The maximum total input sequence length after WordPiece tokenization. "
		"Sequences longer than this will be truncated, and sequences shorter "
		"than this will be padded. Must match data generation.")

## Other parameters
flags.DEFINE_string(
		"init_checkpoint", "./cased_L-12_H-768_A-12/bert_model.ckpt",
		"Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string(
		"pretrain_nvdm_path", None, "path of the pretrained nvdm_only")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 1, "Total batch size for eval.")

flags.DEFINE_integer("num_labels", 0, "Total number of labels")

#flags.DEFINE_integer("test_batch_size", 1, "Total batch size for test.")

flags.DEFINE_integer("num_cores", 8, "the number of CPU cores to use")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float(
		"warmup_proportion", 0.1,
		"Proportion of training to perform linear learning rate warmup for. "
		"E.g., 0.1 = 10% of training.")

flags.DEFINE_integer(
		"patience", 10,
		"training stops if eval loss increases after these num of epochs ")

flags.DEFINE_float("save_after_epoch", 1,
									 "save and validate after ever save_after_every epoch")

flags.DEFINE_float("num_train_epochs", 3.0,
									 "Total number of training epochs to perform.")

flags.DEFINE_bool("concat", False, "projection after hidden layer concatenation")
flags.DEFINE_bool("projection", False, "projection after hidden layer concatenation")

flags.DEFINE_bool("gating", False, "gating after hidden layer concatenation")

flags.DEFINE_integer("hidden_size_TM", 256, "hidden size for TM model")

flags.DEFINE_integer("n_topic", 150, "number of topics")

flags.DEFINE_integer("n_sample", 1, "number of sample")


flags.DEFINE_integer("proj_size", 100, "projection after concatenation")

flags.DEFINE_float("alpha", 1.0, "fraction of language model to be added")
flags.DEFINE_float("gsm_lr_factor", 1.0 , "factor to be multiplied with gsm learning rate")

flags.DEFINE_bool("finetune_bert", False, "weather to pretrain bert")

flags.DEFINE_bool("use_static_topic", False, "weather to add static topics while finetuning")

flags.DEFINE_string("static_topic_path", "", "Path of the static topic proportions to be used")

flags.DEFINE_bool("load_wiki_during_pretrain", True, "during pretraining load wikipedia model")


flags.DEFINE_bool("pretrain_nvdm", False, "weather to pretrain nvdm")

flags.DEFINE_bool("combined_train", False, "weather to combined_train")
flags.DEFINE_bool("sparse_topics", False, "select topics with higher values before combine")

flags.DEFINE_integer("nvdm_alternate_epoch", 10, "epochs for alternate optimisation")
flags.DEFINE_integer("nvdm_train_epoch", 30, "epoch for train optimisation")

flags.DEFINE_integer("nvdm_validation_ppl_freq", 1, "validate for ppl after this many epochs")

flags.DEFINE_integer("nvdm_validation_f1_freq", 1, "validate for F1 after this many epochs, used in supervised TM models only")

flags.DEFINE_integer("nvdm_validation_ir_freq", 100000000, "validate for ir after this many epochs")

flags.DEFINE_float("nvdm_learning_rate", 5e-5, "The initial learning rate for Adam. for nvdm model")

flags.DEFINE_integer("nvdm_batch_size", 64, "The initial learning rate for Adam. for nvdm model")

flags.DEFINE_integer("nvdm_patience", 10, "patience for pretraining nvdm")


flags.DEFINE_bool("multilabel", False, "wheather the labels are multi label")

flags.DEFINE_integer("nvdm_num_class",  2 , "number of class in the label")

flags.DEFINE_bool("use_gpu", True, "wheather to use GPU or not") 


flags.DEFINE_string("topic_model", "gsm", "wheather to use gsm or nvdm or not")

flags.DEFINE_bool("supervised_TM", False, "if supervised TM is to be used")

flags.DEFINE_bool("pretrain_supervised_TM", False, "pretrain a supervised TM")

flags.DEFINE_string("validate_supervised_TM", "ppl", "validate supervised TM on 'ppl' or 'f1's")

flags.DEFINE_float("beta", 0.1, "TM : fraction of supervised loss in final TM loss")

flags.DEFINE_bool("max_softmax", False, "take the max softmax of the predictions")
flags.DEFINE_bool("avg_softmax", False, "take the average softmax of the prediction")



if FLAGS.use_gpu:
	os.environ['CUDA_VISIBLE_DEVICES'] = '0'
else:
	os.environ['CUDA_VISIBLE_DEVICES'] = '1'

if FLAGS.topic_model == "nvdm":

	if FLAGS.supervised_TM or FLAGS.pretrain_supervised_TM:
		from nvdm import model_NVDM_supervised as model_TM
	else:
		from nvdm import model_NVDM as model_TM

elif FLAGS.topic_model == "gsm":

	if FLAGS.supervised_TM or FLAGS.pretrain_supervised_TM:
		from nvdm import model_GSM_supervised as model_TM
	else:
		from nvdm import model_GSM as model_TM

else:
	print("topic model not found")
	sys.exit()

def batch_generator(filename, topic_vocab_size  , is_training):

	def decode(serialized_example):

		if FLAGS.multilabel:
			label_size = FLAGS.num_labels
		else:
			label_size = 1

		features = tf.parse_single_example( serialized_example, features={
			
			"input_ids":
					tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
			
			"input_mask":
					tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),

			"segment_ids":
					tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),

			"label_ids":
					tf.FixedLenFeature([label_size], tf.int64),

			"doc_id":
					tf.FixedLenFeature([1], tf.int64),		
		
			})
		

		return features["input_ids"], features["input_mask"], features["segment_ids"], features["label_ids"], features["doc_id"]
		

	if is_training:
		dataset = tf.data.TFRecordDataset(filename)
		dataset = dataset.map(decode)
		dataset = dataset.shuffle(buffer_size=100, seed = tf_op_seed)
		dataset = dataset.repeat()
		dataset = dataset.batch(FLAGS.train_batch_size)
		iterator = dataset.make_one_shot_iterator()

	else:
		dataset = tf.data.TFRecordDataset(filename)
		dataset = dataset.map(decode)
		#dataset = dataset.shuffle(buffer_size=100, seed = tf_op_seed )
		#dataset = dataset.batch(FLAGS.eval_batch_size)
		dataset = dataset.batch(FLAGS.eval_batch_size)
		iterator = dataset.make_one_shot_iterator()
		### use end of sequence to end the loop for validation and test 
	return iterator


def get_topic_vocab(data_dir):
	with open(os.path.join(data_dir, "vocab_docnade.vocab")) as f:
		vocab = f.readlines()
		vocab = [word.strip() for word in vocab]
	return vocab

def extract_topic_data(doc_ids, data_set, data_doc_ids, topic_vocab_size):

	data_set_dict = {}
	for i in range(len(data_set)):
		data_set_dict[str(data_doc_ids[i])] = data_set[i]

	topic_bow = []
	topic_mask = []
	for doc_id in doc_ids:
		bow = np.zeros(topic_vocab_size, dtype = np.float32)

		if str(doc_id) in data_set_dict:
			bow_dict = data_set_dict[str(doc_id)]
			for key, value in bow_dict.items():
				bow[key] = value
			topic_bow.append(bow)
			topic_mask.append(1.0)

		else:
			topic_bow.append(bow)
			topic_mask.append(0.0)

	topic_bow = np.asarray(topic_bow)
	topic_mask = np.asarray(topic_mask)

	return topic_bow, topic_mask

def extract_topics(topic_dist, topic_matrix, labels, preds , input_dir , output_dir, num_topic, name_string = ""):

	assert len(labels) == len(topic_dist) == len(preds)

	with open(os.path.join(input_dir, "label_dict.pkl"), "rb" ) as handle:
		label_maps = pickle.load(handle)

	inverted_label_maps  = {value: key for key, value in label_maps.items()}


	topic_file = os.path.join(output_dir, "topics_top_20_words.txt")
	best_topic_log = os.path.join(output_dir, "best_topic_log_"+ str(name_string) + ".txt")

	topic_vocab_path  = os.path.join(input_dir, "vocab_docnade.vocab")
	with open(topic_vocab_path, "r")  as f:
		lines = f.readlines()
	topic_vocab = [word.strip() for word in lines]

	sorted_mat = np.argsort(topic_matrix, axis = 1)[:,::-1]

	with open(topic_file, "w") as f:
		for topic_vec in sorted_mat:
			topic_words = []
			top_ids = topic_vec[:num_topic]
			for id in top_ids:
				topic_words.append(topic_vocab[id])
			f.write(" ".join(topic_words) + "\n")
			f.write("\n")


	with open(best_topic_log, "w") as f:
		#for doc_num, dist in enumerate(topic_dist):
		for doc_num, (doc_id, dist) in enumerate(topic_dist.items()):
			sorted_topic_dist = np.argsort(dist, axis = 0)[::-1]
			top_3_topic_ids  = sorted_topic_dist[:5]

			top_topics = sorted_mat[top_3_topic_ids][:, :num_topic]

			top_topic_words = []
			for topic in top_topics:
				topic_words = []
				for id in topic:
					topic_words.append(topic_vocab[id])
				top_topic_words.append(topic_words)

			#doc_id = str(doc_num)
			dist = [str(value) for value in dist]
			topic_dist_str = " ".join(dist)

			if FLAGS.multilabel:
				label_str = []
				for label_num, l in enumerate(labels[doc_num]):
					if l != 0:
						label_str.append(inverted_label_maps[label_num])

				label =":".join(label_str)


				pred_str = []
				for pred_num, l in enumerate(preds[doc_num]):
					if l != 0:
						pred_str.append(inverted_label_maps[pred_num])

				prediction =":".join(pred_str)

			else:
				label = inverted_label_maps[int(labels[doc_num])]

				prediction = inverted_label_maps[int(preds[doc_num])]

			f.write("Doc id: " + str(doc_id) + "\n") 
			f.write("Topic_distribution: " + str(topic_dist_str) + "\n")
			f.write("Doc label: " + str(label)  + "\n")
			f.write("Doc pred: " + str(prediction)  + "\n")
			for num, topic_words in enumerate(top_topic_words):
				f.write("Best topic " + str(top_3_topic_ids[num]) + ": " + " ".join(topic_words) + "(" + str(dist[top_3_topic_ids[num]]) +  ")" + "\n") 
			f.write("\n")
	print("Topics extracted!")


def predict_softmax(pred_prob, unique_ids):

	pred_cands = []
	for id, idx_list in unique_ids.items():
		pred_repeated = []
		for idx in idx_list:
			try:
				pred_repeated.append(pred_prob[idx])
			except:
				import pdb; pdb.set_trace()

		pred_cands.append(pred_repeated)

	assert len(pred_cands) == len(unique_ids)

	post_pred_labels = []
	pred_prob_docwise  = []
	for cand in pred_cands:
		pred_probs_per_cand  = np.array(cand)

		if FLAGS.multilabel:
			if FLAGS.avg_softmax: 
				post_pred_labels.append(list(np.where(np.mean(pred_probs_per_cand, axis = 0) >= 0.5, 1, 0)))

			elif FLAGS.max_softmax:
				## same as avg softmax for the case of multilabel.
				post_pred_labels.append(list(np.where(np.mean(pred_probs_per_cand, axis = 0) >= 0.5, 1, 0)))

		else:
			if FLAGS.avg_softmax: 
				post_pred_labels.append(np.argmax(np.mean(pred_probs_per_cand, axis = 0)))
				pred_prob_docwise.append(np.mean(pred_probs_per_cand, axis = 0))

			elif FLAGS.max_softmax:
				index_max_softmax = np.argmax(np.amax(pred_probs_per_cand, axis = 1))
				post_pred_labels.append(np.argmax(pred_probs_per_cand, axis = 1)[index_max_softmax])

	assert len(post_pred_labels) == len(unique_ids)

	return post_pred_labels, pred_prob_docwise



def postprocess_pred(true_label, pred_label, doc_ids, pred_prob = None):
	
	unique_doc_ids = list(set(doc_ids))
	
	cnt_doc_ids = Counter(doc_ids)
	non_unique = {}
	for key, value in  cnt_doc_ids.items():
		if value > 1:
			non_unique[key] = value

	if len(doc_ids) == len(unique_doc_ids):
		print("all documents are unique!")
		return true_label, pred_label



	print("Repeated doc found")

	unique_ids = {}
	for num, id in enumerate(doc_ids):
		if id not in unique_ids:
			unique_ids[id] = [num]
		else:
			unique_ids[id].append(num)

	pred_cands = []
	true_cands = []
	for id, idx_list in unique_ids.items():
		pred_repeated = []
		true_repeated = []
		for idx in idx_list:
			pred_repeated.append(pred_label[idx])
			true_repeated.append(true_label[idx])

		pred_cands.append(pred_repeated)
		true_cands.append(true_repeated)

	assert len(true_cands) == len(pred_cands) == len(unique_doc_ids)
	
	post_true_labels = []
	post_pred_labels = []
	for true_cand, pred_cand, doc_id in zip(true_cands, pred_cands, unique_doc_ids):

		if len(pred_cand) > 1:
			if FLAGS.multilabel:
				for cand in true_cand:
					assert cand == true_cand[0]
				
				true_cand_class  = true_cand[0]
				post_pred_class = [0]*len(true_cand_class)
				
				
				majority = int(np.ceil(0.5*len(pred_cand)))
				pred_cand_array = np.asarray(pred_cand)
				pred_cand_maj = pred_cand_array.sum(axis = 0)
				for i in range(len(pred_cand_maj)):
					if pred_cand_maj[i] >= majority:
						post_pred_class[i] = 1
					else:
						post_pred_class[i] = 0

				
				"""
				for j in range(len(true_cand_class)):
					class_post_processed = False
					for i in range(len(pred_cand)):
						if pred_cand[i][j] == true_cand_class[j]:
							post_pred_class[j] = true_cand_class[j]
							class_post_processed = True

						if class_post_processed == False:
							post_pred_class[j] = pred_cand[0][j] 	
				"""	

				post_true_labels.append(true_cand_class)
				post_pred_labels.append(post_pred_class)

			else:
				assert all(x==true_cand[0] for x in true_cand)

				if true_cand[0] in pred_cand:
					post_true_labels.append(true_cand[0])
					post_pred_labels.append(true_cand[0])
			
				else:
					post_true_labels.append(true_cand[0])
					post_pred_labels.append(pred_cand[0])

		else:
			post_true_labels.append(true_cand[0])
			post_pred_labels.append(pred_cand[0])

	assert len(post_true_labels) == len(post_pred_labels) == len(unique_doc_ids)

	if FLAGS.avg_softmax or FLAGS.max_softmax:
		post_pred_labels, pred_probs_docwise = predict_softmax(pred_prob, unique_ids)

	return post_true_labels, post_pred_labels, pred_probs_docwise

def save_to_s3():
	pass

def extract_static_topics(doc_ids_batch, topic_dist_dict):

	topic_dist_batch = np.zeros((len(doc_ids_batch), FLAGS.n_topic), dtype = np.float32)
	for num, id in enumerate(doc_ids_batch):
		if str(id) in topic_dist_dict:
			topic_dist_batch[num] = topic_dist_dict[str(id)]

	return topic_dist_batch

def sparsify_topics(doc_repres):

	min_allowed_val = 0.5*tf.math.reduce_max(doc_repres, axis = 1)
	min_allowed_val = tf.expand_dims(min_allowed_val, axis = 1)
	min_allowed = tf.tile(min_allowed_val, [1, FLAGS.n_topic] )
	boolean_allowed = tf.cast(tf.math.greater_equal(doc_repres, min_allowed), dtype = tf.float32)

	doc_repres_allowed =  doc_repres * boolean_allowed

	return doc_repres_allowed


class joint_model(object):
	def __init__(self, bert_config ,input_ids, input_mask, segment_ids, label_ids, doc_ids , topic_bow, topic_mask, topic_vocab_size, is_training, num_train, num_labels, static_topic = None):

		self.input_ids = input_ids
		self.input_mask = input_mask
		self.segment_ids = segment_ids
		self.topic_bow = topic_bow
		self.topic_mask = topic_mask
		self.is_training = is_training
		self.label_ids = label_ids
		self.doc_ids = doc_ids

		if FLAGS.use_static_topic:
			self.static_topic = static_topic

		optimizer_var_exist = []

		num_train_steps = int((num_train / FLAGS.train_batch_size) * FLAGS.num_train_epochs)
		num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)


		##language model 
		bert_model = modeling.BertModel(
				config=bert_config,
				is_training=is_training,
				input_ids=input_ids,
				input_mask=input_mask,
				token_type_ids=segment_ids,
				use_one_hot_embeddings=False)

		sequence_layer = bert_model.get_sequence_output()
		output_layer = bert_model.get_pooled_output()
		hidden_size = output_layer.shape[-1].value

		self.prob_1 = bert_model.prob_1
		self.prob_2 = bert_model.prob_2

		output_layer = tf.nn.dropout(output_layer, keep_prob= 1-is_training, seed=tf_op_seed)

		with tf.variable_scope("bert_loss"):
	
			### adding static embeddings while finetuning bert
			if FLAGS.use_static_topic:
				if FLAGS.concat:
					joint_output_layer = tf.concat([output_layer, self.static_topic], axis = 1)
					
					if FLAGS.projection:
						output_layer = tf.layers.dense(
								joint_output_layer,
								units=hidden_size,
								activation=modeling.get_activation(bert_config.hidden_act),
								kernel_initializer=modeling.create_initializer(bert_config.initializer_range))


			#####################################################			


			"""
			output_weights = tf.get_variable(
				"output_weights", [num_labels, hidden_size],
				initializer=tf.truncated_normal_initializer(stddev=0.02, seed = tf_op_seed))

			output_bias = tf.get_variable(
				"output_bias", [num_labels], initializer=tf.zeros_initializer())		

			logits = tf.matmul(output_layer, output_weights, transpose_b=True)
			self.bert_logits = tf.nn.bias_add(logits, output_bias)

			"""

			### wx+b ###
			output_weights = tf.get_variable(
			"output_weights", [hidden_size, num_labels],
			initializer=tf.truncated_normal_initializer(stddev=0.02, seed = tf_op_seed))

			output_bias = tf.get_variable(
				"output_bias", [num_labels], initializer=tf.zeros_initializer())

			self.bert_logits  = tf.nn.xw_plus_b(output_layer, output_weights, output_bias)
			####


			#######
			if FLAGS.multilabel:
				self.bert_probabilities = tf.nn.sigmoid(self.bert_logits)
				self.bert_per_example_loss = tf.reduce_sum(tf.losses.sigmoid_cross_entropy(label_ids, self.bert_logits, reduction="none"), axis=-1)	
				
			#########

			else:
				self.bert_probabilities = tf.nn.softmax(self.bert_logits, axis=-1)
				self.log_probs = tf.nn.log_softmax(self.bert_logits, axis=-1)

				self.one_hot_labels = tf.one_hot(label_ids, depth=num_labels, dtype=tf.float32)

				self.bert_per_example_loss = -tf.reduce_sum(self.one_hot_labels * self.log_probs, axis=-1)
			self.bert_obj_loss = tf.reduce_mean(self.bert_per_example_loss)

			self.bert_train_op, _, _ = optimization.create_optimizer(
						self.bert_obj_loss, FLAGS.learning_rate , num_train_steps, num_warmup_steps,  optimizer_var_exist, use_tpu = False)



		#with tf.variable_scope("combined_loss"):
			##topic model
		if FLAGS.supervised_TM or FLAGS.pretrain_supervised_TM:
			self.topic_model = model_TM.NVDM(
					topic_params = FLAGS,
					x = topic_bow,
					mask = topic_mask,
					topic_vocab_size = topic_vocab_size,
					label_ids = label_ids,
					n_labels = num_labels,
					initializer_nvdm = [[None]*2]*4
					)


		else:
			self.topic_model = model_TM.NVDM(
					topic_params = FLAGS,
					x = topic_bow,
					mask = topic_mask,
					topic_vocab_size = topic_vocab_size,
					initializer_nvdm = [[None]*2]*4
					)

		self.topic_loss = self.topic_model.final_loss 
		self.doc_repres = self.topic_model.doc_vec

		
		if FLAGS.sparse_topics:
			#self.check_doc_repres = self.doc_repres 
			self.doc_repres = sparsify_topics(self.doc_repres)

		if FLAGS.concat:
			joint_output_layer = tf.concat([output_layer, self.doc_repres], axis = 1)
			
			if FLAGS.projection:
				joint_output_layer = tf.layers.dense(
						joint_output_layer,
						units=hidden_size,
						activation=modeling.get_activation(bert_config.hidden_act),
						kernel_initializer=modeling.create_initializer(bert_config.initializer_range))

		"""
		output_weights = tf.get_variable(
		"output_weights", [num_labels, hidden_size],
		initializer=tf.truncated_normal_initializer(stddev=0.02, seed = tf_op_seed))

		output_bias = tf.get_variable(
			"output_bias", [num_labels], initializer=tf.zeros_initializer())

		logits = tf.matmul(joint_output_layer, output_weights, transpose_b=True)

		self.comb_logits = tf.nn.bias_add(logits, output_bias)

		"""

		### wx+b  ###
		output_weights = tf.get_variable(
		"output_weights", [hidden_size, num_labels],
		initializer=tf.truncated_normal_initializer(stddev=0.02, seed = tf_op_seed))

		output_bias = tf.get_variable(
			"output_bias", [num_labels], initializer=tf.zeros_initializer())

		self.comb_logits = tf.nn.xw_plus_b(joint_output_layer, output_weights, output_bias )
		####

		if FLAGS.multilabel:
				self.comb_probabilities = tf.nn.sigmoid(self.comb_logits)
				self.comb_per_example_loss = tf.reduce_sum(tf.losses.sigmoid_cross_entropy(label_ids, self.comb_logits, reduction="none"), axis=-1)	

		else:
			self.comb_probabilities = tf.nn.softmax(self.comb_logits, axis=-1)
			self.comb_log_probs = tf.nn.log_softmax(self.comb_logits, axis=-1)

			self.one_hot_labels = tf.one_hot(label_ids, depth=num_labels, dtype=tf.float32)

			self.comb_per_example_loss = -tf.reduce_sum(self.one_hot_labels * self.comb_log_probs, axis=-1)

		self.comb_lm_loss = tf.reduce_mean(self.comb_per_example_loss)
		self.topic_loss = tf.reduce_mean(self.topic_loss)

		self.comb_obj_loss = tf.add( (1.0-FLAGS.alpha)*self.comb_lm_loss, FLAGS.alpha*(FLAGS.gsm_lr_factor)*self.topic_loss)

		self.combined_train_op , self.comb_global_step, self.comb_lr = optimization.create_optimizer(
					self.comb_obj_loss, FLAGS.learning_rate , num_train_steps, num_warmup_steps,  optimizer_var_exist, use_tpu = False)


def run_evaluation_bert(model, session, file, topic_vocab_size, topic_dist_path=None ):

	batch_iter = batch_generator(file, topic_vocab_size , is_training=False)
	input_ids, input_mask, segment_ids, label_ids, doc_ids  = batch_iter.get_next()

	pred_label_list = []
	true_label_list = []
	loss_list = []
	val_inst_count = 0
	doc_ids_list = []
	prob_list = []
	while True:
		try:
			input_ids_batch, input_mask_batch, segment_ids_batch, label_ids_batch, doc_ids_batch  = session.run([input_ids, input_mask, segment_ids,  label_ids, doc_ids ])

			if not FLAGS.multilabel:
				label_ids_batch = np.squeeze(label_ids_batch, axis = 1 )
			
				
			doc_ids_batch =  np.squeeze(doc_ids_batch, axis = 1 )


			if FLAGS.use_static_topic:
				topic_dist_dict = np.load(topic_dist_path, allow_pickle = True).item()
				static_topic_batch = extract_static_topics(doc_ids_batch, topic_dist_dict)

				input_feed = {model.input_ids:input_ids_batch, 
							model.input_mask:input_mask_batch, 
							model.segment_ids:segment_ids_batch, 
							model.label_ids:label_ids_batch,  
							model.doc_ids:doc_ids_batch,
							model.static_topic: static_topic_batch,
							model.is_training:0.0}


			else:
				input_feed = {model.input_ids:input_ids_batch, 
							model.input_mask:input_mask_batch, 
							model.segment_ids:segment_ids_batch, 
							model.label_ids:label_ids_batch,  
							model.doc_ids:doc_ids_batch, 
							model.is_training:0.0}

			prob, bert_per_example_loss = session.run([model.bert_probabilities, model.bert_per_example_loss], feed_dict = input_feed )

			prob_list.extend(prob)

			if FLAGS.multilabel:
				prob_arr  = np.asarray(prob)
				multilabel_pred = np.where(prob_arr >= 0.5, 1, 0)
				pred_label_list.extend(np.ndarray.tolist(multilabel_pred))
				true_label_list.extend(np.ndarray.tolist(label_ids_batch))

			else:
				if FLAGS.propD: 
					pred_label_list.extend(list(np.array((np.array(prob, dtype = np.float)[:, 1]  > FLAGS.prob_thres), dtype = np.int)))
				else:
					pred_label_list.extend(list(np.argmax(prob, axis = 1)))

				true_label_list.extend(list(label_ids_batch))

			doc_ids_list.extend(list(doc_ids_batch))
			loss_list.extend(bert_per_example_loss)
				
			val_inst_count += 1
			print("Validation: val instance count: " + str(val_inst_count))

			
		except tf.errors.OutOfRangeError:
			print("Validation End of sequence reached !")
			break


	if not FLAGS.propD:
		true_label_list, pred_label_list, pred_probs_docwise =  postprocess_pred(true_label_list, pred_label_list, doc_ids_list , pred_prob = prob_list)

		if FLAGS.multilabel:
			true_label_list = np.asarray(true_label_list)
			pred_label_list = np.asarray(pred_label_list)

	macro_prec, macro_recall, macro_f1_score, _  = precision_recall_fscore_support(true_label_list, pred_label_list,  average = "macro")
	micro_prec, micro_recall, micro_f1_score, _  = precision_recall_fscore_support(true_label_list, pred_label_list,  average = "micro")
	acc = accuracy_score(true_label_list, pred_label_list)
	loss = np.mean(np.asarray(loss_list))

	return macro_prec, macro_recall, macro_f1_score, micro_prec, micro_recall, micro_f1_score, acc, loss, pred_label_list, pred_probs_docwise
				
def run_evaluation_comb(model, session, file, topic_vocab_size , TM_dataset, TM_batch, TM_count, TM_doc_ids,  TM_labels = None):

	batch_iter = batch_generator(file, topic_vocab_size , is_training=False)
	input_ids, input_mask, segment_ids, label_ids, doc_ids  = batch_iter.get_next()

	pred_label_list = []
	true_label_list = []
	loss_list = []
	val_inst_count = 0
	doc_ids_list = []
	prob_list = []
	while True:
		try:
			input_ids_batch, input_mask_batch, segment_ids_batch, label_ids_batch, doc_ids_batch  = session.run([input_ids, input_mask, segment_ids,  label_ids, doc_ids ])
			if not FLAGS.multilabel:
				label_ids_batch = np.squeeze(label_ids_batch, axis = 1 )
			doc_ids_batch =  np.squeeze(doc_ids_batch, axis = 1 )

			topic_bow_batch, topic_mask_batch = extract_topic_data(doc_ids_batch, TM_dataset, TM_doc_ids, topic_vocab_size)

			input_feed = {model.input_ids:input_ids_batch, 
						model.input_mask:input_mask_batch, 
						model.segment_ids:segment_ids_batch, 
						model.label_ids:label_ids_batch,  
						model.doc_ids:doc_ids_batch, 
						model.topic_bow:topic_bow_batch, 
						model.topic_mask:topic_mask_batch,
						model.is_training:0.0}

			prob, comb_per_example_loss = session.run([model.comb_probabilities, model.comb_per_example_loss], feed_dict = input_feed )

			prob_list.extend(prob)

			if FLAGS.multilabel:
				prob_arr  = np.asarray(prob)
				multilabel_pred = np.where(prob_arr >= 0.5, 1, 0)
				pred_label_list.extend(np.ndarray.tolist(multilabel_pred))
				true_label_list.extend(np.ndarray.tolist(label_ids_batch))

			else:	
				if FLAGS.propD: 
					pred_label_list.extend(list(np.array((np.array(prob, dtype = np.float)[:, 1]  > FLAGS.prob_thres), dtype = np.int)))
				else:
					pred_label_list.extend(list(np.argmax(prob, axis = 1)))
					
				true_label_list.extend(list(label_ids_batch))

			doc_ids_list.extend(list(doc_ids_batch))
			loss_list.extend(comb_per_example_loss)
				
			val_inst_count += 1
			print("Validation: val instance count: " + str(val_inst_count))

			
		except tf.errors.OutOfRangeError:
			print("Validation End of sequence reached !")
			break

	if not FLAGS.propD:
		true_label_list, pred_label_list, pred_prob_docwise =  postprocess_pred(true_label_list, pred_label_list, doc_ids_list, pred_prob = prob_list )

		if FLAGS.multilabel:
			true_label_list = np.asarray(true_label_list)
			pred_label_list = np.asarray(pred_label_list)

	macro_prec, macro_recall, macro_f1_score, _  = precision_recall_fscore_support(true_label_list, pred_label_list,  average = "macro")
	micro_prec, micro_recall, micro_f1_score, _  = precision_recall_fscore_support(true_label_list, pred_label_list,  average = "micro")
	acc = accuracy_score(true_label_list, pred_label_list)
	loss = np.mean(np.asarray(loss_list))

	if FLAGS.supervised_TM:
		topic_ppx, topic_ppx_perdoc, topic_kld, topic_sup_loss,  topic_macro_prec, topic_macro_recall, topic_macro_f1_score, topic_acc = \
		model.topic_model.run_epoch( TM_batch, TM_dataset,  TM_count,  FLAGS, session,  input_labels = TM_labels)

	else:
		topic_ppx, topic_ppx_perdoc, topic_kld = model.topic_model.run_epoch( TM_batch, TM_dataset,  TM_count,  FLAGS, session)

	return macro_prec, macro_recall, macro_f1_score, micro_prec, micro_recall, micro_f1_score, acc, loss, topic_ppx, topic_ppx_perdoc, topic_kld, pred_label_list, pred_prob_docwise



def interpret_TP(model, bert_config,  train_file, validation_file, test_file, topic_vocab_size , num_train , session):

	
	#### evaluationg finetune bert model #####		
	log_dir = os.path.join(FLAGS.output_dir , 'logs')
	if not os.path.isdir(log_dir):
		os.mkdir(os.path.join(log_dir))
	
	tf.local_variables_initializer().run()
	tf.global_variables_initializer().run()
	saver_PPL = tf.train.Saver(tf.global_variables())

	path_name = "finetuned_bert_model"
	saver_PPL.restore(session,  str(FLAGS.output_dir) + "/" + path_name  + "/" + path_name + "-1" )
	print("Best bert Model restored.")


	_, _, _, _, _, _, _ , _ , bert_train_preds, bert_train_probs = run_evaluation_bert(model, session, train_file, topic_vocab_size, topic_dist_path= False)

	np.save(os.path.join(FLAGS.output_dir, "bert_train_preds.npy" ), bert_train_preds , allow_pickle = True )
	np.save(os.path.join(FLAGS.output_dir, "bert_train_probs.npy" ), bert_train_probs , allow_pickle = True )


	_, _, _, _, _, _, _ , _ , bert_val_preds, bert_val_probs = run_evaluation_bert(model, session, validation_file, topic_vocab_size, topic_dist_path= False)

	np.save(os.path.join(FLAGS.output_dir, "bert_val_preds.npy" ), bert_val_preds , allow_pickle = True )
	np.save(os.path.join(FLAGS.output_dir, "bert_val_probs.npy" ), bert_val_probs , allow_pickle = True )

	_, _, bert_test_f1_score, _, _, _, _, _, bert_test_preds, bert_test_probs = run_evaluation_bert(model, session, test_file, topic_vocab_size, topic_dist_path= None)
	print("Bert Test F1 score: " + str(bert_test_f1_score))

	np.save(os.path.join(FLAGS.output_dir, "bert_test_preds.npy" ), bert_test_preds , allow_pickle = True )
	np.save(os.path.join(FLAGS.output_dir, "bert_test_probs.npy" ), bert_test_probs , allow_pickle = True )


	##### evaluating combined model #######
	comb_model_name = 'combined_model' 
	saver_PPL = tf.train.Saver(tf.global_variables())
	
	nvdm_datadir = FLAGS.data_dir

	train_url = os.path.join(nvdm_datadir, 'training_nvdm_docs_non_replicated.csv')
	train_set, train_count, train_labels, train_doc_ids = utils.data_set(train_url, FLAGS)
	train_batches = utils.create_batches(len(train_set), FLAGS.nvdm_batch_size, shuffle=False)

	dev_url = os.path.join(nvdm_datadir, 'validation_nvdm_docs_non_replicated.csv')
	dev_set, dev_count, dev_labels, dev_doc_ids = utils.data_set(dev_url, FLAGS)
	dev_batches = utils.create_batches(len(dev_set), FLAGS.nvdm_batch_size, shuffle=False)

	test_url = os.path.join(nvdm_datadir, 'test_nvdm_docs_non_replicated.csv')
	test_set, test_count, test_labels, test_doc_ids = utils.data_set(test_url, FLAGS)
	test_batches = utils.create_batches(len(test_set), FLAGS.nvdm_batch_size, shuffle=False)
	
	
	path_name = "combined_model"

	if os.path.isdir(os.path.join(FLAGS.output_dir, path_name)):

		saver_PPL.restore(session,  str(FLAGS.output_dir) + "/" + path_name  + "/" + path_name + "-1" )
		print("Best combined Model restored.")


		topic_matrix = session.run(["TM_decoder/projection/Matrix:0"])

		train_topic_dist, train_mask_list = model.topic_model.topic_dist(train_batches, train_set, train_doc_ids,  train_count,  FLAGS, session) 
		val_topic_dist, val_mask_list = model.topic_model.topic_dist(dev_batches, dev_set, dev_doc_ids,  dev_count,  FLAGS, session) 
		test_topic_dist, test_mask_list = model.topic_model.topic_dist(test_batches, test_set, test_doc_ids , test_count,  FLAGS, session) 
		
		##save the matrix for logistic
		np.save(os.path.join(FLAGS.output_dir, "topic_matrix.npy" ),  topic_matrix, allow_pickle = True )
		np.save(os.path.join(FLAGS.output_dir, "train_topic_dist.npy" ),  train_topic_dist, allow_pickle = True )
		np.save(os.path.join(FLAGS.output_dir, "val_topic_dist.npy" ),  val_topic_dist, allow_pickle = True )
		np.save(os.path.join(FLAGS.output_dir, "test_topic_dist.npy" ),  test_topic_dist, allow_pickle = True )

		## Evaluation validation data ####

		_, _, comb_train_f1_score, _, _, _, _, _, _, _, _, comb_train_preds, comb_train_probs = run_evaluation_comb(model, session, train_file, topic_vocab_size , train_set, train_batches, train_count, train_doc_ids, TM_labels = train_labels)

		np.save(os.path.join(FLAGS.output_dir, "comb_train_preds.npy" ), comb_train_preds , allow_pickle = True )
		np.save(os.path.join(FLAGS.output_dir, "comb_train_probs.npy" ), comb_train_probs , allow_pickle = True )

		_, _, comb_val_f1_score, _, _, _, _, _, _, _, _, comb_val_preds, comb_val_probs = run_evaluation_comb(model, session, validation_file, topic_vocab_size , dev_set, dev_batches, dev_count, dev_doc_ids, TM_labels = dev_labels)

		np.save(os.path.join(FLAGS.output_dir, "comb_val_preds.npy" ), comb_val_preds , allow_pickle = True )
		np.save(os.path.join(FLAGS.output_dir, "comb_val_probs.npy" ), comb_val_probs , allow_pickle = True )

		#### running test evaluations 
		_, _, comb_test_f1_score, _, _, _, _, _, _, _, _, comb_test_preds, comb_test_probs = run_evaluation_comb(model, session, test_file, topic_vocab_size,test_set, test_batches, test_count, test_doc_ids, TM_labels = test_labels)

		np.save(os.path.join(FLAGS.output_dir, "comb_test_preds.npy" ), comb_test_preds , allow_pickle = True )
		np.save(os.path.join(FLAGS.output_dir, "comb_test_probs.npy" ), comb_test_probs , allow_pickle = True )

		print("Combined Test F1 score: " + str(comb_test_f1_score))

		extract_topics(val_topic_dist, topic_matrix[0], dev_labels , comb_val_preds,  FLAGS.data_dir , FLAGS.output_dir, num_topic=50, name_string= "val")
		extract_topics(test_topic_dist, topic_matrix[0], test_labels , comb_test_preds,  FLAGS.data_dir, FLAGS.output_dir, num_topic=50 , name_string= "test")
		extract_topics(train_topic_dist, topic_matrix[0], train_labels , comb_train_preds,  FLAGS.data_dir, FLAGS.output_dir, num_topic=50 , name_string= "train")

	
	"""
	bert_train_preds = np.load(os.path.join(FLAGS.output_dir, "bert_train_preds.npy" ), allow_pickle = True)
	bert_train_probs = np.load(os.path.join(FLAGS.output_dir, "bert_train_probs.npy" ), allow_pickle = True)

	bert_val_preds = np.load(os.path.join(FLAGS.output_dir, "bert_val_preds.npy" ), allow_pickle = True)
	bert_val_probs = np.load(os.path.join(FLAGS.output_dir, "bert_val_probs.npy" ), allow_pickle = True)

	bert_test_preds = np.load(os.path.join(FLAGS.output_dir, "bert_test_preds.npy" ), allow_pickle = True)
	bert_test_probs = np.load(os.path.join(FLAGS.output_dir, "bert_test_probs.npy" ), allow_pickle = True)

	comb_train_preds = np.load(os.path.join(FLAGS.output_dir, "comb_train_preds.npy" ), allow_pickle = True)
	comb_train_probs = np.load(os.path.join(FLAGS.output_dir, "comb_train_probs.npy" ), allow_pickle = True)

	comb_val_preds = np.load(os.path.join(FLAGS.output_dir, "comb_val_preds.npy" ), allow_pickle = True)
	comb_val_probs = np.load(os.path.join(FLAGS.output_dir, "comb_val_probs.npy" ), allow_pickle = True)

	comb_test_preds = np.load(os.path.join(FLAGS.output_dir, "comb_test_preds.npy" ), allow_pickle = True)
	comb_test_probs = np.load(os.path.join(FLAGS.output_dir, "comb_test_probs.npy" ), allow_pickle = True)
	

	train_topic_dist = np.load(os.path.join(FLAGS.output_dir, "train_topic_dist.npy" ), allow_pickle = True ).item()
	val_topic_dist = np.load(os.path.join(FLAGS.output_dir, "val_topic_dist.npy" ), allow_pickle = True ).item()
	test_topic_dist = np.load(os.path.join(FLAGS.output_dir, "test_topic_dist.npy" ), allow_pickle = True ).item()
	"""

	val_text_file = os.path.join(FLAGS.data_dir, "validation.txt")
	write_pred_files(val_text_file, bert_val_preds, bert_val_probs, comb_val_preds, comb_val_probs, val_topic_dist, FLAGS.data_dir, log_dir, dev_labels, dev_doc_ids, name_string = "val")

	test_text_file = os.path.join(FLAGS.data_dir, "test.txt")
	write_pred_files(test_text_file, bert_test_preds, bert_test_probs, comb_test_preds, comb_test_probs, test_topic_dist, FLAGS.data_dir, log_dir, test_labels, test_doc_ids, name_string = "test")

	train_text_file = os.path.join(FLAGS.data_dir, "training.txt")
	write_pred_files(train_text_file, bert_train_preds, bert_train_probs, comb_train_preds, comb_train_probs, train_topic_dist, FLAGS.data_dir, log_dir, train_labels, train_doc_ids, name_string = "train")

	save_to_s3()


def write_pred_files(text_file, bert_preds, bert_probs, comb_preds, comb_probs, topic_dist, data_dir, log_dir, labels_ids, doc_ids, name_string = ""):

	with open(text_file) as  f:
		lines = f.readlines()
		docs = []
		labels = []
		for i, line in enumerate(lines):
			if i != doc_ids[i]:
				import pdb; pdb.set_trace()

			labels.append(line.split("\t")[0])
			docs.append(line.split("\t")[1].strip())

	assert len(docs) == len(labels) == len(labels_ids) == len(bert_preds) == len(bert_probs) == len(comb_preds) == len(comb_probs) == len(topic_dist)

	with open(os.path.join(data_dir, "label_dict.pkl"), "rb" ) as handle:
		label_maps = pickle.load(handle)
	inverted_label_maps  = {value: key for key, value in label_maps.items()}

	bert_pred_class = [inverted_label_maps[i] for i in bert_preds]
	comb_pred_class = [inverted_label_maps[i] for i in comb_preds]

	all_preds = os.path.join(log_dir, "all_prediction_tb_vs_b_" + name_string + ".txt")
	with open(all_preds, "w") as f:
		f.write("Doc_id" + "\t" + "True label" + "\t" + "bert_class" + "\t" + "tbert_class" + "\t" + "Document" + "\n" )
		for num in range(len(docs)):
			f.write(str(doc_ids[num]) + "\t" + labels[num] + "\t" + bert_pred_class[num] + "\t" + comb_pred_class[num] + "\t" + docs[num] + "\n" )

	preds_tb =  os.path.join(log_dir, "prediction_tb_better_b_" + name_string + ".txt")

	with open(preds_tb, "w") as f:
		f.write("doc_id" + "\t" + "true_label" + "\t" + "pred_by_bert" + "\t" + "prob_by_bert" + "\t" + "pred_by_tbert" + "\t" +  "prob_by_tbert" + "\n" )
		for num in range(len(docs)):
			if  comb_pred_class[num] == labels[num]:
				if bert_pred_class[num] != labels[num]:
					f.write( str(doc_ids[num]) + "\t" + str(labels_ids[num]) + "\t" + str(bert_preds[num]) + "\t" + str(np.max(bert_probs[num])) + "\t" + str(comb_preds[num]) + "\t" +  str(np.max(comb_probs[num])) + "\n" )


	interpret_file  = os.path.join(log_dir, "interpred_tp_" + name_string + ".npy")
	interpret_dict = {}
	for num in range(len(docs)):
		temp = { "bert_softmax": bert_probs[num], "comb_softmax":comb_probs[num], "tp": topic_dist[str(doc_ids[num])]}
		interpret_dict[doc_ids[num]] = temp

	np.save(interpret_file, interpret_dict, allow_pickle = True)



def train_phases(model, bert_config, train_file, validation_file, test_file,  num_train, topic_vocab_size):

	
	with tf.Session(config = tf.ConfigProto(inter_op_parallelism_threads=FLAGS.num_cores, intra_op_parallelism_threads=FLAGS.num_cores, gpu_options=tf.GPUOptions(allow_growth=True))) as session:
	#with tf.Session(config = tf.ConfigProto(inter_op_parallelism_threads=FLAGS.num_cores, intra_op_parallelism_threads=FLAGS.num_cores)) as session:
	
		if FLAGS.finetune_bert:
			sys.exit()

		if FLAGS.combined_train:
			sys.exit()
			#combined_train(model, bert_config,  train_file, validation_file, test_file, topic_vocab_size , num_train , session)
		interpret_TP(model, bert_config,  train_file, validation_file, test_file, topic_vocab_size , num_train , session)



def main(_):	

	FLAGS.bert_config_file = "cased_L-12_H-768_A-12/bert_config.json" 
	FLAGS.bert_vocab_file = "cased_L-12_H-768_A-12/vocab.txt"
	FLAGS.init_checkpoint = "cased_L-12_H-768_A-12/bert_model.ckpt"
	
	bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

	tf.gfile.MakeDirs(FLAGS.output_dir)

	print("write_flags")
	if not os.path.isdir(FLAGS.output_dir):
		os.mkdir(os.path.join(FLAGS.output_dir))	

	train_file = os.path.join(FLAGS.data_dir, "training.tfrecord")
	validation_file = os.path.join(FLAGS.data_dir, "validation.tfrecord")
	test_file = os.path.join(FLAGS.data_dir, "test.tfrecord")

	label_file  = os.path.join(FLAGS.data_dir, "labels.txt") 

	with open(label_file, "r") as f:
		labels = f.readlines()

	label_list = [label.strip() for label in labels]  
	num_labels = len(label_list)
	FLAGS.num_labels = num_labels

	topic_vocab = get_topic_vocab(FLAGS.data_dir)
	topic_vocab_size = len(topic_vocab)
	num_train = sum(1 for _ in tf.python_io.tf_record_iterator(train_file))

	input_ids = tf.placeholder(tf.int32, shape=(None, None), name='input_ids')
	input_mask = tf.placeholder(tf.int32, shape=(None, None), name='input_mask')
	segment_ids = tf.placeholder(tf.int32, shape=(None, None), name='segment_ids')
	
	if FLAGS.multilabel:
		label_ids = tf.placeholder(tf.int32, shape=(None, None), name='label_ids')

	else:
		label_ids = tf.placeholder(tf.int32, shape=(None), name='label_ids')
	doc_ids = tf.placeholder(tf.int32, shape=(None), name='doc_ids')
	topic_bow = tf.placeholder(tf.float32, shape=(None, topic_vocab_size), name='x')
	topic_mask = tf.placeholder(tf.float32, [None], name='mask')  # mask paddings
	is_training = tf.placeholder(tf.float32, shape=(), name='is_training')

	if FLAGS.use_static_topic:
		static_topic = tf.placeholder(tf.float32, shape=(None, FLAGS.n_topic), name='static_topics')
	else:
		static_topic = None

	model = joint_model(bert_config ,input_ids, input_mask, segment_ids, label_ids, doc_ids, topic_bow, topic_mask, topic_vocab_size, is_training, num_train, FLAGS.num_labels, static_topic=static_topic)
	train_phases(model, bert_config, train_file, validation_file, test_file, num_train , topic_vocab_size)


if __name__ == "__main__":
	#flags.mark_flag_as_required("input_file")
	flags.mark_flag_as_required("data_dir")
	flags.mark_flag_as_required("bert_config_file")
	flags.mark_flag_as_required("output_dir")
	tf.app.run()