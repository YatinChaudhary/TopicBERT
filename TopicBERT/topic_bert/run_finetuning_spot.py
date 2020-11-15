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

def extract_topics(topic_dist, topic_matrix, labels, input_dir , output_dir, num_topic):
	##### args #################
	## topic_dist: topic distribution from nvdm/gsm
	## topic_matrix: decoder matrix for topics
	## labels: document labels 
	## output_dir: output directory
	## number of best topics

	###### returns ##########
	## extracts and writes: 
	## 1. top num_topic using topic_matrix
	## 2. best topic distribution


	assert len(labels) == len(topic_dist)

	with open(os.path.join(input_dir, "label_dict.pkl"), "rb" ) as handle:
		label_maps = pickle.load(handle)

	inverted_label_maps  = {value: key for key, value in label_maps.items()}


	topic_file = os.path.join(output_dir, "topics_top_20_words.txt")
	best_topic_log = os.path.join(output_dir, "best_topic_log.txt")

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

			else:
				label = inverted_label_maps[int(labels[doc_num])]

			f.write("Doc id: " + str(doc_id) + "\n") 
			f.write("Topic_distribution: " + str(topic_dist_str) + "\n")
			f.write("Doc label: " + str(label)  + "\n")
			for num, topic_words in enumerate(top_topic_words):
				f.write("Best topic " + str(top_3_topic_ids[num]) + ": " + " ".join(topic_words) + "(" + str(dist[top_3_topic_ids[num]]) +  ")" + "\n") 
			f.write("\n")
	print("Topics extracted!")


def predict_softmax(pred_prob, unique_ids):

	#### args ###########################
	##pred_prob: prediction probabilities 
	## unique_ids: unique document ids 


	####output########################
	## predictions with average/maximum softmax of document fragments

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

			elif FLAGS.max_softmax:
				index_max_softmax = np.argmax(np.amax(pred_probs_per_cand, axis = 1))
				post_pred_labels.append(np.argmax(pred_probs_per_cand, axis = 1)[index_max_softmax])

	assert len(post_pred_labels) == len(unique_ids)

	return post_pred_labels



def postprocess_pred(true_label, pred_label, doc_ids, pred_prob = None):

	### args #############################
	## true_label: true label for each document fragment
	## pred_label: prediction for each document fragment
	## doc ids: document ids
	## pred_prob: prediction probabilities

	### output #################
	## post process true_label and pred_label of document fragments into single document 

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
	for true_cand, doc_id in zip(true_cands, unique_doc_ids):
		if len(true_cand) > 1:

			if FLAGS.multilabel:
				post_true_labels.append(true_cand[0])

			else:
				assert all(x==true_cand[0] for x in true_cand)
				post_true_labels.append(true_cand[0])

		else:
			post_true_labels.append(true_cand[0])

	assert len(post_true_labels) == len(unique_doc_ids)			
	if FLAGS.avg_softmax or FLAGS.max_softmax:
		post_pred_labels = predict_softmax(pred_prob, unique_ids)

	return post_true_labels, post_pred_labels

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
	#### joint model BERT and nvdm #####

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

			output_weights = tf.get_variable(
			"output_weights", [hidden_size, num_labels],
			initializer=tf.truncated_normal_initializer(stddev=0.02, seed = tf_op_seed))

			output_bias = tf.get_variable(
				"output_bias", [num_labels], initializer=tf.zeros_initializer())

			self.bert_logits  = tf.nn.xw_plus_b(output_layer, output_weights, output_bias)

			
			if FLAGS.multilabel:
				self.bert_probabilities = tf.nn.sigmoid(self.bert_logits)
				self.bert_per_example_loss = tf.reduce_sum(tf.losses.sigmoid_cross_entropy(label_ids, self.bert_logits, reduction="none"), axis=-1)	
				

			else:
				self.bert_probabilities = tf.nn.softmax(self.bert_logits, axis=-1)
				self.log_probs = tf.nn.log_softmax(self.bert_logits, axis=-1)

				self.one_hot_labels = tf.one_hot(label_ids, depth=num_labels, dtype=tf.float32)

				self.bert_per_example_loss = -tf.reduce_sum(self.one_hot_labels * self.log_probs, axis=-1)
			self.bert_obj_loss = tf.reduce_mean(self.bert_per_example_loss)

			self.bert_train_op, _, _ = optimization.create_optimizer(
						self.bert_obj_loss, FLAGS.learning_rate , num_train_steps, num_warmup_steps,  optimizer_var_exist, use_tpu = False)


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

	### This function evaluates the finetuning BERT model 
	############ args ########################
	## model: current tensorflow model graph
	## session: running session
	## file: input file for evaluation, validation or test
	## topic_vocab_size: size of the topic vocabulary
	## topic_dist_path: path of the extracted topic distributions

	############ output #####################
	## macro_prec: macro precision 
	## macro_recall: macro recall
	## macro_f1_score: macro F1 score 
	## micro_prec: micro precision 
	## micro_recall: micro recall
	## micro_f1_score: micro f1 score
	## acc: average accuracy on evaluation dataset
	## loss: average loss on evaluation dataset


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
		true_label_list, pred_label_list =  postprocess_pred(true_label_list, pred_label_list, doc_ids_list , pred_prob = prob_list)

		if FLAGS.multilabel:
			true_label_list = np.asarray(true_label_list)
			pred_label_list = np.asarray(pred_label_list)

	macro_prec, macro_recall, macro_f1_score, _  = precision_recall_fscore_support(true_label_list, pred_label_list,  average = "macro")
	micro_prec, micro_recall, micro_f1_score, _  = precision_recall_fscore_support(true_label_list, pred_label_list,  average = "micro")
	acc = accuracy_score(true_label_list, pred_label_list)
	loss = np.mean(np.asarray(loss_list))

	return macro_prec, macro_recall, macro_f1_score, micro_prec, micro_recall, micro_f1_score, acc, loss

				
def run_evaluation_comb(model, session, file, topic_vocab_size , TM_dataset, TM_batch, TM_count, TM_doc_ids,  TM_labels = None):

	### This function evaluates the joint Finetuning model of BERT and nvdm. 
	
	############ args ########################
	## model: current tensorflow model graph
	## session: running session
	## file: input file for evaluation, validation or test
	## topic_vocab_size: size of the topic vocabulary
	## topic_dist_path: path of the extracted topic distributions
	## TM_dataset: topic model dataset for evaluation
	## TM_batch: topic model batches 
	## TM_doc_ids: document ids
	## TM_labels: document labels

	############ output #####################
	## macro_prec: macro precision 
	## macro_recall: macro recall
	## macro_f1_score: macro F1 score 
	## micro_prec: micro precision 
	## micro_recall: micro recall
	## micro_f1_score: micro f1 score
	## acc: average accuracy on evaluation dataset
	## loss: average loss on evaluation dataset
	## topic_ppx: perplexity of evaluation corpus
	## topic_ppx_perdoc:perplexity per doc 
	## topic_kld: kl-divergence


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
		true_label_list, pred_label_list =  postprocess_pred(true_label_list, pred_label_list, doc_ids_list, pred_prob = prob_list )

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

	return macro_prec, macro_recall, macro_f1_score, micro_prec, micro_recall, micro_f1_score, acc, loss, topic_ppx, topic_ppx_perdoc, topic_kld


def train_supervised_TM(model, topic_vocab_size , num_train , session):

	###########args##############
	## model: tensorflow model
	## topic_vocab_size: topic model vocabulary
	## num_train: number of tranining instances 
	## session: initialized session

	###### output##############
	## trains topic model in spuervised way

	print("Train supervised TM !!!!!")
	saver_PPL = tf.train.Saver(tf.global_variables())

	if not os.path.isdir(FLAGS.output_dir +  '/supervised_TM'):
		os.mkdir(os.path.join(FLAGS.output_dir, 'supervised_TM'))

	with open(os.path.join(FLAGS.output_dir, 'supervised_TM', "FLAGS.txt"), "w") as f:
		for key, value in tf.flags.FLAGS.flag_values_dict().items():
			f.write(str(key) +"\t" + str(value) + "\n")
	best_val_model_file = os.path.join(FLAGS.output_dir + '/supervised_TM' , 'supervised_TM')

	
	log_dir = os.path.join(FLAGS.output_dir + '/supervised_TM', 'logs')
	log_PPL = os.path.join(log_dir, "log_supervised_TM.txt")

	summary_writer = tf.summary.FileWriter(log_dir, session.graph)
	summaries = tf.summary.merge_all()

	tf.local_variables_initializer().run()
	tf.global_variables_initializer().run()

	nvdm_datadir = FLAGS.data_dir

	train_url = os.path.join(nvdm_datadir, 'training_nvdm_docs_non_replicated.csv')
	train_set, train_count, train_labels, train_doc_ids= utils.data_set(train_url, FLAGS)
	train_batches = utils.create_batches(len(train_set), FLAGS.nvdm_batch_size, shuffle=False) ### only for logistic regression 

	dev_url = os.path.join(nvdm_datadir, 'validation_nvdm_docs_non_replicated.csv')
	dev_set, dev_count, dev_labels, dev_doc_ids = utils.data_set(dev_url, FLAGS)
	dev_batches = utils.create_batches(len(dev_set), FLAGS.nvdm_batch_size, shuffle=False)

	test_url = os.path.join(nvdm_datadir, 'test_nvdm_docs_non_replicated.csv')
	test_set, test_count, test_labels, test_doc_ids = utils.data_set(test_url, FLAGS)
	test_batches = utils.create_batches(len(test_set), FLAGS.nvdm_batch_size, shuffle=False)


	print("Training supervised TM!")
	dataset = data_lstm.Dataset(nvdm_datadir)
	model.topic_model.pretrain( dataset, FLAGS, nvdm_datadir , session, training_epochs=FLAGS.nvdm_train_epoch, alternate_epochs=FLAGS.nvdm_alternate_epoch)


	topic_matrix = session.run(["TM_decoder/projection/Matrix:0"])
	train_topic_dist, train_mask_list = model.topic_model.topic_dist(train_batches, train_set, train_doc_ids, train_count,  FLAGS, session) 
	val_topic_dist, val_mask_list = model.topic_model.topic_dist(dev_batches, dev_set,  dev_doc_ids, dev_count,  FLAGS, session) 
	test_topic_dist, test_mask_list = model.topic_model.topic_dist(test_batches, test_set, test_doc_ids, test_count,  FLAGS, session) 
	##save the matrix for logistic
	np.save(os.path.join(FLAGS.output_dir, "topic_matrix.npy" ),  topic_matrix, allow_pickle = True )
	np.save(os.path.join(FLAGS.output_dir, "train_topic_dist.npy" ),  train_topic_dist, allow_pickle = True )
	np.save(os.path.join(FLAGS.output_dir, "val_topic_dist.npy" ),  val_topic_dist, allow_pickle = True )
	np.save(os.path.join(FLAGS.output_dir, "test_topic_dist.npy" ),  test_topic_dist, allow_pickle = True )

	extract_topics(val_topic_dist, topic_matrix[0], dev_labels , FLAGS.data_dir , FLAGS.output_dir, num_topic=20)

	

def finetune_bert(model, bert_config, train_file, validation_file, test_file, topic_vocab_size ,  num_train , session):

		### This function is training scheduler for finetuning BERT. It: 
		## 1. fine-tunes the BERT model, 
		## 2. validates after each epoch, 
		## 3. Reloads the best model after finetuning and 
		## 4. Evaluates on validation and test set

		if FLAGS.load_wiki_during_pretrain:
			tvars =  tf.trainable_variables()
			init_checkpoint = FLAGS.init_checkpoint
			(assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
			tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

			print("Best bert Model restored from the bert config check points wiki model. !!!!!!!!!")

		
		saver_PPL = tf.train.Saver(tf.global_variables())

		if not os.path.isdir(FLAGS.output_dir +  '/finetuned_bert_model'):
			os.mkdir(os.path.join(FLAGS.output_dir, 'finetuned_bert_model'))

		with open(os.path.join(FLAGS.output_dir, 'finetuned_bert_model', "FLAGS.txt"), "w") as f:
			for key, value in tf.flags.FLAGS.flag_values_dict().items():
				f.write(str(key) +"\t" + str(value) + "\n")
		best_val_model_file = os.path.join(FLAGS.output_dir + '/finetuned_bert_model' , 'finetuned_bert_model')

		with open(os.path.join(FLAGS.output_dir, 'finetuned_bert_model', "bert_config.json"), "w") as f:
			json.dump(bert_config.to_dict(), f, sort_keys=True)


		log_dir = os.path.join(FLAGS.output_dir + '/finetuned_bert_model', 'logs')
		log_PPL = os.path.join(log_dir, "log_pretain_bert.txt")
		train_loss_file =  os.path.join(log_dir, "log_train_loss.txt")

		summary_writer = tf.summary.FileWriter(log_dir, session.graph)
		summaries = tf.summary.merge_all()
		tf.local_variables_initializer().run()
		tf.global_variables_initializer().run()

		train_losses = []
		patience = FLAGS.patience


		training_iter = batch_generator(train_file, topic_vocab_size , is_training=True)
		input_ids, input_mask, segment_ids, label_ids, doc_ids = training_iter.get_next()

		num_train_steps = int((num_train / FLAGS.train_batch_size) * FLAGS.num_train_epochs)
		steps_1_epoch = int(num_train//FLAGS.train_batch_size)
		validate_save_every = int(FLAGS.save_after_epoch * steps_1_epoch)
		num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

		best_val_f1 = -np.Inf
		patience_count = 0
		best_step = 0

		evaluate_num_epochs = 5
		steps_5_epoch = int(evaluate_num_epochs * steps_1_epoch)
		check_every = int(evaluate_num_epochs * steps_1_epoch)
		best_till_now = evaluate_num_epochs

		save_to_s3()

		epoch_start_time = time.time()
		epoch_time_list = []	

		
		for step in range(num_train_steps):

			input_ids_batch, input_mask_batch, segment_ids_batch, label_ids_batch, doc_ids_batch = session.run([input_ids, input_mask, segment_ids, label_ids, doc_ids])

			if not FLAGS.multilabel:
				label_ids_batch = np.squeeze(label_ids_batch , axis = 1)		

			doc_ids_batch =  np.squeeze(doc_ids_batch, axis = 1 )

			if FLAGS.use_static_topic:
				train_topic_dist_dict = np.load(os.path.join(FLAGS.static_topic_path, "train_topic_dist.npy"), allow_pickle = True).item()
				static_topic_batch = extract_static_topics(doc_ids_batch, train_topic_dist_dict)

				input_feed = {model.input_ids:input_ids_batch,
				 model.input_mask:input_mask_batch, 
				 model.segment_ids:segment_ids_batch, 
				 model.label_ids:label_ids_batch, 
				 model.doc_ids:doc_ids_batch,
				 model.static_topic:static_topic_batch, 
				 model.is_training:0.1} 

			else:
				input_feed = {model.input_ids:input_ids_batch,
				 model.input_mask:input_mask_batch, 
				 model.segment_ids:segment_ids_batch, 
				 model.label_ids:label_ids_batch, 
				 model.doc_ids:doc_ids_batch, 
				 model.is_training:0.1} 

			_ , train_obj_loss = session.run([model.bert_train_op, model.bert_obj_loss], \
				feed_dict = input_feed )

			print("Step number: ", step, "  | train_obj_loss: ", train_obj_loss)
			log_string =  "Step number: " + str(step) + "|" + "train_obj_loss: " + str(float(train_obj_loss)) + "\n" 

			with open(train_loss_file, 'a') as file_for_train_loss:
				file_for_train_loss.write(log_string)

		
			if (step+1) % validate_save_every == 0:
				epoch_time = time.time() - epoch_start_time
				epoch_time_list.append(epoch_time)

				if FLAGS.use_static_topic:
					val_topic_dist_path = os.path.join(FLAGS.static_topic_path, "val_topic_dist.npy")
					dev_prec, dev_recall, dev_f1_score, micro_dev_prec, micro_dev_recall, micro_dev_f1_score, dev_acc, dev_loss = run_evaluation_bert(model, session , validation_file, topic_vocab_size, topic_dist_path= val_topic_dist_path )

				else:
					dev_prec, dev_recall, dev_f1_score, micro_dev_prec, micro_dev_recall, micro_dev_f1_score, dev_acc, dev_loss = run_evaluation_bert(model, session , validation_file, topic_vocab_size, topic_dist_path = None )

				if dev_f1_score > best_val_f1:
					best_val_f1 = dev_f1_score
					best_step = step
					best_loss = dev_loss
					best_prec = dev_prec
					best_recall = dev_recall
					best_acc = dev_acc

					print("validation updated on the basis of PPL ")
					string_to_write = "best step: " + str(best_step) + " |" + "Macro F1: " + str(best_val_f1)\
					+" |" + "Macro Precision: " + str(best_prec)   + " |" + "Macro Recall " + str(best_recall) \
					+" |" + "loss: " + str(best_loss) + " |" + "acc: " + str(best_acc)\
					+" |" + "Micro F1 " + str(micro_dev_f1_score) + " |" + "Micro prec: " + str(micro_dev_prec)\
					+" |" + "Micro recall: " + str(micro_dev_recall) + " |" + "Epoch time: " + str(epoch_time) +  "\n"

					with open( log_PPL,'a') as log_PPL_file: 
						log_PPL_file.write(string_to_write)

					saver_PPL.save(session, best_val_model_file, global_step=1)
					patience_count  = 0

					if (step+1) <= steps_5_epoch:
						best_till_5 = os.path.join(FLAGS.output_dir  + '/best_till_' + str(best_till_now) , 'best_till_' + str(best_till_now))
						saver_PPL.save(session, best_till_5, global_step=1)

				else:
					patience_count += 1
					print("Patience count=" + str(patience_count)  +  " |" + "F1: "  + str(dev_f1_score))

					string_to_write = "Current step: " + str(step) + " |" + "Macro F1: " + str(dev_f1_score)\
					+ " |" + "Macro Precision: " + str(dev_prec)   + " |" + "Macro Recall " + str(dev_recall)   \
					+" |" + "loss: " + str(dev_loss) +   " |" + "acc: " + str(dev_acc)\
					+" |" + "Micro F1 " + str(micro_dev_f1_score) + " |" + "Micro prec: " + str(micro_dev_prec)\
					+" |" + "Micro recall: " + str(micro_dev_recall)  + " |" + "Epoch time: " + str(epoch_time)  + "\n"


					with open( log_PPL,'a') as log_PPL_file: 
						log_PPL_file.write(string_to_write)
					

				if (step+1)%steps_5_epoch == 0:
					steps_5_epoch += check_every
					best_till_now += evaluate_num_epochs
					save_to_s3()
			
					
				if patience_count > FLAGS.patience:

					with open(train_loss_file, 'a') as file_for_train_loss:
						file_for_train_loss.write('Preraining Bert: Early stopping criterion for f1 satisfied.'+ '\n')

					with open(log_PPL, 'a') as log_PPL_file: 
						log_PPL_file.write('Preraining Bert: Early stopping criterion for f1 satisfied.'+ '\n')

					print('Preraining Bert: Early stopping criterion for f1 satisfied.' + '\n')
					break

				epoch_start_time = time.time()

		avg_epoch_time = sum(epoch_time_list)/len(epoch_time_list)
		with open(log_PPL, 'a') as log_PPL_file: 
			log_PPL_file.write('Average epoch time'+ str(avg_epoch_time)+ '\n')

		
		save_to_s3()
		if FLAGS.avg_softmax:
			bert_result = os.path.join(log_dir, "best_model_result_avg_softmax" +".txt")

		elif FLAGS.max_softmax:
			bert_result = os.path.join(log_dir, "best_model_result_max_softmax" + ".txt")
		else:
			bert_result = os.path.join(log_dir, "best_model_result.txt")


		evaluate_list_dirs = sorted(os.listdir(FLAGS.output_dir))
		for path_name in evaluate_list_dirs:
			
			print("Loading best model")

			saver_PPL.restore(session,  str(FLAGS.output_dir) + "/" + path_name  + "/" + path_name + "-1" )
			print("Best bert Model restored.")
		
			## Evaluation validation data ####

			with open(bert_result , "a") as f:
				f.write("Evaluation for :" + path_name + "\n")
			
			if FLAGS.use_static_topic:
				val_topic_dist_path = os.path.join(FLAGS.static_topic_path, "val_topic_dist.npy")
				val_prec, val_recall, val_f1_score, micro_val_prec, micro_val_recall, micro_val_f1_score, val_acc, val_loss = run_evaluation_bert(model, session, validation_file, topic_vocab_size, topic_dist_path= val_topic_dist_path )
			else:
				val_prec, val_recall, val_f1_score, micro_val_prec, micro_val_recall, micro_val_f1_score, val_acc, val_loss = run_evaluation_bert(model, session, validation_file, topic_vocab_size, topic_dist_path= False)


			with open(bert_result , "a") as f:

				eval_string =  "Validation Macro F1: " + str(val_f1_score)\
							+ " |" + "Validation Macro Precision: " + str(val_prec)   + " |" + "Validation Macro Recall " + str(val_recall)   \
							+" |" + "Validation loss: " + str(val_loss) +" |" + "Validation acc: " + str(val_acc) \
							+" |" + "Micro F1 " + str(micro_val_f1_score) + " |" + "Micro prec: " + str(micro_val_prec)  + " |" + "Micro recall: " + str(micro_val_recall) + "\n"

				f.write(eval_string)
			
			
			#### running test evaluations 
			if FLAGS.use_static_topic:
				test_topic_dist_path = os.path.join(FLAGS.static_topic_path, "test_topic_dist.npy")
				test_prec, test_recall, test_f1_score, micro_test_prec, micro_test_recall, micro_test_f1_score, test_acc, test_loss = run_evaluation_bert(model, session, test_file, topic_vocab_size, topic_dist_path= test_topic_dist_path )
			else:
				test_prec, test_recall, test_f1_score, micro_test_prec, micro_test_recall, micro_test_f1_score, test_acc, test_loss = run_evaluation_bert(model, session, test_file, topic_vocab_size, topic_dist_path= None)

			with open(bert_result , "a") as f:

				eval_string =  "Test Macro F1: " + str(test_f1_score)\
							+ " |" + "Test Macro Precision: " + str(test_prec)   + " |" + "Test Macro Recall " + str(test_recall)   \
							+" |" + "Test loss: " + str(test_loss)  +" |" + "Test acc: " + str(test_acc) \
							+" |" + "Micro F1 " + str(micro_test_f1_score) + " |" + "Micro prec: " + str(micro_test_prec)  + " |" + "Micro recall: " + str(micro_test_recall) + "\n\n"

				f.write(eval_string)


		save_to_s3()



def combined_train(model, bert_config, train_file, validation_file, test_file, topic_vocab_size,  num_train, session):

	### This function is training scheduler for joint finetuning of BERT and nvdm. It: 
		## 1. fine-tunes the joint model, 
		## 2. validates after each epoch, 
		## 3. Reloads the best model after finetuning and 
		## 4. Evaluates on validation and test set
	
	print("Starting Combined Training !!!!!!!!!!!")
	saver_PPL = tf.train.Saver(tf.global_variables())
	

	if not os.path.isdir(FLAGS.output_dir +  '/finetuned_bert_model'):

		if FLAGS.load_wiki_during_pretrain:
			tvars =  tf.trainable_variables()
			init_checkpoint = FLAGS.init_checkpoint
			(assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
			tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
			print("Best bert Model restored from the bert config check points wiki model. !!!!!!!!!")

		else:
			print("Initializing bert model with random weights!!!!!!!!!!!!!!!!!")

		tf.local_variables_initializer().run()
		tf.global_variables_initializer().run()


	else:
		ckpt = tf.train.get_checkpoint_state(str(FLAGS.output_dir) + "/finetuned_bert_model")
		saver_PPL.restore(session, ckpt.model_checkpoint_path)
		print("Best bert Model restored from the pretrained bert model!!!!!!!!!!!!!!!!!!")

		#### initialising global step to 1 again ###
		global_step_init = tf.initialize_variables([model.comb_global_step])
		session.run(global_step_init)

		### intialise classifier weights ###
		bert_clf_weights = session.run("bert_loss/output_weights:0")
		bert_clf_bias = session.run("bert_loss/output_bias:0")
		
		with tf.variable_scope("", reuse=True):
			comb_clf_weights = tf.get_variable("output_weights")
			comb_clf_bias = tf.get_variable("output_bias")
		
		clf_weights_assign_op =  tf.assign(comb_clf_weights, bert_clf_weights )
		clf_bias_assign_op =  tf.assign(comb_clf_bias, bert_clf_bias )
		session.run(clf_weights_assign_op)
		session.run(clf_bias_assign_op)
		##
	
	comb_model_name = 'combined_model'  

	if not os.path.isdir(FLAGS.output_dir +  '/' + comb_model_name):
			os.mkdir(os.path.join(FLAGS.output_dir, comb_model_name))

	with open(os.path.join(FLAGS.output_dir, comb_model_name, "FLAGS.txt"), "w") as f:
		for key, value in tf.flags.FLAGS.flag_values_dict().items():
			f.write(str(key) +"\t" + str(value) + "\n")


	with open(os.path.join(FLAGS.output_dir, comb_model_name, "bert_config.json"), "w") as f:
		json.dump(bert_config.to_dict(), f, sort_keys=True)

	best_val_model_file = os.path.join(FLAGS.output_dir + '/' + comb_model_name , 'combined_model')

	log_dir = os.path.join(FLAGS.output_dir + '/' + comb_model_name, 'logs')
	log_PPL = os.path.join(log_dir, "log_combined_model.txt")
	train_loss_file =  os.path.join(log_dir, "log_train_loss.txt")

	summary_writer = tf.summary.FileWriter(log_dir, session.graph)
	summaries = tf.summary.merge_all()

	nvdm_datadir = FLAGS.data_dir

	train_url = os.path.join(nvdm_datadir, 'training_nvdm_docs_non_replicated.csv')
	train_set, train_count, train_labels, train_doc_ids = utils.data_set(train_url, FLAGS)
	train_batches = utils.create_batches(len(train_set), FLAGS.nvdm_batch_size, shuffle=False) ### only for logistic regression 

	dev_url = os.path.join(nvdm_datadir, 'validation_nvdm_docs_non_replicated.csv')
	dev_set, dev_count, dev_labels, dev_doc_ids = utils.data_set(dev_url, FLAGS)
	dev_batches = utils.create_batches(len(dev_set), FLAGS.nvdm_batch_size, shuffle=False)

	test_url = os.path.join(nvdm_datadir, 'test_nvdm_docs_non_replicated.csv')
	test_set, test_count, test_labels, test_doc_ids = utils.data_set(test_url, FLAGS)
	test_batches = utils.create_batches(len(test_set), FLAGS.nvdm_batch_size, shuffle=False)


	if  FLAGS.pretrain_nvdm:
		if FLAGS.pretrain_nvdm_path:

			if FLAGS.supervised_TM:
				print_ppx, print_ppx_perdoc, print_kld, print_sup_loss,  print_macro_prec, print_macro_recall, print_macro_f1_score, print_acc\
				= model.topic_model.run_epoch(dev_batches, dev_set,  dev_count , FLAGS, session, input_labels = dev_labels)
				print("Corpus ppl: " + str(print_ppx) +  " Per doc ppl: " + str(print_ppx_perdoc) + " kld: " + str(print_kld) )

			else:
				print_ppx, print_ppx_perdoc, print_kld = model.topic_model.run_epoch(dev_batches, dev_set,  dev_count,  FLAGS, session)
				print("Corpus ppl: " + str(print_ppx) +  " Per doc ppl: " + str(print_ppx_perdoc) + " kld: " + str(print_kld) )

			model.topic_model.pretrain_saver.restore(session, FLAGS.pretrain_nvdm_path)

			if FLAGS.supervised_TM:
				print_ppx, print_ppx_perdoc, print_kld, print_sup_loss,  print_macro_prec, print_macro_recall, print_macro_f1_score, print_acc\
				= model.topic_model.run_epoch(dev_batches, dev_set,  dev_count , FLAGS, session, input_labels = dev_labels)
				print("Corpus ppl: " + str(print_ppx) +  " Per doc ppl: " + str(print_ppx_perdoc) + " kld: " + str(print_kld) )

			else:
				print_ppx, print_ppx_perdoc, print_kld = model.topic_model.run_epoch( dev_batches, dev_set,  dev_count,  FLAGS, session)
				print(" Corpus ppl: " + str(print_ppx) +  " Per doc ppl: " + str(print_ppx_perdoc) + " kld: " + str(print_kld) )

			topic_matrix = session.run(["TM_decoder/projection/Matrix:0"])

			train_topic_dist, train_mask_list = model.topic_model.topic_dist(train_batches, train_set, train_doc_ids,  train_count,  FLAGS, session) 
			val_topic_dist, val_mask_list = model.topic_model.topic_dist(dev_batches, dev_set, dev_doc_ids,  dev_count,  FLAGS, session) 
			test_topic_dist, test_mask_list = model.topic_model.topic_dist(test_batches, test_set, test_doc_ids , test_count,  FLAGS, session) 
			##save the matrix for logistic
			np.save(os.path.join(FLAGS.output_dir, "topic_matrix.npy" ),  topic_matrix, allow_pickle = True )
			np.save(os.path.join(FLAGS.output_dir, "train_topic_dist.npy" ),  train_topic_dist, allow_pickle = True )
			np.save(os.path.join(FLAGS.output_dir, "val_topic_dist.npy" ),  val_topic_dist, allow_pickle = True )
			np.save(os.path.join(FLAGS.output_dir, "test_topic_dist.npy" ),  test_topic_dist, allow_pickle = True )

			extract_topics(val_topic_dist, topic_matrix[0], dev_labels , FLAGS.data_dir , FLAGS.output_dir, num_topic=20)
		
		else:
			print("Pretraining nvdm !")
			dataset = data_lstm.Dataset(nvdm_datadir)
			model.topic_model.pretrain( dataset, FLAGS, nvdm_datadir , session, training_epochs=FLAGS.nvdm_train_epoch, alternate_epochs=FLAGS.nvdm_alternate_epoch)
			topic_matrix = session.run(["TM_decoder/projection/Matrix:0"])
			train_topic_dist, train_mask_list = model.topic_model.topic_dist(train_batches, train_set,   train_doc_ids, train_count,  FLAGS, session) 
			val_topic_dist, val_mask_list = model.topic_model.topic_dist(dev_batches, dev_set,  dev_doc_ids, dev_count,  FLAGS, session) 
			test_topic_dist, test_mask_list = model.topic_model.topic_dist(test_batches, test_set, test_doc_ids,  test_count,  FLAGS, session) 
			##save the matrix for logistic
			np.save(os.path.join(FLAGS.output_dir, "topic_matrix.npy" ),  topic_matrix, allow_pickle = True )
			np.save(os.path.join(FLAGS.output_dir, "train_topic_dist.npy" ),  train_topic_dist, allow_pickle = True )
			np.save(os.path.join(FLAGS.output_dir, "val_topic_dist.npy" ),  val_topic_dist, allow_pickle = True )
			np.save(os.path.join(FLAGS.output_dir, "test_topic_dist.npy" ),  test_topic_dist, allow_pickle = True )

			extract_topics(val_topic_dist, topic_matrix[0], dev_labels , FLAGS.data_dir , FLAGS.output_dir, num_topic=20)
	
	
	if FLAGS.supervised_TM:
		val_topic_ppx, val_topic_ppx_perdoc, val_topic_kld, val_topic_sup_loss,  val_topic_macro_prec, val_topic_macro_recall, val_topic_macro_f1_score, val_topic_acc\
				= model.topic_model.run_epoch(dev_batches, dev_set,  dev_count , FLAGS, session, input_labels = dev_labels)

	else:
		val_topic_ppx, val_topic_ppx_perdoc, val_topic_kld = model.topic_model.run_epoch(dev_batches, dev_set,  dev_count,  FLAGS, session)

	string_to_write = "Corpus ppl: " + str(val_topic_ppx) + " |" +  "Per doc ppl: " + str(val_topic_ppx_perdoc) + " |"   + "kld: " + str(val_topic_kld) + "\n"
	with open( log_PPL,'a') as log_PPL_file: 
		log_PPL_file.write(string_to_write)
		
	train_losses = []
	patience = FLAGS.patience


	training_iter = batch_generator(train_file, topic_vocab_size , is_training=True)
	input_ids, input_mask, segment_ids, label_ids, doc_ids = training_iter.get_next()

	num_train_steps = int((num_train / FLAGS.train_batch_size) * FLAGS.num_train_epochs)
	steps_1_epoch = int(num_train//FLAGS.train_batch_size)
	validate_save_every = int(FLAGS.save_after_epoch * steps_1_epoch)
	num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

	best_val_f1 = -np.Inf
	patience_count = 0
	best_step = 0

	evaluate_num_epochs= 5
	steps_5_epoch = int(evaluate_num_epochs * steps_1_epoch)
	check_every = int(evaluate_num_epochs * steps_1_epoch)
	best_till_now = evaluate_num_epochs

	save_to_s3()
	
	epoch_start_time = time.time()
	epoch_time_list = []

	
	for step in range(num_train_steps):

		input_ids_batch, input_mask_batch, segment_ids_batch, label_ids_batch, doc_ids_batch = session.run([input_ids, input_mask, segment_ids, label_ids, doc_ids])

		if not FLAGS.multilabel:
			label_ids_batch = np.squeeze(label_ids_batch , axis = 1)

		doc_ids_batch = np.squeeze(doc_ids_batch , axis = 1)

		topic_bow_batch, topic_mask_batch = extract_topic_data(doc_ids_batch, train_set, train_doc_ids, topic_vocab_size)

		_ , train_obj_loss = session.run([model.combined_train_op, model.comb_obj_loss], \
			feed_dict = {model.input_ids:input_ids_batch, model.input_mask:input_mask_batch, model.segment_ids:segment_ids_batch, model.label_ids:label_ids_batch, model.topic_bow:topic_bow_batch, model.topic_mask:topic_mask_batch, model.is_training:0.1})

		print("Step number: ", step, "  | train_obj_loss: ", train_obj_loss)
		log_string =  "Step number: " + str(step) + "|" + "train_obj_loss: " + str(train_obj_loss) + "\n" 

		with open(train_loss_file, 'a') as file_for_train_loss:
			file_for_train_loss.write(log_string)


		if (step+1) % validate_save_every == 0:
		#if (step) % validate_save_every == 0:
			epoch_time = time.time() - epoch_start_time
			epoch_time_list.append(epoch_time)

			dev_prec, dev_recall, dev_f1_score, micro_dev_prec, micro_dev_recall, micro_dev_f1_score, dev_acc, dev_loss, dev_topic_ppx, dev_topic_ppx_perdoc, dev_topic_kld = run_evaluation_comb(model, session, validation_file, topic_vocab_size , dev_set, dev_batches, dev_count, dev_doc_ids, TM_labels = dev_labels)

			if dev_f1_score > best_val_f1:
				best_val_f1 = dev_f1_score
				best_step = step
				best_loss = dev_loss
				best_prec = dev_prec
				best_recall = dev_recall
				best_acc = dev_acc
				print("validation updated on the basis of PPL ")

				string_to_write = "best step: " + str(best_step) + " |" + "Macro F1: " + str(best_val_f1)\
				+ " |" + "Macro Precision: " + str(best_prec)   + " |" + "Macro Recall " + str(best_recall)   \
				+" |" + "loss: " + str(best_loss)  +" |" + "acc: " + str(best_acc) \
				+ " |" + "Micro F1 " + str(micro_dev_f1_score) + " |" + "Micro prec: " + str(micro_dev_prec)  + " |" + "Micro recall: " + str(micro_dev_recall) \
				+ " |"  + "Corpus ppl: " + str(dev_topic_ppx) + " |" +  "Per doc ppl: " + str(dev_topic_ppx_perdoc) + " |"   + "kld: " + str(dev_topic_kld) + " |" + "Epoch time: " + str(epoch_time) + "\n"

				with open( log_PPL,'a') as log_PPL_file: 
					log_PPL_file.write(string_to_write)

				saver_PPL.save(session, best_val_model_file, global_step=1)
				
				patience_count  = 0

				if (step+1) <= steps_5_epoch:
					best_till_5 = os.path.join(FLAGS.output_dir  + '/best_till_' + str(best_till_now) , 'best_till_' + str(best_till_now))
					saver_PPL.save(session, best_till_5, global_step=1)

			else:
				patience_count += 1
				print("Patience count=" + str(patience_count)  +  " |" + " F1: " + str(dev_f1_score))

				string_to_write = "Current step: " + str(step) + " |" + "Macro F1: " + str(dev_f1_score)\
				+ " |" + "Macro Precision: " + str(dev_prec)   + " |" + "Macro Recall " + str(dev_recall)   \
				+" |" + "loss: " + str(dev_loss) +" |" + "acc: " + str(dev_acc) \
				+ " |" + "Micro F1 " + str(micro_dev_f1_score) + " |" + "Micro prec: " + str(micro_dev_prec)  + " |" + "Micro recall: " + str(micro_dev_recall) \
				+ " |"  + "Corpus ppl: " + str(dev_topic_ppx) + " |" +  "Per doc ppl: " + str(dev_topic_ppx_perdoc) + " |"   + "kld: " + str(dev_topic_kld) + " |" + "Epoch time: " + str(epoch_time) + "\n"

				with open( log_PPL,'a') as log_PPL_file: 
					log_PPL_file.write(string_to_write)

			

			if (step+1)%steps_5_epoch == 0:
				steps_5_epoch += check_every
				best_till_now += evaluate_num_epochs
				save_to_s3()
			

			if patience_count > FLAGS.patience:

				with open(train_loss_file, 'a') as file_for_train_loss:
					file_for_train_loss.write('Combined Bert: Early stopping criterion for f1 satisfied.'+ '\n')


				with open(log_PPL, 'a') as log_PPL_file: 
					log_PPL_file.write('Combined Bert: Early stopping criterion for f1 satisfied.'+ '\n')

				print('Combined Bert: Early stopping criterion for f1 satisfied.' + '\n')
				break

			epoch_start_time = time.time()

	avg_epoch_time = sum(epoch_time_list)/len(epoch_time_list)
	with open(log_PPL, 'a') as log_PPL_file: 
		log_PPL_file.write('Average epoch time'+ str(avg_epoch_time)+ '\n')
	
	save_to_s3()
	if FLAGS.avg_softmax:
		bert_result = os.path.join(log_dir, "best_model_result_avg_softmax" +".txt")

	elif FLAGS.max_softmax:
		bert_result = os.path.join(log_dir, "best_model_result_max_softmax" + ".txt")
	else:
		bert_result = os.path.join(log_dir, "best_model_result.txt")

	evaluate_list_dirs = sorted(os.listdir(FLAGS.output_dir))
	for path_name in evaluate_list_dirs:


		print("Loading best model")
		
		if os.path.isdir(os.path.join(FLAGS.output_dir, path_name)):

			saver_PPL.restore(session,  str(FLAGS.output_dir) + "/" + path_name  + "/" + path_name + "-1" )
			print("Best combined Model restored.")

			with open(bert_result , "a") as f:
				f.write("Evaluation for: " + path_name + "\n")
			
			## Evaluation validation data ####
			val_prec, val_recall, val_f1_score, micro_val_prec, micro_val_recall, micro_val_f1_score, val_acc, val_loss, val_topic_ppx, val_topic_ppx_perdoc, val_topic_kld = run_evaluation_comb(model, session, validation_file, topic_vocab_size , dev_set, dev_batches, dev_count, dev_doc_ids, TM_labels = dev_labels)

			with open(bert_result , "a") as f:

				eval_string =  "Validation Macro F1: " + str(val_f1_score)\
							+ " |" + "Validation Macro Precision: " + str(val_prec)   + " |" + "Validation Macro Recall " + str(val_recall)   \
							+" |" + "Validation loss: " + str(val_loss)+ " |" + "Validation acc: " + str(val_acc) \
							+ " |" + "Micro F1 " + str(micro_val_f1_score) + " |" + "Micro prec: " + str(micro_val_prec)  + " |" + "Micro recall: " + str(micro_val_recall) + "\n"

				f.write(eval_string)



			#### running test evaluations 
			test_prec, test_recall, test_f1_score, micro_test_prec, micro_test_recall, micro_test_f1_score, test_acc, test_loss, test_topic_ppx, test_topic_ppx_perdoc, test_topic_kld = run_evaluation_comb(model, session, test_file, topic_vocab_size,test_set, test_batches, test_count, test_doc_ids, TM_labels = test_labels)

			with open(bert_result , "a") as f:

				eval_string =  "Test Macro F1: " + str(test_f1_score)\
							+ " |" + "Test Macro Precision: " + str(test_prec)   + " |" + "Test Macro Recall " + str(test_recall)\
							+" |" + "Test loss: " + str(test_loss) +" |" + "Test acc: " + str(test_acc)  \
							+ " |" + "Micro F1 " + str(micro_test_f1_score) + " |" + "Micro prec: " + str(micro_test_prec)  + " |" + "Micro recall: " + str(micro_test_recall) +  "\n\n"

				f.write(eval_string)
	save_to_s3()



def train_phases(model, bert_config, train_file, validation_file, test_file,  num_train, topic_vocab_size):

	
	with tf.Session(config = tf.ConfigProto(inter_op_parallelism_threads=FLAGS.num_cores, intra_op_parallelism_threads=FLAGS.num_cores, gpu_options=tf.GPUOptions(allow_growth=True))) as session:
		if FLAGS.pretrain_supervised_TM:
			train_supervised_TM(model, topic_vocab_size , num_train , session)

		if FLAGS.finetune_bert:
			finetune_bert(model, bert_config,  train_file, validation_file, test_file, topic_vocab_size , num_train , session)


		if FLAGS.combined_train:
			combined_train(model, bert_config,  train_file, validation_file, test_file, topic_vocab_size , num_train , session)



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
	flags.mark_flag_as_required("data_dir")
	flags.mark_flag_as_required("bert_config_file")
	flags.mark_flag_as_required("output_dir")
	tf.app.run()
