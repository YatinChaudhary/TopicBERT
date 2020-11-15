from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random
import tokenization
import tensorflow as tf
from collections import Counter
import os
import shutil
import codecs
from collections import defaultdict
import sys
import operator

import nltk
from nltk.corpus import stopwords

import re
import csv
import pickle
from nltk.tokenize import RegexpTokenizer

reg_tokenizer = RegexpTokenizer(r'\w+')

cachedStopWords = stopwords.words('english')


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("task_name", None, "name of the task to fine tune")


flags.DEFINE_string("input_folder", None,
										"Input document folder for bert and nvdm.")

flags.DEFINE_string("input_file", None,
										"Input raw text file (or comma-separated list of files).")

flags.DEFINE_string(
		"output_file", None,
		"Output TF example file (or comma-separated list of files).")

flags.DEFINE_string("vocab_file", None,
										"The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
		"do_lower_case", False,
		"Whether to lower case the input text. Should be True for uncased "
		"models and False for cased models.")

flags.DEFINE_bool("clean_string", True, "wheather to clean data before topic modelling")

flags.DEFINE_bool("multilabel", False, "multilabel")

flags.DEFINE_string( "data_specific", "20NS", "[20NS|BBC|R21578|dbpedia|ohsumed]")

flags.DEFINE_integer("max_seq_length", 512, "Maximum sequence length.")


flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

def isalpha_num(token):
	char_set = set(token)

	num_found = False
	alpha_found = False
	for char in char_set:
		if char.isalpha():
			alpha_found = True

		if char.isnumeric():
			num_found = True

	if  alpha_found == True and num_found==True:
		return True
	else:
		return False


class InputExample(object):
	"""A single training/test example for simple sequence classification."""

	def __init__(self, guid, text_a, text_b=None, label=None):
		"""Constructs a InputExample.

		Args:
			guid: Unique id for the example.
			text_a: string. The untokenized text of the first sequence. For single
				sequence tasks, only this sequence must be specified.
			text_b: (Optional) string. The untokenized text of the second sequence.
				Only must be specified for sequence pair tasks.
			label: (Optional) string. The label of the example. This should be
				specified for train and dev examples, but not for test examples.
		"""
		self.guid = guid
		self.text_a = text_a
		self.text_b = text_b
		self.label = label


def update_vocab(symbol, idxvocab, vocabxid):

	idxvocab.append(symbol)

	vocabxid[symbol] = len(idxvocab) - 1 

 
def gen_vocab_docnade(dummy_symbols, corpus, stopwords, vocab_minfreq, vocab_maxfreq, verbose):

	idxvocab = []
	vocabxid = defaultdict(int)
	vocab_freq = defaultdict(int)

	for doc_id, doc in enumerate(corpus):

		"""
		doc_tokens = []
		for doc_sent in doc.strip().lower().split("\t"):
			sent_tokens = doc_sent.strip().lower().split()
			doc_tokens.extend(sent_tokens)
		"""
		doc_tokens = doc.strip().lower().split()
		for word in doc_tokens:
			vocab_freq[word] += 1

		if doc_id % 1000 == 0 and verbose:
			sys.stdout.write(str(doc_id) + " processed\r")
			sys.stdout.flush()

 
	print("Vocab Frequency extracted!")
	#add in dummy symbols into vocab
	for s in dummy_symbols:
		update_vocab(s, idxvocab, vocabxid)

 
	TM_ignore = []
	ignored_min_freq= []
	#remove low fequency words
	for word, f in sorted(vocab_freq.items(), key=operator.itemgetter(1), reverse=True):
		
		if f < vocab_minfreq:
			ignored_min_freq.append(word)
			continue

		else:
			if f < 100:
				TM_ignore.append(word)
				
			update_vocab(word, idxvocab, vocabxid)

 
	#ignore stopwords, frequent words and symbols for the document input for topic model

	#stopwords = set([item.strip().lower() for item in open(stopwords)])
	print("Before removing words")

	stopwords = set([item.strip().lower() for item in stopwords])
	freqwords = set([item[0] for item in sorted(vocab_freq.items(), key=operator.itemgetter(1), \
		reverse=True)[:int(float(len(vocab_freq))*vocab_maxfreq)]]) #ignore top N% most frequent words for topic model
	
	alpha_check = re.compile("[a-zA-Z]")
	symbols = set([ w for w in vocabxid.keys() if ((alpha_check.search(w) == None) or w.startswith("'")) ])
	ignore = stopwords | freqwords | symbols | set(dummy_symbols) | set(["n't"])
	ignore = [vocabxid[w] for w in ignore if w in vocabxid]
	ignore.extend([vocabxid[w] for w in TM_ignore])
	ignore = set(ignore)


	print("after removing words")
	return idxvocab, vocabxid, ignore



class InputFeatures(object):
	"""A single set of features of data."""
	def __init__(self,
							 input_ids,
							 input_mask,
							 segment_ids,
							 label_id,
							 doc_id,
							 start_idx,
							 is_real_example=True):
		self.input_ids = input_ids
		self.input_mask = input_mask
		self.segment_ids = segment_ids
		self.label_id = label_id
		self.is_real_example = is_real_example
		self.doc_id = doc_id
		self.start_idx = start_idx


class DataProcessor(object):
	"""Base class for data converters for sequence classification data sets."""

	def get_train_examples(self, data_dir):
		"""Gets a collection of `InputExample`s for the train set."""
		raise NotImplementedError()

	def get_dev_examples(self, data_dir):
		"""Gets a collection of `InputExample`s for the dev set."""
		raise NotImplementedError()

	def get_test_examples(self, data_dir):
		"""Gets a collection of `InputExample`s for prediction."""
		raise NotImplementedError()

	def get_labels(self):
		"""Gets the list of labels for this data set."""
		raise NotImplementedError()

	@classmethod
	def _read_tsv(cls, input_file, quotechar=None):
		"""Reads a tab separated value file."""
		with tf.gfile.Open(input_file, "r") as f:
			reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
			lines = []
			for line in reader:
				lines.append(line)
			return lines


	@classmethod
	def _read_txt(cls, input_file):
		with codecs.open(input_file, "r", encoding = "utf-8", errors = "ignore") as f:
			lines = f.readlines()

		return lines



## added a binary classifier class BinaryClassificationProcessor
class ClassificationProcessor(DataProcessor):
		"""Processor for binary classification dataset."""

		def get_train_examples(self, data_dir):
				"""See base class."""
				return self._create_examples(
						self._read_txt(os.path.join(data_dir, "training.txt")), "train")

		def get_dev_examples(self, data_dir):
				"""See base class."""
				return self._create_examples(
						self._read_txt(os.path.join(data_dir, "validation.txt")), "validation")

		def get_test_examples(self, data_dir):
				"""See base class."""
				return self._create_examples(
						self._read_txt(os.path.join(data_dir, "test.txt")), "test")


		def get_labels(self,data_dir):
				train_examples = self._create_examples(
					self._read_txt(os.path.join(data_dir, "training.txt")), "train")

				label_list = []
				if FLAGS.multilabel:
					for example in train_examples:
						label_list.extend(example.label.split(":"))

					label_set = set(label_list)

					return sorted(list(label_set))

				else:
					for example in train_examples:
						label_list.append(example.label)

					label_set = set(label_list)

					return sorted(list(label_set))

					"""See base class."""
					"""
					return ["0", "1"]
					"""

		def _create_examples(self, lines, set_type):
				"""Creates examples for the training and dev sets."""
				examples = []
				for (i, text_line) in enumerate(lines):
						line = text_line.split("\t")
						guid = "%s-%s" % (set_type, i)
						text_a = line[1].strip()
						label = line[0] 
						examples.append(
								InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
				return examples




def extract_docs(file_name, output_folder):

	with open(os.path.join(output_folder, "label_dict.pkl"), "rb" ) as handle:
		label_maps = pickle.load(handle)


	with codecs.open(file_name, "r", encoding = "utf-8", errors = "ignore") as f:
		lines = f.readlines()

	label_list = []
	doc_list = []

	counter_check = 0 
	for line in lines:
		counter_check = counter_check + 1 

		if counter_check % 100 == 0:
			print(str(counter_check))

		label_list.append(line.split("\t")[0])
		if FLAGS.clean_string == True:
			doc = line.split("\t")[1].strip()
			tokens = reg_tokenizer.tokenize(doc.lower().strip())
			
			if FLAGS.data_specific == "20NS":
				for i, token in enumerate(tokens):
					if len(set(token)) and "_" in set(token):
						tokens[i] = ""

					if bool(re.match('^(?=.*[0-9]$)(?=.*[a-zA-Z])',token)) == True:
						tokens[i] = "<alphanum>"

				cleaned_tokens = []
				for token in tokens:
					if token == "":
						continue
					else:
						cleaned_tokens.append(token)

				tokens = cleaned_tokens 

			if FLAGS.data_specific == "BBC" or FLAGS.data_specific == "R21578" or FLAGS.data_specific == "dbpedia" or FLAGS.data_specific == "ohsumed":
				for i, token in enumerate(tokens):
					if isalpha_num(token) == True:
						tokens[i] = "<alphanum>"


			if FLAGS.data_specific =="proppy":

				for i, token in enumerate(tokens):
					if isalpha_num(token) == True:
						tokens[i] = "<alphanum>"

					if len(token) == 1:
						tokens[i] = ""

				cleaned_tokens = []
				for token in tokens:
					if token == "":
						continue
					else:
						cleaned_tokens.append(token)

				tokens = cleaned_tokens

			doc = " ".join(tokens)

		else:
			doc = line.split("\t")[1].strip()

		doc_list.append(doc)

	if FLAGS.multilabel:
		label_ids = []
		for labels in label_list:
			m_label = labels.split(":")
			m_label_id = []
			for label in m_label:
				m_label_id.append(str(label_maps[label]))
				
			label_ids.append(":".join(m_label_id))

	else:
		label_ids = []
		for label in label_list:
			label_ids.append(label_maps[label])

	print("Extracted docs from:" + str(file_name))
	return label_ids, doc_list

def write_topic_data(docs, labels, vocab_to_id, file_name):

	assert len(labels) == len(docs)
	with open(file_name, "w", newline='') as f:
		writer = csv.writer(f,  delimiter=",")
		for doc_id, doc in enumerate(docs):
			doc_tokens = doc.strip().lower().split()
			ids = []
			tokens_found = []
			for token in doc_tokens:
				if token in vocab_to_id:
					ids.append(vocab_to_id[token])
					tokens_found.append(token)
			ids = [str(id) for id in ids]
			if len(ids) != 0:
				writer.writerow([str(labels[doc_id]), " ".join(ids), str(doc_id)])


def prepare_doc_data(input_folder, output_folder):

	train_input = os.path.join(input_folder, "training.txt")
	validation_input= os.path.join(input_folder, "validation.txt")
	test_input= os.path.join(input_folder, "test.txt")
	

	train_label, train_doc = extract_docs(train_input, output_folder)
	validation_label, validation_doc = extract_docs(validation_input, output_folder)
	test_label, test_doc = extract_docs(test_input, output_folder)


	all_doc = train_doc + validation_doc + test_doc

	vocab_list, vocab_dict, ignore_words = gen_vocab_docnade(["_bos_", "_eos_", "_unk_"], all_doc, cachedStopWords, 10, 0.001, True)
	vocab_to_id = dict(zip(vocab_list, range(len(vocab_list))))
	vocab_list_docnade = [vocab_list[index] for index in range(len(vocab_list)) if not index in ignore_words]
	vocab_to_id_docnade = dict(zip(vocab_list_docnade, range(len(vocab_list_docnade))))

	docnade_vocab_filename = os.path.join(output_folder, "vocab_docnade.vocab")
	with open(docnade_vocab_filename, "w") as f:
		f.write('\n'.join(vocab_list_docnade))
		
	print("Size of the topic vocab: " + str(len(vocab_to_id_docnade)))


	file_name = os.path.join(output_folder, "training_nvdm_docs_non_replicated.csv")
	write_topic_data(train_doc, train_label, vocab_to_id_docnade, file_name )

	file_name = os.path.join(output_folder, "validation_nvdm_docs_non_replicated.csv")
	write_topic_data(validation_doc, validation_label, vocab_to_id_docnade, file_name)
	
	file_name = os.path.join(output_folder, "test_nvdm_docs_non_replicated.csv")
	write_topic_data(test_doc, test_label, vocab_to_id_docnade, file_name)


def convert_single_example(ex_index, example, label_map, max_seq_length,
													 tokenizer, doc_id_repeating):
	"""Converts a single `InputExample` into a single `InputFeatures`."""
	"""
	label_map = {}
	for (i, label) in enumerate(label_list):
		label_map[label] = i
	"""

	tokens_a = tokenizer.tokenize(example.text_a)
	tokens_b = None
	if example.text_b:
		tokens_b = tokenizer.tokenize(example.text_b)


	
	print("converting example:" + str(ex_index))
	"""
	if tokens_b:
		# Modifies `tokens_a` and `tokens_b` in place so that the total
		# length is less than the specified length.
		# Account for [CLS], [SEP], [SEP] with "- 3"
		_truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
	else:
		# Account for [CLS] and [SEP] with "- 2"
		if len(tokens_a) > max_seq_length - 2:
			tokens_a = tokens_a[0:(max_seq_length - 2)]
	"""


	# The convention in BERT is:
	# (a) For sequence pairs:
	#	tokens:	 [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
	#	type_ids: 0		 0	0		0		0		 0			 0 0		 1	1	1	1	 1 1
	# (b) For single sequences:
	#	tokens:	 [CLS] the dog is hairy . [SEP]
	#	type_ids: 0		 0	 0	 0	0		 0 0
	#
	# Where "type_ids" are used to indicate whether this is the first
	# sequence or the second sequence. The embedding vectors for `type=0` and
	# `type=1` were learned during pre-training and are added to the wordpiece
	# embedding vector (and position vector). This is not *strictly* necessary
	# since the [SEP] token unambiguously separates the sequences, but it makes
	# it easier for the model to learn the concept of sequences.
	#
	# For classification tasks, the first vector (corresponding to [CLS]) is
	# used as the "sentence vector". Note that this only makes sense because
	# the entire model is fine-tuned.

	########### divide docs in 512 ############

	assert tokens_b == None

	max_num_tokens = max_seq_length - 2

	if len(tokens_a) <= max_num_tokens:
		segment_dict = {}
		num_chunk = 0
		chunk_dict = {}
		chunk_dict["doc_id"] = str(ex_index)
		chunk_dict["segment_id"] = str(num_chunk)
		chunk_dict["bert_tokens"] = tokens_a
		chunk_dict["start_index"] = 0
		segment_dict[num_chunk] = chunk_dict

	else:
		doc_id_repeating.append(ex_index)
		print("Lenght exceeded: " + str(ex_index) + " !!!!!!!!")
		actual_doc_len = len(tokens_a)
		start = 0
		end = max_num_tokens
		num_chunk = 0
		segment_dict = {}
		unique_len = 0

		while end <= len(tokens_a):
			
			current_end = end
			if end != len(tokens_a):	 
				if tokens_a[end][:2] == "##":
					while tokens_a[end][:2] == "##":
						end = end - 1 

			### 20NS-64 31/3
			#if start == end:
			#	end = current_end
			####

			chunk_dict = {}
			chunk_dict["doc_id"] = str(ex_index)
			chunk_dict["segment_id"] = str(num_chunk)
			chunk_dict["bert_tokens"] = tokens_a[start:end]
			chunk_dict["start_index"] = 0
			unique_len = unique_len +  len(tokens_a[start:end])
			segment_dict[num_chunk] = chunk_dict
			start = end
			end = max_num_tokens + start
			num_chunk += 1

			if  len(tokens_a[start:]) < max_num_tokens and  len(tokens_a[start:]) != 0:			
				addtional_chunk_len = max_num_tokens - len(tokens_a[start:end])
				#previous_segment = segment_dict[num_chunk -1]["bert_tokens"][:-1]
				previous_segment = segment_dict[num_chunk -1]["bert_tokens"]
				additional_chunk = previous_segment[-addtional_chunk_len:]

				
				if additional_chunk[0][:2] == "##":
					while  additional_chunk[0][:2] == "##":
						additional_chunk = additional_chunk[1:]
						 
						####
						if len(additional_chunk) == 0:
							break

						#####



						
				chunk_dict = {}
				chunk_dict["doc_id"] = str(ex_index)
				chunk_dict["segment_id"] = str(num_chunk)
				chunk_dict["bert_tokens"] = additional_chunk  +  tokens_a[start:end]
				chunk_dict["start_index"] = addtional_chunk_len
				unique_len = unique_len +  len(tokens_a[start:end])
				segment_dict[num_chunk] = chunk_dict



		assert actual_doc_len == unique_len

	#####################################################
	features = []
	for key, value in segment_dict.items():
		doc_id = segment_dict[key]["doc_id"]
		segment_id = segment_dict[key]["segment_id"]
		bert_tokens = segment_dict[key]["bert_tokens"]
		start_index =  segment_dict[key]["start_index"]

		tokens = []
		segment_ids = []
		tokens.append("[CLS]")
		segment_ids.append(0)
		for token in bert_tokens:
			tokens.append(token)
			segment_ids.append(0)
		tokens.append("[SEP]")
		segment_ids.append(0)

		### because of addition of CLS
		if start_index != 0:
			start_index +=1 

		input_ids = tokenizer.convert_tokens_to_ids(tokens)
		# The mask has 1 for real tokens and 0 for padding tokens. Only real
		# tokens are attended to.
		input_mask = [1] * len(input_ids)

		# Zero-pad up to the sequence length.
		while len(input_ids) < max_seq_length:
			input_ids.append(0)
			input_mask.append(0)
			segment_ids.append(0)

		assert len(input_ids) == max_seq_length
		assert len(input_mask) == max_seq_length
		assert len(segment_ids) == max_seq_length

		if FLAGS.multilabel:
			label_id = [0]*len(label_map)
			m_labels = example.label.split(":")
			for label in m_labels:
				label_id[label_map[label]] = 1


		else:
			label_id = label_map[example.label]


		if ex_index == 0:
			tf.logging.info("*** Example ***")
			tf.logging.info("guid: %s" % (example.guid))
			tf.logging.info("tokens: %s" % " ".join(
					[tokenization.printable_text(x) for x in tokens]))
			tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
			tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
			tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
			#tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

		feature = InputFeatures(
				input_ids=input_ids,
				input_mask=input_mask,
				segment_ids=segment_ids,
				label_id=label_id,
				doc_id = int(doc_id),
				start_idx = start_index,
				is_real_example=True
				)

		## +1 in start index because of the CLS addition ! 

		features.append(feature)

	return features


def file_based_convert_examples_to_features(
		examples, label_map, max_seq_length, tokenizer, output_file, log_file):
	"""Convert a set of `InputExample`s to a TFRecord file."""

	writer = tf.python_io.TFRecordWriter(output_file)
	
	def create_int_feature(values):
			f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
			return f

	doc_id_repeating = []
	for (ex_index, example) in enumerate(examples):
		if ex_index % 50 == 0:
			tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

		features = convert_single_example(ex_index, example, label_map, max_seq_length, tokenizer, doc_id_repeating)

		for feature in features:
			features = collections.OrderedDict()
			features["input_ids"] = create_int_feature(feature.input_ids)
			features["input_mask"] = create_int_feature(feature.input_mask)
			features["segment_ids"] = create_int_feature(feature.segment_ids)

			if FLAGS.multilabel:
				features["label_ids"] = create_int_feature(feature.label_id)
			else:
				features["label_ids"] = create_int_feature([feature.label_id])

			features["is_real_example"] = create_int_feature([int(feature.is_real_example)])
			features["doc_id"] = create_int_feature([int(feature.doc_id)])
			features["start_idx"] = create_int_feature([int(feature.start_idx)])

			tf_example = tf.train.Example(features=tf.train.Features(feature=features))
			writer.write(tf_example.SerializeToString())
	writer.close()

	with open(log_file, "w") as f:
		doc_id_repeating = [str(id) for id in doc_id_repeating]
		f.write("\n".join(doc_id_repeating))




def main(_):

	### debug original vs new ############
	"""
	with open(os.path.join(FLAGS.input_folder, "r8-train.txt"), "r") as f:
		new_text = f.readlines()
	new_text = [text.strip() for text in new_text]

	with open(os.path.join(FLAGS.input_folder, "r8-train_original.txt"), "r") as f:
		orig_text = f.readlines()
	orig_text = [text.strip() for text in orig_text]

	for o_num, text in enumerate(orig_text):
		if text not in new_text:
			import pdb; pdb.set_trace()

	print("all found")
	"""
	##########################################

	output_folder = os.path.join(FLAGS.input_folder, "bert_and_gsm_doc_classification_"  + str(FLAGS.max_seq_length))
	
	if not os.path.isdir(output_folder):
		os.mkdir(output_folder)

	processors = {
			"doc_clf" : ClassificationProcessor
	}

	task_name = FLAGS.task_name.lower()	
	if task_name not in processors:
		raise ValueError("Task not found: %s" % (task_name))
	
	processor = processors[task_name]()
	label_list = processor.get_labels(FLAGS.input_folder)

	label_map = {}
	for num, label in enumerate(label_list):
		label_map[label] = num

	with open(os.path.join(output_folder, "label_dict.pkl"), "wb") as handle:
		pickle.dump(label_map, handle)

	with open(os.path.join(output_folder, "labels.txt"), "w") as f:
		f.write("\n".join(label_list))
	
	prepare_doc_data(FLAGS.input_folder, output_folder)


	tokenizer = tokenization.FullTokenizer(
			vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

	###### TRAINING ###############################################################
	
	tf.logging.info("*** Reading from training input files ***")
	tf.logging.info("	%s", os.path.join(FLAGS.input_folder, "training.txt"))

	train_file = os.path.join(output_folder, "training.tfrecord" )
	tf.logging.info("*** Writing to training output files ***")

	#topic_file =	os.path.join(output_folder, "training_nvdm_docs_non_replicated.csv" )
	train_examples = processor.get_train_examples(FLAGS.input_folder)
	train_log =  os.path.join(output_folder, "training_log.txt" )
	#file_based_convert_examples_to_features(train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file, train_log)
	file_based_convert_examples_to_features(train_examples, label_map, FLAGS.max_seq_length, tokenizer, train_file, train_log)

	print("train_file written!")
	

	########### VALIDATION ###################################################

	tf.logging.info("*** Reading from validation input files ***")
	tf.logging.info("	%s", os.path.join(FLAGS.input_folder, "validation.txt"))

	val_file = os.path.join(output_folder, "validation.tfrecord" )
	tf.logging.info("*** Writing to validation output files ***")

	val_examples = processor.get_dev_examples(FLAGS.input_folder)
	val_log =  os.path.join(output_folder, "validation_log.txt" )
	#file_based_convert_examples_to_features( val_examples, label_list, FLAGS.max_seq_length, tokenizer, val_file, val_log)
	file_based_convert_examples_to_features( val_examples, label_map, FLAGS.max_seq_length, tokenizer, val_file, val_log)

	print("validation file written!")
	

	########### TEST ###################################################

	tf.logging.info("*** Reading from test input files ***")
	tf.logging.info("	%s", os.path.join(FLAGS.input_folder, "test.txt"))

	test_file = os.path.join(output_folder, "test.tfrecord" )
	tf.logging.info("*** Writing to test output files ***")

	test_log =  os.path.join(output_folder, "test_log.txt" )
	test_examples = processor.get_test_examples(FLAGS.input_folder)
	#file_based_convert_examples_to_features( test_examples, label_list, FLAGS.max_seq_length, tokenizer, test_file, test_log)
	file_based_convert_examples_to_features( test_examples, label_map, FLAGS.max_seq_length, tokenizer, test_file, test_log)

	print("test file written!")

if __name__ == "__main__":
	flags.mark_flag_as_required("vocab_file")
	tf.app.run()