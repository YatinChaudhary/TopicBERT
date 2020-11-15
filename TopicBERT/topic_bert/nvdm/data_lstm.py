import os
import glob
import random
import csv
import itertools
import numpy as np
import keras.preprocessing.sequence as pp
import tensorflow as tf
import math
import model.utils as u
#from itertools import izip_longest

seed = 42
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

csv.field_size_limit(2**28)


def file_name(abs_path):
	return os.path.basename(abs_path).replace('.csv', '')


def format_row(row, shuffle=True, multilabel=False):
	y, raw_x = row
	x = np.array([int(v) for v in raw_x.split()])
	if shuffle:
		np.random.shuffle(x)
	
	if multilabel:
		return [y, x, len(x)]
	else:
		return [int(y), x, len(x)]

def split_data_LM(x_LM, y_LM, lm_sent_len=30):
	x_LM_split = []
	y_LM_split = []
	seq_len_LM_split = []
	split_indices = []
	
	for i, s in enumerate(x_LM):
		for si in range(int(math.ceil(len(s) * 1.0 / lm_sent_len))):
			seq = s[si*lm_sent_len:((si+1)*lm_sent_len+1)]
			if len(seq) > 1:
				x_LM_split.append(seq)
				y_LM_split.append(y_LM[i])
				seq_len_LM_split.append(len(seq))
				split_indices.append(i)
	
	return x_LM_split, y_LM_split, seq_len_LM_split, split_indices

def split_data_TM(x_TM, y_TM, seq_len_TM, split_indices):
	x_TM_split = []
	y_TM_split = []
	seq_len_TM_split = []

	for i in split_indices:
		x_TM_split.append(x_TM[i])
		y_TM_split.append(y_TM[i])
		seq_len_TM_split.append(seq_len_TM[i])
	
	return np.array(x_TM_split), np.array(y_TM_split), np.array(seq_len_TM_split)


class Dataset(object):
	def __init__(self, path):
		self.path = path
		files = glob.glob(path + '/*.csv')
		self.collections = {file_name(file): file for file in files}

	def rows(self, collection_name, num_epochs=None):
		if collection_name not in self.collections:
			raise ValueError(
				'Collection not found: {}'.format(collection_name)
			)
		epoch = 0
		while True:
			#with open(self.collections[collection_name], 'rb', newline='') as f:
			with open(self.collections[collection_name], 'r') as f:
				r = csv.reader(f)
				for row in r:
					yield row
			epoch += 1
			if num_epochs and (epoch >= num_epochs):
				#raise StopIteration
				break

	def _batch_iter(self, collection_name, batch_size, num_epochs):
		gen = [self.rows(collection_name, num_epochs)] * batch_size
		return itertools.zip_longest(fillvalue=None, *gen)
		#return izip_longest(fillvalue=None, *gen)

	def batches(self, collection_name, batch_size, num_epochs=None, shuffle=False, max_len=None, multilabel=False):
		for batch in self._batch_iter(collection_name, batch_size, num_epochs):
			data = [format_row(row, shuffle, multilabel) for row in batch if row]
			y, x, seq_lengths = zip(*data)
			if not max_len is None:
				x = pp.pad_sequences(x, maxlen=max_len, padding='post')
			else:
				x = pp.pad_sequences(x, padding='post')
			#yield np.array(y), x, np.array(seq_lengths)
			yield np.array(y), x, np.array(seq_lengths), None, None, None

	def batches_split(self, collection_name, batch_size, 
					num_epochs=None, shuffle=False, 
					max_len=None, multilabel=False, 
					lm_sent_len=30):
		for batch in self._batch_iter(collection_name, batch_size, num_epochs):
			data = [format_row(row, shuffle, multilabel) for row in batch if row]
			y, x, seq_lengths = zip(*data)
			
			x_split, y_split, seq_lengths_split, split_indices = split_data_LM(x, y, lm_sent_len=lm_sent_len)
			
			if max_len is None:
				x_split = pp.pad_sequences(x_split, padding='post')
			else:
				x_split = pp.pad_sequences(x_split, maxlen=max_len, padding='post')
			
			yield np.array(y_split), x_split, np.array(seq_lengths_split), split_indices, x, seq_lengths

	## Function for NVDM dataset

	def batches_nvdm_LM(self, collection_name, batch_size, vocab_size, num_epochs=None, max_len=None, multilabel=False):
		for batch in self._batch_iter(collection_name, batch_size, num_epochs):
			#data_batch = np.zeros((batch_size, vocab_size))
			data_batch = []
			count_batch = []
			y_batch = []
			#mask = np.zeros(batch_size)
			mask = []
			for i, row in enumerate(batch):
				if row:
					count = 0
					y_batch.append(row[0].strip())
					raw_x = row[1].strip()
					id_freqs = u.format_doc(raw_x).split()
					doc = np.zeros(vocab_size)
					for value in id_freqs:
						index, freq = value.strip().split(":")
						#data_batch[i, int(index)] = float(freq)
						doc[int(index)] = float(freq)
						count += int(freq)
					count_batch.append(count)
					#mask[i] = float(1)
					mask.append(float(1))
					data_batch.append(doc)
			#if len(count_batch) < batch_size:
			#	count_batch += [0] * (batch_size - len(count_batch))
			#	y_batch += [-1] * (batch_size - len(count_batch))
			data_batch = np.array(data_batch, dtype=np.float32)
			mask = np.array(mask, dtype=np.float32)
			"""
			if shuffle:
				shuffle_indices = np.random.permutation(len(count_batch))
				data_batch = data_batch[shuffle_indices]
				count_batch = [count_batch[index] for index in shuffle_indices]
				y_batch = [y_batch[index] for index in shuffle_indices]
				mask = mask[shuffle_indices]
			"""
			
			yield y_batch, data_batch, count_batch, mask