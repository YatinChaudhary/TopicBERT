import os, sys
import string, csv
import pickle
import copy
from glob import glob
import pandas as pd
import numpy as np
import tensorflow as tf
import itertools
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
#import keras.preprocessing.sequence as pp


class Dataset:
	"""Utility class for dataset handling and preprocessing"""

	def __init__(self, datainputdir, datasetname, 
				filenames, dataoutputdir, processing_params):
		
		self.datainputdir      = datainputdir
		self.datasetname       = datasetname
		self.filenames         = filenames
		self.dataoutputdir     = dataoutputdir
		self.processing_params = processing_params

		self.input_filenames = {
			filename: os.path.join(datainputdir, datasetname, filename)
				for filename in filenames
		}
		self.processed_output_filenames = {
			filename: os.path.join(dataoutputdir, datasetname, ".".join(filename.split(".")[:-1]) + ".pkl")
				for filename in filenames
		}

		self.punc_table = str.maketrans({key: " " for key in string.punctuation})
		self.tokenizer = RegexpTokenizer(r'\w+')

		if not os.path.exists(os.path.join(dataoutputdir, datasetname)):
			os.makedirs(os.path.join(dataoutputdir, datasetname))

	def clean_str(self, text):
		text = text.strip()

		if self.processing_params.do_lower:
			# do lowercase
			text = text.lower()

		if self.processing_params.remove_punctuation:
			# remove punctuations
			text = text.replace("_", " ")
			#text = text.translate(self.punc_table)
			text = " ".join(self.tokenizer.tokenize(text))
		
		tokens = text.split()

		if self.processing_params.remove_numbers:
			# remove numbers
			tokens = [token if not token.isdigit() else "[unused0]" for token in tokens]

		text = " ".join(tokens)
		return text

	def get_raw_data(self, filename):
		with open(self.input_filenames[filename], "r", encoding="utf-8") as f:
			data = pd.read_csv(f, sep=self.processing_params.delimiter)
		return {
			self.processing_params.label_header: data[self.processing_params.label_header],
			self.processing_params.document_header: data[self.processing_params.document_header],
		}

	def get_cleaned_data(self, filename):
		with open(self.input_filenames[filename], "r", encoding="utf-8") as f:
			data = pd.read_csv(f, sep=self.processing_params.delimiter)
		cleaned_docs_list = []
		for doc in data[self.processing_params.document_header]:
			doc = self.clean_str(doc)
			cleaned_docs_list.append(doc.strip())
		return cleaned_docs_list, data[self.processing_params.label_header]

	def get_encoded_labels(self, training_labels, validation_labels, test_labels, multilabel=False):
		if multilabel:
			training_labels = [label.strip().split(self.processing_params.multilabelsplitter) for label in training_labels]
			validation_labels = [label.strip().split(self.processing_params.multilabelsplitter) for label in validation_labels]
			test_labels = [label.strip().split(self.processing_params.multilabelsplitter) for label in test_labels]

			total_labels = []
			total_labels.extend([label for jointlabel in training_labels for label in jointlabel])
			total_labels.extend([label for jointlabel in validation_labels for label in jointlabel])
			total_labels.extend([label for jointlabel in test_labels for label in jointlabel])
			unique_labels = [np.unique(total_labels)]
			self.label_transform = preprocessing.MultiLabelBinarizer()
		else:
			training_labels = [label.strip() for label in training_labels]
			validation_labels = [label.strip() for label in validation_labels]
			test_labels = [label.strip() for label in test_labels]

			total_labels = training_labels + validation_labels + test_labels
			unique_labels = np.unique(total_labels)
			self.label_transform = preprocessing.LabelEncoder()
		
		self.label_transform.fit(unique_labels)

		training_labels = self.label_transform.transform(training_labels)
		validation_labels = self.label_transform.transform(validation_labels)
		test_labels = self.label_transform.transform(test_labels)

		if multilabel:
			return training_labels.astype(np.float32), \
					validation_labels.astype(np.float32), \
					test_labels.astype(np.float32), \
					self.label_transform.classes_
		else:
			return training_labels.astype(np.int32), \
					validation_labels.astype(np.int32), \
					test_labels.astype(np.int32), \
					self.label_transform.classes_