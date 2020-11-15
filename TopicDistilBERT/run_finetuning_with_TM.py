import os, sys
import argparse
import pickle
import datetime
import logging
import numpy as np
import pandas as pd

#sys.path.append(os.environ["SRC_MODULE_PATH"])

from src.data.data import Dataset
from src.model.model_classification_with_TM import Transformer_Model
from config import params, model_params, data_params, AttributeDict
from src.model_TM.model_NVDM_TF2 import data_set

def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--datainputdir', type=str, # required=True,
						default="./datasets/raw", help='path of input dataset directory')
	parser.add_argument('--datasetname', type=str, required=True,
						help='path of dataset directory')
	parser.add_argument('--trainfilename', type=str, required=True,
						help='name of the training file')
	parser.add_argument('--valfilename', type=str, required=True,
						help='name of the training file')
	parser.add_argument('--testfilename', type=str, required=True,
						help='name of the training file')
	parser.add_argument('--modeldir', type=str, # required=True,
						default="./output", help='path of output model directory')
	parser.add_argument('--dataoutputdir', type=str, # required=True,
						default="./datasets/processed", help='path of output dataset directory')
	parser.add_argument('--multilabel', type=str, # required=True,
						default="False", help='name of the training file')
	parser.add_argument('--max_length', type=str, # required=True,
						default="32", help='max sequence length')
	parser.add_argument('--batch_size', type=str, # required=True,
						default="32", help='batch size')

	# TM params
	parser.add_argument('--TMtrainfilename', type=str, required=True,
						help='name of the training file for topic model')
	parser.add_argument('--TMvalfilename', type=str, required=True,
						help='name of the training file for topic model')
	parser.add_argument('--TMtestfilename', type=str, required=True,
						help='name of the training file for topic model')
	parser.add_argument('--TM_vocab_length', type=int, required=True,
						default=0, help='vocab size for TM')
	parser.add_argument('--hidden_size_TM', type=int, required=True,
						default=0, help='hidden size for TM')
	parser.add_argument('--n_topic_TM', type=int, required=True,
						default=0, help='number of topics for TM')
	parser.add_argument('--n_sample_TM', type=int, #required=True,
						default=1, help='sampling frequency for TM')
	parser.add_argument('--learning_rate_TM', type=float, #required=True,
						default=0.001, help='learning rate for TM')
	parser.add_argument('--alpha', type=float, #required=True,
						default=0.9, help='learning rate for TM')
	parser.add_argument('--TM_pretrained_model_path', type=str, #required=True,
						default="", help='pretrained model path for TM')
	
	args = parser.parse_args()
	
	args_TM = dict(
		TM_vocab_length = args.TM_vocab_length,
		hidden_size_TM = args.hidden_size_TM,
		n_topic = args.n_topic_TM,
		n_sample = args.n_sample_TM,
		learning_rate = args.learning_rate_TM,
		TM_pretrained_model_path = args.TM_pretrained_model_path,
	)
	params_TM = AttributeDict(args_TM)

	args.multilabel = str2bool(args.multilabel)

	now = datetime.datetime.now()
	datestring = "-date-" + str(now.day) + "-" + str(now.month) + "-" + str(now.year)

	params.output_dir = args.modeldir
	params.savemodeldir = args.datasetname + "-model_finetuning-" + datestring
	params.max_length = int(args.max_length)
	params.batch_size = int(args.batch_size)
	params.alpha = args.alpha

	if not os.path.exists(os.path.join(params.output_dir, params.savemodeldir)):
		os.makedirs(os.path.join(params.output_dir, params.savemodeldir))

	# logging config
	logging_filename = os.path.join(params.output_dir, params.savemodeldir, "finetuning.log")
	logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s',
						filename=logging_filename)

	dataset = Dataset(
		args.datainputdir, 
		args.datasetname, 
		[args.trainfilename, args.valfilename, args.testfilename], 
		args.dataoutputdir, 
		data_params
	)
	
	logging.info("Reading data.")
	training_docs, training_labels = dataset.get_cleaned_data(args.trainfilename)
	validation_docs, validation_labels = dataset.get_cleaned_data(args.valfilename)
	test_docs, test_labels = dataset.get_cleaned_data(args.testfilename)
	
	training_labels = [str(label) for label in training_labels]
	validation_labels = [str(label) for label in validation_labels]
	test_labels = [str(label) for label in test_labels]

	TM_train_url = os.path.join(args.datainputdir, args.datasetname, args.TMtrainfilename)
	TM_val_url = os.path.join(args.datainputdir, args.datasetname, args.TMvalfilename)
	TM_test_url = os.path.join(args.datainputdir, args.datasetname, args.TMtestfilename)
	
	TM_train_set, TM_train_count = data_set(TM_train_url)
	TM_val_set, TM_val_count = data_set(TM_val_url)
	TM_test_set, TM_test_count = data_set(TM_test_url)

	assert(len(TM_train_set) == len(training_labels))
	assert(len(TM_val_set) == len(validation_labels))
	assert(len(TM_test_set) == len(test_labels))

	logging.info("Labels encoded.")
	training_labels, validation_labels, \
	test_labels, unique_labels = dataset.get_encoded_labels(
									training_labels,
									validation_labels,
									test_labels,
									multilabel=args.multilabel,
								)
	
	logging.info("Creating TF model.")
	model = Transformer_Model(params, model_params, params_TM, num_labels=len(unique_labels))
	vocab_token2id = model.tokenizer.get_vocab()
	vocab_id2token = {id: word for word, id in vocab_token2id.items()}
	
	logging.info("Training started.")
	model.fit(
		((training_labels, training_docs), (TM_train_set, TM_train_count)),
		((validation_labels, validation_docs), (TM_val_set, TM_val_count)),
		((test_labels, test_docs), (TM_test_set, TM_test_count)),
		15,
		shuffle=False,
		multilabel=args.multilabel
	)
	
	logging.info("Saving vocabulary.")
	with open(os.path.join(args.dataoutputdir, args.datasetname, "vocab_id2token.pkl"), "wb") as f:
		pickle.dump(vocab_id2token, f, protocol=3)