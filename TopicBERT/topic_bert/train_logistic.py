
import glob
import os.path
import numpy as np
import sys
import string 
import re
import string 
import csv
import argparse
from scipy import sparse
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
import codecs
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
#from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
import json
from collections import Counter
from shutil import copyfile
import pickle
import dill

seed = 42


def prepare_token_data(bert_emb_dict_file, multilabel_flag, bert_emb_type= None ):

	with open(bert_emb_dict_file, "rb" ) as handle:
		bert_emb_dict = dill.load(handle)

	doc_id = []
	labels = []
	embeddings = []
	for key, value in bert_emb_dict.items():
		doc_id.append(key)
		num_tokens = 0
		sum_emb = np.zeros(768, dtype = np.float32)
		temp_labels = []

		for inst in value:
			sum_emb += sum(inst["embedding"]) ## sum of all the token emb 
			num_tokens += len(inst["embedding"])
			if len(temp_labels) != 0:
				if not multilabel_flag:
					assert inst["label"] == temp_labels[0]
				else:
					assert (inst["label"] == temp_labels[0]).all()
			else:
				temp_labels.append(inst["label"]) 
				labels.append(inst["label"])

		if bert_emb_type == "token_avg":
			emb = sum_emb/num_tokens

		elif bert_emb_type == "token_sum":
			emb = sum_emb

		embeddings.append(emb)
	
	embeddings = np.asarray(embeddings)
	assert len(embeddings) == len(labels)

	print("token emb extracted from: " + str(bert_emb_dict_file))

	return embeddings, labels	


def prepare_doc_emb_data(bert_emb_dict, multilabel_flag ):

	doc_id = []
	labels = []
	embeddings = []
	for key, value in bert_emb_dict.items():
		doc_id.append(key)
		sum_emb = np.zeros(768, dtype = np.float32)
		temp_labels = []

		for inst in value:
			sum_emb += inst["embedding"]
			if len(temp_labels) != 0:
				if not multilabel_flag:
					assert inst["label"] == temp_labels[0]
				else:
					assert (inst["label"] == temp_labels[0]).all()
				
			else:
				temp_labels.append(inst["label"]) 
				labels.append(inst["label"])
		avg_emb = sum_emb/len(value)
		embeddings.append(avg_emb)

	embeddings = np.asarray(embeddings)
	assert len(embeddings) == len(labels)

	return embeddings, labels

def get_dist_mat(bert_emb, topic_dist_dict, label, n_topic):

	assert len(bert_emb) == len(label)
	topic_dist = []
	for i in range(len(bert_emb)):
		if str(i) in topic_dist_dict.keys():
			topic_prop = topic_dist_dict[str(i)]
		else:
			topic_prop = np.random.rand(n_topic)
			topic_prop = topic_prop / sum(topic_prop)

		topic_dist.append(topic_prop)

	assert len(bert_emb) == len(label) == len(topic_dist)

	topic_dist = np.asarray(topic_dist)

	return topic_dist




bert_dir = "/home/ubuntu/topic_bert_spot_ohsumed/contexualised_embeddings/ohsumed_tokenwise"

#topic_input_list = ["ohsumed_sigmoid_n_topics_50", "ohsumed_sigmoid_n_topics_100", "ohsumed_sigmoid_n_topics_200" ]
#output_dir_list = ["nvdm_n_topic_50", "nvdm_n_topic_100", "nvdm_n_topic_200"]
#n_topic_list = [50, 100, 200]

topic_input_list = [ "ohsumed_sigmoid_n_topics_200"]
output_dir_list = ["ohsumed_nvdm_n_topic_200"]
n_topic_list = [200]

assert len(topic_input_list) == len(output_dir_list) == len(n_topic_list)

topic_repres_path = "/home/ubuntu/topic_bert_spot_ohsumed/outputs_nvdm_only/ohsumed"
output_path = "/home/ubuntu/topic_bert_spot_ohsumed/logistic_outputs/ohsumed/bert_and_nvdm"

classifier = "logistic"
f1 = "macro"
bert_emb_type = "token_avg" #[doc|token_avg|token_sum]
bert_and_topic = True
only_topic = False
only_bert = False
multilabel= True

if bert_emb_type == "doc":

	train_file_bert = os.path.join(bert_dir, "train_emb.npy")
	train_bert = np.load(train_file_bert, allow_pickle = True).item()

	validation_file_bert = os.path.join(bert_dir, "validation_emb.npy")
	validation_bert = np.load(validation_file_bert, allow_pickle = True).item()

	test_file_bert = os.path.join(bert_dir, "test_emb.npy")
	test_bert =  np.load(test_file_bert, allow_pickle = True).item()

	train_bert_emb, train_label   = prepare_doc_emb_data(train_bert, multilabel_flag= multilabel)
	dev_bert_emb, dev_label   = prepare_doc_emb_data(validation_bert, multilabel_flag= multilabel)
	test_bert_emb, test_label   = prepare_doc_emb_data(test_bert, multilabel_flag= multilabel)

if bert_emb_type == "token_avg" or bert_emb_type == "token_sum":

	train_file_bert = os.path.join(bert_dir, "train_token_emb.pkl")
	#train_bert = np.load(train_file_bert, allow_pickle = True).item()

	validation_file_bert = os.path.join(bert_dir, "validation_token_emb.pkl")
	#validation_bert = np.load(validation_file_bert, allow_pickle = True).item()

	test_file_bert = os.path.join(bert_dir, "test_token_emb.pkl")
	#test_bert =  np.load(test_file_bert, allow_pickle = True).item()

	train_bert_emb, train_label   = prepare_token_data(train_file_bert, multilabel_flag= multilabel, bert_emb_type = bert_emb_type)
	dev_bert_emb, dev_label   = prepare_token_data(validation_file_bert, multilabel_flag= multilabel,  bert_emb_type = bert_emb_type)
	test_bert_emb, test_label   = prepare_token_data(test_file_bert, multilabel_flag= multilabel, bert_emb_type = bert_emb_type)

if bert_and_topic == True or only_topic == True:
	for i in range(len(topic_input_list)):

		topic_repres_dir = os.path.join(topic_repres_path, topic_input_list[i])	
		output_dir = os.path.join(output_path, output_dir_list[i])	
		n_topic = n_topic_list[i]

		if not os.path.isdir(output_dir):
			os.mkdir(output_dir)


		train_topic_dist_file =  os.path.join(topic_repres_dir, "train_topic_dist.npy")
		train_topic_dist_dict  = np.load(train_topic_dist_file, allow_pickle = True).item()
		train_topic_dist = get_dist_mat(train_bert_emb, train_topic_dist_dict, train_label, n_topic)

		dev_topic_dist_file =  os.path.join(topic_repres_dir, "val_topic_dist.npy")
		dev_topic_dist_dict =  np.load(dev_topic_dist_file, allow_pickle = True).item()
		dev_topic_dist = get_dist_mat(dev_bert_emb, dev_topic_dist_dict, dev_label, n_topic)

		test_topic_dist_file =  os.path.join(topic_repres_dir, "test_topic_dist.npy")
		test_topic_dist_dict =  np.load(test_topic_dist_file, allow_pickle = True).item()
		test_topic_dist = get_dist_mat(test_bert_emb, test_topic_dist_dict, test_label, n_topic)


		if bert_and_topic:
			train_data = np.concatenate([train_bert_emb, train_topic_dist], axis = 1)
			dev_data = np.concatenate([dev_bert_emb, dev_topic_dist], axis = 1)
			test_data = np.concatenate([test_bert_emb, test_topic_dist], axis = 1)

		if only_bert:
			train_data = train_bert_emb
			dev_data = dev_bert_emb
			test_data = test_bert_emb

		if only_topic:
			train_data = train_topic_dist
			dev_data = dev_topic_dist
			test_data = test_topic_dist

		if classifier == "logistic":
			if multilabel:
				C = [0.000001,0.00001,  0.0001, 0.001, 0.01 , 0.1, 1, 5.0, 10.0, 100.0, 1000.0]
			else:
				C = [0.000001,0.00001,  0.0001, 0.001, 0.01 , 0.1, 1, 5.0, 10.0, 100.0, 1000.0, 10000.0]

			best_f1 =  - np.inf
			for c in C:
				log_file = os.path.join(output_dir, str(classifier)  + "_log.txt")
				#clf = LogisticRegression(C = c, random_state = seed, class_weight="balanced")
				if multilabel:
					#clf = MLPClassifier(hidden_layer_sizes = (), activation = "logistic", alpha = c, random_state=seed ) 
					#clf.fit(train_data, train_label)
					#dev_pred = clf.predict(dev_data)
					train_label = np.asarray(train_label)
					#clf = OneVsRestClassifier(SVC(C=c, kernel='linear', random_state = seed ))
					clf = OneVsRestClassifier(LinearSVC(C=c, random_state = seed , dual=False))
					clf.fit(train_data, train_label)
					dev_pred = clf.predict(dev_data)

				else:
					clf = LogisticRegression(C = c, random_state = seed,  max_iter=1000)
					clf.fit(train_data, train_label)
					dev_pred = clf.predict(dev_data)


				#dev_f1_score = f1_score(dev_label, dev_pred)

				if multilabel:
					dev_label = np.asarray(dev_label)
					dev_pred = np.asarray(dev_pred)

				dev_prec, dev_recall, dev_f1_score, _ = precision_recall_fscore_support(dev_label, dev_pred,  pos_label=None , average=f1)

				with open(log_file, "a") as log:
					print( "f1 score: " + str(dev_f1_score))
					log.write( str(c) + "\t" + str(dev_f1_score) + "\n")

				if dev_f1_score > best_f1:
					best_f1 = dev_f1_score
					best_prec = dev_prec
					best_recall = dev_recall
					best_clf = clf
					best_c = c


			dev_pred_best_clf = best_clf.predict(dev_data)
			
			if multilabel:
				dev_pred_best_clf = np.asarray(dev_pred_best_clf)

			micro_dev_prec, micro_dev_recall, micro_dev_f1_score, _ = precision_recall_fscore_support(dev_label, dev_pred_best_clf,  pos_label=None, average="micro")
			#bin_dev_prec, bin_dev_recall, bin_dev_f1_score, _ = precision_recall_fscore_support(dev_label, dev_pred_best_clf,  pos_label= 1, average="binary")
			dev_acc = accuracy_score(dev_label, dev_pred_best_clf)

			print("Best C: " + str(best_c))
			print("Best dev f1 score: " + str(best_f1))
			print("Best dev precision: " + str(best_prec))
			print("Best dev recall: " + str(best_recall))

			with open(log_file, "a") as log:
				log.write("Best C: " + str(best_c) +  "  Best dev Macro f1: " + str(best_f1)\
				+ "  Best dev Macro precision score: " + str(best_prec)  +  "  Best dev Macro recall: " + str(best_recall) \
				+ "  Micro f1: " + str(micro_dev_f1_score) + "  Micro prec: " + str(micro_dev_prec) \
				+ "  Micro recall: " + str(micro_dev_recall)  + "  Accuracy: " + str(dev_acc) +  "\n")


			test_pred_best_clf = best_clf.predict(test_data)

			if multilabel:
				test_label = np.asarray(test_label)
				test_pred_best_clf = np.asarray(test_pred_best_clf)


			test_prec, test_recall, test_f1_score, _ = precision_recall_fscore_support(test_label, test_pred_best_clf,  pos_label=None , average=f1)
			micro_test_prec, micro_test_recall, micro_test_f1_score, _ = precision_recall_fscore_support(test_label, test_pred_best_clf,  pos_label=None, average="micro")
			#bin_test_prec, bin_test_recall, bin_test_f1_score, _ = precision_recall_fscore_support(test_label, test_pred_best_clf,  pos_label=1, average="binary")
			test_acc = accuracy_score(test_label, test_pred_best_clf)

			with open(log_file, "a") as log:
				log.write("Best C: " + str(best_c) +  "Best test Macro f1: " + str(test_f1_score)\
				+ "  Best test Macro precision score: " + str(test_prec)  +  "  Best test Macro recall: " + str(test_recall) \
				+ "  Test Micro f1: " + str(micro_test_f1_score) + "  Test Micro prec: " + str(micro_test_prec) \
				+ "  Test Micro recall: " + str(micro_test_recall)  + "  Test Accuracy: " + str(test_acc) +  "\n")

elif only_bert:

	train_data = train_bert_emb
	dev_data = dev_bert_emb
	test_data = test_bert_emb

	output_dir = os.path.join(output_path, str(bert_emb_type) + "_emb_768")	

	if not os.path.isdir(output_dir):
		os.mkdir(output_dir)

	if classifier == "logistic":

			if multilabel:
				C = [0.000001,0.00001,  0.0001, 0.001, 0.01 , 0.1, 1, 5.0, 10.0, 100.0, 1000.0]
			else:
				C = [0.000001,0.00001,  0.0001, 0.001, 0.01 , 0.1, 1, 5.0, 10.0, 100.0, 1000.0, 10000.0]

			best_f1 =  - np.inf
			for c in C:
				log_file = os.path.join(output_dir, str(classifier)  + "_log.txt")
				#clf = LogisticRegression(C = c, random_state = seed, class_weight="balanced")

				if multilabel:
					#clf = MLPClassifier(hidden_layer_sizes = (), activation = "logistic", alpha = c, random_state=seed ) 
					#clf.fit(train_data, train_label)
					#dev_pred = clf.predict(dev_data)

					train_label = np.asarray(train_label)
					#clf = OneVsRestClassifier(SVC(C=c, kernel='linear', random_state = seed ))
					clf = OneVsRestClassifier(LinearSVC(C=c, random_state = seed, dual=False ))
					clf.fit(train_data, train_label)
					dev_pred = clf.predict(dev_data)

				else:
					clf = LogisticRegression(C = c, random_state = seed,  max_iter=1000)
					clf.fit(train_data, train_label)
					dev_pred = clf.predict(dev_data)

				if multilabel:
					dev_label = np.asarray(dev_label)
					dev_pred = np.asarray(dev_pred)
				#dev_f1_score = f1_score(dev_label, dev_pred)

				dev_prec, dev_recall, dev_f1_score, _ = precision_recall_fscore_support(dev_label, dev_pred,  pos_label=None , average=f1)

				with open(log_file, "a") as log:
					print( "f1 score: " + str(dev_f1_score))
					log.write( str(c) + "\t" + str(dev_f1_score) + "\n")

				if dev_f1_score > best_f1:
					best_f1 = dev_f1_score
					best_prec = dev_prec
					best_recall = dev_recall
					best_clf = clf
					best_c = c


			dev_pred_best_clf = best_clf.predict(dev_data)

			if multilabel:
				dev_pred_best_clf = np.asarray(dev_pred_best_clf)

			micro_dev_prec, micro_dev_recall, micro_dev_f1_score, _ = precision_recall_fscore_support(dev_label, dev_pred_best_clf,  pos_label=None, average="micro")
			#bin_dev_prec, bin_dev_recall, bin_dev_f1_score, _ = precision_recall_fscore_support(dev_label, dev_pred_best_clf,  pos_label= 1, average="binary")
			dev_acc = accuracy_score(dev_label, dev_pred_best_clf)

			print("Best C: " + str(best_c))
			print("Best dev f1 score: " + str(best_f1))
			print("Best dev precision: " + str(best_prec))
			print("Best dev recall: " + str(best_recall))

			with open(log_file, "a") as log:
				log.write("Best C: " + str(best_c) +  "  Best dev Macro f1: " + str(best_f1)\
				+ "  Best dev Macro precision score: " + str(best_prec)  +  "  Best dev Macro recall: " + str(best_recall) \
				+ "  Micro f1: " + str(micro_dev_f1_score) + "  Micro prec: " + str(micro_dev_prec) \
				+ "  Micro recall: " + str(micro_dev_recall)  + "  Accuracy: " + str(dev_acc) +  "\n")


			test_pred_best_clf = best_clf.predict(test_data)
			
			if multilabel:
				test_label = np.asarray(test_label)
				test_pred_best_clf = np.asarray(test_pred_best_clf)

			test_prec, test_recall, test_f1_score, _ = precision_recall_fscore_support(test_label, test_pred_best_clf,  pos_label=None , average=f1)
			micro_test_prec, micro_test_recall, micro_test_f1_score, _ = precision_recall_fscore_support(test_label, test_pred_best_clf,  pos_label=None, average="micro")
			#bin_test_prec, bin_test_recall, bin_test_f1_score, _ = precision_recall_fscore_support(test_label, test_pred_best_clf,  pos_label=1, average="binary")
			test_acc = accuracy_score(test_label, test_pred_best_clf)

			with open(log_file, "a") as log:
				log.write("Best C: " + str(best_c) +  "Best test Macro f1: " + str(test_f1_score)\
				+ "  Best test Macro precision score: " + str(test_prec)  +  "  Best test Macro recall: " + str(test_recall) \
				+ "  Test Micro f1: " + str(micro_test_f1_score) + "  Test Micro prec: " + str(micro_test_prec) \
				+ "  Test Micro recall: " + str(micro_test_recall)  + "  Test Accuracy: " + str(test_acc) +  "\n")


else:
	import pdb; pdb.set_trace()



