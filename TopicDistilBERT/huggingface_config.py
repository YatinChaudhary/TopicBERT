import tensorflow as tf
from transformers import *

BERT_model_config = dict(
	tokenizer_class       = BertTokenizer,
	model_class_unsup     = TFBertModel,
	model_class_sup       = TFBertForSequenceClassification,
	pretrained_model_name = "bert-base-cased",
	special_token_begin   = 1,
	special_token_end     = 1,
	seq_start_from_begin  = 1,
)

DistilBERT_model_config = dict(
	tokenizer_class       = DistilBertTokenizer,
	model_class_unsup     = TFDistilBertModel,
	model_class_sup       = TFDistilBertForSequenceClassification,
	pretrained_model_name = "distilbert-base-cased",
	special_token_begin   = 1,
	special_token_end     = 1,
	seq_start_from_begin  = 1,
)