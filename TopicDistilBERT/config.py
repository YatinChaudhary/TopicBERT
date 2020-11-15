import os, sys
from huggingface_config import *

class AttributeDict(dict):
	def __getattr__(self, attr):
		return self[attr]
	def __setattr__(self, attr, value):
		self[attr] = value


model = DistilBERT_model_config

config = dict(
	output_dir            = "./output",
	savemodeldir          = "temp",
	model_class_unsup     = model["model_class_unsup"],
	model_class_sup       = model["model_class_sup"],
	tokenizer_class       = model["tokenizer_class"],
	pretrained_model_name = model["pretrained_model_name"],
	special_token_begin   = model["special_token_begin"],
	special_token_end     = model["special_token_end"],
	seq_start_from_begin  = model["seq_start_from_begin"],
	cls_comb_strategy     = "sum",
	copy_to_s3            = False,
)
params = AttributeDict(config)

model_config = dict(
	# Transformer model parameters
	output_hidden_states  = False,
	output_attentions     = False,
)
model_params = AttributeDict(model_config)

data_config = dict(
	do_lower           = False,
	remove_punctuation = False,
	remove_numbers     = False,
	label_header       = "LABEL",
	document_header    = "DOCUMENT",
	delimiter          = "\t",
	multilabelsplitter = ":",
)
data_params = AttributeDict(data_config)

tokenizer_config = dict()
tokenizer_params = AttributeDict(tokenizer_config)