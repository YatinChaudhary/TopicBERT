# TopicBERT: Topic-aware BERT for Efficient Document Classification

## Topic BERT setup:

### Download and setup BERT-base cased pretrained model as shown below:

-> chmod +x download_bert_base_model.sh
-> ./download_bert_base_model.sh


### Run create_env.sh for creating the environment for installing dependencies.

-> chmod +x create_env.sh
-> ./create_env.sh


### For Finetuning BERT: 
	1. set the corresponding parameters (default parameters are set) 
	1. run the following commands

	-> source projectenv/bin/activate
	-> cd topic_bert
	-> python run_finetuning_spot.py --use_gpu=True --topic_model='nvdm' --data_dir='../dataset/reuters8/bert_and_gsm_doc_classification_512' --propD=False --prob_thres=0.5 --output_dir='../outputs_doc_classification/reuters8/nvdm/reuters8_finetune_bert_100_512_e_25' --multilabel=False --pretrain_supervised_TM=False --beta=0.9 --validate_supervised_TM='f1' --finetune_bert=True --load_wiki_during_pretrain=True --use_static_topic=False --pretrain_nvdm=False --supervised_TM=False --combined_train=False --sparse_topics=False --max_seq_length=512 --learning_rate=2e-5 --train_batch_size=4 --eval_batch_size=4 --alpha=0.5 --gsm_lr_factor=1.0 --num_train_epochs=25 --patience=50 --concat=True --projection=True --nvdm_learning_rate=0.001 --nvdm_batch_size=64 --nvdm_patience=10 --nvdm_train_epoch=1000 --nvdm_alternate_epoch=10 --n_sample=10 --n_topic=100 --pretrain_nvdm_path='' --static_topic_path='' --avg_softmax=True


### For TopicBERT (Topic model + BERT) finetuning:
	1. set the corresponding parameters (default parameters are set) 
	2. run the following commands

	-> source projectenv/bin/activate
	-> cd topic_bert
	-> python run_finetuning_spot.py --use_gpu=True --topic_model='nvdm' --data_dir='../dataset/reuters8/bert_and_gsm_doc_classification_512' --propD=False --prob_thres=0.5 --output_dir='../outputs_doc_classification/reuters8/nvdm/reuters8_comb_finetune_bert_100_512_e_25_alpha_0.9' --multilabel=False --pretrain_supervised_TM=False --beta=0.9 --validate_supervised_TM='f1' --finetune_bert=False --load_wiki_during_pretrain=True --use_static_topic=False --pretrain_nvdm=True --supervised_TM=False --combined_train=True --sparse_topics=False --max_seq_length=512 --learning_rate=2e-5 --train_batch_size=4 --eval_batch_size=4 --alpha=0.9 --gsm_lr_factor=1.0 --num_train_epochs=25 --patience=50 --concat=True --projection=True --nvdm_learning_rate=0.001 --nvdm_batch_size=64 --nvdm_patience=10 --nvdm_train_epoch=1000 --nvdm_alternate_epoch=10 --n_sample=10 --n_topic=100 --pretrain_nvdm_path='../outputs_nvdm_only/reuters8/reuters8_sigmoid_n_topics_100/model_ppl_nvdm_pretrain/model_ppl_nvdm_pretrain-1' --static_topic_path='' --avg_softmax=True


### The results can be found in "outputs_doc_classification" folder.


