import os 
import tensorflow as tf
import numpy as np

model_path  = "../outputs_nvdm_only/reuters8/reuters8_sigmoid_n_topics_100_arr_to_dict"
ppl_model =  os.path.join(model_path,  "model_ppl_nvdm_pretrain/model_ppl_nvdm_pretrain-1" )
ppl_model_meta = os.path.join(model_path, "model_ppl_nvdm_pretrain/model_ppl_nvdm_pretrain-1.meta")
dataset = "../dataset/reuters8/bert_and_gsm_doc_classification"
vocab_path = "../dataset/reuters8/bert_and_gsm_doc_classification"
#ckpt_path = os.path.join(model_path, ppl_model)

with tf.Session() as session:
	saver = tf.train.import_meta_graph(ppl_model_meta, clear_devices=True)
	saver.restore(session, ppl_model) 
	print("Best bert Model restored from the pretrained bert model")
	decoder_Mat = session.run("TM_decoder/projection/Matrix:0") 

print("Matrix extracted")

top_n_topic_words = 20
w_h_top_words_indices = []
W_topics = decoder_Mat
topics_list_W = []

log_dir  = os.path.join(model_path , "logs_nvdm_pretrain")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

for h_num in range(np.array(W_topics).shape[0]):
    w_h_top_words_indices.append(np.argsort(W_topics[h_num, :])[::-1][:top_n_topic_words])

with open(vocab_path + "/vocab_docnade.vocab", 'r') as f:
    vocab_docnade = [w.strip() for w in f.readlines()]


with open(os.path.join(log_dir, "TM_topics.txt"), "w") as f:
    for w_h_top_words_indx, h_num in zip(w_h_top_words_indices, range(len(w_h_top_words_indices))):
        w_h_top_words = [vocab_docnade[w_indx] for w_indx in w_h_top_words_indx]
        
        topics_list_W.append(w_h_top_words)
        
        #print('h_num: %s' % h_num)
        #print('w_h_top_words_indx: %s' % w_h_top_words_indx)
        #print('w_h_top_words:%s' % w_h_top_words)
        #print('----------------------------------------------------------------------')

        f.write('h_num: %s\n' % h_num)
        f.write('w_h_top_words_indx: %s\n' % w_h_top_words_indx)
        f.write('w_h_top_words:%s\n' % w_h_top_words)
        f.write('----------------------------------------------------------------------\n')

# Compute Topic Coherence with internal corpus
    
topic_file = os.path.join(log_dir, "TM_topics.txt")
ref_corpus = dataset
coherence_file_path = os.path.join(log_dir, "window_size_document")
if not os.path.exists(coherence_file_path):
    os.makedirs(coherence_file_path)
#coherence_file = os.path.join(coherence_file_path, "topics-oc_bnc_internal.txt")
#wordcount_file = os.path.join(coherence_file_path, "wc-oc_bnc_internal.txt")
#os.system('python ./topic_coherence_code_python3/ComputeTopicCoherence.py ' + topic_file + ' ' + ref_corpus + ' ' + wordcount_file + ' ' + coherence_file)

# Compute Topic Coherence with external corpus
ref_corpus =  "./topic_coherence_code/wiki_corpus"
coherence_file = os.path.join(coherence_file_path, "topics-oc_bnc_external.txt")
wordcount_file = os.path.join(coherence_file_path, "wc-oc_bnc_external.txt")
os.system('python ./topic_coherence_code_python3/ComputeTopicCoherence.py ' + topic_file + ' ' + ref_corpus + ' ' + wordcount_file + ' ' + coherence_file)
