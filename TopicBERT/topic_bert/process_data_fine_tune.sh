export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=10
python -m pdb process_data_fine_tune.py --task_name="doc_clf" --input_folder="../dataset/AGnews" --vocab_file=cased_L-12_H-768_A-12/vocab.txt --do_lower_case=False --max_seq_length=32 --clean_string=True --data_specific="" --multilabel=False