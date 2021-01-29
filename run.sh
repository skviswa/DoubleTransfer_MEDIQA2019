# train with MT-DNN
python train.py --train_datasets mednli,rqe,mediqa,medquad --save_last --save_best --mediqa_score adjusted --mediqa_score_offset -2.0 --batch_size 16 --max_seq_len 384 --data_dir ../data/mediqa_processed/mt_dnn_mediqa_384/ --init_checkpoint ../bert_models/mt_dnn_base.pt --float_medquad --external_datasets mnli --mtl_opt 0 --output_dir ../model_data/
# train with SciBERT
python train.py --train_datasets mednli,rqe,mediqa,medquad --save_last --save_best --mediqa_score adjusted --mediqa_score_offset -2.0 --batch_size 16 --max_seq_len 384 --data_dir ../data/mediqa_processed/mt_dnn_mediqa_384_scibert/ --init_checkpoint ../bert_models/scibert/scibert_scivocab_uncased/pytorch_model.bin --init_config ../bert_models/scibert/scibert_scivocab_uncased/bert_config.json --float_medquad --external_datasets mnli --mtl_opt 0 --output_dir ../model_data/
# train with BlueBERT
python train.py --train_datasets mednli,rqe,mediqa,medquad --save_last --save_best --mediqa_score adjusted --mediqa_score_offset -2.0 --batch_size 16 --max_seq_len 384 --data_dir ../data/mediqa_processed/mt_dnn_mediqa_384_bluebert/ --init_checkpoint ../bert_models/NCBI_BERT_pubmed_mimic_uncased_L-12_H-768_A-12/pytorch_model.bin --init_config ../bert_models/NCBI_BERT_pubmed_mimic_uncased_L-12_H-768_A-12/bert_config.json --float_medquad --external_datasets mnli --mtl_opt 0 --output_dir ../model_data_bluebert/

# train with BioBERT base cased v1.1
python train.py --train_datasets mednli,rqe,mediqa,medquad --save_last --save_best --mediqa_score adjusted --mediqa_score_offset -2.0 --batch_size 16 --max_seq_len 384 --data_dir ../data/mediqa_processed/mt_dnn_mediqa_384_cased_biobert/ --init_checkpoint ../bert_models/biobert-base-cased-v1.1/pytorch_model.bin --init_config ../bert_models/biobert-base-cased-v1.1/config.json --float_medquad --external_datasets mnli --mtl_opt 0 --output_dir ../model_data_biobert/


transformers-cli convert --model_type bert \
  --tf_checkpoint CQA_Example/bert_models/NCBI_BERT_pubmed_mimic_uncased_L-12_H-768_A-12/bert_model.ckpt \
  --config CQA_Example/bert_models/NCBI_BERT_pubmed_mimic_uncased_L-12_H-768_A-12/bert_config.json \
  --pytorch_dump_output CQA_Example/bert_models/NCBI_BERT_pubmed_mimic_uncased_L-12_H-768_A-12/pytorch_model.bin
