#!/bin/bash

set -e

BERT_BASE_DIR="./bert/models/uncased_L-12_H-768_A-12"
COQA_DIR="./data/coqa"

rm -rf bexp/debug
python bert/run_coqa.py \
    --vocab_file "$BERT_BASE_DIR/vocab.txt" \
    --bert_config_file "$BERT_BASE_DIR/bert_config.json" \
    --init_checkpoint "$BERT_BASE_DIR.pth" \
    --do_train \
    --do_predict \
    --do_lower_case \
    --train_file $COQA_DIR/coqa-dev-v1.0-processed-bert-uncased.json \
    --predict_file $COQA_DIR/coqa-dev-v1.0-processed-bert-uncased.json \
    --train_batch_size 2 \
    --predict_batch_size 2 \
    --learning_rate 3e-5 \
    --num_train_epochs 1.0 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir bexp/debug

python "$COQA_DIR/evaluate-v1.0.py" --data-file "$COQA_DIR/coqa-dev-v1.0.json" --pred-file bexp/debug/predictions.json
