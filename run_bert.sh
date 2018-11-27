#!/bin/bash

set -e

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 EXP_DIR"
    exit 1
fi

if [ -d "$1" ]; then
    read -p "$1 already exists. Remove? [yN] " -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -r $1
    else
        exit 0
    fi
fi

BERT_BASE_DIR="./bert/models/uncased_L-12_H-768_A-12"
COQA_DIR="./data/coqa"

python bert/run_coqa.py \
    --vocab_file "$BERT_BASE_DIR/vocab.txt" \
    --bert_config_file "$BERT_BASE_DIR/bert_config.json" \
    --init_checkpoint "$BERT_BASE_DIR.pth" \
    --do_train \
    --do_predict \
    --do_lower_case \
    --train_file $COQA_DIR/coqa-train-v1.0-processed-bert-uncased.json \
    --predict_file $COQA_DIR/coqa-dev-v1.0-processed-bert-uncased.json \
    --train_batch_size 12 \
    --learning_rate 1e-5 \
    --num_train_epochs 2.0 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir "$1"

python "$COQA_DIR/evaluate-v1.0.py" --data-file "$COQA_DIR/coqa-dev-v1.0.json" --pred-file "$1/predictions.json"
