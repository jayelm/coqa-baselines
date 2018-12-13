#!/bin/bash

set -e

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 MODEL_CKPT"
    exit 1
fi

BERT_BASE_DIR="./bert/models/uncased_L-12_H-768_A-12"
COQA_DIR="./data/coqa"

python bert/rc2.py \
    --bert_model bert-base-uncased \
    --init_checkpoint "$1" \
    --do_predict \
    --do_lower_case \
    --train_file $COQA_DIR/coqa-train-v1.0-processed-bert-uncased.json \
    --predict_file $COQA_DIR/coqa-dev-v1.0-processed-bert-uncased.json \
    --predict_batch_size 2 \
    --learning_rate 1e-5 \
    --num_train_epochs 8.0 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --save_checkpoints_steps 4000 \
    --output_dir "`dirname $1`"

python "$COQA_DIR/evaluate-v1.0.py" --data-file "$COQA_DIR/coqa-dev-v1.0.json" --pred-file "$(dirname $1)/predictions.json" | tee "$(dirname $1)/$(basename $1)-metrics.json"
