
#!/bin/bash

set -e

BERT_BASE_DIR="./bert/models/cased_L-12_H-768_A-12"
SQUAD_DIR="./data/squad"

rm -rf bexp/debug
python bert/run_squad.py \
    --vocab_file "$BERT_BASE_DIR/vocab.txt" \
    --bert_config_file "$BERT_BASE_DIR/bert_config.json" \
    --init_checkpoint "$BERT_BASE_DIR.pth" \
    --do_train \
    --do_predict \
    --do_lower_case \
    --train_file $SQUAD_DIR/train-v1.1.json \
    --predict_file $SQUAD_DIR/dev-v1.1.json \
    --train_batch_size 12 \
    --learning_rate 3e-5 \
    --num_train_epochs 2.0 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir bexp/debug

# Evaluate
echo "==== EVALUATING ===="
python data/squad/evaluate-v1.1.py data/squad/dev-v1.1.json "bexp/debug/predictions.json" | tee "bexp/debug/metrics.json"
