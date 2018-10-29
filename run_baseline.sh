#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 EXP_DIR"
    exit 1
fi

python rc/main.py --trainset data/coqa/coqa-train-v1.0-processed.json --devset data/coqa/coqa-dev-v1.0-processed.json --embed_file data/wordvecs/coqa.crawl-300d-2M.vec --embed_type fasttext --f_history y \
    --dir "$1" \
    --dialog_batched true --batch_size 2 \
    --q_dialog_history true --q_dialog_attn word_hidden_incr \
    --q_dialog_attn_incr_merge average \
    --doc_dialog_history false --doc_dialog_attn word_hidden \
    --history_dialog_answer_f false \
    --history_dialog_time_f false \
    --recency_bias true
