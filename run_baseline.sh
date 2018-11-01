#!/bin/bash

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

python rc/main.py --trainset data/coqa/coqa-train-v1.0-processed.json --devset data/coqa/coqa-dev-v1.0-processed.json --embed_file data/wordvecs/coqa.crawl-300d-2M.vec --embed_type fasttext --f_history y \
    --dir "$1" \
    --dialog_batched true --batch_size 2 \
    --q_dialog_history true --q_dialog_attn word_hidden_incr \
    --attn_hidden_size 250 \
    --q_dialog_attn_scoring linear_relu \
    --q_dialog_attn_incr_merge linear_both \
    --recency_bias true
