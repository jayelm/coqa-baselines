#!/bin/bash

# TODO: Make this load the relevant config values from the config.
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 EXP_DIR"
    exit 1
fi

python rc/main.py --pretrained "$1" --testset data/coqa/coqa-dev-v1.0-processed.json --embed_file data/wordvecs/coqa.crawl-300d-2M.vec --embed_type fasttext --f_history y --dialog_batched true --batch_size 1 --recency_bias true \
           --doc_dialog_history true --doc_dialog_attn word \
           --q_dialog_history true --q_dialog_attn word \
           --save_params false \
           --recency_bias true
