#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 EXP_DIR"
    exit 1
fi

python rc/main.py --trainset data/coqa/coqa-train-v1.0-processed.json --devset data/coqa/coqa-dev-v1.0-processed.json --embed_file data/wordvecs/coqa.crawl-300d-2M.vec --embed_type fasttext --f_history y \
    --dir "$1" \
    --dialog_batched true --batch_size 1 \
    --use_history_qhidden false --qhidden_attn word \
    --use_history_qemb false --qemb_attn qhidden \
    --use_history_aemb false --aemb_attn qhidden \
    --use_history_dialog true --dialog_attn word \
    --history_dialog_answer_f false \
    --history_dialog_time_f false \
    --recency_bias false
