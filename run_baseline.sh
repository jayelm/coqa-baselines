#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 EXP_DIR"
    exit 1
fi

python rc/main.py --trainset data/coqa/coqa-train-v1.0-processed.json --devset data/coqa/coqa-dev-v1.0-processed.json --embed_file data/wordvecs/coqa.crawl-300d-2M.vec --embed_type fasttext --f_history y \
    --dir "$1" \
    --dialog_batched true --batch_size 1 \
    --use_history_qhidden false --qhidden_attn word \  # Augment q at sentence level with weighted average of past qs?
    --use_history_qemb false --qemb_attn qhidden \  # Add averages of past sentence-level q representations to each word?
    --use_history_aemb false --aemb_attn qhidden \  # Add averages of past sentence-level a representations to each word?
    --use_history_dialog true --dialog_attn word \  # Add averages of word embeddings of past dialog history to each word? (WORD-LEVEL)
    --history_dialog_answer_f false \   # Add binary marker features distinguishing qs from as in dialog history?
    --history_dialog_time_f false \  # Add features corresponding to timestep differential in dialog history?
    --recency_bias false  # Bias recent qs in attn mechanisms w/ a linear term?
