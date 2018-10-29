#!/bin/bash

# A simple CoQA recipe for debugging without cuda.

rm -rf exp/debug
python rc/main.py --trainset data/coqa/coqa-train-v1.0-processed.json --devset data/coqa/coqa-dev-v1.0-processed.json --dir exp/debug --embed_file data/wordvecs/coqa.glove.6B.50d.txt --embed_type fasttext --f_history y --cuda false --debug false \
    --dialog_batched true --batch_size 2 --num_layers 1 --hidden_size 50 \
    --q_dialog_history true --q_dialog_attn word_hidden_incr \
    --q_dialog_attn_incr_merge linear_both \
    --doc_dialog_history false --doc_dialog_attn word_hidden \
    --recency_bias true \
    --history_dialog_answer_f false \
    --history_dialog_time_f false
