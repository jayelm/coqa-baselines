#!/bin/bash

# A simple CoQA recipe for debugging without cuda.

rm -rf exp/debug
python rc/main.py --trainset data/coqa/coqa-dev-v1.0-processed.json --devset data/coqa/coqa-dev-v1.0-processed.json --dir exp/debug --embed_file data/wordvecs/coqa.glove.6B.50d.txt --embed_type fasttext --f_history y --cuda false --debug false \
    --dialog_batched true --batch_size 2 --num_layers 1 --hidden_size 50 \
    --q_dialog_history true --q_dialog_attn word_hidden_incr \
    --attn_hidden_size 30 \
    --q_dialog_attn_scoring linear_relu \
    --q_dialog_attn_incr_merge linear_both_lstm \
    --recency_bias true \
    --qa_emb_markers true \
    --standardize_endings artificial
