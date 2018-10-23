#!/bin/bash

# A simple CoQA recipe for debugging without cuda.

rm -rf exp/debug
python rc/main.py --trainset data/coqa/coqa-train-v1.0-processed.json --devset data/coqa/coqa-dev-v1.0-processed.json --dir exp/debug --embed_file data/wordvecs/coqa.glove.6B.50d.txt --embed_type fasttext --f_history y --cuda false --debug false \
    --dialog_batched true --batch_size 1 --num_layers 1 --hidden_size 50 \
    --use_history_qhidden false --qhidden_attn qa_sentence \
    --use_history_qemb false --qemb_attn qa_sentence \
    --use_history_aemb false --aemb_attn qemb \
    --use_history_dialog true --dialog_attn word \
    --recency_bias true \
    --history_dialog_answer_f false \
    --history_dialog_time_f false
