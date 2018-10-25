#!/bin/bash

python rc/main.py --pretrained exp/debug --testset data/coqa/coqa-dev-v1.0-processed.json --embed_file data/wordvecs/coqa.crawl-300d-2M.vec --embed_type fasttext --f_history y \
           --dialog_batched true --batch_size 2 --num_layers 1 --hidden_size 50 --cuda false \
           --q_dialog_history true --q_dialog_attn word_hidden \
           --doc_dialog_history false --doc_dialog_attn word_hidden \
           --save_params false \
           --recency_bias true \
           --save_attn_weights q_dialog_attn
