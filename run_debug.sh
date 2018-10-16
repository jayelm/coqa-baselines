#!/bin/bash

# A simple CoQA recipe for debugging without cuda.

rm -rf exp/debug
python rc/main.py --trainset data/coqa/coqa-train-v1.0-processed.json --devset data/coqa/coqa-dev-v1.0-processed.json --dir exp/debug --embed_file data/wordvecs/glove.6B.50d.txt --embed_type fasttext --f_history y --cuda false --dialog_batched true --debug true --batch_size 1 --num_layers 1 --hidden_size 50 --recency_bias true --use_history_qemb true
