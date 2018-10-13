#!/bin/bash

rm -rf exp_simple
python rc/main_simple.py --trainset data/coqa/coqa-train-v1.0-processed.json --devset data/coqa/coqa-dev-v1.0-processed.json --dir exp_simple --embed_file data/wordvecs/glove.6B.50d.txt --embed_type fasttext --f_history y --cuda false --dialog_batched true --debug true --batch_size 1

# 83283 2835
# 17579
