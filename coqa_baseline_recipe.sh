#!/bin/bash

python rc/main.py --trainset data/coqa/coqa-train-v1.0-processed.json --devset data/coqa/coqa-dev-v1.0-processed.json --dir exp --embed_file data/wordvecs/crawl-300d-2M.vec --embed_type fasttext --f_history y --dialog_batched true --batch_size 1
