#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 EXP_DIR"
    exit 1
fi

python rc/main.py --trainset data/coqa/coqa-train-v1.0-processed.json --devset data/coqa/coqa-dev-v1.0-processed.json --dir "$1" --embed_file data/wordvecs/coqa.crawl-300d-2M.vec --embed_type fasttext --f_history y --dialog_batched true --batch_size 1 --cuda_id 0 --recency_bias true
