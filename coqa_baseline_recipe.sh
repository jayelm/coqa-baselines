#!/bin/bash

python rc/main.py --trainset data/coqa/coqa-train-v1.0-processed.json --devset data/coqa/coqa-dev-v1.0-processed.json --dir exp --embed_file data/wordvecs/ --embed_type fasttext --f_history y
