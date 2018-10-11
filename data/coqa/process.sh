#!/bin/bash

CORENLP_DIR="../../../../stanford-corenlp-full-2018-10-05/"

pushd $CORENLP_DIR
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
popd

python gen_drqa_data_wt_heuristics.py -d coqa-train-v1.0.json -o coqa-train-v1.0-processed.json
python gen_drqa_data_wt_heuristics.py -d coqa-dev-v1.0.json -o coqa-dev-v1.0-processed.json
