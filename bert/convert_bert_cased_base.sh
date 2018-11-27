#!/bin/bash

python convert_tf_checkpoint_to_pytorch.py \
    --tf_checkpoint_path models/cased_L-12_H-768_A-12/bert_model.ckpt \
    --bert_config_file models/cased_L-12_H-768_A-12/bert_config.json
