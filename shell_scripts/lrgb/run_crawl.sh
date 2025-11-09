#!/bin/bash

python3 -m train.train \
  --dataset COCO-SP \
  --architecture CRAWL \
  --num_layers 12 \
  --activation_function ReLU \
  --batch_size 64 \
  --batch_norm BatchNorm \
  --num_attention_heads 2 \
  --dropout_rate 0.1 \
  --hidden_size 128 \
  --edge_aggregator NONE \
  --optimizer Adam \
  --lr 0.001 \
  --epochs 750 \
  --weight_decay 0.0 \
  --window_size 4 \
  --receptive_field 5 \
  --save_dir output \
  --log_rq
