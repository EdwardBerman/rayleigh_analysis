#!/bin/bash

python3 -m train.train \
  --dataset PascalVOC-SP \
  --architecture GCN \
  --num_layers 8 \
  --activation_function ReLU \
  --batch_size 64 \
  --batch_norm None \
  --num_attention_heads 2 \
  --dropout_rate 0.1 \
  --hidden_size 128 \
  --edge_aggregator GATED \
  --optimizer Muon \
  --lr 0.001 \
  --epochs 200 \
  --weight_decay 0.01 \
  --window_size 4 \
  --receptive_field 5 \
  --save_dir output \
  --log_rq

python3 -m train.train \
  --dataset PascalVOC-SP \
  --architecture GAT \
  --num_layers 8 \
  --activation_function ReLU \
  --batch_size 64 \
  --batch_norm None \
  --num_attention_heads 2 \
  --dropout_rate 0.1 \
  --hidden_size 128 \
  --edge_aggregator GATED \
  --optimizer Muon \
  --lr 0.001 \
  --epochs 200 \
  --weight_decay 0.01 \
  --window_size 4 \
  --receptive_field 5 \
  --save_dir output \
  --log_rq

python3 -m train.train \
  --dataset PascalVOC-SP \
  --architecture Sage \
  --num_layers 8 \
  --activation_function ReLU \
  --batch_size 64 \
  --batch_norm None \
  --num_attention_heads 2 \
  --dropout_rate 0.1 \
  --hidden_size 128 \
  --edge_aggregator GATED \
  --optimizer Muon \
  --lr 0.001 \
  --epochs 200 \
  --weight_decay 0.01 \
  --window_size 4 \
  --receptive_field 5 \
  --save_dir output \
  --log_rq

python3 -m train.train \
  --dataset PascalVOC-SP \
  --architecture Uni \
  --num_layers 8 \
  --activation_function ReLU \
  --batch_size 64 \
  --batch_norm None \
  --num_attention_heads 2 \
  --dropout_rate 0.1 \
  --hidden_size 128 \
  --edge_aggregator GATED \
  --optimizer Muon \
  --lr 0.001 \
  --epochs 200 \
  --weight_decay 0.01 \
  --window_size 4 \
  --receptive_field 5 \
  --save_dir output \
  --log_rq
