#!/bin/bash

# Define search spaces
layers=(1 2 3 4 5)
datasets=("PascalVOC-SP" "COCO-SP" "Peptides-func" "Peptides-struct")
architectures=("GCN" "GAT" "MPNN" "Sage" "Uni" "Crawl")

# Global parameters
BATCH_SIZE=64
ACTIVATION="ReLU"
BATCH_NORM="BatchNorm"
NUM_HEADS=2
DROPOUT=0.1
HIDDEN=128
EDGE_AGG="GATED"
OPTIMIZER="Cosine"
LR=0.001
EPOCHS=1000
WEIGHT_DECAY=0.0
WINDOW_SIZE=4
RECEPTIVE_FIELD=5
SAVE_DIR="network_depth_experiment"

# Loop through all combinations
for dataset in "${datasets[@]}"; do
  for arch in "${architectures[@]}"; do
    for layer in "${layers[@]}"; do
      echo "Running: dataset=${dataset}, architecture=${arch}, layers=${layer}"

      python3 -m train.train \
        --dataset "${dataset}" \
        --architecture "${arch}" \
        --num_layers "${layer}" \
        --skip_connections False \
        --activation_function "${ACTIVATION}" \
        --batch_size "${BATCH_SIZE}" \
        --batch_norm "${BATCH_NORM}" \
        --num_attention_heads "${NUM_HEADS}" \
        --dropout_rate "${DROPOUT}" \
        --hidden_size "${HIDDEN}" \
        --edge_aggregator "${EDGE_AGG}" \
        --optimizer "${OPTIMIZER}" \
        --lr "${LR}" \
        --epochs "${EPOCHS}" \
        --weight_decay "${WEIGHT_DECAY}" \
        --window_size "${WINDOW_SIZE}" \
        --receptive_field "${RECEPTIVE_FIELD}" \
        --save_dir "${SAVE_DIR}/${dataset}_${arch}_${layer}layers" \
        --verbose
    done
  done
done
