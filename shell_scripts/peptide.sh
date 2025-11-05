#!/bin/bash

# Grid search ranges
layers=(8)
datasets=("Peptides-struct")
architectures=("GCN" "GAT" "Sage" "Uni" "Crawl")

# Default hyperparameters (based on your provided baseline)
ACTIVATION="ReLU"
BATCH_SIZE=64
BATCH_NORM="None"
NUM_HEADS=2
DROPOUT=0.1
HIDDEN_SIZE=128
EDGE_AGG="GATED"
OPTIMIZER="Cosine"
LR=0.001
EPOCHS=500
WEIGHT_DECAY=0.0
WINDOW_SIZE=4
RECEPTIVE_FIELD=5
SAVE_DIR="peptide_experiment"
VERBOSE="--verbose"
# Uncomment the next line if you want skip connections enabled
# SKIP="--skip_connections"

# Loop through all combinations
for dataset in "${datasets[@]}"; do
  for arch in "${architectures[@]}"; do
    for layer in "${layers[@]}"; do

      if [ "${arch}" = "Crawl" ]; then
        CURRENT_EDGE_AGG="NONE"
      else
        CURRENT_EDGE_AGG="${EDGE_AGG}"
      fi
      
      if [ "${arch}" = "GCN" ]; then
        CURRENT_HIDDEN_SIZE=512
      else
        CURRENT_HIDDEN_SIZE="${HIDDEN_SIZE}"
      fi

      run_dir="${SAVE_DIR}/${dataset}_${arch}_${layer}layers"

      echo "=============================================="
      echo "Running: dataset=${dataset}, architecture=${arch}, layers=${layer}"
      echo "Output dir: ${run_dir}"
      echo "=============================================="

      python3 -m train.train \
        --dataset "${dataset}" \
        --architecture "${arch}" \
        --num_layers "${layer}" \
        --activation_function "${ACTIVATION}" \
        --batch_size "${BATCH_SIZE}" \
        --batch_norm "${BATCH_NORM}" \
        --num_attention_heads "${NUM_HEADS}" \
        --dropout_rate "${DROPOUT}" \
        --hidden_size "${CURRENT_HIDDEN_SIZE}" \
        --edge_aggregator "${CURRENT_EDGE_AGG}" \
        --optimizer "${OPTIMIZER}" \
        --lr "${LR}" \
        --epochs "${EPOCHS}" \
        --weight_decay "${WEIGHT_DECAY}" \
        --window_size "${WINDOW_SIZE}" \
        --receptive_field "${RECEPTIVE_FIELD}" \
        --save_dir "${run_dir}" \
        ${VERBOSE} \
        ${SKIP}
    done
  done
done
