#!/bin/bash

# GCN baseline for Peptides-struct (Table 2) https://arxiv.org/pdf/2309.00367

DATASET="Peptides-struct"
ARCHITECTURE="GCN"

# Hyperparameters from the table
NUM_LAYERS=6
LR=0.001
DROPOUT=0.1
HIDDEN_SIZE=235
BATCH_SIZE=200
EPOCHS=250

# Other defaults
ACTIVATION="ReLU"
BATCH_NORM="None"
NUM_HEADS=1         
EDGE_AGG="GATED"
OPTIMIZER="Cosine"
WEIGHT_DECAY=0.0
WINDOW_SIZE=4
RECEPTIVE_FIELD=5
SAVE_DIR="peptide_experiment"
VERBOSE="--verbose"
# Uncomment if you want skip connections
# SKIP="--skip_connections"

run_dir="${SAVE_DIR}/${DATASET}_${ARCHITECTURE}_${NUM_LAYERS}"

echo "=============================================="
echo "Running GCN baseline from Table 2"
echo "dataset=${DATASET}, architecture=${ARCHITECTURE}, layers=${NUM_LAYERS}"
echo "Output dir: ${run_dir}"
echo "=============================================="

python3 -m train.train \
  --dataset "${DATASET}" \
  --architecture "${ARCHITECTURE}" \
  --num_layers "${NUM_LAYERS}" \
  --activation_function "${ACTIVATION}" \
  --batch_size "${BATCH_SIZE}" \
  --batch_norm "${BATCH_NORM}" \
  --num_attention_heads "${NUM_HEADS}" \
  --dropout_rate "${DROPOUT}" \
  --hidden_size "${HIDDEN_SIZE}" \
  --edge_aggregator "${EDGE_AGG}" \
  --optimizer "${OPTIMIZER}" \
  --lr "${LR}" \
  --epochs "${EPOCHS}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --window_size "${WINDOW_SIZE}" \
  --receptive_field "${RECEPTIVE_FIELD}" \
  --save_dir "${run_dir}" \
  ${VERBOSE} \
  ${SKIP}

