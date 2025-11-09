#!/bin/bash

python3 -m external.custom_hermes.eval_rollout dataset=heat backbone=hermes model_save_path=external/hermes/pretrained_checkpoints/Heat_Hermes_model.pt
python3 -m external.custom_hermes.eval_rollout dataset=wave backbone=hermes model_save_path=external/hermes/pretrained_checkpoints/Wave_Hermes_model.pt

python3 -m external.custom_hermes.eval_rollout dataset=heat backbone=eman model_save_path=external/hermes/pretrained_checkpoints/Heat_EMAN_model.pt
python3 -m external.custom_hermes.eval_rollout dataset=wave backbone=eman model_save_path=external/hermes/pretrained_checkpoints/Wave_EMAN_model.pt

python3 -m external.custom_hermes.eval_rollout dataset=heat backbone=gem_cnn model_save_path=external/hermes/pretrained_checkpoints/Heat_GemCNN_model.pt
python3 -m external.custom_hermes.eval_rollout dataset=wave backbone=gem_cnn model_save_path=external/hermes/pretrained_checkpoints/Wave_GemCNN_model.pt
