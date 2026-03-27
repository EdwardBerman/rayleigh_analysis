# python3 -m external.custom_hermes.eval_rollout_local_metric  dataset=wave backbone=gem_cnn  model_save_path=external/hermes/pretrained_checkpoints/Wave_GemCNN_model.pt  
# python3 -m external.custom_hermes.eval_rollout_local_metric  dataset=wave backbone=EMAN  model_save_path=external/hermes/pretrained_checkpoints/Wave_EMAN_model.pt
# python3 -m external.custom_hermes.eval_rollout_local_metric  dataset=wave backbone=uni  model_save_path=model_checkpoints/Wave_Uni_seed1_model.pt   
# python3 -m external.custom_hermes.eval_rollout_local_metric  dataset=wave backbone=egnn model_save_path=model_checkpoints/Wave_EGNN_h32_model.pt
# python3 -m external.custom_hermes.eval_rollout_local_metric  dataset=wave backbone=gcn  model_save_path=model_checkpoints/Wave_GCN_seed1_model.pt
# python3 -m external.custom_hermes.eval_rollout_local_metric  dataset=wave backbone=mpnn  model_save_path=model_checkpoints/Wave_MPNN_seed1_model.pt
# python3 -m external.custom_hermes.eval_rollout_local_metric  dataset=wave backbone=hermes  model_save_path=external/hermes/pretrained_checkpoints/Wave_Hermes_model.pt
python3 -m external.custom_hermes.eval_rollout_local_metric  dataset=wave backbone=transformer  model_save_path=model_checkpoints/Wave_transformer_seed1_model.pt