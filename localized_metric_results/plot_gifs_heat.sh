python3 -m external.custom_hermes.eval_rollout_local_metric  dataset=heat backbone=gem_cnn  model_save_path=external/hermes/pretrained_checkpoints/Heat_GemCNN_model.pt  
python3 -m external.custom_hermes.eval_rollout_local_metric  dataset=heat backbone=EMAN  model_save_path=external/hermes/pretrained_checkpoints/Heat_EMAN_model.pt
python3 -m external.custom_hermes.eval_rollout_local_metric  dataset=heat backbone=uni  model_save_path=model_checkpoints/Heat_Uni_seed1_model.pt   
python3 -m external.custom_hermes.eval_rollout_local_metric  dataset=heat backbone=egnn model_save_path=model_checkpoints/Heat_EGNN_h32_model.pt
python3 -m external.custom_hermes.eval_rollout_local_metric  dataset=heat backbone=gcn  model_save_path=model_checkpoints/Heat_GCN_seed1_model.pt
python3 -m external.custom_hermes.eval_rollout_local_metric  dataset=heat backbone=mpnn  model_save_path=model_checkpoints/Heat_MPNN_seed1_model.pt
python3 -m external.custom_hermes.eval_rollout_local_metric  dataset=heat backbone=hermes  model_save_path=external/hermes/pretrained_checkpoints/Heat_Hermes_model.pt
python3 -m external.custom_hermes.eval_rollout_local_metric  dataset=heat backbone=transformer  model_save_path=model_checkpoints/Heat_transformer_seed1_model.pt