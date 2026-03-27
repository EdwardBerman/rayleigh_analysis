python3 -m external.custom_hermes.eval_rollout_local_metric  dataset=cahn_hilliard backbone=gem_cnn  model_save_path=external/hermes/pretrained_checkpoints/Cahn-Hilliard_GemCNN_model.pt  
python3 -m external.custom_hermes.eval_rollout_local_metric  dataset=cahn_hilliard backbone=EMAN  model_save_path=external/hermes/pretrained_checkpoints/Cahn-Hilliard_EMAN_model.pt
python3 -m external.custom_hermes.eval_rollout_local_metric  dataset=cahn_hilliard backbone=uni  model_save_path=model_checkpoints/Cahn-Hilliard_Uni_seed1_model.pt   
python3 -m external.custom_hermes.eval_rollout_local_metric  dataset=cahn_hilliard backbone=egnn model_save_path=model_checkpoints/Cahn-Hilliard_EGNN_h32_model.pt
python3 -m external.custom_hermes.eval_rollout_local_metric  dataset=cahn_hilliard backbone=gcn  model_save_path=model_checkpoints/Cahn-Hilliard_GCN_seed1_model.pt
python3 -m external.custom_hermes.eval_rollout_local_metric  dataset=cahn_hilliard backbone=mpnn  model_save_path=model_checkpoints/Cahn-Hilliard_MPNN_seed1_model.pt
python3 -m external.custom_hermes.eval_rollout_local_metric  dataset=cahn_hilliard backbone=hermes  model_save_path=external/hermes/pretrained_checkpoints/Cahn-Hilliard_Hermes_model.pt
python3 -m external.custom_hermes.eval_rollout_local_metric  dataset=cahn_hilliard backbone=uni  model_save_path=model_checkpoints/Cahn-Hilliard_Uni_seed1_model.pt   