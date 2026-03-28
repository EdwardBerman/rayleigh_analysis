python3 -m external.custom_hermes.eval_rollout_local_metric  dataset=wave backbone=uni  model_save_path=model_checkpoints/adaptiveuni/Wave_Uni_seed1_model.pt

python3 -m external.custom_hermes.eval_rollout_local_metric  dataset=heat backbone=uni  model_save_path=model_checkpoints/adaptiveuni/Heat_Uni_seed1_model.pt

python3 -m external.custom_hermes.eval_rollout_local_metric  dataset=cahn_hilliard backbone=uni  model_save_path=model_checkpoints/adaptiveuni/Cahn-Hilliard_Uni_seed1_model.pt