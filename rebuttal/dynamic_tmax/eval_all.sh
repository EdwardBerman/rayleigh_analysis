
python3 -m external.custom_hermes.eval_rollout dataset=cahn_hilliard backbone=uni model_save_path=model_checkpoints/regadaptive_uni/Cahn-Hilliard_Uni_seed1_model.pt

python3 -m external.custom_hermes.eval_rollout dataset=heat backbone=uni model_save_path=model_checkpoints/regadaptive_uni/Heat_Uni_seed1_model.pt

python3 -m external.custom_hermes.eval_rollout dataset=wave backbone=uni model_save_path=model_checkpoints/regadaptive_uni/Wave_Uni_seed1_model.pt