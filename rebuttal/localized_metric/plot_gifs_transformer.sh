python3 -m external.custom_hermes.eval_rollout_local_metric  dataset=wave backbone=transformer  model_save_path=/projects/gllab/berman.ed/rayleigh_analysis/model_checkpoints/transformer/Wave_transformer_seed1_model.pt

python3 -m external.custom_hermes.eval_rollout_local_metric  dataset=heat backbone=transformer  model_save_path=/projects/gllab/li.tao/rayleigh_analysis/model_checkpoints/Heat_transformer_seed1_model.pt

python3 -m external.custom_hermes.eval_rollout_local_metric  dataset=cahn_hilliard backbone=transformer  model_save_path=/projects/gllab/li.tao/rayleigh_analysis/model_checkpoints/Cahn-Hilliard_transformer_seed1_model.pt