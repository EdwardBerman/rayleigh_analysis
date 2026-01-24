module load python/3.13.5

source weatherbench2/bin/activate

python3 -m external.custom_hermes.eval_weatherbench backbone=transformer paths.forecast=/projects/gllab/li.tao/rayleigh_analysis/rollouts/transformer/t850/forecasts_20260121_010836_932942.zarr eval.variable=temperature eval.level=850
python3 -m external.custom_hermes.eval_weatherbench backbone=egnn_local paths.forecast=/projects/gllab/li.tao/rayleigh_analysis/rollouts/EGNN_Local/t850/forecasts_20260120_173828_643026.zarr eval.variable=temperature eval.level=850
python3 -m external.custom_hermes.eval_weatherbench backbone=eman paths.forecast=/projects/gllab/li.tao/rayleigh_analysis/rollouts/EMAN/t850/forecasts_20260120_170858_239641.zarr eval.variable=temperature eval.level=850
python3 -m external.custom_hermes.eval_weatherbench backbone=gcn paths.forecast=/projects/gllab/li.tao/rayleigh_analysis/rollouts/GCN/t850/forecasts_20260120_170807_640487.zarr eval.variable=temperature eval.level=850
python3 -m external.custom_hermes.eval_weatherbench backbone=gem_cnn paths.forecast=/projects/gllab/li.tao/rayleigh_analysis/rollouts/GemCNN/t850/forecasts_20260120_173825_280960.zarr eval.variable=temperature eval.level=850
python3 -m external.custom_hermes.eval_weatherbench backbone=hermes paths.forecast=/projects/gllab/li.tao/rayleigh_analysis/rollouts/Hermes/t850/forecasts_20260120_171137_768832.zarr eval.variable=temperature eval.level=850
python3 -m external.custom_hermes.eval_weatherbench backbone=mpnn paths.forecast=/projects/gllab/li.tao/rayleigh_analysis/rollouts/MPNN/t850/forecasts_20260120_170807_640102.zarr eval.variable=temperature eval.level=850
python3 -m external.custom_hermes.eval_weatherbench backbone=uni paths.forecast=/projects/gllab/li.tao/rayleigh_analysis/rollouts/Uni/t850/forecasts_20260121_014254_299199.zarr eval.variable=temperature eval.level=850

python3 -m external.custom_hermes.eval_weatherbench backbone=uni paths.forecast=/projects/gllab/li.tao/rayleigh_analysis/rollouts/Uni/z500/forecasts_20260123_125314_686783.zarr eval.variable=geopotential eval.level=500
python3 -m external.custom_hermes.eval_weatherbench backbone=transformer paths.forecast=/projects/gllab/li.tao/rayleigh_analysis/rollouts/transformer/z500/forecasts_20260123_141330_556286.zarr eval.variable=geopotential eval.level=500
python3 -m external.custom_hermes.eval_weatherbench backbone=egnn_local paths.forecast=/projects/gllab/li.tao/rayleigh_analysis/rollouts/EGNN_Local/z500/forecasts_20260123_135826_079177.zarr eval.variable=geopotential eval.level=500
python3 -m external.custom_hermes.eval_weatherbench backbone=eman paths.forecast=/projects/gllab/li.tao/rayleigh_analysis/rollouts/EMAN/z500/forecasts_20260123_131131_063878.zarr eval.variable=geopotential eval.level=500
python3 -m external.custom_hermes.eval_weatherbench backbone=gcn paths.forecast=/projects/gllab/li.tao/rayleigh_analysis/rollouts/GCN/z500/forecasts_20260123_123156_726718.zarr eval.variable=geopotential eval.level=500
python3 -m external.custom_hermes.eval_weatherbench backbone=gem_cnn paths.forecast=/projects/gllab/li.tao/rayleigh_analysis/rollouts/GemCNN/z500/forecasts_20260123_131138_667277.zarr eval.variable=geopotential eval.level=500
python3 -m external.custom_hermes.eval_weatherbench backbone=hermes paths.forecast=/projects/gllab/li.tao/rayleigh_analysis/rollouts/Hermes/z500/forecasts_20260123_125434_070322.zarr eval.variable=geopotential eval.level=500
python3 -m external.custom_hermes.eval_weatherbench backbone=mpnn paths.forecast=/projects/gllab/li.tao/rayleigh_analysis/rollouts/MPNN/z500/forecasts_20260123_123259_420654.zarr eval.variable=geopotential eval.level=500