module load python/3.13.5

source weatherbench2/bin/activate

python3 -m external.custom_hermes.eval_weatherbench paths.forecast=/projects/gllab/li.tao/rayleigh_analysis/rollouts/transformer/t850/forecasts_20260128_130900_460035.zarr eval.variable=temperature eval.level=850 eval.model=transformer

python3 -m external.custom_hermes.eval_weatherbench paths.forecast=/projects/gllab/li.tao/rayleigh_analysis/rollouts/EGNN_Local/t850/forecasts_20260128_122644_907120.zarr eval.variable=temperature eval.level=850 eval.model=egnn

python3 -m external.custom_hermes.eval_weatherbench paths.forecast=/projects/gllab/li.tao/rayleigh_analysis/rollouts/EMAN/t850/forecasts_20260128_184246_378231.zarr eval.variable=temperature eval.level=850 eval.model=eman

python3 -m external.custom_hermes.eval_weatherbench paths.forecast=/projects/gllab/li.tao/rayleigh_analysis/rollouts/GCN/t850/forecasts_20260128_112519_809646.zarr eval.variable=temperature eval.level=850 eval.model=gcn

python3 -m external.custom_hermes.eval_weatherbench paths.forecast=/projects/gllab/li.tao/rayleigh_analysis/rollouts/GemCNN/t850/forecasts_20260128_200202_338998.zarr eval.variable=temperature eval.level=850 eval.model=gemcnn

python3 -m external.custom_hermes.eval_weatherbench paths.forecast=/projects/gllab/li.tao/rayleigh_analysis/rollouts/Hermes/t850/forecasts_20260128_153024_888021.zarr eval.variable=temperature eval.level=850 eval.model=hermes

python3 -m external.custom_hermes.eval_weatherbench paths.forecast=/projects/gllab/li.tao/rayleigh_analysis/rollouts/MPNN/t850/forecasts_20260128_120647_978377.zarr eval.variable=temperature eval.level=850 eval.model=mpnn

python3 -m external.custom_hermes.eval_weatherbench paths.forecast=/projects/gllab/li.tao/rayleigh_analysis/rollouts/Uni/t850/forecasts_20260128_124703_141875.zarr eval.variable=temperature eval.level=850 eval.model=uni

