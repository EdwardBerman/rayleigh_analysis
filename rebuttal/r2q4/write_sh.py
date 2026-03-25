import os

SBATCH_HEADER = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=07:59:59
#SBATCH --job-name=truncation_reb
#SBATCH --mem=32GB
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200
#SBATCH --output=slurm/%j.out
#SBATCH --error=slurm/%j.err

module load python/3.13.5

eval "$(poetry env activate)"
"""


def parse_config(filepath):
    config = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()

            if value.lower() in ('true', 'false'):
                value = value.lower() == 'true'
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass

            config[key] = value
    return config


def build_command(config, truncation, trial):
    save_dir = f"rebuttal/r2q4/lie_unitary_trunc{truncation}_trial{trial}"
    return f"""python3 -m toy_heat_diffusion.train \\
    --data_dir {config['data_dir']} \\
    --train_steps {config['train_steps']} \\
    --eval_steps {config['eval_steps']} \\
    --model {config['model']} \\
    --layers {config['layers']} \\
    --act {config['act']} \\
    --hidden 1 \\
    --epochs 200 \\
    --lr {config['lr']} \\
    --batch_size {config['batch_size']} \\
    --dropout {config['dropout']} \\
    --truncation_level {truncation} \\
    --save_dir {save_dir} \\
    --entity_name {config['entity_name']} \\
    --project_name truncation_rebuttal"""


if __name__ == "__main__":
    config = parse_config("rebuttal/r2q4/lie_args.txt")

    scripts_dir = "rebuttal/r2q4/scripts"
    os.makedirs(scripts_dir, exist_ok=True)

    for truncation in [1, 3, 5, 7, 9]:
        script_path = f"{scripts_dir}/run_trunc{truncation}.sh"

        with open(script_path, 'w') as f:
            f.write(SBATCH_HEADER)

            for trial in range(5):
                f.write(build_command(config, truncation, trial))
                f.write("\n\n")

        os.chmod(script_path, 0o755)
        print(f"Generated: {script_path}")
