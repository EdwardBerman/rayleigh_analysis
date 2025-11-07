import os
import sys

BASH_TEMPLATE = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=07:59:59
#SBATCH --job-name={job_name}
#SBATCH --mem=32GB
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200
#SBATCH --output=slurm/%j.out
#SBATCH --error=slurm/%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=li.tao@northeastern.edu

{command}
"""


def make_job_script(command: str, path: str):
    job_name = command.split()[1] if len(command.split()) > 1 else "job"
    job_name = os.path.basename(job_name).replace(".py", "")

    script_content = BASH_TEMPLATE.format(command=command, job_name=job_name)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        f.write(script_content)

    os.chmod(path, 0o755)
    return path


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python writesh.py \"<command>\" <output_path>")
        sys.exit(1)

    command = sys.argv[1]
    path = sys.argv[2]
    make_job_script(command, path)
