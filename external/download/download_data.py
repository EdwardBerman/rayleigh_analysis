import logging
import os
import shutil
from pathlib import Path

import gdown
import requests
from typer import Typer

app = Typer()
logger = logging.getLogger(__name__)


def download_file(url, out_path):
    """See https://stackoverflow.com/a/39217788/3790116."""
    with requests.get(url, stream=True) as r:
        with open(out_path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

@app.command()
def meshgraphnets():
    """Download the meshgraphnets datasets from DeepMind."""
    settings = ['airfoil', 'cylinder_flow', 'deforming_plate', 'flag_minimal',
                'flag_simple', 'flag_dynamic', 'flag_dynamic_sizing',
                'sphere_simple', 'sphere_dynamic', 'sphere_dynamic_sizing']
    files = ['meta.json', 'train.tfrecord', 'valid.tfrecord', 'test.tfrecord']
    base_url = 'https://storage.googleapis.com/dm-meshgraphnets'

    for setting in settings:
        data_dir = Path('data') / 'meshgraphnets' / setting
        data_dir.mkdir(parents=True, exist_ok=True)

        for file in files:
            url = f'{base_url}/{setting}/{file}'
            out_path = data_dir / file
            logger.info(f'Getting {out_path}')
            if not out_path.exists():
                download_file(url, out_path)


if __name__ == "__main__":
    app()
