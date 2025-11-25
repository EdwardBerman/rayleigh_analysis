import os
import sys
from pathlib import Path

parent_dir = str(Path().resolve().parents[1])

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

os.chdir('../')
