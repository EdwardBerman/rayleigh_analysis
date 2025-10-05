import random

from torch_geometric.data import Data
from torch_geometric.datasets import LRGBDataset
from torch_geometric.loader import DataLoader


def one_percent_filter(data: Data) -> bool:
    """Pre-filter for the PyG datasets that just randomly picks ~1% of the data."""
    return random.random() < 0.01  # 1% chance to be True


root = "data/LRGB"
datasets = ["PascalVOC-SP", "COCO-SP", "PCQM-Contact",
            "Peptides-func", "Peptides-struct"]

for name in datasets:
    dataset = LRGBDataset(root=root, name=name,
                          split="train", pre_filter=one_percent_filter)
    print(f"Dataset: {name}, Data points: {len(dataset)}")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for data in loader:
        print(data)
        break
