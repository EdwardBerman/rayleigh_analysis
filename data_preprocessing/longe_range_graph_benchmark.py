from torch_geometric.datasets import LRGBDataset
from torch_geometric.loader import DataLoader

root = "data/LRGB"
datasets = ["PascalVOC-SP", "COCO-SP", "PCQM-Contact",
            "Peptides-func", "Peptides-struct"]

for name in datasets:
    dataset = LRGBDataset(root=root, name=name, split="train")

    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for data in loader:
        print(data)
        break
