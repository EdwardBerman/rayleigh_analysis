from torch_geometric.datasets import LRGBDataset
from torch_geometric.loader import DataLoader

root = "data/LRGB"
dataset_name = "PascalVOC-SP"  # e.g. one of "PascalVOC-SP", "COCO-SP", "PCQM-Contact", "Peptides-func", "Peptides-struct"

# 2. Create the dataset object
dataset = LRGBDataset(root=root, name=dataset_name, split="train")

loader = DataLoader(dataset, batch_size=32, shuffle=True)

for data in loader:
    # data is a PyG Data object (or a batch of them)
    # e.g. data.x, data.edge_index, data.y, etc.
    print(data)
    break

