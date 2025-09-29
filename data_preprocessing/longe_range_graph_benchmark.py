from torch_geometric.datasets import LRGBDataset
from torch_geometric.loader import DataLoader

root = "data/LRGB"
dataset_pascal = "PascalVOC-SP"  # e.g. one of "PascalVOC-SP", "COCO-SP", "PCQM-Contact", "Peptides-func", "Peptides-struct"
dataset_coco = "COCO-SP"
dataset_PCQM = "PCQM-Contact"
dataset_peptides_func = "Peptides-func"
dataset_peptides_struct = "Peptides-struct"

dataset_name = dataset_pascal  
dataset = LRGBDataset(root=root, name=dataset_name, split="train")

loader = DataLoader(dataset, batch_size=32, shuffle=True)
for data in loader:
    print(data)
    break

dataset_name = dataset_coco  
dataset = LRGBDataset(root=root, name=dataset_name, split="train")

loader = DataLoader(dataset, batch_size=32, shuffle=True)
for data in loader:
    print(data)
    break

dataset_name = dataset_PCQM  
dataset = LRGBDataset(root=root, name=dataset_name, split="train")

loader = DataLoader(dataset, batch_size=32, shuffle=True)
for data in loader:
    print(data)
    break

dataset_name = dataset_peptides_func  
dataset = LRGBDataset(root=root, name=dataset_name, split="train")

loader = DataLoader(dataset, batch_size=32, shuffle=True)
for data in loader:
    print(data)
    break

dataset_name = dataset_peptides_struct  
dataset = LRGBDataset(root=root, name=dataset_name, split="train")

loader = DataLoader(dataset, batch_size=32, shuffle=True)
for data in loader:
    print(data)
    break

