from external.homiphily import our_measure
from parsers.parser_lrgb import LongeRangeGraphBenchmarkParser
from torch.utils.data import ConcatDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import numpy as np

if __name__ == "__main__":
    parser = LongeRangeGraphBenchmarkParser(name="PascalVOC-SP", verbose=True)
    train_dataset, val_dataset, test_dataset = parser.return_datasets()
    full_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])
    num_graphs = len(full_dataset)
    
    homiphily = 0
    loader = DataLoader(full_dataset, batch_size=num_graphs, shuffle=True)
    for data in tqdm(loader):
        edge_indices = data.edge_index
        node_labels = data.y
        homiphily += our_measure(edge_indices, node_labels)
    print("Homophily PascalVOC-SP:", homiphily)
    parser = LongeRangeGraphBenchmarkParser(name="COCO-SP", verbose=True)
    train_dataset, val_dataset, test_dataset = parser.return_datasets()
    full_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])
    num_graphs = len(full_dataset)
    
    homiphily = 0
    loader = DataLoader(full_dataset, batch_size=num_graphs, shuffle=True)
    for data in tqdm(loader):
        edge_indices = data.edge_index
        node_labels = data.y
        homiphily += our_measure(edge_indices, node_labels)
    print("Homophily COCO-SP:", homiphily)
