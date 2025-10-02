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
    
    homiphily = []
    loader = DataLoader(full_dataset, batch_size=1, shuffle=True)
    pbar = tqdm(loader)
    for data in pbar:
        edge_indices = data.edge_index
        node_labels = data.y
        h = our_measure(edge_indices, node_labels)
        homiphily.append(h.item() if hasattr(val, "item") else h)
        pbar.set_postfix({"Homophily": f"{val:.4f}"})

    print("Homophily PascalVOC-SP:", np.mean(homiphily), "+/-", np.std(homiphily))
    parser = LongeRangeGraphBenchmarkParser(name="COCO-SP", verbose=True)
    train_dataset, val_dataset, test_dataset = parser.return_datasets()
    full_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])
    
    homiphily = []
    loader = DataLoader(full_dataset, batch_size=1, shuffle=True)
    pbar = tqdm(loader)
    for data in pbar:
        edge_indices = data.edge_index
        node_labels = data.y
        h = our_measure(edge_indices, node_labels)
        homiphily.append(h.item() if hasattr(val, "item") else h)
        pbar.set_postfix({"Homophily": f"{val:.4f}"})

    print("Homophily COCO-SP:", np.mean(homiphily), "+/-", np.std(homiphily))
