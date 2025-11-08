import os

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import ConcatDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from external.homiphily import our_measure
from parsers.parser_lrgb import LongRangeGraphBenchmarkParser

if __name__ == "__main__":
    save_dir = "data_preprocessing/data/LRGB"
    current_dir = os.getcwd()
    os.makedirs(save_dir, exist_ok=True)

    parser = LongRangeGraphBenchmarkParser(name="PascalVOC-SP", transform=None, verbose=True)
    train_dataset, val_dataset, test_dataset = parser.return_datasets()
    full_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])

    homiphily = []
    loader = DataLoader(full_dataset, batch_size=1, shuffle=True)
    pbar = tqdm(loader)
    for data in pbar:
        edge_indices = data.edge_index
        node_labels = data.y
        h = our_measure(edge_indices, node_labels)
        homiphily.append(h.item() if hasattr(h, "item") else h)
        pbar.set_postfix({"Homophily": f"{h:.4f}"})

    num_nans = sum(1 for h in homiphily if np.isnan(h))
    print(
        f"Number of NaNs in homophily values: {num_nans} out of {len(homiphily)}")
    homiphily = [h for h in homiphily if not np.isnan(h)]
    print("Homophily PascalVOC-SP:", np.mean(homiphily), "+/-", np.std(homiphily))

    plt.figure(figsize=(8, 6))
    plt.hist(homiphily, bins=30, color='skyblue', edgecolor='black')
    plt.title('Homophily Distribution for PascalVOC-SP')
    plt.xlabel('Homophily')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(os.path.join(
        save_dir, 'homophily_distribution_pascalvoc_sp.pdf'))
    plt.close()

    parser = LongRangeGraphBenchmarkParser(name="COCO-SP", verbose=True)
    train_dataset, val_dataset, test_dataset = parser.return_datasets()
    full_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])

    homiphily = []
    loader = DataLoader(full_dataset, batch_size=1, shuffle=True)
    pbar = tqdm(loader)
    for data in pbar:
        edge_indices = data.edge_index
        node_labels = data.y
        h = our_measure(edge_indices, node_labels)
        homiphily.append(h.item() if hasattr(h, "item") else h)
        pbar.set_postfix({"Homophily": f"{h:.4f}"})

    num_nans = sum(1 for h in homiphily if np.isnan(h))
    print(
        f"Number of NaNs in homophily values: {num_nans} out of {len(homiphily)}")
    homiphily = [h for h in homiphily if not np.isnan(h)]
    print("Homophily COCO-SP:", np.mean(homiphily), "+/-", np.std(homiphily))

    plt.figure(figsize=(8, 6))
    plt.hist(homiphily, bins=30, color='skyblue', edgecolor='black')
    plt.title('Homophily Distribution for COCO-SP')
    plt.xlabel('Homophily')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(os.path.join(save_dir, 'homophily_distribution_coco_sp.pdf'))
    plt.close()
