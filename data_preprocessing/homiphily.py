from external.homiphily import our_measure
from parsers.parser_lrgb import LongeRangeGraphBenchmarkParser
from torch.utils.data import ConcatDataset

if __name__ == "__main__":
    parser = LongeRangeGraphBenchmarkParser(name="PascalVOC-SP", verbose=True)
    train_dataset, val_dataset, test_dataset = parser.return_datasets()
    full_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])
    edge_indices = full_dataset.data.edge_index
    node_labels = full_dataset.data.y
    print("Homophily PascalVOC-SP:", our_measure(edge_indices, node_labels))
    parser = LongeRangeGraphBenchmarkParser(name="COCO-SP", verbose=True)
    train_dataset, val_dataset, test_dataset = parser.return_datasets()
    full_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])
    edge_indices = full_dataset.data.edge_index
    node_labels = full_dataset.data.y
    print("Homophily COCO-SP:", our_measure(edge_indices, node_labels))
