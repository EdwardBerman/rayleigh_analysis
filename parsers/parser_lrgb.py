import pprint

from torch_geometric.datasets import LRGBDataset

from parsers.parser_base import Parser


class LongRangeGraphBenchmarkParser(Parser):
    def __init__(self, name: str, path: str | None = None, verbose: bool = True):
        self._level = "node_level"

        root = 'data_preprocessing/data/LRGB/' if path is None else path
        self.verbose = verbose

        match name:
            case 'PascalVOC-SP':
                self.train_dataset = LRGBDataset(
                    root=root, name="PascalVOC-SP", split="train")
                self.val_dataset = LRGBDataset(
                    root=root, name="PascalVOC-SP", split="val")
                self.test_dataset = LRGBDataset(
                    root=root, name="PascalVOC-SP", split="test")
                self._is_classification = True
                self._level = "node_level"
                self.num_classes = self.train_dataset.num_classes
            case 'COCO-SP':
                self.train_dataset = LRGBDataset(
                    root=root, name="COCO-SP", split="train")
                self.val_dataset = LRGBDataset(
                    root=root, name="COCO-SP", split="val")
                self.test_dataset = LRGBDataset(
                    root=root, name="COCO-SP", split="test")
                self._is_classification = True
                self._level = "node_level"
                self.num_classes = self.train_dataset.num_classes
            case 'Peptides-func':
                self.train_dataset = LRGBDataset(
                    root=root, name="Peptides-func", split="train")
                self.val_dataset = LRGBDataset(
                    root=root, name="Peptides-func", split="val")
                self.test_dataset = LRGBDataset(
                    root=root, name="Peptides-func", split="test")
                self._is_classification = True
                self._level = "graph_level"
                self.num_classes = self.train_dataset.num_classes
            case 'Peptides-struct':
                self.train_dataset = LRGBDataset(
                    root=root, name="Peptides-struct", split="train")
                self.val_dataset = LRGBDataset(
                    root=root, name="Peptides-struct", split="val")
                self.test_dataset = LRGBDataset(
                    root=root, name="Peptides-struct", split="test")
                self._is_classification = False
                self._level = "graph_level"
                self.num_classes = self.train_dataset.num_classes
            case _:
                raise ValueError(
                    f"Dataset {name} not recognized. Available datasets are 'PascalVOC-SP' and ''.")

        self._node_dim = self.train_dataset.num_node_features
        self._edge_dim = self.train_dataset.num_edge_features

    @property
    def node_dim(self):
        return self._node_dim

    @property
    def edge_dim(self):
        return self._edge_dim if hasattr(self, '_edge_dim') else None

    @property
    def is_classification(self):
        return self._is_classification

    @property
    def level(self):
        return self._level

    def return_datasets(self):
        return self.train_dataset, self.val_dataset, self.test_dataset

    def parse(self):
        datasets_info = {
            'train_dataset': self.train_dataset,
            'val_dataset': self.val_dataset,
            'test_dataset': self.test_dataset,
            'node_dim': self.node_dim,
            'edge_dim': self.edge_dim,
            'is_classification': self.is_classification,
            'level': self.level,
            'num_classes': self.num_classes if self.is_classification else None
        }
        if self.verbose:
            print("Dataset Information:")
            pprint.pprint(datasets_info)
        return datasets_info
