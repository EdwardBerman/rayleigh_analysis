import torch
import torch.nn as nn
from torch_geometric.datasets import LRGBDataset

class Parser(ABC):

    @abstractmethod
    def loss_fn(self) -> nn.Module:
        """Returns the loss function used for training the model."""
        pass

    @property
    @abstractmethod
    def node_dim(self):
        """Return the node dimension."""
        pass

    @property
    @abstractmethod
    def edge_dim(self):
        """Return the edge dimension."""
        pass
    
    @property
    @abstractmethod
    def is_classification(self):
        """Allows us to check if the dataset is for classification or regression."""
        pass
    
    @property
    @abstractmethod
    def level(self):
        """Graph Level or Node Level Task."""
        pass

class LongeRangeGraphBenchmarkParser(Parser):
    def __init__(self, path: str, name: str):
        self.path = path
        self.name = name
        self._level = "node_level"
        self._loss_fn = nn.CrossEntropyLoss()

        match name:
            case 'PascalVOC-SP':
                self.train_dataset = LRGBDataset(root=root, name="PascalVOC-SP", split="train")
                self.val_dataset = LRGBDataset(root=root, name="PascalVOC-SP", split="val")
                self.test_dataset = LRGBDataset(root=root, name="PascalVOC-SP", split="test")
                self._node_dim = self.train_dataset.num_node_features
                self._edge_dim = self.train_dataset.num_edge_features
                self._is_classification = True
                self._level = "node_level"
                # loss_fn = ...
                # acc_fn = ...
            case 'COCO-SP':
                self.train_dataset = LRGBDataset(root=root, name="COCO-SP", split="train")
                self.val_dataset = LRGBDataset(root=root, name="COCO-SP", split="val")
                self.test_dataset = LRGBDataset(root=root, name="COCO-SP", split="test")
                self._node_dim = self.train_dataset.num_node_features
                self._edge_dim = self.train_dataset.num_edge_features
                self._is_classification = True
                self._level = "node_level"
            case 'Peptides-func':
                self.train_dataset = LRGBDataset(root=root, name="Peptides-func", split="train")
                self.val_dataset = LRGBDataset(root=root, name="Peptides-func", split="val")
                self.test_dataset = LRGBDataset(root=root, name="Peptides-func", split="test")
                self._node_dim = self.train_dataset.num_node_features
                self._edge_dim = self.train_dataset.num_edge_features
                self._is_classification = True
                self._level = "graph_level"
            case 'Peptides-struct':
                self.train_dataset = LRGBDataset(root=root, name="Peptides-struct", split="train")
                self.val_dataset = LRGBDataset(root=root, name="Peptides-struct", split="val")
                self.test_dataset = LRGBDataset(root=root, name="Peptides-struct", split="test")
                self._node_dim = self.train_dataset.num_node_features
                self._edge_dim = self.train_dataset.num_edge_features
                self._is_classification = False
                self._level = "graph_level"
                self._loss_fn = nn.MSELoss()
            case _:
                raise ValueError(f"Dataset {name} not recognized. Available datasets are 'PascalVOC-SP' and ''.")

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

    @property
    def loss_fn(self):
        return self._loss_fn

    def return_datasets(self):
        return self.train_dataset, self.val_dataset, self.test_dataset


    

