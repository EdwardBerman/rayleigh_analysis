import torch
import torch.nn as nn
from torch_geometric.datasets import LRGBDataset

@ABC
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

