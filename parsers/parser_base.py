import torch
import torch.nn as nn

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


