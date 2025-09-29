import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool

# TODO, classifier wrappers

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

class Regressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Regressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class GraphLevelRegressor(nn.Module):
    def __init__(self, base_model: nn.Module, node_dim: int):
        super(GraphLevelRegressor, self).__init__()
        self.base_model = base_model
        self.Regressor = Regressor(node_dim, node_dim // 2, 1)
    
    def forward(self, x: Data):
        x = self.base_model(x)
        x = global_mean_pool(x, x.batch)
        x = self.Regressor(x)
        return x

class NodeLevelRegressor(nn.Module):
    def __init__(self, base_model: nn.Module, node_dim: int):
        super(NodeLevelRegressor, self).__init__()
        self.base_model = base_model
        self.Regressor = Regressor(node_dim, node_dim // 2, 1)
    
    def forward(self, x: Data):
        # Input [n, d] treated by the regressor as a batch of n samples of dimension d
        x = self.Regressor(x)
        return x
