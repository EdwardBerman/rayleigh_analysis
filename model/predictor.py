import torch
import torch.nn as nn
from torch_geometric.data import Data

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
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
        x = x.mean(dim=0, keepdim=True)
        x = self.Regressor(x)
        return x

class NodeLevelRegressor(nn.Module):
    def __init__(self, base_model: nn.Module, node_dim: int):
        super(NodeLevelRegressor, self).__init__()
        self.base_model = base_model
        self.Regressor = Regressor(node_dim, node_dim // 2, 1)
    
    def forward(self, x: Data):
        # Input [n, d] treated by the regressor as a batch of n samples of dimension d
        x = self.base_model(x)
        x = self.Regressor(x)
        return x

class GraphLevelClassifier(nn.Module):
    def __init__(self, base_model: nn.Module, node_dim: int, num_classes: int):
        super(GraphLevelClassifier, self).__init__()
        self.base_model = base_model
        self.Classifier = Classifier(node_dim, node_dim // 2, num_classes)
    
    def forward(self, x: Data):
        x = self.base_model(x)
        x = x.mean(dim=0, keepdim=True)
        x = self.Classifier(x)
        return x

class NodeLevelClassifier(nn.Module):
    def __init__(self, base_model: nn.Module, node_dim: int, num_classes: int):
        super(NodeLevelClassifier, self).__init__()
        self.base_model = base_model
        self.Classifier = Classifier(node_dim, node_dim // 2, num_classes)
    
    def forward(self, x: Data):
        # Input [n, d] treated by the classifier as a batch of n samples of dimension d
        x = self.base_model(x)
        x = self.Classifier(x)
        return x
