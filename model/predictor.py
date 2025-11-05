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

class ComplexClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ComplexClassifier, self).__init__()
        self.fc1_complex = nn.Linear(2*input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x_real, x_imag = x.real, x.imag
        x = torch.cat([x_real, x_imag], dim=-1).float()
        x = self.fc1_complex(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

class ComplexRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ComplexRegressor, self).__init__()
        self.fc1 = nn.Linear(2*input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x_real, x_imag = x.real, x.imag
        x = torch.cat([x_real, x_imag], dim=-1).float()
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

class GraphLevelRegressor(nn.Module):
    def __init__(self, base_model: nn.Module, node_dim: int, complex_floats: bool = False):
        super(GraphLevelRegressor, self).__init__()
        self.base_model = base_model
        self.Regressor = Regressor(node_dim, node_dim // 2, 1) if not complex_floats else ComplexRegressor(node_dim, node_dim // 2, 1)

    def forward(self, x: Data):
        x = self.base_model(x)
        x = x.mean(dim=0, keepdim=True)
        x = self.Regressor(x)
        return x


class NodeLevelRegressor(nn.Module):
    def __init__(self, base_model: nn.Module, node_dim: int, output_dim: int, complex_floats: bool = False):
        super(NodeLevelRegressor, self).__init__()
        self.base_model = base_model
        hidden_dim = node_dim // 2 if node_dim // 2 > 0 output_dim else node_dim
        self.Regressor = Regressor(node_dim, node_dim // 2, output_dim) if not complex_floats else ComplexRegressor(node_dim, node_dim // 2, 1)

    def forward(self, x: Data):
        # Input [n, d] treated by the regressor as a batch of n samples of dimension d
        x = self.base_model(x)
        x = self.Regressor(x)
        return x


class GraphLevelClassifier(nn.Module):
    def __init__(self, base_model: nn.Module, node_dim: int, num_classes: int, complex_floats: bool = False):
        super(GraphLevelClassifier, self).__init__()
        self.base_model = base_model
        self.Classifier = Classifier(node_dim, node_dim // 2, num_classes) if not complex_floats else ComplexClassifier(node_dim, node_dim // 2, num_classes)

    def forward(self, x: Data):
        x = self.base_model(x)
        x = x.mean(dim=0, keepdim=True)
        x = self.Classifier(x)
        return x


class NodeLevelClassifier(nn.Module):
    def __init__(self, base_model: nn.Module, node_dim: int, num_classes: int, complex_floats: bool = False):
        super(NodeLevelClassifier, self).__init__()
        self.base_model = base_model
        self.Classifier = Classifier(node_dim, node_dim // 2, num_classes) if not complex_floats else ComplexClassifier(node_dim, node_dim // 2, num_classes)

    def forward(self, x: Data):
        # Input [n, d] treated by the classifier as a batch of n samples of dimension d
        x = self.base_model(x)
        x = self.Classifier(x)
        return x
