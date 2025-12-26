import torch.nn as nn

from external.custom_hermes.nn.spiralconv import SpiralConv

class SpiralNet(nn.Module):
    def __init__(self, dims, seq_length, final_activation):
        super().__init__()

        self.dims = dims
        self.seq_length = seq_length
        self.final_activation = final_activation

        self.out_dim = self.dims[-1]

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 2):
            self.layers.append(SpiralConv(self.dims[i], self.dims[i + 1], seq_length))
            self.layers.append(nn.ELU())

        self.layers.append(SpiralConv(self.dims[-2], self.dims[-1], seq_length))
        if self.final_activation:
            self.layers.append(nn.ELU())

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.named_modules():
            if isinstance(layer, SpiralConv):
                layer.reset_parameters()

    def forward(self, data):
        x = data.x.squeeze(-1)

        for layer in self.layers:
            if isinstance(layer, SpiralConv):
                x = layer(x, data.spiral_indices)
            else:
                x = layer(x)

        return x[:, :, None]
