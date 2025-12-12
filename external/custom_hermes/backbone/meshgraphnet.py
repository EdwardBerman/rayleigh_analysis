"""
Reference: https://github.com/CCSI-Toolset/MGN/blob/main/GNN/GNNComponents/GNNComponents.py

Modified by Hermes authors
"""
from torch import nn

from external.custom_hermes.nn.meshgraphnet import MLP, GraphProcessor


class MeshGraphNet(nn.Module):
    def __init__(
        self,
        num_mp_layers,
        mp_node_in_dim,
        mp_edge_in_dim,
        mp_node_hid_dim,
        mp_node_num_layers,
        mp_edge_hid_dim,
        mp_edge_num_layers,
    ):
        """
        MeshGraphNets model (arXiv:2010.03409); default values are based on the paper/supplement

        in_dim_node: input node feature dimension
        in_dim_edge: input edge feature dimension
        out_dim: output dimension
        out_dim_node: encoded node feature dimension
        out_dim_edge: encoded edge feature dimension
        hidden_dim_node: node encoder MLP dimension
        hidden_dim_edge: edge encoder MLP dimension
        hidden_layers_node: number of node encoder MLP layers
        hidden_layers_edge: number of edge encoder MLP layers
        mp_iterations: number of message passing iterations
        hidden_dim_processor_node: MGN node processor MLP dimension
        hidden_dim_processor_edge: MGN edge processor MLP dimension
        hidden_layers_processor_node: number of MGN node processor MLP layers
        hidden_layers_processor_edge: number of MGN edge processor MLP layers
        mlp_norm_type: MLP normalization type ('LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm')
        hidden_dim_decoder: decoder MLP dimension
        hidden_layers_decoder: decoder MLP layers
        output_type: output type ('state', 'velocity', 'acceleration')
        use_adaptive_mesh: if True, use adaptive; not functional yet
        """

        super().__init__()

        self.graph_processor = GraphProcessor(
            num_mp_layers,
            mp_node_in_dim,
            mp_edge_in_dim,
            mp_node_hid_dim,
            mp_edge_hid_dim,
            mp_node_num_layers,
            mp_edge_num_layers,
            norm_type="LayerNorm",
        )

        # graph: torch_geometric.data.Data object with the following attributes:
        #       x: node x feature array (volume fraction, pressure, node type, inlet velocity, etc.)
        #       edge_index: 2 x edge array
        #       edge_attr: edge x feature matrix (distance, relative coordinates)

        self.out_dim = mp_node_in_dim

    def forward(self, data):
        x = data.x.squeeze(-1)
        # print data 
        print(data)

        pos = data.pos                      # [N, 3]
        src, dst = data.edge_index          # [E], [E]
        rel = pos[dst] - pos[src]
        data.edge_attr = (rel ** 2).sum(dim=-1, keepdim=True)

        # message passing
        x, _ = self.graph_processor(x, data.edge_index, data.edge_attr)

        return x.unsqueeze(-1)
