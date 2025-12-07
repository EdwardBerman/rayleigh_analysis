import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self, input_size, output_size=128, layer_norm=True, n_hidden=2, hidden_size=128):
        super(MLP, self).__init__()
        if hidden_size == 0:
            f = [nn.Linear(input_size, output_size)]
        else:
            f = [nn.Linear(input_size, hidden_size), nn.ReLU()]
            h = 1
            for i in range(h, n_hidden):
                f.append(nn.Linear(hidden_size, hidden_size))
                f.append(nn.ReLU())
            f.append(nn.Linear(hidden_size, output_size))
            if layer_norm:
                f.append(nn.LayerNorm(output_size))

        self.f = nn.Sequential(*f)

    def forward(self, x):
        return self.f(x)


class GNN(nn.Module):
    def __init__(self, n_hidden=2, node_size=128, edge_size=128, output_size=None, layer_norm=False):
        super(GNN, self).__init__()
        output_size = output_size or node_size
        self.f_edge = MLP(input_size=edge_size + node_size * 2, n_hidden=n_hidden, layer_norm=layer_norm,
                          output_size=edge_size)
        self.f_node = MLP(input_size=edge_size + node_size, n_hidden=n_hidden, layer_norm=layer_norm,
                          output_size=output_size)

    def forward(self, V, E, edges):
        senders = torch.gather(V, -2, edges[..., 0].unsqueeze(-1).repeat(1, 1, V.shape[-1]))
        receivers = torch.gather(V, -2, edges[..., 1].unsqueeze(-1).repeat(1, 1, V.shape[-1]))

        edge_inpt = torch.cat([senders, receivers, E], dim=-1)
        edge_embeddings = self.f_edge(edge_inpt)

        col = edges[..., 0].unsqueeze(-1).repeat(1, 1, edge_embeddings.shape[-1])
        edge_sum = scatter_sum(edge_embeddings, col, dim=-2)

        node_inpt = torch.cat([V, edge_sum], dim=-1)
        node_embeddings = self.f_node(node_inpt)

        return node_embeddings, edge_embeddings


class GraphViT(nn.Module):
    def __init__(self, state_size=1, w_size=512, n_attention=4, nb_gn=4, n_heads=4):
        super(GraphViT, self).__init__()
        pos_start = -3
        pos_length = 8
        self.encoder = Encoder(nb_gn, state_size, pos_length)
        self.graph_pooling = GraphPooling(w_size, pos_length=pos_length)
        self.graph_retrieve = GraphRetrieveSimple(w_size, pos_length, state_size)
        self.attention = nn.ModuleList([AttentionBlock_PreLN(w_size, pos_length, n_heads) for _ in range(n_attention)])
        self.ln = nn.LayerNorm(w_size)

        self.noise_std = 0.0
        self.positional_encoder = Positional_Encoder(pos_start, pos_length)

    def build_cluster_index_and_mask(self, cluster_labels, cluster_centers, device):
        """
        cluster_labels  [N]     -> label per node  (0 .. C-1)
        cluster_centers [C,3]  -> only needed to know how many clusters exist
        Returns:
            clusters      [1,C,N_max] Node indices for each cluster
            cluster_mask  [1,C,N_max] 1 where real, 0 where padded
        """
        B = 1
        N = cluster_labels.size(0)
        C = cluster_centers.size(0)

        # collect node indices belonging to each cluster
        idx_per_cluster = [
            (cluster_labels == c).nonzero(as_tuple=False).view(-1).to(device)
            for c in range(C)
        ]

        # max cluster size for padding shape
        N_max = max(len(t) for t in idx_per_cluster)

        clusters = torch.zeros(B, C, N_max, dtype=torch.long, device=device)
        cluster_mask = torch.zeros(B, C, N_max, dtype=torch.bool, device=device)

        for c, idx in enumerate(idx_per_cluster):
            count = len(idx)
            if count > 0:
                clusters[0,c,:count] = idx
                cluster_mask[0,c,:count] = True   # valid entries

        return clusters, cluster_mask, N_max

    def forward(self, data):
        
        # if first forward call print data keys
        if not hasattr(self, 'printed_keys'):
            print("Data keys:", data.keys)
            self.printed_keys = True

        device = data.pos.device

        mesh_pos        = data.pos.unsqueeze(0)      # [1,2501,3]
        state           = data.x                     # [2501,5, 1]
        cluster_labels  = data.cluster_labels        # [2501]
        cluster_centers = data.cluster_centers       # [127,3]
        edges = data.edge_index
        clusters = 120

        clusters, cluster_mask, N_max = self.build_cluster_index_and_mask(
            cluster_labels, cluster_centers, device
        )

        if hasattr(data, 'node_type'):
            node_type = data.node_type.float()                    # assume user provided one-hot or integer class
            if node_type.dim() == 3: node_type = node_type.unsqueeze(1)   # → [B,1,N,C]
        else:
            # default to “all NORMAL nodes” = 1-hot with 1 channel
            node_type = torch.zeros(state.shape[2], state.shape[1], state.shape[0], 1,
                                    device=state.device, dtype=state.dtype).long()
            data.node_type = node_type

        # Removed apply noise flag for fair comparison and bc paper says it hurt and didnt help
        
        #if apply_noise:
            # Following MGN, this add noise to the input. Better results are obtained with longer windows and no noise
            #mask = torch.logical_or(node_type[:, 0, :, NODE_NORMAL] == 1, node_type[:, 0, :, NODE_OUTPUT] == 1)
            #noise = torch.randn_like(state[:, 0]).to(state[:, 0].device) * self.noise_std
            #state[:, 0][mask] = state[:, 0][mask] + noise[mask]
        mesh_posenc, cluster_posenc = self.positional_encoder(mesh_pos, clusters, cluster_mask)

        output_hat = []
        
        for t in range(1, state.shape[1]):


            V, E = self.encoder(mesh_pos, edges, state[:, t - 1], node_type, mesh_posenc)
            W = self.graph_pooling(V, clusters, mesh_posenc, clusters_mask)

            # We use batch size 1 so no need to adjust attention mask for multiple simulations

            # This attention_mask deals with the ghost nodes needed to batch multiple simulations
            #attention_mask = clusters_mask[:, t - 1].sum(-1, keepdim=True) == 0
            #attention_mask = attention_mask.unsqueeze(1).repeat(1, len(self.attention), 1, W.shape[1]).view(-1, W.shape[1], W.shape[1])
            #attention_mask[:, torch.eye(W.shape[1], dtype=torch.bool)] = False
            #attention_mask = attention_mask.transpose(-1, -2)

            attention_mask = None

            for i, a in enumerate(self.attention):
                W = a(W, attention_mask, cluster_posenc)
            W = self.ln(W)

            next_output = self.graph_retrieve(W, V, clusters[:, t - 1], mesh_posenc, edges[:, t - 1], E)

            # NO BOUNDARY CONDITIONS IN OUR SETUP. Commenting out prior mask
            
            # Following MGN, we force the boundary conditions at each steps
            #mask = torch.logical_or(node_type[:, t, :, NODE_INPUT] == 1, node_type[:, t, :, NODE_WALL] == 1)
            #mask = torch.logical_or(mask, node_type[:, t, :, NODE_DISABLE] == 1)
            #next_state[mask, :] = state[:, t][mask, :]
            output_hat.append(next_output)

        output_hat = torch.stack(output_hat, dim=1)
        print("Output hat shape:", output_hat.shape)
        breakpoint()
        return output_hat.permute(0,2,1,3).squeeze(0)  # [N, T-1, F]


class AttentionBlock_PreLN(nn.Module):
    def __init__(self, w_size, pos_length, n_heads):
        super(AttentionBlock_PreLN, self).__init__()
        self.ln1 = nn.LayerNorm(w_size)

        embed_dim = w_size + 4 * pos_length

        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads, batch_first=True)
        self.linear = nn.Linear(embed_dim, w_size)
        self.ln2 = nn.LayerNorm(w_size)
        self.mlp = MLP(input_size=w_size, n_hidden=1, output_size=w_size, hidden_size=w_size, layer_norm=False)

    def forward(self, W, attention_mask, posenc):
        W1 = self.ln1(W)
        W1_posenc = torch.cat([W1, posenc], dim=-1)
        W2 = self.attention(W1_posenc, W1_posenc, W1_posenc, attn_mask=attention_mask)[0]
        W3 = W + self.linear(W2)

        W4 = self.ln2(W3)
        W5 = self.mlp(W4)

        W6 = W3 + W5
        return W6


class GraphPooling(nn.Module):
    def __init__(self, w_size, pos_length):
        super(GraphPooling, self).__init__()
        input_size = 128 + pos_length * 8

        self.rnn_pooling = nn.GRU(input_size=input_size, hidden_size=w_size, batch_first=True)
        self.linear_rnn = MLP(input_size=w_size, output_size=w_size, n_hidden=1, layer_norm=False)

    def forward(self, V, clusters, positional_encoding, cluster_mask):
        pos_by_cluster = torch.gather(positional_encoding, -2, clusters.reshape(clusters.shape[0], -1, 1).repeat(1, 1,
                                                                                                                 positional_encoding.shape[
                                                                                                                     -1]))
        pos_features = pos_by_cluster.reshape(*clusters.shape, -1)

        V_by_cluster = torch.gather(V, -2, clusters.reshape(clusters.shape[0], -1, 1).repeat(1, 1, V.shape[-1]))
        V_by_cluster = V_by_cluster.reshape(*clusters.shape, -1)

        inpt_by_cluster = torch.cat([V_by_cluster, pos_features], dim=-1)

        B, C, N, S = inpt_by_cluster.shape
        output, h = self.rnn_pooling(inpt_by_cluster.reshape(B * C, N, S))
        indices = (cluster_mask.sum(-1).long() - 1).reshape(B * C)
        indices[indices == -1] = output.shape[-2] - 1
        w = torch.gather(output, 1, indices.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, output.shape[-1]))

        w = self.linear_rnn(w)
        W = w.reshape(B, C, -1)

        return W


class GraphRetrieveSimple(nn.Module):
    def __init__(self, w_size, pos_length, state_size):
        pos_size = pos_length * 8
        super(GraphRetrieveSimple, self).__init__()
        node_size = w_size + 128 + pos_size
        self.gnn = GNN(node_size=node_size, output_size=128)
        self.final_mlp = nn.Sequential(
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, state_size)
        )

    def forward(self, W, V, clusters, positional_encoding, edges, E):
        B, N, S = V.shape
        C = W.shape[1]
        K = clusters.shape[-1]

        W = W.unsqueeze(-2).repeat(1, 1, K, 1).view(B, C * K, -1)
        W = W.scatter(-2, clusters.reshape(B, -1, 1).repeat(1, 1, W.shape[-1]), W)
        W = W[:, :N]

        nodes = torch.cat([V, W, positional_encoding], dim=-1)
        nodes, _ = self.gnn(nodes, E, edges)
        final_state = self.final_mlp(nodes)
        return final_state


class Encoder(nn.Module):
    def __init__(self, nb_gn=4, state_size=3, pos_length=7):
        super(Encoder, self).__init__()
        self.encoder_node = MLP(input_size=9 + state_size, output_size=128, n_hidden=1, layer_norm=False)
        self.encoder_edge = MLP(input_size=3, output_size=128, n_hidden=1, layer_norm=False)

        node_size = 128 + pos_length * 8
        self.encoder_gn = nn.ModuleList(
            [GNN(node_size=node_size, edge_size=128, output_size=128, layer_norm=True) for _ in
             range(nb_gn)])

    def forward(self, mesh_pos, edges, states, node_type, pos_enc):
        breakpoint()
        V = torch.cat([states, node_type], dim=-1)

        senders = torch.gather(mesh_pos, -2, edges[..., 0].unsqueeze(-1).repeat(1, 1, 2))
        receivers = torch.gather(mesh_pos, -2, edges[..., 1].unsqueeze(-1).repeat(1, 1, 2))

        distance = senders - receivers
        norm = torch.sqrt((distance ** 2).sum(-1, keepdims=True))
        E = torch.cat([distance, norm], dim=-1)

        V = self.encoder_node(V)
        E = self.encoder_edge(E)

        for i in range(len(self.encoder_gn)):
            inpt = torch.cat([V, pos_enc], dim=-1)
            v, e = self.encoder_gn[i](inpt, E, edges)
            V = V + v
            E = E + e

        return V, E


class Positional_Encoder(nn.Module):
    def __init__(self, pos_start, pos_length):
        super(Positional_Encoder, self).__init__()
        self.pos_length = pos_length
        self.pos_start = pos_start

    def forward(self, mesh_pos, clusters, cluster_mask):
        B, N, _ = mesh_pos.shape
        _, K, C = clusters.shape

        meshpos_by_cluster = torch.gather(mesh_pos, -2, clusters.reshape(B, -1, 1).repeat(1, 1, 2))
        meshpos_by_cluster = meshpos_by_cluster.reshape(*clusters.shape, -1)

        clusters_centers = meshpos_by_cluster.sum(dim=-2)
        clusters_centers = clusters_centers / (cluster_mask.sum(-1, keepdim=True) + 1e-8)

        distances_to_cluster = clusters_centers.unsqueeze(-2) - meshpos_by_cluster
        pos_embeddings = self.embed(distances_to_cluster)
        S = pos_embeddings.shape[-1]
        pos_embeddings = pos_embeddings.reshape(B, -1, S)
        relative_positions = pos_embeddings.scatter(-2, clusters.reshape(B, -1, 1).repeat(1, 1, S),
                                                    pos_embeddings.view(B, -1, S))
        relative_positions = relative_positions[:, :N]

        nodes_embedding = torch.cat([self.embed(mesh_pos), relative_positions], dim=-1)

        return nodes_embedding, self.embed(clusters_centers)

    def embed(self, pos):
        original_shape = pos.shape
        pos = pos.reshape(-1, original_shape[-1])
        index = torch.arange(self.pos_start, self.pos_start + self.pos_length, device=pos.device)
        index = index.float()
        freq = 2 ** index * torch.pi
        cos_feat = torch.cos(freq.view(1, 1, -1) * pos.unsqueeze(-1))
        sin_feat = torch.sin(freq.view(1, 1, -1) * pos.unsqueeze(-1))
        embedding = torch.cat([cos_feat, sin_feat], dim=-1)
        embedding = embedding.view(*original_shape[:-1], -1)
        return embedding
