import torch
import torch.nn as nn
from torch.utils.data import Dataset


def enn_collate_fn(data):
    """
    TODO: Incorporate multiple y's
    """
    x, forces, energies = zip(*data)
    forces = torch.stack(forces)
    energies = torch.stack(energies)

    x = list(zip(*x))
    if len(x) == 5:
        node_position, node_feature, edge_index, edge_feature, node_distances = x
        assert (len(node_position) == len(node_feature)
                == len(edge_index) == len(edge_feature)
                == len(node_distances))
    else:
        raise ValueError("Incorrect number of arguments provided")

    node_num = torch.tensor([node_pos_i.shape[0]
                            for node_pos_i in node_position])
    # Use list comprehension as node sizes are different
    node_position = torch.cat(node_position)
    node_feature = torch.cat(node_feature)
    edge_num = torch.tensor([e_i.shape[-1] for e_i in edge_index])
    edge_index = list(edge_index)
    edge_feature = torch.cat(edge_feature)

    to_add = torch.cumsum(node_num, dim=0) - node_num
    shifted_edges = []
    to_add = to_add.to(node_position.device)
    for (shift, graph_ei) in zip(to_add, edge_index):
        shifted_edges.append(graph_ei + shift)
    shifted_edges = torch.cat(shifted_edges, dim=-1)
    node_num = node_num.to(node_position.device)
    edge_num = edge_num.to(node_position.device)
    batch_size = forces.shape[0]
    if node_distances[0] is not None:
        node_distances = torch.stack(node_distances)
        return [node_position, [node_feature, shifted_edges, node_num, edge_feature, edge_num, node_distances]], forces, energies, batch_size
    else:
        return [node_position, [node_feature, shifted_edges, node_num, edge_feature, edge_num]], forces, energies, batch_size


class EGNNDataset(Dataset):
    def __init__(self, energies, forces, node_positions, node_features,
                 edge_indices, edge_features, node_distances=None):
        self.energies = energies
        self.forces = forces
        self.node_positions = node_positions
        self.node_features = node_features
        self.edge_indices = edge_indices
        self.edge_features = edge_features
        self.node_distances = node_distances

        assert (self.energies.shape[0]
                == self.forces.shape[0]
                == self.node_positions.shape[0]
                == len(self.edge_indices)
                == len(edge_features))

    def __len__(self):
        return self.node_positions.shape[0]

    def __getitem__(self, idx):
        if self.node_distances is not None:
            x = [self.node_positions[idx], self.node_features,
                 self.edge_indices[idx], self.edge_features[idx], self.node_distances[idx]]
        else:
            node_distances = torch.linalg.norm((self.node_positions[idx].unsqueeze(-2)
                                                - self.node_positions[idx].unsqueeze(-3)), axis=-1)
            x = [self.node_positions[idx], self.node_features,
                 self.edge_indices[idx], self.edge_features[idx], node_distances]
        return x, self.forces[idx], self.energies[idx]



def unsorted_segment_sum(data, row_index, num_nodes: int):
    result_shape = (num_nodes, data.size(-1))
    row_index = row_index.unsqueeze(-1).expand(-1, data.size(-1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    result.scatter_add_(0, row_index, data)
    return result


def unsorted_segment_mean(data, row_index, num_nodes: int):
    result_shape = (num_nodes, data.size(-1))
    row_index = row_index.unsqueeze(-1).expand(-1, data.size(-1))
    result = data.new_full(result_shape, 0)
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, row_index, data)
    count.scatter_add_(0, row_index, torch.ones_like(data))
    return result / count.clamp(min=1)


class EGCL(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0,
                 update_pos=True, use_attention=True, activation=nn.SiLU()):
        super().__init__()
        input_edge = input_nf * 2
        edge_coords_nf = 1
        #h_i + h_j + distance + edge_features
        self.edge_mlp_1 = nn.Linear((input_nf * 2) + 1 + edges_in_d,
                                    hidden_nf)
        self.edge_mlp_2 = nn.Linear(hidden_nf, hidden_nf)

        self.update_pos = update_pos
        #hi(input_nf) + mi(hidden_nf)
        self.node_mlp_1 = nn.Linear(hidden_nf + input_nf, hidden_nf)
        self.node_mlp_2 = nn.Linear(hidden_nf, output_nf)

        if (self.update_pos):
            # mi(hidden_nf)
            self.pos_mlp_1 = nn.Linear(hidden_nf, hidden_nf)
            self.pos_mlp_2 = nn.Linear(hidden_nf, 1, bias=False)
            nn.init.xavier_uniform_(self.pos_mlp_2.weight, gain=0.001)

        self.use_attention = use_attention
        if (self.use_attention):
            self.attention = nn.Linear(hidden_nf, 1)

        self.activation = activation
        self.sigmoid = nn.Sigmoid()

    def edge_model(self, source_h, target_h, dist_sq, edge_attr):
        """Eq. 3
        """
        if (edge_attr is None):
            """No edge attributes
            """
            out = torch.cat([source_h, target_h, dist_sq], dim=-1)
        else:
            out = torch.cat([source_h, target_h, dist_sq, edge_attr], dim=-1)
        out = self.activation(self.edge_mlp_1(out))
        edge_feat = self.activation(self.edge_mlp_2(out))
        if (self.use_attention):
            # Uses sigmoid not softmax independent of edge num
            att_val = self.sigmoid(self.attention(edge_feat))
            edge_feat = edge_feat * att_val
        return edge_feat

    def node_model(self, h, edge_index, edge_attr):
        row = edge_index[0, :]
        col = edge_index[1, :]
        agg = unsorted_segment_sum(edge_attr, row, num_nodes=h.size(0))  # Eq.5
        agg = torch.cat([h, agg], dim=1)
        # Eq. 6
        out = self.activation(self.node_mlp_1(agg))
        h_new = self.node_mlp_2(out)
        h_new = h_new + h  # Residual Layer
        return h_new

    def pos_model(self, pos, edge_index, pos_diff, edge_feat):
        """Eq. 4 (Never used but worth incorporating updates authors made if using)
        """
        row = edge_index[0, :]
        col = edge_index[1, :]
        trans = self.activation(self.pos_mlp_1(edge_feat))
        trans = self.pos_mlp_2(edge_feat)
        trans = pos_diff * trans
        agg = unsorted_segment_mean(trans, row, num_nodes=pos.size(0))
        pos = pos + agg
        return pos

    def get_radial_information(self, edge_index, pos):
        """Get relative positions and distances
        (pos_diff never used but worth incorporating updates authors made if using)
        """
        row = edge_index[0, :]
        col = edge_index[1, :]
        pos_diff = pos[row] - pos[col]
        dist_sq = torch.sum(pos_diff**2, dim=-1).unsqueeze(-1)

        return dist_sq, pos_diff

    def forward(self, h, pos, edge_index, edge_attr):
        row = edge_index[0, :]
        col = edge_index[1, :]

        dist_sq, pos_diff = self.get_radial_information(edge_index, pos)
        edge_attr = self.edge_model(h[row], h[col], dist_sq, edge_attr)
        if (self.update_pos):
            pos = self.pos_model(pos, edge_index, pos_diff, edge_attr)
        h = self.node_model(h, edge_index, edge_attr)
        return h, pos, edge_attr


class EGNN(nn.Module):
    def __init__(self, input_nf=5, hidden_nf=32, final_nf=32, n_layers=4, update_pos=False, use_attention=True,
                 do_sum_pool=False, num_edge_types=2, activation=nn.SiLU()):
        super().__init__()
        self.node_embedding_in = nn.Embedding(input_nf, hidden_nf)
        self.edge_embedding_in = nn.Embedding(num_edge_types, hidden_nf)
        self.egcl_layers = []
        for _ in range(n_layers):
            self.egcl_layers.append(EGCL(input_nf=hidden_nf,
                                         output_nf=hidden_nf, hidden_nf=hidden_nf,
                                         edges_in_d=hidden_nf, update_pos=update_pos,
                                         use_attention=use_attention,
                                         activation=activation))
        self.egcl_layers = nn.ModuleList(self.egcl_layers)
        self.node_dec_1 = nn.Linear(hidden_nf, hidden_nf)
        self.do_sum_pool = do_sum_pool
        if self.do_sum_pool:
            """Used for a scalar output
            """
            self.edge_attr_dec_1 = nn.Linear(hidden_nf, hidden_nf)
            self.edge_attr_dec_2 = nn.Linear(hidden_nf, hidden_nf)
            self.node_dec_2 = nn.Linear(hidden_nf, hidden_nf)
            self.graph_dec_1 = nn.Linear(2 * hidden_nf, final_nf)
        else:
            """Used for a vector output (i.e. DiffPool)
            """
            self.node_dec_2 = nn.Linear(hidden_nf, final_nf)
        self.activation = activation

    def forward(self, pos, h, edge_index, node_num, edge_attr, edge_num):
        h = self.node_embedding_in(h)
        edge_attr = self.edge_embedding_in(edge_attr)
        if edge_attr.shape[0] > 0:
            for egcl_layer in self.egcl_layers:
                h, pos, edge_attr = egcl_layer(h, pos, edge_index, edge_attr)

        h = self.activation(self.node_dec_1(h))
        h = self.node_dec_2(h)

        # Split nodes into individual graphs
        indices = torch.cumsum(node_num, dim=0)[:-1].cpu()
        h = torch.tensor_split(h, indices)

        if not self.do_sum_pool:
            h_all = torch.stack(h)
            return h_all

        else:
            # All graphs should be the same size so can use sum/mean interchangeably
            # Sum over all the nodes in a graph
            pool_h = torch.stack([h_i.sum(dim=-2) for h_i in h])
            edge_attr = self.activation(self.edge_attr_dec_1(edge_attr))
            edge_attr = self.edge_attr_dec_2(edge_attr)

            # Split edges into individual graphs
            indices = torch.cumsum(edge_num, dim=0)[:-1].tolist()
            edge_attr = torch.tensor_split(edge_attr, indices)
            # Sum over all edges_per graph
            pool_edge_attr = torch.stack([ea_i.sum(dim=-2)
                                          for ea_i in edge_attr])

            node_edge_attr = torch.cat((pool_h, pool_edge_attr), dim=-1)
            out = self.graph_dec_1(node_edge_attr)
            return out
