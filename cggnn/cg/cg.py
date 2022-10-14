import torch
import torch.nn as nn
from .egnn import EGNN


class BeadInfo(nn.Module):
    def __init__(self, num_beads=6):
        super().__init__()
        self.num_beads = num_beads
        bead_edge_indices = torch.stack(torch.where(torch.zeros((num_beads,
                                                                 num_beads)) == 0))
        bead_edge_features = torch.tensor([0] * bead_edge_indices.shape[-1])
        node_features = torch.tensor(list(range(num_beads))).long()

        self.register_buffer('bead_edge_indices', bead_edge_indices)
        self.register_buffer('bead_edge_features', bead_edge_features)
        self.register_buffer('node_features', node_features)

    def get_bead_node_features(self, batch_size):
        """Returns bead node features
        Returns:
            all_node_features: A torch tensor of shape (self.num_beads) consisting of bead features
        """
        all_node_features = torch.stack([self.node_features.clone()
                                         for _ in range(batch_size)])
        return all_node_features

    def get_bead_info_sch(self, batch_size):
        """Returns bead information as an input to an SchNet model
        """
        all_node_features = torch.stack([self.node_features.clone()
                                         for _ in range(batch_size)])

        return all_node_features

    def get_bead_info(self, batch_size):
        """Returns bead information as an input to an E(N) GNN model
        """

        all_node_features = torch.cat([self.node_features.clone()
                                       for _ in range(batch_size)])

        all_bead_edge_indices = [self.bead_edge_indices.clone()
                                 for _ in range(batch_size)]
        all_bead_edge_features = torch.cat([self.bead_edge_features.clone()
                                            for _ in range(batch_size)])

        node_num = torch.tensor([self.num_beads] * batch_size,
                                device=self.bead_edge_indices.device)
        edge_num = torch.tensor([b_e_i.shape[-1] for b_e_i in all_bead_edge_indices],
                                device=self.bead_edge_indices.device)

        to_add = torch.cumsum(node_num, dim=0).to(
            self.bead_edge_indices.device) - node_num
        all_bead_edge_shifted_indices = []
        for (shift, b_e_i) in zip(to_add, all_bead_edge_indices):
            all_bead_edge_shifted_indices.append(b_e_i + shift)

        all_bead_edge_shifted_indices = torch.cat(
            all_bead_edge_shifted_indices, dim=-1).long()

        inputs = [all_node_features, all_bead_edge_shifted_indices,
                  node_num, all_bead_edge_features, edge_num]
        return inputs


class CGFixed(nn.Module):
    def __init__(self, num_atoms_fg, atoms_r):
        """
        A Coarse-Graining approach, where the map is fixed a priori
        """
        super().__init__()
        proj = torch.zeros((num_atoms_fg, len(atoms_r))).float()
        proj[atoms_r, torch.arange(len(atoms_r))] = 1
        self.register_buffer("proj", proj)
        inv_proj = torch.eye(len(atoms_r)).float()
        self.register_buffer("inv_proj", inv_proj)
        self.num_atoms = num_atoms_fg
        self.num_beads = len(atoms_r)
        self.num_atoms_r = len(atoms_r)
        self.bead_info = BeadInfo(num_beads=self.num_beads)

    def forward(self, x_data, batch_size, return_proj=False, fix_inv_proj=False, backbone_r_indices=None, train_inv_proj_only=False):
        """Computes coarse-graining embedding
        Argumemnts:
            x_data: A list of [x, inputs], where x is a PyTorch tensor of shape (batch_size*n_atoms, 3) representing the fine-grained positions
            batch_size: An int representing the batch size
            return_proj: A boolean specifying whether to return projection matrices
            train_inv_proj_only: A boolean specifying whether to train the inverse projection only (feature not used)
            fix_inv_proj: A boolean specifying whether to fix_inverse_projection_matrix to backbone_r_indices (Not used)
            backbone_r_indices: A PyTorch tensor of shape (backbone_r_indices.shape[0]) corresponding to indices of backbone atoms (Not used )
        Returns:
            mean_x: A PyTorch tensor of shape (batch_size, num_r_indices, 3) representing the reconstructed positions
            link_loss: A torch tensor of shape (,) representing the link loss
            ent_loss: A torch tensor of shape (,) representing the entropy loss
            diag_loss: A torch tensor of shape (,) representing the assignment loss
            proj (if return_proj): A PyTorch tensor of (batch_size, n_beads, n_atoms) representing the projection matrix P_x
            inv_proj (if return_proj): A PyTorch tensor of (batch_size, self.num_atoms_r, n_beads) representing the inverse projection matrix P_z
        """
        proj_t = (self.proj.T).unsqueeze(0).repeat((batch_size, 1, 1))
        inv_proj_t = (self.inv_proj).unsqueeze(0).repeat((batch_size, 1, 1))
        x, inputs = x_data
        x = x.reshape((batch_size, self.num_atoms, 3))
        bead_x = torch.bmm(proj_t, x)
        bead_x = (bead_x - bead_x.mean(-2).unsqueeze(-2))
        mean_x = torch.bmm(inv_proj_t, bead_x)
        ent_loss = torch.tensor([0], device=x.device)
        link_loss = 0
        diag_loss = 0
        if return_proj:
            return mean_x, link_loss, ent_loss, diag_loss, proj_t, inv_proj_t
        else:
            return mean_x, link_loss, ent_loss, diag_loss


class CGAPB(nn.Module):
    """
    A Coarse-Graining Function where the atomic contributions from all atoms to a bead should be 1
    """

    def __init__(self, width=64, n_layers=2,
                 num_atoms=22, num_beads=6, num_atoms_r=10, min_detach_loss=-5):
        super().__init__()
        self.bead_info = BeadInfo(num_beads=num_beads)
        self.pool = EGNN(input_nf=4, hidden_nf=width, final_nf=num_beads,
                         num_edge_types=3, n_layers=n_layers)

        self.depool = EGNN(input_nf=num_beads, hidden_nf=width, final_nf=num_atoms_r,
                           num_edge_types=3, n_layers=n_layers)
        self.num_atoms = num_atoms
        self.num_beads = num_beads
        self.num_atoms_r = num_atoms_r
        self.min_detach_loss = min_detach_loss

    def forward(self, x_data, batch_size, train_inv_proj_only=False, return_proj=False, fix_inv_proj=False, backbone_r_indices=None):
        """Computes coarse-graining embedding
        Argumemnts:
            x_data: A list of [x, inputs], where x is a PyTorch tensor of shape (batch_size*n_atoms, 3) representing the fine-grained positions
            batch_size: An int representing the batch size
            train_inv_proj_only: A boolean specifying whether to train the inverse projection only (feature not used)
            return_proj: A boolean specifying whether to return projection matrices
            fix_inv_proj: A boolean specifying whether to fix_inverse_projection_matrix to backbone_r_indices
            backbone_r_indices: A PyTorch tensor of shape (backbone_r_indices.shape[0]) corresponding to indices of backbone atoms
        Returns:
            mean_x: A PyTorch tensor of shape (batch_size, num_r_indices, 3) representing the reconstructed positions
            link_loss: A torch tensor of shape (,) representing the link loss
            ent_loss: A torch tensor of shape (,) representing the entropy loss
            diag_loss: A torch tensor of shape (,) representing the assignment loss
            proj (if return_proj): A PyTorch tensor of (batch_size, n_beads, n_atoms) representing the projection matrix P_x
            inv_proj (if return_proj): A PyTorch tensor of (batch_size, self.num_atoms_r, n_beads) representing the inverse projection matrix P_z
        """
        ###########POOL###########
        x, inputs = x_data  # inputs[-1] includes distances
        # Get projection matrices
        # (batch_size, self.num_atoms, self.num_beads)
        proj = nn.Softmax(dim=-2)(self.pool(x, *inputs[:-1]))
        if train_inv_proj_only:
            proj = proj.detach()
        # (batch_size, self.num_beads, self.num_atoms)
        proj_t = proj.transpose(-1, -2)

        # (batch_size, self.num_atoms, 3)
        x = x.reshape((batch_size, self.num_atoms, 3))

        # Get auxilary losses
        ppt = torch.matmul(proj, proj_t)
        distances = inputs[-1]
        diagonal = torch.diagonal(ppt, dim1=-1, dim2=-2)  # diag

        link_loss = torch.norm(distances * ppt, p="fro")  # linkloss
        ent_loss = ((-proj
                     * torch.log(proj + 1E-15))).sum(dim=-2).mean()  # entloss

        diag_loss = (diagonal[diagonal > 1] - 1).sum()
        if (link_loss < 10**(self.min_detach_loss)):
            link_loss = link_loss.detach()
        if (ent_loss < 10**(self.min_detach_loss)):
            ent_loss = ent_loss.detach()

        ###########DEPOOL###########
        # Get new positions and inputs
        bead_x = torch.bmm(proj_t, x)
        bead_x = (bead_x - bead_x.mean(-2).unsqueeze(-2))
        bead_x = bead_x.reshape((-1, 3))

        bead_inputs = self.bead_info.get_bead_info(batch_size=batch_size)
        # (batch_size, self.num_beads, self.num_atoms_r)
        inv_proj = self.depool(bead_x, *bead_inputs)
        # (batch_size, self.num_atoms_r, self.num_beads)
        inv_proj_t = inv_proj.transpose(-1, -2)
        if fix_inv_proj:
            inv_proj_t = torch.zeros_like(inv_proj_t)
            inv_proj_t[:, backbone_r_indices, torch.arange(backbone_r_indices.shape[0])] = 1

        # (batch_size, self.num_beads, 3)
        bead_x = bead_x.reshape((batch_size, self.num_beads, 3))
        # (batch_size, self.num_atoms_r, 3)
        mean_x = torch.bmm(inv_proj_t, bead_x)

        if return_proj:
            return mean_x, link_loss, ent_loss, diag_loss, proj_t, inv_proj_t
        else:
            return mean_x, link_loss, ent_loss, diag_loss