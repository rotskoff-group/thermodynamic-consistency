import torch
import os

class CGModel:
    def __init__(self, cg, cg_optimizer,  cg_scheduler,
                 r_indices, backbone_original_indices, atom_names, 
                 masses, num_atoms=22,
                 backbone_weight=10, is_mass_normalized=False,
                 device=torch.device("cuda:0")):
        """
        Args:
            r_indices: A List of indices of atoms to reconstruct
            backbone_orignial_indices: A List of indices of the backbone (C_alpha) atoms 
            backbone_weight: An int representing how much to favor backbone reconstruction
        """
        self.device = device
        self.cg = cg
        self.cg_optimizer = cg_optimizer
        self.cg_scheduler = cg_scheduler
        self.num_atoms = num_atoms
        self.masses = masses
        self.is_mass_normalized = is_mass_normalized
        self.atom_names = atom_names
        self.r_indices = torch.tensor(r_indices,
                                      device=self.device)
        backbone_original_indices = torch.tensor(backbone_original_indices,
                                                 device=device).long()
        backbone_r_indices = torch.cat([torch.where(boi_i == self.r_indices)[0]
                                        for boi_i in backbone_original_indices])
        self.r_weight = torch.ones_like(self.r_indices,
                                        device=self.device).float()
        self.r_weight[backbone_r_indices] = backbone_weight
        self.r_weight = self.r_weight.unsqueeze(0)
        self.cg.set_backbone_r_indices(backbone_r_indices)

    def save_networks(self, folder_name="./"):
        """Saves cg and u networks
        """
        torch.save({"model_state_dict": self.cg.state_dict(),
                    "optimizer_state_dict": self.cg_optimizer.state_dict(),
                    "scheduler_state_dict": self.cg_scheduler.state_dict()
                    }, folder_name + "cg")

    def load_networks(self, folder_name="./", epoch=None):
        if epoch is None:
            epoch = max([int(f) for f in os.listdir(folder_name) if f.isnumeric()])
            if "cg" not in os.listdir(folder_name + str(epoch) + "/"):
                epoch = epoch - 1

            print("CG Epoch Loaded:", epoch)
        cg_checkpoint = torch.load(folder_name + str(epoch) + "/cg",
                                   map_location=self.device)
        self.cg.load_state_dict(cg_checkpoint["model_state_dict"])
        self.cg_optimizer.load_state_dict(
            cg_checkpoint["optimizer_state_dict"])
        self.cg_scheduler.load_state_dict(
            cg_checkpoint["scheduler_state_dict"])

    def get_reconstruction_loss(self, pred_x, init_x):
        """
        Args:
            pred_x: A torch tensor of shape (n_batch, n_r_atoms, 3) representing the coordinates of the reconstructed atoms
            init_x: A torch tensor of shape (n_batch, n_atoms, 3) representing the coordinates of the original configuration
        """
        diff = pred_x - init_x[:, self.r_indices, :]
        diff = (diff - diff.mean(-2).unsqueeze(-2))
        r_loss = (torch.linalg.norm(diff, axis=-1) * self.r_weight).sum(-1).mean(0)
        # r_loss = (torch.linalg.norm((pred_x - init_x[:, self.r_indices, :]), axis=-1) * self.r_weight).sum(-1).mean(0)
        return r_loss

    def get_proj_forces_bead(self, x_data, x_forces, batch_size,
                             detach=False):
        """Given fine-grained configuration and fine-grained forces computes instantaneous coarse-grained force
        Arguments:
            x_data: A list of [x, inputs], where x is a PyTorch tensor of shape (batch_size*n_atoms, 3) representing the fine-grained positions
            x_forces: A PyTorch tensor of shape (batch_size, n_atoms, 3) representing the fine-grained forces
            batch_size: An int representing the batch size
            detach: A boolean specifying whether to destroy the computational graph (of gradient information)
        Returns:
            proj_forces: A PyTorch tensor of shape (batch_size, n_beads, 3)
            bead_x: A tuple (bead_position, [bead_info]), where bead_position is a tensor of shape (batch_size, n_beads, 3) 
                    and where bead_info is a tensor of shape (batch_size, n_features)
            ent_loss: A PyTorch tensor of shape (1, ) representing the entropy loss
        """
        pred_x, link_loss, ent_loss, diag_loss, proj, inv_proj = self.cg(x_data,
                                                                         batch_size,
                                                                         return_proj=True)
        x = x_data[0].reshape((batch_size, self.num_atoms, 3))
        bead_x = torch.bmm(proj, x)

        proj_forces = torch.bmm(torch.bmm(torch.linalg.inv(torch.bmm(proj,
                                                                     torch.transpose(proj, dim1=-1, dim0=-2))),
                                          proj),
                                x_forces)
        if (self.is_mass_normalized):
            proj_forces = proj_forces/(proj@self.masses)

        bead_x_input = self.cg.bead_info.get_bead_info_sch(batch_size)
        bead_x = (bead_x.detach(), [bead_x_input])  # Detach x always
        if detach:
            proj_forces = proj_forces.detach()

        return proj_forces, bead_x, ent_loss.detach()
    
    def get_reconstructed_x(self, bead_x, batch_size):
        """Computes the reconstructed target atoms given a coarse-grained configuration
        Arguments:
            bead_x: A PyTorch tensor of shape (batch_size, n_beads, 3) corresponding to bead positions
            batch_size: An int representing the batch size
        Returns:
            mean_x: A PyTorch tensor of shape (batch_size, self.r_indices.shape[0], 3) corresponding 
            to the positions of the reconstructed atoms
            inv_proj_t: A PyTorch tensor of shape (batch_size, self.r_indices.shape[0], n_beads) corresponding to the inverse 
            projection matrices (P_z) used
        """
        assert bead_x.shape[0] == batch_size
        bead_x = (bead_x - bead_x.mean(-2).unsqueeze(-2))
        bead_x = bead_x.reshape((-1, 3))
        bead_inputs = self.cg.bead_info.get_bead_info(batch_size=batch_size)
        # (batch_size, self.num_beads, self.num_atoms_r)
        inv_proj = self.cg.depool(bead_x, *bead_inputs)
        # (batch_size, self.num_atoms_r, self.num_beads)
        inv_proj_t = inv_proj.transpose(-1, -2)
        # (batch_size, self.num_beads, 3)
        bead_x = bead_x.reshape((batch_size, self.cg.num_beads, 3))
        # (batch_size, self.num_atoms_r, 3)
        mean_x = torch.bmm(inv_proj_t, bead_x)
        return mean_x, inv_proj_t


    def get_proj(self, x_data, batch_size, get_more_info=False):
        """Gets the projection matrices
        Arguments:
            x_data: A list of [x, inputs], where x is a PyTorch tensor of shape (batch_size*n_atoms, 3) representing the fine-grained positions
            batch_size: An int representing the batch size
            get_more_info: A boolean specifying whether to return more info
        Returns:
            proj: A PyTorch tensor of (batch_size, n_beads, n_atoms) representing the projection matrix P_x
            inv_proj: A PyTorch tensor of (batch_size, self.r_indices.shape[0], n_beads) representing the inverse projection matrix P_z
            init_x(if get_more_info): A PyTorch tensor of shape (batch_size, n_atoms, 3) representing the fine-grained positions
            pred_x(if get_more_info):A PyTorch tensor of shape (batch_size, self.r_indices.shape[0], 3) representing the reconstructed positions
        """
        pred_x, _, _, _, proj, inv_proj = self.cg(x_data,
                                                  batch_size,
                                                  train_inv_proj_only=False,
                                                  return_proj=True)
        init_x = x_data[0].reshape((batch_size, self.num_atoms, 3))

        if get_more_info:
            return proj, inv_proj, init_x.detach(), pred_x.detach()
        else:
            return proj, inv_proj

    def get_loss(self, x_data, x_forces, u_model,
                 batch_size, fix_inv_proj=False, add_mean_force_reg=None):
        """Computes reconstruction loss and a suite of auxiliary losses
        Arguments:
            x_data: A list of [x, inputs], where x is a PyTorch tensor of shape (batch_size*n_atoms, 3) representing the fine-grained positions
            x_forces: A PyTorch tensor of shape (batch_size, n_atoms, 3) representing the fine-grained forces
            u_model: A UModel object corresponding to the coarse-grained potential energy function
            batch_size: An int representing the batch size
            fix_inv_proj: A boolean specifiying whether to fix inverse projection
            add_mean_force_reg: A boolean specifiying whether to add a mean_force loss
        Returns:
            r_loss: A torch tensor of shape (,) representing the reconstructing loss
            mse_force_loss: A torch tensor of shape (,) representing the mean_forceloss
            link_loss: A torch tensor of shape (,) representing the link loss
            ent_loss: A torch tensor of shape (,) representing the entropy loss
            diag_loss: A torch tensor of shape (,) representing the assignment loss
        """
        pred_x, link_loss, ent_loss, diag_loss, proj, inv_proj = self.cg(x_data,
                                                                         batch_size,
                                                                         fix_inv_proj=fix_inv_proj,
                                                                         return_proj=True,
                                                                         backbone_r_indices=self.backbone_r_indices)
        init_x = x_data[0].reshape((batch_size, self.num_atoms, 3))
        if add_mean_force_reg:
            prefactor = torch.linalg.inv(torch.bmm(proj, torch.transpose(proj, dim1=-1, dim0=-2))).detach()
            proj_forces = torch.bmm(torch.bmm(prefactor, proj), x_forces)
            if (self.is_mass_normalized):
                proj_forces = proj_forces/(proj@self.masses)

            x = x_data[0].reshape((batch_size, self.num_atoms, 3))
            bead_x = torch.bmm(proj, x)
            bead_x_input = self.cg.bead_info.get_bead_info_sch(batch_size)
            bead_x = (bead_x.detach(), [bead_x_input])  # Detach x always
            _, pred_forces = u_model.get_force(bead_x)
            pred_forces = pred_forces.detach()
            mse_force_loss = ((proj_forces - pred_forces)
                              ** 2).sum((-1, -2)).mean()
        else:
            mse_force_loss = torch.zeros((), device=self.device)
        r_loss = self.get_reconstruction_loss(pred_x, init_x)
        return r_loss, mse_force_loss, link_loss, ent_loss, diag_loss
