import torch
import os
from torch.autograd import grad
import copy
torch.set_printoptions(sci_mode=False)


class UModel():
    def __init__(self, u, u_optimizer, u_scheduler, w_e=0.0, 
                 device=torch.device("cuda:0"),
                 init_u=None):
        self.u = u
        self.u_optimizer = u_optimizer
        self.u_scheduler = u_scheduler
        self.init_u = init_u
        self.w_e = w_e
        self.device = device

    def reset_u(self):
        """Re-Initializes self.u, self.u_optimizer and self.u_scheduler
        """
        if self.init_u is not None:
            self.u, self.u_optimizer, self.u_scheduler = self.init_u()

    def save_networks(self, folder_name="./"):
        """Saves cg and u networks
        """
        torch.save({"model_state_dict": self.u.state_dict(),
                    "optimizer_state_dict": self.u_optimizer.state_dict(),
                    "scheduler_state_dict": self.u_scheduler.state_dict()
                    }, folder_name + "u")

    def load_networks(self, folder_name="./", epoch=0):
        """Given folder name loads cg and u network
        Arguments:
            folder_name: folder name to load network from
            u_file_name: a file name for the potential energy network to be saved
        """
        if epoch is None:
            epoch = max([int(f)
                        for f in os.listdir(folder_name) if f.isnumeric()])
            if "u" not in os.listdir(folder_name + str(epoch) + "/"):
                epoch = epoch - 1
            print("CG Epoch Loaded:", epoch)
        u_checkpoint = torch.load(folder_name + str(epoch) + "/u",
                                  map_location=self.device)
        self.u.load_state_dict(u_checkpoint["model_state_dict"])
        self.u_optimizer.load_state_dict(u_checkpoint["optimizer_state_dict"])
        self.u_scheduler.load_state_dict(u_checkpoint["scheduler_state_dict"])

    def get_force(self, x_data, info=None):
        """Computes Coarse-Grained Force
        Arguments:
            x_data: A list of [x, inputs], where x is a PyTorch tensor of shape (batch_size, n_beads, 3) representing the coarse-grained positions
            info: A string ["verbose", "force_only" or None] specifying how much information to return 
        Returns:
            u_x: A PyTorch tensor of shape (batch_size, ) corresponding to the computed coarse-grained potential energy
            pred_force: A PyTorch tensor of shape (batch_size, n_beads, 4) corresponding to the computed coarse-grained forces
        """
        x, inputs = x_data
        x.requires_grad = True
        if info == "verbose":
            u_x, hidden_features = self.u(x, *inputs, verbose=True)
        else:
            u_x = self.u(x, *inputs)
        pred_force = -grad(u_x, x, create_graph=True,
                           grad_outputs=torch.ones_like(u_x))[0]
        if info == "force_only":
            return pred_force
        if info == "verbose":
            return u_x, pred_force, hidden_features
        else:
            return u_x, pred_force

    def get_loss(self, x_data, target_force, target_energy):
        """Computes force-matching loss (can use for coarse-graining or fine-graining training)
        Arguments:
            x_data: A list of [x, inputs], where x is a PyTorch tensor of shape (batch_size, n_beads, 3) representing the coarsse-grained positions
            target_force: A PyTorch tensor of shape (batch_size, n_beads, 3) representing the coarse-grained force
            target_energy: A PyTorch tensor of shape (batch_size, ) representing the coarse-grained energy. Input tensor of zeros if coarse-grained energy is not known (almost always the case)
        Returns:
            mse_force_loss: A PyTorch tensor of shape (,) representing the mean squared error between predicted force and target force
            mae_force_loss: A PyTorch tensor of shape (,) representing the mean absolute error between predicted force and target force
            mse_energy_loss: A PyTorch tensor of shape (,) representing the mean squared error between predicted force and target force
            mae_energy_loss: A PyTorch tensor of shape (,) representing the mean absolute error between predicted force and target force
            mse_total_loss: A PyTorch tensor of shape (,) representing the total mse loss weighted by self.w_e
            mae_total_loss: A PyTorch tensor of shape (,) representing the total mae loss weighted by self.w_e
        """
        pred_energy, pred_force = self.get_force(x_data)
        assert pred_force.shape[0] == pred_energy.shape[0] == target_force.shape[0] == target_energy.shape[0]
        mse_force_loss = ((pred_force - target_force)
                          ** 2).sum((-1, -2)).mean()
        mae_force_loss = ((pred_force - target_force).abs()
                          ).sum((-1, -2)).mean()

        mse_energy_loss = ((pred_energy - target_energy) ** 2).mean()
        mae_energy_loss = ((pred_energy - target_energy).abs()).mean()

        mse_total_loss = mse_force_loss + self.w_e * mse_energy_loss
        mae_total_loss = mae_force_loss + self.w_e * mae_energy_loss

        return mse_force_loss, mae_force_loss, mse_energy_loss, mae_energy_loss, mse_total_loss, mae_total_loss
