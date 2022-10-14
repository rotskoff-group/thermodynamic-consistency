import torch
from torch.distributions import MultivariateNormal
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import argparse

from cggnn.reconstruct import NFModel, NSFAR, NSFARC, train_forward_loss

device = torch.device("cuda:0")
parser = argparse.ArgumentParser()
parser.add_argument("--n_layers", type=int, default=2)
parser.add_argument("--width", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=int, default=3)
parser.add_argument("--num_epochs", type=int, default=100000)
parser.add_argument("--num_datapoints", type=int, default=50000)

parser.add_argument("--protein_name", type=str,
                    default="chignolin", choices=["chignolin", "adp"])

device = torch.device("cuda:0")

config = parser.parse_args()
n_layers = config.n_layers
width = config.width
batch_size = config.batch_size
num_epochs = config.num_epochs
lr = config.lr
num_datapoints = config.num_datapoints
protein_name = config.protein_name

save_folder_name = "./"
if protein_name == "chignolin":
    data_folder_name = "/scratch/groups/rotskoff/cg/CoarseGrainingDataset/ChignolinDataset/"
    # data_folder_name = "/home/shriramc/Documents/outputs/CoarseGrainingDataset/ChignolinDataset/"
    all_d_angles = np.concatenate((np.load(data_folder_name + "all_dihedral_angles_to_reconstruct.npy"),
                                   np.load(data_folder_name + "all_seed_dihedral_angles.npy")), axis=-1)
    in_dim = np.load(data_folder_name + "reconstruct_z_matrix.npy").shape[0]
    n_bb_ic = np.load(data_folder_name + "seed_z_matrix.npy").shape[0]
    flow = NSFARC(in_dim=in_dim, n_bb_ic=n_bb_ic,
                  n_layers=n_layers,
                  num_to_condition=32,
                  device=device).to(device)

    class BaseDistribuion:
        def __init__(self, n_ic):
            self.upper_base_dist = MultivariateNormal(loc=torch.zeros(n_ic).to(device),
                                                      covariance_matrix=torch.eye(n_ic).to(device))

        def sample(self, n_batches, seed_angle_samples=None):
            n_batches = n_batches[0]
            reconstruct_angle_samples = self.upper_base_dist.sample(
                (n_batches, ))
            return torch.cat((reconstruct_angle_samples, seed_angle_samples), axis=-1)

        def log_prob(self, batch_z):
            return self.upper_base_dist.log_prob(batch_z)

    base_dist = BaseDistribuion(n_ic=in_dim)


elif protein_name == "adp":
    data_folder_name = "/scratch/groups/rotskoff/cg/CoarseGrainingDataset/ADPSolventDataset/"
    # data_folder_name = "/home/shriramc/Documents/outputs/CoarseGrainingDataset/ADPSolventDataset/"
    all_d_angles = np.load(data_folder_name + "all_dihedral_angles_to_reconstruct.npy")
    in_dim = np.load(data_folder_name + "reconstruct_z_matrix.npy").shape[0]
    flow = NSFAR(in_dim=in_dim, n_layers=n_layers, device=device).to(device)
    base_dist = MultivariateNormal(loc=torch.zeros(in_dim).to(device),
                                   covariance_matrix=torch.eye(in_dim).to(device))
else:
    raise ValueError("Incorrect Protein Name Supplied")

rand_indices = np.random.permutation(all_d_angles.shape[0])
all_d_angles = torch.tensor(all_d_angles[rand_indices[0:num_datapoints]], device=device).float()

flow_opt = Adam(flow.parameters(), lr=lr * 1e-4)
flow_scheduler = ReduceLROnPlateau(flow_opt, patience=5,
                                   factor=0.8, min_lr=1e-6)


class DihedralDataset(Dataset):
    def __init__(self, all_d_angles):
        self.all_d_angles = all_d_angles

    def __len__(self):
        return self.all_d_angles.shape[0]

    def __getitem__(self, idx):
        return self.all_d_angles[idx]


dihedral_dataset = DihedralDataset(all_d_angles)
dihedral_dataloader = DataLoader(dihedral_dataset,
                                 batch_size=batch_size,
                                 shuffle=True)

tag = (str(protein_name) + "_" + str(n_layers) + "_n_layers_" + str(width) + "_width_"
       + str(batch_size) + "_bs_" + str(lr) + "_lr_"
       + str(num_datapoints) + "_num_datapoints")

nf_model = NFModel(base_dist=base_dist, flow=flow,
                   flow_optimizer=flow_opt, flow_scheduler=flow_scheduler,
                   folder_name=save_folder_name + tag + "/", device=device)

train_forward_loss(nf_model, ic_dataloader=dihedral_dataloader,
                   num_epochs=num_epochs, folder_name="./", tag=tag)