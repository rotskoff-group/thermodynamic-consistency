import argparse
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from cggnn.cg import CG, CGModel, UModel, EGNNDataset, enn_collate_fn, SchNet, train_cg_u
from cggnn.omm import ProteinInfo
np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)


parser = argparse.ArgumentParser()
parser.add_argument("--protein_name", type=str,
                    default="chignolin", choices=["chignolin", "adp"])
parser.add_argument("--precompute_node_distances",
                    action='store_true', default=False)


parser.add_argument("--u_n_interaction_blocks", type=int, default=2)
parser.add_argument("--u_width", type=int, default=128)
parser.add_argument("--u_n_gaussians", type=int, default=25)
parser.add_argument("--u_cutoff", type=int, default=10)
parser.add_argument("--u_lr", type=float, default=3.0)
parser.add_argument("--is_mass_normalized", action='store_true', default=False)
parser.add_argument("--u_w_e", type=float, default=0.0)

parser.add_argument("--cg_num_beads", type=int, default=10)
parser.add_argument("--cg_cutoff", type=float, default=5)
parser.add_argument("--cg_width", type=int, default=128)
parser.add_argument("--cg_n_layers", type=int, default=2)
parser.add_argument("--cg_lr", type=float, default=1.0)
parser.add_argument("--cg_reg_aux", type=float, default=0.1)
parser.add_argument("--cg_reg_force", type=float, default=0.001)
parser.add_argument("--cg_backbone_weight", type=float, default=1.)
parser.add_argument("--add_mean_force_reg", action='store_true', default=False)

parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--num_data_points", type=int, default=50000)


parser.add_argument("--u_num_sub_epochs", type=int, default=10)
parser.add_argument("--cg_num_sub_epochs", type=int, default=10)
parser.add_argument("--freeze_cg_epoch", type=int, default=15)
parser.add_argument("--max_proj_ent", type=int, default=2)

device = torch.device("cuda:0")


config = parser.parse_args()
protein_name = config.protein_name
precompute_node_distances = config.precompute_node_distances


u_n_interaction_blocks = config.u_n_interaction_blocks
u_width = config.u_width
u_n_gaussians = config.u_n_gaussians
u_cutoff = config.u_cutoff
u_lr = config.u_lr
u_w_e = config.u_w_e

cg_num_beads = config.cg_num_beads
cg_cutoff = config.cg_cutoff
cg_width = config.cg_width
cg_n_layers = config.cg_n_layers
cg_lr = config.cg_lr
cg_reg_aux = config.cg_reg_aux
cg_reg_force = config.cg_reg_force
cg_backbone_weight = config.cg_backbone_weight
add_mean_force_reg = config.add_mean_force_reg
is_mass_normalized = config.is_mass_normalized

batch_size = config.batch_size
num_data_points = config.num_data_points

u_num_sub_epochs = config.u_num_sub_epochs
cg_num_sub_epochs = config.cg_num_sub_epochs
freeze_cg_epoch = config.freeze_cg_epoch
max_proj_ent = config.max_proj_ent

p_info = ProteinInfo(protein_name=protein_name)


if protein_name == "chignolin":
    data_folder_name = "/scratch/groups/rotskoff/cg/CoarseGrainingDataset/ChignolinDataset/"
    # data_folder_name = "/home/shriramc/Documents/outputs/CoarseGrainingDataset/ChignolinDataset/"
    assert not precompute_node_distances
    min_detach_loss = -5
    detach_link_loss=5
    fix_inv_proj_epoch=5
elif protein_name == "adp":
    data_folder_name = "/scratch/groups/rotskoff/cg/CoarseGrainingDataset/ADPSolventDataset/"
    # data_folder_name = "/home/shriramc/Documents/outputs/CoarseGrainingDataset/ADPSolventDataset/"
    min_detach_loss = -3
    detach_link_loss=10000
    fix_inv_proj_epoch=5

    assert precompute_node_distances
else:
    raise ValueError("Incorrect Protein Name Supplied")

positions = torch.tensor(np.load(data_folder_name + "positions_" +
                         str(num_data_points) + ".npy"), device=device).float()
masses = torch.tensor(p_info.masses, device=device).unsqueeze(-1)
center_of_mass = ((masses * positions).sum(dim=1) / masses.sum(1))
center_of_mass = center_of_mass.unsqueeze(1)
positions = positions - center_of_mass
forces = torch.tensor(np.load(data_folder_name + "forces_" +
                              str(num_data_points) + ".npy"), device=device).float()
pe = torch.tensor(np.load(data_folder_name + "pe_" + str(num_data_points) + ".npy"),
                  device=device).float()

num_train = int(num_data_points * 0.8)
rand_indices = torch.randperm(num_data_points)

positions = positions[rand_indices]
forces = forces[rand_indices]
pe = pe[rand_indices]

atomic_numbers = torch.tensor(p_info.atomic_numbers, device=device)
_, atomic_numbers = torch.unique(atomic_numbers, return_inverse=True)
self_index = torch.stack([torch.tensor([i, i])
                         for i in range(p_info.num_atoms)]).to(device).T

bond_edge_index = torch.tensor(p_info.bond_edge_index, device=device).long().T
bond_edge_index2 = torch.stack((bond_edge_index[1, :], bond_edge_index[0, :]))
bond_edge_index = torch.cat((bond_edge_index, bond_edge_index2), dim=-1)


temp_batch_size = 1000
num_batches = (num_data_points + temp_batch_size - 1) // temp_batch_size
all_nonbonded_edge_index = []
for batch_num in range(num_batches):
    batch_positions = positions[batch_num *
                                temp_batch_size: (batch_num + 1) * temp_batch_size]
    batch_distances = torch.linalg.norm((batch_positions.unsqueeze(-2)
                                        - batch_positions.unsqueeze(-3)), axis=-1)
    nonbonded_edge_index = [torch.stack(torch.where((0.0 < distance)
                                                    & (distance < cg_cutoff))) for distance in batch_distances]
    all_nonbonded_edge_index.extend(nonbonded_edge_index)


all_edge_indices = []
all_edge_features = []
for nonbonded_edge_index in all_nonbonded_edge_index:
    edge_index = torch.cat([self_index,
                            bond_edge_index,
                            nonbonded_edge_index], dim=-1)
    all_edge_indices.append(edge_index)
    edge_feature = torch.tensor([0] * self_index.shape[-1] +
                                [1] * bond_edge_index.shape[-1] +
                                [2] * nonbonded_edge_index.shape[-1], device=device)
    all_edge_features.append(edge_feature)

for edge_index, edge_feature in zip(all_edge_indices, all_edge_features):
    assert edge_feature.shape[0] == edge_index.shape[1]

if precompute_node_distances:
    all_original_distances = torch.linalg.norm((positions.unsqueeze(-2)
                                                - positions.unsqueeze(-3)), axis=-1)

    train_dataset = EGNNDataset(pe[0:num_train], forces[0:num_train], positions[0:num_train], atomic_numbers,
                                all_edge_indices[0:num_train], all_edge_features[0:num_train], all_original_distances[0:num_train])
    val_dataset = EGNNDataset(pe[num_train:num_data_points], forces[num_train:num_data_points], positions[num_train:num_data_points], atomic_numbers,
                              all_edge_indices[num_train:num_data_points], all_edge_features[num_train:num_data_points], all_original_distances[num_train:num_data_points])
else:
    train_dataset = EGNNDataset(pe[0:num_train], forces[0:num_train], positions[0:num_train], atomic_numbers,
                                all_edge_indices[0:num_train], all_edge_features[0:num_train], None)
    val_dataset = EGNNDataset(pe[num_train:num_data_points], forces[num_train:num_data_points], positions[num_train:num_data_points], atomic_numbers,
                              all_edge_indices[num_train:num_data_points], all_edge_features[num_train:num_data_points], None)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, collate_fn=enn_collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, collate_fn=enn_collate_fn)

cg = CG(width=cg_width, num_beads=cg_num_beads,
        n_layers=cg_n_layers, num_atoms = p_info.num_atoms,
        num_atoms_r=len(p_info.r_indices), min_detach_loss=min_detach_loss).to(device)

cg_optimizer = Adam(cg.parameters(), lr=cg_lr * 1E-4)
cg_scheduler = ReduceLROnPlateau(
    cg_optimizer, patience=5, factor=0.8, min_lr=1e-6)
use_metric_scheduler = True


cg_model = CGModel(cg, cg_optimizer, cg_scheduler,
                   atom_names=p_info.atom_names,
                   r_indices=p_info.r_indices,
                   backbone_original_indices=p_info.backbone_original_indices,
                   masses=masses,
                   num_atoms=p_info.num_atoms,
                   backbone_weight=cg_backbone_weight,
                   is_mass_normalized=is_mass_normalized, device=device)


###############SET UP U INPUTS##########################
bead_atomic_numbers = torch.arange(cg_num_beads).to(device)

def init_u():
    u = SchNet(n_atom_basis=u_width, n_filters=u_width,
            n_interaction_blocks=u_n_interaction_blocks,
            n_gaussians=u_n_gaussians, cutoff=u_cutoff,
            max_z=bead_atomic_numbers.max() + 1).to(device)

    u_optimizer = Adam(u.parameters(), lr=u_lr * 1E-4)
    u_scheduler = ReduceLROnPlateau(u_optimizer,
                                    patience=5,
                                    factor=0.8,
                                    min_lr=1e-6)
    
    return u, u_optimizer, u_scheduler

use_metric_scheduler = True

u, u_optimizer, u_scheduler = init_u()

u_model = UModel(u, u_optimizer, u_scheduler,
                 w_e=u_w_e, device=device, init_u=init_u)


tag = (str(protein_name) + "_" + str(cg_num_beads) + "_cg_num_beads_" +
       str(cg_n_layers) + "_cg_n_layers_" +
       str(cg_width) + "_cg_width_" + str(cg_cutoff) +
       "_cg_cutoff_" + str(cg_lr) +
       "_cg_lr_" + str(cg_reg_aux) + "_cg_reg_aux_" +
       str(cg_reg_force) + "_cg_reg_force_" +
       str(cg_num_sub_epochs) + "_cg_num_sub_epochs_" +
       str(u_num_sub_epochs) + "_u_num_sub_epochs_" +
       str(cg_backbone_weight) + "_cg_backbone_weight_" + "pmf_sch_" +
       str(u_n_interaction_blocks) + "_u_block_" + str(u_width) + "_u_width_" +
       str(u_n_gaussians) + "_u_gaussians_" + str(u_cutoff) +
       "_u_cutoff_" + str(u_lr) +
       "_u_lr_" + str(batch_size) + "_bs")
folder_name = "./"
train_cg_u(train_dataloader, val_dataloader, cg_model,
           u_model, max_proj_ent=max_proj_ent,
           cg_num_sub_epochs=cg_num_sub_epochs,
           u_num_sub_epochs=u_num_sub_epochs,
           folder_name="./", tag=tag, num_epochs=1000,
           reg=[cg_reg_aux, cg_reg_force], freeze_cg_epoch=freeze_cg_epoch,
           add_mean_force_reg=add_mean_force_reg,
           save_epochs=1, detach_link_loss=detach_link_loss, fix_inv_proj_epoch=fix_inv_proj_epoch)