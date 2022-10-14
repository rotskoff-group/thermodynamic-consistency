import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from cggnn.cg import CG, CGModel, UModel, EGNNDataset, enn_collate_fn, SchNet, SchNetSimInfo
from cggnn.integrators import OMMOVRVO, OVRVO, SecondOrder, VRORV
from cggnn.omm import ProteinInfo
import matplotlib.pyplot as plt
import matplotlib as mpl
import mdtraj as md
import seaborn as sb
import warnings
import pandas as pd
from openmmtorch import TorchForce
import tensorboard as tb
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def tflog2pandas(path):
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data


network_folder_name = "/scratch/groups/rotskoff/cg/092522ChignolinTraining/"
save_folder_name = "./"
np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)
plt.style.use("shriram")

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

parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--num_data_points", type=int, default=50000)


parser.add_argument("--u_num_sub_epochs", type=int, default=10)
parser.add_argument("--cg_num_sub_epochs", type=int, default=10)
parser.add_argument("--freeze_cg_epoch", type=int, default=15)
parser.add_argument("--max_proj_ent", type=int, default=2)



parser.add_argument("--integrator_name", type=str, default="ovrvo", choices=["ovrvo", "second_order", "vrorv"])
parser.add_argument("--dt", type=float, default=0.001)
parser.add_argument("--friction", type=float, default=10.0)
parser.add_argument("--num_trajectories_per_basin", type=int, default=3)









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
is_mass_normalized = config.is_mass_normalized

batch_size = config.batch_size
num_data_points = config.num_data_points

u_num_sub_epochs = config.u_num_sub_epochs
cg_num_sub_epochs = config.cg_num_sub_epochs
freeze_cg_epoch = config.freeze_cg_epoch
max_proj_ent = config.max_proj_ent


integrator_name = config.integrator_name
dt = config.dt
friction = config.friction
num_trajectories_per_basin = config.num_trajectories_per_basin





p_info = ProteinInfo(protein_name=protein_name)

device = torch.device("cuda:0")
if protein_name == "chignolin":
    data_folder_name = "/scratch/groups/rotskoff/cg/CoarseGrainingDataset/ChignolinDataset/"
    # data_folder_name = "/home/shriramc/Documents/outputs/CoarseGrainingDataset/ChignolinDataset/"
    assert not precompute_node_distances
    temperature = 350

elif protein_name == "adp":
    data_folder_name = "/scratch/groups/rotskoff/cg/CoarseGrainingDataset/ADPSolventDataset/"
    # data_folder_name = "/home/shriramc/Documents/outputs/CoarseGrainingDataset/ADPSolventDataset/"
    temperature = 300
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

if protein_name == "chignolin":
    warnings.warn("Using only 25000 points")
    positions = positions[0:25000]
    forces = forces[0:25000]
    pe = pe[0:25000]


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

    val_dataset = EGNNDataset(pe, forces, positions, atomic_numbers,
                              all_edge_indices, all_edge_features,
                              all_original_distances)
else:
    val_dataset = EGNNDataset(pe, forces, positions, atomic_numbers,
                              all_edge_indices, all_edge_features, None)

val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, collate_fn=enn_collate_fn)

cg = CG(width=cg_width, num_beads=cg_num_beads,
        n_layers=cg_n_layers, num_atoms=p_info.num_atoms,
        num_atoms_r=len(p_info.r_indices)).to(device)

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
u = SchNet(n_atom_basis=u_width, n_filters=u_width,
           n_interaction_blocks=u_n_interaction_blocks,
           n_gaussians=u_n_gaussians, cutoff=u_cutoff,
           max_z=bead_atomic_numbers.max().item() + 1).to(device)

u_optimizer = Adam(u.parameters(), lr=u_lr * 1E-4)
u_scheduler = ReduceLROnPlateau(u_optimizer,
                                patience=5,
                                factor=0.8,
                                min_lr=1e-6)
use_metric_scheduler = True
u_model = UModel(u, u_optimizer, u_scheduler,
                 w_e=u_w_e, device=device)

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

all_filenames = os.listdir(network_folder_name + tag + "/")
filename = [filename for filename in all_filenames
            if filename.startswith("events")]
assert len(filename) == 1
filename = filename[0]
tb_df = tflog2pandas(network_folder_name + tag + "/")
tb_df_val_mse_f = tb_df[tb_df["metric"] == "val_u/MSEForces"]

burn_in = freeze_cg_epoch * u_num_sub_epochs
post_burn_in_indices = tb_df_val_mse_f["step"].values >= burn_in
min_index = tb_df_val_mse_f["value"].values[post_burn_in_indices].argmin()
min_sub_epoch = int(tb_df_val_mse_f["step"].values[post_burn_in_indices][min_index])
epoch_to_load = int(min_sub_epoch // u_num_sub_epochs)


cg_model.load_networks(folder_name=network_folder_name +
                       tag + "/", epoch=epoch_to_load)
u_model.load_networks(folder_name=network_folder_name +
                      tag + "/", epoch=epoch_to_load)
print("Loaded Epoch:", epoch_to_load)

try:
    os.mkdir(save_folder_name + tag)
except:
    pass

############GET PROJECTIONS######################
all_proj = []
all_init_x = []
all_pred_x = []
all_inv_proj = []
for (val_x, _, _, batch_size) in val_dataloader:
    proj, inv_proj, init_x, pred_x = cg_model.get_proj(val_x,
                                                       batch_size,
                                                       get_more_info=True)
    all_proj.append(proj.detach())
    all_inv_proj.append(inv_proj.detach())
    all_init_x.append(init_x.detach())
    all_pred_x.append(pred_x.detach())


all_proj = torch.cat(all_proj)
all_inv_proj = torch.cat(all_inv_proj)
all_init_x = torch.cat(all_init_x)
all_pred_x = torch.cat(all_pred_x)

###################PLOT RECONSTRUCTION LOSSES##################################

diff = all_pred_x - all_init_x[:, cg_model.r_indices, :]
diff = (diff - diff.mean(-2).unsqueeze(-2))
mean_rloss = (torch.linalg.norm(diff, axis=-1)).mean(0).cpu().numpy()
atom_names_r = [p_info.atom_names[i] for i in cg_model.r_indices.tolist()]
plt.rcParams["figure.figsize"] = (12, 9)
plt.bar(range(len(atom_names_r)), mean_rloss)  # , color=bar_colors)
plt.ylabel("Mean Reconstruction Loss (Å)")
plt.xlabel("Atom Name")
plt.xticks(range(len(atom_names_r)), atom_names_r)
plt.ylim(0, 1.0)
plt.savefig(save_folder_name + tag + "/r_loss.pdf")
plt.close()
######################## Plot Projection and Inverse Projection matrices#####################
mp = all_proj.mean(0)
torch.save(mp, save_folder_name + tag + "/mean_proj.pt")
vp = all_proj.var(0)
torch.save(vp, save_folder_name + tag + "/var_proj.pt")

inv_mp = all_inv_proj.mean(0)
torch.save(inv_mp, save_folder_name + tag + "/mean_inv_proj.pt")
inv_vp = all_inv_proj.var(0)
torch.save(inv_vp, save_folder_name + tag + "/var_inv_proj.pt")


plt.rcParams["figure.figsize"] = (12, 9)
sb.heatmap(all_proj.mean(0).cpu().numpy(), cmap="coolwarm",
           xticklabels=p_info.atom_names, square=True)
plt.title("Mean Proj")
plt.savefig(save_folder_name + tag + "/mean_proj.pdf")
plt.close()

plt.rcParams["figure.figsize"] = (12, 9)
sb.heatmap(vp.cpu().numpy(), cmap="coolwarm",
           xticklabels=p_info.atom_names, square=True)
plt.title("Var Proj}")
plt.savefig(save_folder_name + tag + "/var_proj.pdf")
plt.close()

plt.rcParams["figure.figsize"] = (12, 9)
plt.title("Mean Inv Proj")
sb.heatmap(all_inv_proj.mean(0).cpu().numpy(), cmap="coolwarm",
           yticklabels=atom_names_r, square=True)
plt.savefig(save_folder_name + tag + "/mean_inv_proj.pdf")
plt.close()

plt.rcParams["figure.figsize"] = (12, 9)
plt.title("Mean Inv Proj")
sb.heatmap(all_inv_proj.mean(0).cpu().numpy(), cmap="coolwarm",
           yticklabels=atom_names_r, square=True)
plt.savefig(save_folder_name + tag + "/var_inv_proj.pdf")
plt.close()


##########################Save Mass Distribution#############################
masses = torch.tensor(p_info.masses, device=device).float().T
bead_masses = all_proj @ masses
for i in range(bead_masses.shape[-2]):
    plt.hist(bead_masses[:, i, 0].cpu().numpy(), label=str(i))
plt.legend()
plt.savefig(save_folder_name + tag + "/mass_dist.pdf")
plt.close()


######################## Save Reconstructed Protein#############################
r_bond_edge_index = []
for edge_index in bond_edge_index.T:
    if (edge_index[0] in cg_model.r_indices and edge_index[1] in cg_model.r_indices):
        r_bond_edge_index.append(torch.cat([torch.where(edge_index[0] == cg_model.r_indices)[0],
                                            torch.where(edge_index[1] == cg_model.r_indices)[0]]))
r_bond_edge_index = torch.stack(r_bond_edge_index).T

topology = md.Topology()
topology.add_chain()
topology.add_residue(name="r_protein", chain=topology.chain(0))
atom_names_r = [cg_model.atom_names[i] for i in cg_model.r_indices]
for (index, atom_name) in enumerate(atom_names_r):
    topology.add_atom(atom_name, element=md.element.Element.getBySymbol(atom_name[0]),
                      residue=topology.residue(0),
                      serial=index + 1)

for bond in r_bond_edge_index.T:
    topology.add_bond(topology.atom(bond[0].item()),
                      topology.atom(bond[1].item()),
                      order=1)

rand_indices = torch.randperm(all_pred_x.shape[0])
traj = md.Trajectory(all_pred_x[rand_indices[0:100]].cpu().numpy(),
                     topology=topology)
traj.save_pdb(save_folder_name + tag + "/traj_rec.pdb")


all_init_x_r = all_init_x[:, cg_model.r_indices]
traj = md.Trajectory(all_init_x_r[rand_indices[0:100]].cpu().numpy(),
                     topology=topology)
traj.save_pdb(save_folder_name + tag + "/traj_r.pdb")


torch.save(all_pred_x, save_folder_name + tag + "/all_pred_x.pt")
torch.save(all_init_x, save_folder_name + tag + "/all_init_x.pt")

###################PLOT REACTION COORDINATES###################################
if protein_name == "adp":
    pred_rc = p_info.get_reaction_coordinates(all_pred_x.cpu().numpy())
    init_rc = p_info.get_reaction_coordinates(all_init_x_r.cpu().numpy())
elif protein_name == "chignolin":
    box_lengths = torch.load(data_folder_name + "box_lengths.pt")
    tica_eigenvectors = np.load(data_folder_name + "tica_eigenvectors.npy")
    tica_mean = np.load(data_folder_name + "tica_mean.npy")
    pred_rc = p_info.get_reaction_coordinates(all_pred_x.cpu().numpy(), box_lengths=box_lengths,
                                              mean=tica_mean, eigenvectors=tica_eigenvectors)
    init_rc = p_info.get_reaction_coordinates(all_init_x_r.cpu().numpy(), box_lengths=box_lengths,
                                              mean=tica_mean, eigenvectors=tica_eigenvectors)


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
pred_z, pred_rc1_edge, pred_rc2_edge = np.histogram2d(pred_rc[:, 0],
                                                      pred_rc[:, 1],
                                                      bins=100)
pred_rc1 = 0.5 * (pred_rc1_edge[:-1] + pred_rc1_edge[1:])
pred_rc2 = 0.5 * (pred_rc2_edge[:-1] + pred_rc2_edge[1:])
pred_z = pred_z.T
pred_z_density = pred_z/float(pred_z.sum())
pred_free_energy = np.inf * np.ones(shape=pred_z.shape)
nonzero = pred_z_density.nonzero()
pred_free_energy[nonzero] = -np.log(pred_z_density[nonzero])


init_z, init_rc1_edge, init_rc2_edge = np.histogram2d(init_rc[:, 0],
                                                      init_rc[:, 1],
                                                      bins=100)
init_rc1 = 0.5 * (init_rc1_edge[:-1] + init_rc1_edge[1:])
init_rc2 = 0.5 * (init_rc2_edge[:-1] + init_rc2_edge[1:])
init_z = init_z.T
init_z_density = init_z/float(init_z.sum())
init_free_energy = np.inf * np.ones(shape=init_z.shape)
nonzero = init_z_density.nonzero()
init_free_energy[nonzero] = -np.log(init_z_density[nonzero])


img_0 = axes[0].contourf(pred_rc1, pred_rc2, pred_free_energy, cmap="coolwarm")
img_1 = axes[1].contourf(init_rc1, init_rc2, init_free_energy)


axes[0].set_ylabel(p_info.rc1)
axes[0].set_xlabel(p_info.rc2)
axes[1].set_ylabel(p_info.rc1)
axes[1].set_xlabel(p_info.rc2)
fig.colorbar(img_0, ax=axes[0])
fig.colorbar(img_1, ax=axes[1])
axes[0].set_xlim(-p_info.xlim, p_info.xlim)
axes[0].set_ylim(-p_info.xlim, p_info.xlim)
axes[1].set_xlim(-p_info.xlim, p_info.xlim)
axes[1].set_ylim(-p_info.xlim, p_info.xlim)
axes[0].set_title("Predicted")
axes[1].set_title("Original")

plt.savefig(save_folder_name + tag + "/pred_rc.pdf")
plt.close()

######################START CG SIMULATION#########################################
schinfo = SchNetSimInfo(atomic_numbers=bead_atomic_numbers)
if integrator_name == "ovrvo":
    integrator = OMMOVRVO
else:
    raise ValueError("Unknown integrator passed")

class CGEnergy(nn.Module):
    def __init__(self, u, bead_info):
        super().__init__()
        self.u = u
        self.bead_info = bead_info
    def forward(self, x):
        x = x*10
        u_kcal = self.u(x, self.bead_info)
        u_kj = u_kcal * 4.184
        return u_kj

cg_energy = CGEnergy(u_model.u, schinfo.atomic_numbers)
module = torch.jit.script(cg_energy)

module_filename = tag + "/" + "friction_" + str(friction) + "_dt_" + str(dt)
module.save(module_filename + "model.pt")
torch_force = TorchForce(module_filename + "model.pt")

all_starting_basins = np.load(data_folder_name + "all_starting_basins.npy")
num_trajectories_per_basin = 3
num_basins = all_starting_basins.shape[0]

all_starting_points = []
for basin_num in range(num_basins):
    all_starting_points.extend(np.random.choice(all_starting_basins[basin_num, :], size=num_trajectories_per_basin, replace=False).tolist())

def compute_max_distance(x_i):
    distances = torch.linalg.norm((x_i.unsqueeze(-2)
                                   - x_i.unsqueeze(-3)), axis=-1)
    return distances.max()


def compute_min_distance(x_i):
    distances = torch.linalg.norm((x_i.unsqueeze(-2)
                                   - x_i.unsqueeze(-3)), axis=-1)
    distances = distances + torch.eye(distances.shape[0]).to(device) * 10000
    return distances.min()


init_bead_mass = (mp @ masses)
if is_mass_normalized:
    init_bead_mass = torch.ones_like(init_bead_mass).float()

save_freq = 100
steps = 2000000
num_data_points = steps//save_freq

# Convert to kg/mol
init_bead_mass = init_bead_mass/1000
kB = 8.314268278  # In units of J/(mol / K)
for (seed, i) in enumerate(all_starting_points):
    sim_tag = ("sim_" + str(integrator_name) + "_integrator_" + str(dt) + "_dt_" + str(friction) + "_friction_" + str(seed) + "_seed")
    sim_save_folder_name = save_folder_name + tag + "/" + sim_tag + "/"
    os.mkdir(sim_save_folder_name)

    init_pos = all_init_x[i]
    init_proj_matrix = all_proj[i]
    init_bead_pos = init_proj_matrix @ init_pos


    torch_force = TorchForce(module_filename + "model.pt")
    cg_integrator = integrator(init_pos=init_bead_pos, bead_masses=init_bead_mass, torch_force=torch_force,
                               temperature=temperature,
                               dt=dt, friction=friction, tag=sim_save_folder_name)
    cg_integrator.generate_long_trajectory(num_data_points=num_data_points, save_freq=save_freq, tag=sim_save_folder_name)
    del cg_integrator

    sim_bead_x = torch.tensor(np.load(sim_save_folder_name + "positions.npy"), device=device).float()
    sim_bead_v = torch.tensor(np.load(sim_save_folder_name + "velocities.npy"), device=device).float()
    sim_bead_f = torch.tensor(np.load(sim_save_folder_name + "forces.npy"), device=device).float()
    scale = torch.sqrt(kB * temperature / init_bead_mass)
    r_batch_size = 16
    num_r_batches = ((sim_bead_x.shape[0] + r_batch_size - 1)
                        // r_batch_size)
    all_reconstructed_x = []
    all_inv_proj = []
    for batch in range(num_r_batches):
        sim_bead_x_batch = sim_bead_x[r_batch_size *
                                        batch:r_batch_size * (batch + 1)]
        sim_r_x, inv_proj = cg_model.get_reconstructed_x(sim_bead_x_batch,
                                                            sim_bead_x_batch.shape[0])
        all_reconstructed_x.append(sim_r_x.detach())
        all_inv_proj.append(inv_proj.detach())

    all_inv_proj = torch.cat(all_inv_proj)
    plt.rcParams["figure.figsize"] = (12, 9)
    plt.title("Mean Inv Proj")
    sb.heatmap(all_inv_proj.mean(0).cpu().numpy(), cmap="coolwarm",
                yticklabels=atom_names_r, square=True)
    plt.savefig(sim_save_folder_name + "r_mean_inv_proj.pdf")
    plt.close()

    plt.rcParams["figure.figsize"] = (12, 9)
    plt.title("Mean Inv Proj")
    sb.heatmap(all_inv_proj.mean(0).cpu().numpy(), cmap="coolwarm",
                yticklabels=atom_names_r, square=True)
    plt.savefig(sim_save_folder_name + "r_var_inv_proj.pdf")
    plt.close()

    all_reconstructed_x = torch.cat(all_reconstructed_x)
    traj = md.Trajectory(all_reconstructed_x.cpu().numpy(),
                            topology=topology)
    traj.save_pdb(sim_save_folder_name + "traj_sim_r.pdb")
    torch.save(all_reconstructed_x, sim_save_folder_name + "all_reconstructed_x.pt")

    if protein_name == "adp":
        sim_r_angles = p_info.get_reaction_coordinates(all_reconstructed_x.cpu().numpy())
    elif protein_name == "chignolin":
        sim_r_angles = p_info.get_reaction_coordinates(all_reconstructed_x.cpu().numpy(), box_lengths=box_lengths,
                                                        mean=tica_mean, eigenvectors=tica_eigenvectors)

    plt.hist2d(sim_r_angles[:, 0], sim_r_angles[:, 1],
                bins=[100, 100], norm=mpl.colors.LogNorm(), cmap="coolwarm", density=True)
    plt.xlim(-p_info.xlim, p_info.xlim)
    plt.ylim(-p_info.xlim, p_info.xlim)
    plt.ylabel(p_info.rc1)
    plt.xlabel(p_info.rc2)
    plt.savefig(sim_save_folder_name + "sim_rc.pdf")
    plt.close()

    np.save(sim_save_folder_name + "sim_r_angles.npy", sim_r_angles)

    sim_max_distances = [compute_max_distance(x_i).item()
                            for x_i in sim_bead_x]
    sim_min_distances = [compute_min_distance(x_i).item()
                            for x_i in sim_bead_x]

    plt.plot(np.arange(0, steps, save_freq),
                sim_max_distances, label="max distance")
    plt.xlabel("Steps")
    plt.ylabel("Max Distance (Å)")
    plt.savefig(sim_save_folder_name + "sim_max_distance.pdf")
    plt.close()

    plt.plot(np.arange(0, steps, save_freq),
                sim_min_distances, label="min distance")
    plt.xlabel("Steps")
    plt.ylabel("Min Distance (Å)")
    # plt.ylim(0.93, 1.05)
    plt.savefig(sim_save_folder_name + "sim_min_distance.pdf")
    plt.close()

    plt.plot(np.arange(0, steps, save_freq),
                ((0.5 * init_bead_mass.unsqueeze(0) * ((sim_bead_v * 1E2) ** 2)).sum(-1).mean(-1).cpu().numpy())/(1.5 * kB), 
                label="Simulated")
    plt.plot(np.arange(0, steps, save_freq),
                [temperature] * np.arange(0, steps, save_freq).shape[0], label="True")
    plt.xlabel("Steps")
    plt.ylabel("Temperature (K))")
    plt.legend()
    plt.savefig(sim_save_folder_name + "sim_temperature.pdf")
    plt.close()
