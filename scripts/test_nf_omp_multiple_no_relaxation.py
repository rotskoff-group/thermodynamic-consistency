from mpi4py import MPI
import matplotlib as mpl
from cggnn.omm import ADPImplicit, ChignolinImplicit, ProteinInfo
from cggnn.reconstruct import NFModel, NSFAR, NSFARC, Reconstructor, ChainBuffer, Chain
import os
import torch
from torch.distributions import MultivariateNormal
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import argparse
import matplotlib.pyplot as plt
from openmm.unit import *
import mdtraj as md
plt.style.use("shriram")

comm = MPI.COMM_WORLD
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()

device = torch.device("cuda:0")
parser = argparse.ArgumentParser()
parser.add_argument("--n_layers", type=int, default=2)
parser.add_argument("--width", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=int, default=3)
parser.add_argument("--num_epochs", type=int, default=100000)
parser.add_argument("--num_datapoints", type=int, default=50000)

parser.add_argument("--cg_num_beads", type=int, default=10)
parser.add_argument("--cg_cutoff", type=float, default=5)
parser.add_argument("--cg_width", type=int, default=128)
parser.add_argument("--cg_n_layers", type=int, default=2)
parser.add_argument("--cg_lr", type=float, default=1.0)
parser.add_argument("--cg_reg_aux", type=float, default=0.1)
parser.add_argument("--cg_reg_force", type=float, default=0.001)
parser.add_argument("--cg_backbone_weight", type=float, default=1.)

parser.add_argument("--u_n_interaction_blocks", type=int, default=2)
parser.add_argument("--u_width", type=int, default=128)
parser.add_argument("--u_n_gaussians", type=int, default=25)
parser.add_argument("--u_cutoff", type=int, default=10)
parser.add_argument("--u_lr", type=float, default=3.0)

parser.add_argument("--u_num_sub_epochs", type=int, default=10)
parser.add_argument("--cg_num_sub_epochs", type=int, default=10)
parser.add_argument("--cg_batch_size", type=int, default=8)

parser.add_argument("--integrator_name", type=str,
                    default="ovrvo", choices=["ovrvo"])
parser.add_argument("--dt", type=float, default=0.0002)
parser.add_argument("--friction", type=float, default=10.0)

parser.add_argument("--num_data_to_gen", type=int, default=200)
parser.add_argument("--num_relax_steps", type=int, default=2)

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

cg_num_beads = config.cg_num_beads
cg_cutoff = config.cg_cutoff
cg_width = config.cg_width
cg_n_layers = config.cg_n_layers
cg_lr = config.cg_lr
cg_reg_aux = config.cg_reg_aux
cg_reg_force = config.cg_reg_force
cg_backbone_weight = config.cg_backbone_weight


u_n_interaction_blocks = config.u_n_interaction_blocks
u_width = config.u_width
u_n_gaussians = config.u_n_gaussians
u_cutoff = config.u_cutoff
u_lr = config.u_lr

u_num_sub_epochs = config.u_num_sub_epochs
cg_num_sub_epochs = config.cg_num_sub_epochs
cg_batch_size = config.cg_batch_size

integrator_name = config.integrator_name
dt = config.dt
friction = config.friction

num_data_to_gen = config.num_data_to_gen
num_relax_steps = config.num_relax_steps

save_folder_name = "./"
network_folder_name = "/scratch/groups/rotskoff/cg/091722NFTrain/"

sim_tag = (str(protein_name) + "_" + str(cg_num_beads) + "_cg_num_beads_" +
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
           "_u_lr_" + str(cg_batch_size) + "_bs")

sim_tag_2 = ("sim_" + str(integrator_name) + "_integrator_" + str(dt)
             + "_dt_" + str(friction) + "_friction")

sim_tag_3 = "num_relax_steps_" + str(num_relax_steps)

p_info = ProteinInfo(protein_name=protein_name)
if protein_name == "chignolin":
    data_folder_name = "/scratch/groups/rotskoff/cg/CoarseGrainingDataset/ChignolinDataset/"
    # data_folder_name = "/home/shriramc/Documents/outputs/CoarseGrainingDataset/ChignolinDataset/"

    sim_folder_name = "/scratch/groups/rotskoff/cg/092622ChignolinAnalysis/"

    all_d_angles = np.concatenate((np.load(data_folder_name + "all_dihedral_angles_to_reconstruct.npy"),
                                   np.load(data_folder_name + "all_seed_dihedral_angles.npy")), axis=-1)
    in_dim = np.load(data_folder_name + "reconstruct_z_matrix.npy").shape[0]
    n_bb_ic = np.load(data_folder_name + "seed_z_matrix.npy").shape[0]
    seed_z_matrix_reindexed = np.load(
        data_folder_name + "seed_z_matrix_reindexed.npy")
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
    omm_int = ChignolinImplicit(
        dt=0.00002, friction=10.0, integrator_to_use="overdamped", tag=str(np.random.random()) + str(rank))
    chain_batch_num = num_relax_steps
elif protein_name == "adp":
    data_folder_name = "/scratch/groups/rotskoff/cg/CoarseGrainingDataset/ADPSolventDataset/"
    # data_folder_name = "/home/shriramc/Documents/outputs/CoarseGrainingDataset/ADPSolventDataset/"

    sim_folder_name = "/scratch/groups/rotskoff/cg/091922ADPAnalysis/"

    all_d_angles = np.load(
        data_folder_name + "all_dihedral_angles_to_reconstruct.npy")
    in_dim = np.load(data_folder_name + "reconstruct_z_matrix.npy").shape[0]
    flow = NSFAR(in_dim=in_dim, n_layers=n_layers, device=device).to(device)
    base_dist = MultivariateNormal(loc=torch.zeros(in_dim).to(device),
                                   covariance_matrix=torch.eye(in_dim).to(device))
    omm_int = ADPImplicit(dt=0.00002, tag=str(np.random.random()) +
                          str(rank), friction=10.0, integrator_to_use="overdamped")
    chain_batch_num = 1000
else:
    raise ValueError("Incorrect Protein Name Supplied")

flow_opt = Adam(flow.parameters(), lr=lr * 1e-4)
flow_scheduler = ReduceLROnPlateau(flow_opt, patience=5,
                                   factor=0.8, min_lr=1e-6)


tag = (str(protein_name) + "_" + str(n_layers) + "_n_layers_" + str(width) + "_width_"
       + str(batch_size) + "_bs_" + str(lr) + "_lr_"
       + str(num_datapoints) + "_num_datapoints")

nf_model = NFModel(base_dist=base_dist, flow=flow,
                   flow_optimizer=flow_opt, flow_scheduler=flow_scheduler,
                   folder_name=network_folder_name + tag + "/", device=device)
nf_model.load()
##########GENERATE DIHEDRAL DISTRIBUTION (Only do Once)#######################
if rank == 0:
    try:
        os.mkdir(save_folder_name + tag + "/")

        n_samples = 1000

        if protein_name == "adp":
            c_info = None
        elif protein_name == "chignolin":
            sample_indices = np.random.choice(all_d_angles.shape[0], 1000)
            c_info = torch.tensor(all_d_angles[sample_indices, in_dim:],
                                  device=device).float()
        samples, _, _ = nf_model.sample(n_samples=1000, c_info=c_info)
        gen_dihedrals_folder_name = save_folder_name + tag + "/gen_dihedrals/"
        os.mkdir(gen_dihedrals_folder_name)
        for i in range(in_dim):
            plt.title(i)
            plt.hist(all_d_angles[:, i], bins=np.arange(-np.pi *
                                                        1.25, np.pi*1.25, 0.1), density=True, label="Data")
            plt.hist(samples[:, i].cpu().numpy(), bins=np.arange(-np.pi*1.25,
                                                                 np.pi*1.25, 0.1), density=True, label="Generated", alpha=0.5)
            plt.savefig(gen_dihedrals_folder_name + str(i) + ".pdf")
            plt.close()
    except FileExistsError:
        pass

comm.Barrier()

###############GENERATE CONFIGURATIONS##########################
config_save_folder_name = save_folder_name + tag + "/" + sim_tag + "/"
if rank == 0:
    try:
        os.mkdir(config_save_folder_name)
    except FileExistsError:
        pass


config_save_folder_name += sim_tag_2 + "/"
if rank == 0:
    try:
        os.mkdir(config_save_folder_name)
    except FileExistsError:
        pass

config_save_folder_name += sim_tag_3 + "/"
if rank == 0:
    os.mkdir(config_save_folder_name)

comm.Barrier()


all_distances_to_reconstruct = np.load(
    data_folder_name + "all_distances_to_reconstruct.npy")
all_bond_angles_to_reconstruct = np.load(
    data_folder_name + "all_bond_angles_to_reconstruct.npy")
z_matrix = np.load(data_folder_name + "reconstruct_z_matrix.npy")
fixed_atoms = np.load(data_folder_name + "fixed_atoms.npy")
original_pe = np.load(data_folder_name + "pe_50000_implicit.npy")


all_starting_basins = np.load(data_folder_name + "all_starting_basins.npy")
num_basins = all_starting_basins.shape[0]
num_trajectories_per_basin = 3
all_sim_x = []
all_seeds = [
'sim_ovrvo_integrator_0.0001_dt_10.0_friction_17_seed',
 'sim_ovrvo_integrator_0.001_dt_100.0_friction_0_seed',
 'sim_ovrvo_integrator_0.001_dt_100.0_friction_7_seed',
 'sim_ovrvo_integrator_0.001_dt_100.0_friction_3_seed',
 'sim_ovrvo_integrator_0.001_dt_100.0_friction_1_seed',
 'sim_ovrvo_integrator_0.001_dt_100.0_friction_14_seed',
 'sim_ovrvo_integrator_0.001_dt_100.0_friction_10_seed',
 'sim_ovrvo_integrator_0.001_dt_100.0_friction_12_seed',
 'sim_ovrvo_integrator_0.0001_dt_100.0_friction_20_seed',
 'sim_ovrvo_integrator_0.001_dt_100.0_friction_8_seed',
 'sim_ovrvo_integrator_0.001_dt_100.0_friction_6_seed',
 'sim_ovrvo_integrator_0.001_dt_100.0_friction_2_seed',
 'sim_ovrvo_integrator_0.0001_dt_100.0_friction_21_seed',
 'sim_ovrvo_integrator_0.0001_dt_10.0_friction_15_seed',
 'sim_ovrvo_integrator_0.001_dt_100.0_friction_5_seed',
 'sim_ovrvo_integrator_0.001_dt_100.0_friction_9_seed',
 'sim_ovrvo_integrator_0.001_dt_100.0_friction_4_seed',
 'sim_ovrvo_integrator_0.0001_dt_10.0_friction_19_seed',
 'sim_ovrvo_integrator_0.0001_dt_10.0_friction_16_seed',
 'sim_ovrvo_integrator_0.001_dt_100.0_friction_11_seed',
 'sim_ovrvo_integrator_0.0001_dt_10.0_friction_18_seed',
 'sim_ovrvo_integrator_0.001_dt_100.0_friction_13_seed',
 'sim_ovrvo_integrator_0.0001_dt_100.0_friction_23_seed',
 'sim_ovrvo_integrator_0.0001_dt_100.0_friction_22_seed']

for seed in all_seeds:
    sim_x = torch.load("./data/" + seed + "/all_reconstructed_x.pt", map_location="cpu").numpy()
    all_sim_x.append(sim_x)

all_sim_x = np.concatenate(all_sim_x)



bond_distances = np.expand_dims(np.median(all_distances_to_reconstruct, 0), 0)
bond_angles = np.expand_dims(np.median(all_bond_angles_to_reconstruct, 0), 0)
r = Reconstructor(nf_model, bond_distances, bond_angles,
                  fixed_atoms, p_info.num_atoms, z_matrix)


all_scales = []
for i in range(p_info.num_atoms):
    scale = 1/(omm_int.simulation.system.getParticleMass(i) * omm_int.beta)
    all_scales.append(sqrt(scale).in_units_of(angstroms/picosecond)._value)
all_scales = torch.tensor(all_scales).to(device)
all_scales = all_scales.repeat_interleave(3)
velocity_dist = MultivariateNormal(loc=torch.zeros(p_info.num_atoms*3).to(device),
                                   covariance_matrix=torch.diag(all_scales**2).to(device))


num_data_to_gen_per_rank = num_data_to_gen // size
indices = range(rank * num_data_to_gen_per_rank,
                (rank + 1) * num_data_to_gen_per_rank)
sim_indices = np.arange(0,
                        all_sim_x.shape[0],
                        all_sim_x.shape[0]//num_data_to_gen)

if protein_name == "chignolin":
    pred_traj = md.Trajectory(all_sim_x[sim_indices], topology=None)
    seed_dihedral_angles_init = md.compute_dihedrals(pred_traj,
                                                     seed_z_matrix_reindexed)
    seed_dihedral_angles_init = torch.tensor(
        seed_dihedral_angles_init).to(device)


for index in indices:
    chain_buffer = ChainBuffer(100000)
    seed_positions = torch.tensor(
        all_sim_x[sim_indices[index]]).float().unsqueeze(0)
    if protein_name == "chignolin":
        seed_angles = seed_dihedral_angles_init[index].float().unsqueeze(0)
    elif protein_name == "adp":
        seed_angles = None

    c = Chain(r, seed_angles=seed_angles, seed_positions=seed_positions,
              omm_int=omm_int, velocity_dist=velocity_dist,
              chain_buffer=chain_buffer, num_relax_steps=num_relax_steps, skip_metropolize=True)
    with torch.no_grad():
        all_init_pos, all_init_pe = c.generate_batch_wo_relax(num_to_gen=chain_batch_num, 
                                                              batch_size=min(chain_batch_num, 1000))
    np.save(config_save_folder_name + str(index) + "_all_init_x.npy", all_init_pos)
    np.save(config_save_folder_name + str(index) + "_all_init_pe.npy", all_init_pe)

comm.Barrier()

if rank == 0:
    all_pe = []
    all_traj_x = []
    for i in range(num_data_to_gen):
        pe = np.load(config_save_folder_name + str(i) + "_all_init_pe.npy")
        traj_x = np.load(config_save_folder_name + str(i) + "_all_init_x.npy")
        assert pe.shape[0] == traj_x.shape[0]
        if pe.shape[0] == 0:
            continue
        all_pe.append(pe)
        all_traj_x.append(traj_x)

    all_traj_x = np.concatenate(all_traj_x)
    all_pe = np.concatenate(all_pe)

    np.save(config_save_folder_name + "all_traj_x.npy", all_traj_x)
    np.save(config_save_folder_name + "all_pe.npy", all_pe)

 