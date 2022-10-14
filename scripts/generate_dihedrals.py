import torch
import numpy as np
import mdtraj as md
import matplotlib.pyplot as plt
from cggnn.omm import ProteinInfo
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--protein_name", type=str,
                    default="chignolin", choices=["chignolin", "adp"])
parser.add_argument("--num_data_points", type=int, default=50000)
device = torch.device("cuda:0")



config = parser.parse_args()
protein_name = config.protein_name
num_data_points = config.num_data_points

p_info = ProteinInfo(protein_name=protein_name)

if protein_name == "chignolin":
    # data_folder_name = "/scratch/groups/rotskoff/cg/CoarseGrainingDataset/ChignolinDataset/"
    data_folder_name = "/home/shriramc/Documents/outputs/CoarseGrainingDataset/ChignolinDataset/"
elif protein_name == "adp":
    # data_folder_name = "/scratch/groups/rotskoff/cg/CoarseGrainingDataset/ADPSolventDataset/"
    data_folder_name = "/home/shriramc/Documents/outputs/CoarseGrainingDataset/ADPSolventDataset/"
else:
    raise ValueError("Incorrect Protein Name Supplied")


bond_edge_index = torch.tensor(p_info.bond_edge_index, device=device).long()
seed_indices = p_info.r_indices
all_non_seed_atoms = np.array([i for i in range(p_info.num_atoms) 
                               if i not in seed_indices])
def distance_from_seed(bonds_visited, curr_node, target_node, num_visits):
    if curr_node == target_node:
        return [num_visits]
    all_possible_bond_indices = torch.cat((torch.where(bond_edge_index[:, 0] == curr_node)[0], 
                                           torch.where(bond_edge_index[:, 1] == curr_node)[0])).tolist()
    
    for b in bonds_visited:
        if b in all_possible_bond_indices:
            all_possible_bond_indices.remove(b)
    all_nodes_visited = []
    for b in all_possible_bond_indices:
        bond = bond_edge_index[b].tolist()
        bond.remove(curr_node)
        assert len(bond) == 1
        new_bonds_visited = bonds_visited.copy()
        new_bonds_visited.append(b)
        all_nodes_visited.extend(distance_from_seed(new_bonds_visited, curr_node = bond[0], 
                                                    target_node=target_node, num_visits=num_visits+1))
    return all_nodes_visited


################# Get Distances From Closest Fixed Atom ##################3

all_distances = []
all_distances_by_atom = {}

for atom in all_non_seed_atoms:
    distances = []
    for seed_atom in seed_indices:
        distances.extend(distance_from_seed([], atom, seed_atom, 0))
        if np.min(distances) == 1:
            break
    all_distances.append(np.min(distances))
    all_distances_by_atom[atom] = np.min(distances)
    
for atom in seed_indices:
    all_distances_by_atom[atom] = 0

all_distances = np.array(all_distances)

all_unique_distances = np.unique(all_distances)
all_non_seed_atoms_sorted = []
for unique_distance in all_unique_distances:
    all_non_seed_atoms_sorted.extend(all_non_seed_atoms[all_distances == unique_distance])

############################## Get All Z Matrices ##########################

def dfs_util(nodes_visited, bonds_visited, curr_node):
    nodes_visited.append(curr_node)
    if len(nodes_visited) == 4:
        return nodes_visited
    all_possible_bond_indices = torch.cat((torch.where(bond_edge_index[:, 0] == curr_node)[0], 
                                           torch.where(bond_edge_index[:, 1] == curr_node)[0])).tolist()
    
    for b in bonds_visited:
        if b in all_possible_bond_indices:
            all_possible_bond_indices.remove(b)
    all_nodes_visited = []
    for b in all_possible_bond_indices:
        bond = bond_edge_index[b].tolist()
        bond.remove(curr_node)
        assert len(bond) == 1
        new_bonds_visited = bonds_visited.copy()
        new_bonds_visited.append(b)
        all_nodes_visited.extend(dfs_util(nodes_visited.copy(), new_bonds_visited, curr_node = bond[0]))
    return all_nodes_visited
    
all_nodes_visited = []
for i in range(p_info.num_atoms):
    if i in all_non_seed_atoms:
        all_nodes_visited.extend(dfs_util([], [], i))
all_nodes_visited = np.array(all_nodes_visited)
z_matrix = all_nodes_visited.reshape((-1, 4))

positions = np.load(data_folder_name + "positions_"
                    + str(num_data_points) + ".npy")
traj = md.Trajectory(positions, topology=None)

z_matrix_total_distances = []
for zmi in z_matrix:
    z_matrix_total_distances.append(all_distances_by_atom[zmi[0]] + 
                                    all_distances_by_atom[zmi[1]] + 
                                    all_distances_by_atom[zmi[2]] +
                                    all_distances_by_atom[zmi[3]])
z_matrix_sorted = z_matrix[np.argsort(z_matrix_total_distances)]



reconstruct_z_matrix = []
all_added = seed_indices.copy()
for non_seed_atom in all_non_seed_atoms_sorted:
    if non_seed_atom in all_added:
        continue
    z_matrix_sorted_seed = z_matrix_sorted[z_matrix_sorted[:, 0] == non_seed_atom]
    for zmi in z_matrix_sorted_seed:
        if zmi[0] not in all_added and np.all(np.isin(zmi[1:], all_added), axis=-1):
            reconstruct_z_matrix.append(zmi)
            all_added.append(zmi[0])
reconstruct_z_matrix = np.stack(reconstruct_z_matrix)


all_dihedral_angles_to_reconstruct = md.compute_dihedrals(traj, reconstruct_z_matrix)
all_distances_to_reconstruct = md.compute_distances(traj, reconstruct_z_matrix[:, :2])
all_bond_angles_to_reconstruct = md.compute_angles(traj, reconstruct_z_matrix[:, :3])

np.save(data_folder_name + "all_dihedral_angles_to_reconstruct.npy", all_dihedral_angles_to_reconstruct)
np.save(data_folder_name + "all_distances_to_reconstruct.npy", all_distances_to_reconstruct)
np.save(data_folder_name + "all_bond_angles_to_reconstruct.npy", all_bond_angles_to_reconstruct)
np.save(data_folder_name + "reconstruct_z_matrix.npy", reconstruct_z_matrix)
np.save(data_folder_name + "fixed_atoms.npy", seed_indices)

seed_z_matrix = []
for i in range(len(p_info.full_backbone_original_indices) - 3):
    seed_z_matrix.append(p_info.full_backbone_original_indices[i:(i + 4)])
seed_z_matrix = np.stack(seed_z_matrix)

all_seed_dihedral_angles = md.compute_dihedrals(traj, seed_z_matrix)
all_seed_distances = md.compute_distances(traj, seed_z_matrix[:, :2])
all_seed_bond_angles = md.compute_angles(traj, seed_z_matrix[:, :3])

np.save(data_folder_name + "all_seed_dihedral_angles.npy", all_seed_dihedral_angles)
np.save(data_folder_name + "all_seed_distances.npy", all_seed_distances)
np.save(data_folder_name + "all_seed_bond_angles.npy", all_seed_bond_angles)
np.save(data_folder_name + "seed_z_matrix.npy", seed_z_matrix)


seed_z_matrix_flattened = seed_z_matrix.flatten()
seed_z_matrix_reindexed = seed_z_matrix_flattened.copy()


for (index, seed) in enumerate(seed_indices):
    seed_z_matrix_reindexed[seed_z_matrix_flattened == seed] = index

seed_z_matrix_reindexed = seed_z_matrix_reindexed.reshape((-1, 4))

np.save(data_folder_name + "seed_z_matrix_reindexed.npy", seed_z_matrix_reindexed)