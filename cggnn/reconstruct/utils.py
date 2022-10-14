import numpy as np
import torch


def get_reconstruction_blocks(fixed_atoms, z_matrix):
    """Gets the order of reconstruction
    Arguments:
        all_reconstructed_positions: A numpy array of shape (batch_size, fixed_atoms.shape[0], 3) representing positions of seed
        z_matrix: A numpy array of shape (num_ic, 4) corresponding to the z_matrix. The first two columns correspond to bond-distance, 
                  the second three columns correspond to a bond angle, the four columns correspond to a dihedral angle
    Returns:
        reconstruct_blockss: A list of length n_blocks, where each element is of shape (n_block_len, 4)
        all_indices_to_atoms: A numpy array of shape (n_atoms, ) mapping reconstruction order to atom number
        all_atoms_to_indices: A numpy array of shape (n_atoms, ) mapping atom number to reconstruction order
        reconstruct_order: A numpy array of shape (n_atoms-num_fixed_atoms,) specifying reconstruction order (excludes seed atoms)
    """


    all_atoms = [fixed_atoms]
    reconstruct_blocks = []
    reconstruct_order = []
    all_added_atoms = fixed_atoms.copy()
    z_matrix_to_add = np.concatenate((np.expand_dims(np.arange(len(z_matrix)), -1), z_matrix), axis=1)
    while z_matrix_to_add.shape[0] > 0:
        all_to_add = np.all(np.isin(z_matrix_to_add[:, 2:], all_added_atoms), axis=-1)
        if (not np.any(all_to_add) and len(z_matrix_to_add) > 0):
             raise ValueError("The following atoms were not reachable from the fixed atoms:", z_matrix_to_add[:,1])
        atom_indices_to_add = z_matrix_to_add[all_to_add, 0]
        atoms_to_add = z_matrix_to_add[all_to_add, 1]
        
        all_atoms.append(atoms_to_add)
        reconstruct_order.append(atom_indices_to_add)
        reconstruct_blocks.append(z_matrix_to_add[all_to_add, 1:])
        assert not np.any(np.isin(atoms_to_add, all_added_atoms))
        all_added_atoms = np.append(all_added_atoms, atoms_to_add)
        z_matrix_to_add = z_matrix_to_add[~all_to_add]
    
    all_indices_to_atoms = all_added_atoms
    all_atoms_to_indices = np.argsort(all_indices_to_atoms)
    reconstruct_order = np.concatenate(reconstruct_order)

    return reconstruct_blocks, all_indices_to_atoms, all_atoms_to_indices, reconstruct_order

def reconstruct_atoms(all_reconstructed_positions, all_distances, all_angles, all_dihedral_angles, fixed_atoms, z_matrix):
    """Reconstructs positions based on internal coordinates
    Arguments:
        all_reconstructed_positions: A numpy array of shape (batch_size, fixed_atoms.shape[0], 3) representing positions of seed
        all_distances: A numpy array of shape (batch_size, z_matrix.shape[0]) representing bond distances to use
        all_angles: A numpy array of shape (batch_size, z_matrix.shape[0]) representing bond angles to use
        all_dihedral_angles: A numpy array of shape (batch_size, z_matrix.shape[0]) representing dihedral angles to use
        fixed_atoms: A numpy arrray of shape (num_fixed_atoms, ) representing the indices of the fixed atoms
        z_matrix: A numpy array of shape (num_ic, 4) corresponding to the z_matrix. The first two columns correspond to bond-distance, 
                  the second three columns correspond to a bond angle, the four columns correspond to a dihedral angle
    Returns:
        all_reconstructed_positions: A numpy array of shape (batch_size, n_atoms, 3) representing positions of reconstructed atoms

    """


    
    reconstruct_blocks, all_indices_to_atoms, all_atoms_to_indices, reconstruct_order = get_reconstruction_blocks(fixed_atoms, z_matrix)
    current_index = fixed_atoms.shape[0]
    for reconstruct_block in reconstruct_blocks:
        reconstruct_indices = all_atoms_to_indices[reconstruct_block]
        fixed_points = all_reconstructed_positions[:, reconstruct_indices[:, 1:]]
        fp_0 = fixed_points[:, :, 0]
        fp_1 = fixed_points[:, :, 1]
        fp_2 = fixed_points[:, :, 2]
        reconstruct_idx = reconstruct_order[reconstruct_indices[:, 0] - fixed_atoms.shape[0]]
        b = torch.tensor(all_distances[:, reconstruct_idx]).float().unsqueeze(-1)
        a = torch.tensor(all_angles[:, reconstruct_idx]).float().unsqueeze(-1)
        t = torch.tensor(all_dihedral_angles[:, reconstruct_idx]).float().unsqueeze(-1)

        v_1 = fp_0 - fp_1
        v_2 = fp_0 - fp_2

        n = torch.cross(v_1, v_2, dim=-1)
        nn = torch.cross(v_1, n, dim=-1)
        
        n_norm = torch.norm(n, dim=-1, keepdim=True)
        n_normalized = n / n_norm

        nn_norm = torch.norm(nn, dim=-1, keepdim=True)
        nn_normalized = nn / nn_norm

        n_scaled = n_normalized * -torch.sin(t)
        nn_scaled = nn_normalized * torch.cos(t)

        v_3 = n_scaled + nn_scaled
        v_3_norm = torch.norm(v_3, dim=-1, keepdim=True)
        v_3_normalized = v_3 / v_3_norm
        v_3_scaled = v_3_normalized * b * torch.sin(a)
        
        v_1_norm = torch.norm(v_1, dim=-1, keepdim=True)
        v_1_normalized = v_1 / v_1_norm
        v_1_scaled = v_1_normalized * b * torch.cos(a)
        g_pos = fp_0 + v_3_scaled - v_1_scaled
        all_reconstructed_positions[:, current_index:current_index+reconstruct_indices.shape[0], :] = g_pos
        current_index += reconstruct_indices.shape[0]
    all_reconstructed_positions = all_reconstructed_positions[:, all_atoms_to_indices]
    return all_reconstructed_positions

