import numpy as np
import torch
from .utils import reconstruct_atoms


class Reconstructor:
    def __init__(self, nf_model, bond_distances, bond_angles,
                 fixed_atoms, num_atoms, z_matrix):
        self.bond_distances = bond_distances
        self.bond_angles = bond_angles
        self.nf_model = nf_model
        self.fixed_atoms = fixed_atoms
        self.num_atoms = num_atoms
        self.z_matrix = z_matrix

    def generate_new_configs(self, num_to_gen, seed_positions, seed_angles=None):
        """Given a set of seed_positions generates new configurations
        Arguments:
            num_to_gen: An int specifying the number of configurations to generate
            seed_positions: A numpy array of shape (batch_size, fixed_atoms, 3) specifying seed positions
            seed_angles: A numpy array of shape (batch_size, fixed_atoms_ix, 3) specifying seed angles to use for conditioning
        """
        assert seed_positions.shape[0] == num_to_gen
        x, log_px, z = self.nf_model.sample(num_to_gen,
                                            c_info=seed_angles)
        if seed_angles is not None:
            num_c = seed_angles.shape[-1]
            assert torch.abs(z[:, -num_c:] - seed_angles).sum() == 0.0

        all_reconstructed_positions = torch.empty((num_to_gen, self.num_atoms, 3))
        all_reconstructed_positions[:, :self.fixed_atoms.shape[0], :] = seed_positions
        all_distances = np.tile(self.bond_distances, (num_to_gen, 1))
        all_angles = np.tile(self.bond_angles, (num_to_gen, 1))
        all_reconstructed_positions = reconstruct_atoms(all_reconstructed_positions=all_reconstructed_positions,
                                                        all_distances=all_distances, all_angles=all_angles, all_dihedral_angles=x.cpu().numpy(),
                                                        fixed_atoms=self.fixed_atoms, z_matrix=self.z_matrix)
        return all_reconstructed_positions, log_px
