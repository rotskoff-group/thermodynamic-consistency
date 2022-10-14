import torch
import torch.nn as nn
from torch.utils.data import Dataset


def sch_collate_fn(data):
    x_positions, x_atomic_numbers, y_f, y_e  = zip(*data)

    x_positions = torch.stack(x_positions)
    x_atomic_numbers = torch.stack(x_atomic_numbers)
    y_f = torch.stack(y_f)
    y_e = torch.stack(y_e)
    return [x_positions, [x_atomic_numbers]], y_f, y_e


class SchNetDataset(Dataset):
    def __init__(self, energies, forces, positions, atomic_numbers):
        self.energies = energies
        self.forces = forces
        self.positions = positions
        self.atomic_numbers = atomic_numbers
        assert self.energies.shape[0] == self.forces.shape[0] == self.positions.shape[0]

    def __len__(self):
        return self.positions.shape[0]

    def __getitem__(self, idx):
        return self.positions[idx], self.atomic_numbers, self.forces[idx], self.energies[idx]


class SchNetSimInfo:
    def __init__(self, atomic_numbers):
        self.atomic_numbers = atomic_numbers

    def get_info(self, x):
        return x, [self.atomic_numbers]


class SchNetInteraction(nn.Module):
    def __init__(self, n_atom_basis, n_gaussians, n_filters):
        super().__init__()
        self.filter_network_1 = nn.Linear(n_gaussians, n_filters)
        self.filter_network_2 = nn.Linear(n_filters, n_filters)

        self.in2f = nn.Linear(n_atom_basis, n_filters)
        self.f2out = nn.Linear(n_filters, n_atom_basis)

        self.activation = nn.Tanh()
        self.dense = nn.Linear(n_atom_basis, n_atom_basis)

    def forward(self, x, f_ij):
        """
        Arguments:
            x: A torch tensor of (n_configs, n_atoms, n_atom_basis) representing
               the embedding of x
            f_ij: A torch tensor of (n_configs, n_atoms, n_atoms - 1, n_gaussian_basis)
                representing the gaussian expansion. Diagonals are excluded resulting
                in (n_atoms, n_atoms - 1)
        """

        w = self.activation(self.filter_network_1(f_ij))
        w = self.filter_network_2(w)
        y = self.in2f(x)

        n_atoms = x.shape[1]
        n_configs = x.shape[0]

        neighbors = torch.tensor([[list(range(n_atoms))] * n_atoms
                                  for _ in range(n_configs)], device=x.device)

        mask = (torch.eye(n_atoms) == 0).to(x.device)
        neighbors = neighbors.masked_select(mask).view((-1,
                                                        n_atoms,
                                                        n_atoms - 1))
        nbh_size = neighbors.size()
        nbh = neighbors.view(-1, nbh_size[1] * nbh_size[2], 1)
        nbh = nbh.expand(-1, -1, y.size(2))
        nbh = nbh.to(y.device)
        y = torch.gather(y, 1, nbh)
        y = y.view(nbh_size[0], nbh_size[1], nbh_size[2], -1)

        y = y * w
        y = y.sum(dim=2)
        y = self.activation(self.f2out(y))
        y = self.dense(y)
        return y


class SchNet(nn.Module):
    def __init__(self, n_atom_basis=128, n_filters=128, n_interaction_blocks=3,
                 cutoff=5., n_gaussians=25, max_z=5):
        """
        Arguments:
            n_atom_basis: Basis size of atom embedding
            n_filters: Number of filters to use for position convolution
            n_interaction_blocks: Number of SchNet interaction blocks to use
            n_gaussians: Number of gaussian basis functions to use
            cutoff: Cutoff in Å
            max_z: Max number of atomic_numbers that will be used
        """
        super().__init__()
        self.n_atom_basis = n_atom_basis
        self.num_beads = max_z
        self.embedding = nn.Embedding(max_z, n_atom_basis)
        self.interactions = nn.ModuleList([SchNetInteraction(n_atom_basis=n_atom_basis,
                                                             n_gaussians=n_gaussians,
                                                             n_filters=n_filters) for _ in range(n_interaction_blocks)])
        self.register_buffer('centers',
                             torch.linspace(0.0, cutoff, n_gaussians))

        self.variance = (self.centers[1] - self.centers[0]) ** 2
        self.energy_1 = nn.Linear(n_atom_basis, n_atom_basis)
        self.energy_2 = nn.Linear(n_atom_basis, 1)
        self.activation = nn.Tanh()

    def forward(self, atomic_positions, atomic_numbers):
        """
        Arguments:
            atomic_positions: A tensor of size (n_configs, n_atoms, 3)
            atomic_numbers: A tensor of size (n_configs, n_atoms)
        """
        if (len(atomic_positions.shape) == 2):
            """If atomic_positions has shape(n_atoms, 3)
            """
            atomic_positions = atomic_positions.unsqueeze(0)
        if (len(atomic_numbers.shape) == 1):
            """If atomic_positions has shape(n_atoms,)
            """
            atomic_numbers = atomic_numbers.unsqueeze(0)

        n_atoms = atomic_positions.shape[1]
        assert atomic_numbers.shape[0] == atomic_positions.shape[0]
        x = self.embedding(atomic_numbers)
        # Compute distances and gaussian expansion
        pair_dist = (atomic_positions.unsqueeze(-2) -
                     atomic_positions.unsqueeze(-3))
        r_ij = torch.linalg.norm(pair_dist, dim=-1)

        mask = (torch.eye(n_atoms) == 0).to(x.device)
        r_ij = r_ij.masked_select(mask).view((-1, n_atoms, n_atoms - 1))

        dist_centered_squared = (r_ij.unsqueeze(-1) - self.centers) ** 2
        # (n_configs, n_beads, n_beads, n_gaussians)
        f_ij = torch.exp(-(0.5 / self.variance) * dist_centered_squared)

        # Compute Interaction Block
        for interactions in self.interactions:
            v = interactions(x, f_ij)
            x = x + v

        # Compute Energy
        x = self.activation(self.energy_1(x))
        x = self.energy_2(x)
        x = x.sum(1)
        return x
