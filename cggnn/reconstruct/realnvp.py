import torch
import torch.nn as nn


class DenseNet(nn.Module):
    """
    Simple fully connected neural network.
    """

    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                     nn.Tanh(),
                                     nn.Linear(hidden_dim, hidden_dim),
                                     nn.Tanh(),
                                     nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        return self.network(x)


class RealNVPLayer(nn.Module):
    """
    Non-volume preserving flow.
    [Dinh et. al. 2017]
    """

    def __init__(self, in_dim, hidden_dim=256):
        super().__init__()
        self.dim_1, self.dim_2 = in_dim
        self.t1 = DenseNet(self.dim_1, self.dim_2, hidden_dim)
        self.s1 = DenseNet(self.dim_1, self.dim_2, hidden_dim)
        self.t2 = DenseNet(self.dim_2, self.dim_1, hidden_dim)
        self.s2 = DenseNet(self.dim_2, self.dim_1, hidden_dim)

    def forward(self, x):
        lower, upper = x[:, :self.dim_1], x[:, self.dim_1:]
        t1_transformed = self.t1(lower)
        s1_transformed = self.s1(lower)
        upper = t1_transformed + upper * torch.exp(s1_transformed)
        t2_transformed = self.t2(upper)
        s2_transformed = self.s2(upper)
        lower = t2_transformed + lower * torch.exp(s2_transformed)
        z = torch.cat([lower, upper], dim=-1)
        log_det = (torch.sum(s1_transformed, dim=-1)
                   + torch.sum(s2_transformed, dim=-1))
        return z, log_det

    def inverse(self, z):
        lower, upper = z[:, :self.dim_1], z[:, self.dim_1:]
 
        t2_transformed = self.t2(upper)
        s2_transformed = self.s2(upper)
        lower = (lower - t2_transformed) * torch.exp(-s2_transformed)
        t1_transformed = self.t1(lower)
        s1_transformed = self.s1(lower)
        upper = (upper - t1_transformed) * torch.exp(-s1_transformed)
        x = torch.cat([lower, upper], dim=-1)
        log_det = (torch.sum(-s1_transformed, dim=1)
                   + torch.sum(-s2_transformed, dim=1))
        return x, log_det

class RealNVPLayerFL(nn.Module):
    """
    Non-volume preserving flow. Fixed Lower
    [Dinh et. al. 2017]
    """

    def __init__(self, in_dim, hidden_dim=256):
        super().__init__()
        self.dim_1, self.dim_2 = in_dim
        self.t1 = DenseNet(self.dim_1, self.dim_2, hidden_dim)
        self.s1 = DenseNet(self.dim_1, self.dim_2, hidden_dim)

    def forward(self, x):
        lower, upper = x[:, :self.dim_1], x[:, self.dim_1:]
        t1_transformed = self.t1(lower)
        s1_transformed = self.s1(lower)
        upper = t1_transformed + upper * torch.exp(s1_transformed)
        z = torch.cat([lower, upper], dim=-1)
        log_det = torch.sum(s1_transformed, dim=-1)
        return z, log_det

    def inverse(self, z):
        lower, upper = z[:, :self.dim_1], z[:, self.dim_1:]
 
        t1_transformed = self.t1(lower)
        s1_transformed = self.s1(lower)
        upper = (upper - t1_transformed) * torch.exp(-s1_transformed)
        x = torch.cat([lower, upper], dim=-1)
        log_det = torch.sum(-s1_transformed, dim=1)
        return x, log_det


class RealNVP(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, n_layers=1, fix_lower=False):
        super().__init__()
        if fix_lower:
            self.real_nvp_layers = nn.ModuleList([RealNVPLayerFL(in_dim=in_dim, hidden_dim=hidden_dim) for _ in range(n_layers)])
        else:
            self.real_nvp_layers = nn.ModuleList([RealNVPLayer(in_dim=in_dim, hidden_dim=hidden_dim) for _ in range(n_layers)])

    def forward(self, x):
        """(target->base)
        x is target
        z is base
        """
        log_det = 0
        for rnl in self.real_nvp_layers:
            x, ld = rnl.forward(x)
            log_det += ld
        z = x
        return z, log_det
    
    def inverse(self, z):
        """ (base->target)
        x is target
        z is base
        """
        log_det = 0
        for rnl in self.real_nvp_layers[::-1]:
            z, ld = rnl.inverse(z)
            log_det += ld
        x = z
        return x, log_det
    




