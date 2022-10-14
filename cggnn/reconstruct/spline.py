import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(
        inputs[..., None] >= bin_locations,
        dim=-1
    ) - 1


def RQS(inputs, unnormalized_widths, unnormalized_heights,
        unnormalized_derivatives, inverse=False, left=0., right=1.,
        bottom=0., top=1., min_bin_width=DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=DEFAULT_MIN_DERIVATIVE):

    if torch.min(inputs) < left or torch.max(inputs) > right:
        print(torch.min(inputs))
        print(torch.max(inputs))
        raise ValueError("Input outside domain")
    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError('Minimal bin width too large for the number of bins')
    if min_bin_height * num_bins > 1.0:
        raise ValueError('Minimal bin height too large for the number of bins')

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode='constant', value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)
    input_derivatives_plus_one = input_derivatives_plus_one[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (((inputs - input_cumheights) * (input_derivatives
                                             + input_derivatives_plus_one - 2 * input_delta)
              + input_heights * (input_delta - input_derivatives)))
        b = (input_heights * input_derivatives - (inputs - input_cumheights)
             * (input_derivatives + input_derivatives_plus_one
                - 2 * input_delta))
        c = - input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        assert (discriminant >= 0).all()

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta \
            + ((input_derivatives + input_derivatives_plus_one
                - 2 * input_delta) * theta_one_minus_theta)
        derivative_numerator = input_delta.pow(2) \
            * (input_derivatives_plus_one * root.pow(2)
               + 2 * input_delta * theta_one_minus_theta
               + input_derivatives * (1 - root).pow(2))
        logabsdet = torch.log(derivative_numerator) - \
            2 * torch.log(denominator)
        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (input_delta * theta.pow(2)
                                     + input_derivatives * theta_one_minus_theta)
        denominator = input_delta + ((input_derivatives
                                      + input_derivatives_plus_one - 2 * input_delta)
                                     * theta_one_minus_theta)
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) \
            * (input_derivatives_plus_one * theta.pow(2)
               + 2 * input_delta * theta_one_minus_theta
               + input_derivatives * (1 - theta).pow(2))
        logabsdet = torch.log(derivative_numerator) - \
            2 * torch.log(denominator)
        return outputs, logabsdet


def unconstrained_RQS(inputs, unnormalized_widths, unnormalized_heights,
                      unnormalized_derivatives, inverse=False,
                      tail_bound=1., min_bin_width=DEFAULT_MIN_BIN_WIDTH,
                      min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
                      min_derivative=DEFAULT_MIN_DERIVATIVE):
    inside_intvl_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_intvl_mask
    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
    constant = np.log(np.exp(1 - min_derivative) - 1)
    unnormalized_derivatives[..., 0] = constant
    unnormalized_derivatives[..., -1] = constant

    outputs[outside_interval_mask] = inputs[outside_interval_mask]
    logabsdet[outside_interval_mask] = 0
    # if any(inside_intvl_mask):
    outputs[inside_intvl_mask], logabsdet[inside_intvl_mask] = RQS(
        inputs=inputs[inside_intvl_mask],
        unnormalized_widths=unnormalized_widths[inside_intvl_mask, :],
        unnormalized_heights=unnormalized_heights[inside_intvl_mask, :],
        unnormalized_derivatives=unnormalized_derivatives[inside_intvl_mask, :],
        inverse=inverse,
        left=-tail_bound, right=tail_bound, bottom=-tail_bound, top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative
    )
    return outputs, logabsdet


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


class NSFARLayer(nn.Module):
    def __init__(self, input_dim, K=32, B=np.pi, hidden_dim=256, device=torch.device("cuda:0")):
        super().__init__()
        self.input_dim = input_dim
        self.K = K
        self.B = B
        self.register_buffer("pi", torch.tensor(np.pi))
        layers = []
        for i in range(self.input_dim):
            # Add n_bc
            if i == 0:
                layers.append(DenseNet(1, 3*K - 1, hidden_dim))
            else:
                layers.append(DenseNet(i, 3*K - 1, hidden_dim))
        self.layers = nn.ModuleList(layers)
        self.device = device

    def forward(self, x):
        """(target->base)
        x is target
        z is base
        """
        z = torch.zeros_like(x).to(self.device)
        log_det = torch.zeros(z.shape[0]).to(self.device)

        for i in range(self.input_dim):
            if i == 0:
                out = self.layers[i](torch.zeros(
                    x.shape[0], 1).to(self.device))
            else:
                out = self.layers[i](x[:, :i])

            W, H, D = torch.split(out, self.K, dim=-1)
            W, H = torch.softmax(W, dim=1), torch.softmax(H, dim=-1)
            W, H = 2 * self.B * W, 2 * self.B * H
            D = F.softplus(D)
            z[:, i], ld = unconstrained_RQS(x[:, i], W, H, D,
                                            inverse=False, tail_bound=self.B)
            log_det += ld
        return z, log_det

    def inverse(self, z):
        """ (base->target)
        x is target
        z is base
        """
        x = torch.zeros_like(z).to(self.device)
        log_det = torch.zeros(x.shape[0]).to(self.device)
        for i in range(self.input_dim):
            if i == 0:
                out = self.layers[i](torch.zeros(x.shape[0], 1).to(
                    self.device))  # + concatenate bc
            else:
                out = self.layers[i](x[:, :i])  # + concatenate bc
            W, H, D = torch.split(out, self.K, dim=-1)
            W, H = torch.softmax(W, dim=1), torch.softmax(H, dim=1)
            W, H = 2 * self.B * W, 2 * self.B * H
            D = F.softplus(D)
            x[:, i], ld = unconstrained_RQS(z[:, i], W, H, D,
                                            inverse=True, tail_bound=self.B)
            log_det += ld
        return x, log_det


class NSFAR(nn.Module):
    """Spline architecture using autoregressive flows without backbone conditioning
    """
    def __init__(self, in_dim, hidden_dim=256, n_layers=1, device=torch.device("cuda:0")):
        super().__init__()
        self.real_nvp_layers = nn.ModuleList([NSFARLayer(input_dim=in_dim,
                                                         hidden_dim=hidden_dim, 
                                                         device=device) for _ in range(n_layers)])

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


class NSFARCLayer(nn.Module):
    def __init__(self, input_dim, n_bb_ic, num_to_condition=32,
                 K=32, B=np.pi, hidden_dim=256, device=torch.device("cuda:0")):
        super().__init__()
        self.input_dim = input_dim
        self.K = K
        self.B = B
        self.num_to_condition = num_to_condition
        self.register_buffer("pi", torch.tensor(np.pi))
        layers = []
        for i in range(self.input_dim):
            # Add n_bc
            if i < self.num_to_condition:
                layers.append(DenseNet(n_bb_ic + i, 3*K - 1, hidden_dim))
            else:
                layers.append(DenseNet(i, 3*K - 1, hidden_dim))
        self.layers = nn.ModuleList(layers)
        self.device = device

    def forward(self, x, bb_ic):
        """(target->base)
        x is target
        z is base
        """
        z = torch.zeros_like(x).to(self.device)
        log_det = torch.zeros(z.shape[0]).to(self.device)

        for i in range(self.input_dim):
            if i < self.num_to_condition:
                out = self.layers[i](torch.cat((bb_ic, x[:, :i]), dim=-1))
            else:
                out = self.layers[i](x[:, :i])
            W, H, D = torch.split(out, self.K, dim=-1)
            W, H = torch.softmax(W, dim=1), torch.softmax(H, dim=-1)
            W, H = 2 * self.B * W, 2 * self.B * H
            D = F.softplus(D)
            z[:, i], ld = unconstrained_RQS(x[:, i], W, H, D,
                                            inverse=False, tail_bound=self.B)
            log_det += ld
        return z, log_det

    def inverse(self, z, bb_ic):
        """ (base->target)
        x is target
        z is base
        """
        x = torch.zeros_like(z).to(self.device)
        log_det = torch.zeros(x.shape[0]).to(self.device)
        for i in range(self.input_dim):
            if i < self.num_to_condition:
                out = self.layers[i](torch.cat((bb_ic, x[:, :i]), dim=-1))
            else:
                out = self.layers[i](x[:, :i])
            W, H, D = torch.split(out, self.K, dim=1)
            W, H = torch.softmax(W, dim=1), torch.softmax(H, dim=1)
            W, H = 2 * self.B * W, 2 * self.B * H
            D = F.softplus(D)
            x[:, i], ld = unconstrained_RQS(z[:, i], W, H, D,
                                            inverse=True, tail_bound=self.B)
            log_det += ld
        return x, log_det


class NSFARC(nn.Module):
    """Spline architecture using autoregressive flows with backbone conditioning
    """
    def __init__(self, in_dim, n_bb_ic, hidden_dim=256, num_to_condition = 32, 
                 n_layers=1, device=torch.device("cuda:0")):
        super().__init__()
        self.in_dim = in_dim
        self.n_bb_ic = n_bb_ic
        self.real_nvp_layers = nn.ModuleList([NSFARCLayer(input_dim=in_dim,
                                                          n_bb_ic=n_bb_ic,
                                                          num_to_condition = num_to_condition,
                                                          hidden_dim=hidden_dim,
                                                          device=device) for _ in range(n_layers)])

    def forward(self, x):
        """(target->base)
        x is target
        z is base
        """
        bb_ic = x[:, self.in_dim:]
        x = x[:, :self.in_dim]
        log_det = 0
        for rnl in self.real_nvp_layers:
            x, ld = rnl.forward(x, bb_ic)
            log_det += ld
        z = x
        return z, log_det

    def inverse(self, z):
        """ (base->target)
        x is target
        z is base
        """
        bb_ic = z[:, self.in_dim:]
        z = z[:, :self.in_dim]
        log_det = 0
        for rnl in self.real_nvp_layers[::-1]:
            z, ld = rnl.inverse(z, bb_ic)
            log_det += ld
        x = z
        return x, log_det



