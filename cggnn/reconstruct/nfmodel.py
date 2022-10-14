import torch
from torch.distributions import MultivariateNormal


class NFModel:
    def __init__(self, base_dist, flow, flow_optimizer, flow_scheduler,
                 folder_name = "./", device=torch.device("cuda:0")):
        self.base_dist = base_dist
        self.flow = flow
        self.flow_optimizer = flow_optimizer
        self.flow_scheduler = flow_scheduler
        self.folder_name = folder_name
        self.device = device

    def sample(self, n_samples, c_info=None):
        if c_info is None:
            z = self.base_dist.sample((n_samples,))
        else:
            z = self.base_dist.sample((n_samples,), seed_angle_samples=c_info)
        x, log_det = self.flow.inverse(z)
        if c_info is None:
            log_px = self.base_dist.log_prob(z) - log_det
        else:
            num_c = c_info.shape[-1]
            log_px = self.base_dist.log_prob(z[:, :-num_c]) - log_det
        return x.detach(), log_px.detach(), z.detach()

    def compute_log_p_hat(self, x):
        z, log_det = self.flow(x)
        log_p_x_b = self.base_dist.log_prob(z)
        log_p_hat_x = log_p_x_b + log_det
        return log_p_hat_x

    def compute_forward_loss_from_dataset(self, x):
        forward_loss = -self.compute_log_p_hat(x)
        return forward_loss.mean()

    def compute_reverse_loss(self, z):
        raise NotImplementedError
    
    def save(self, epoch_num=None):
        torch.save({"model_state_dict": self.flow.state_dict(),
                    "optimizer_state_dict": self.flow_optimizer.state_dict()
                    }, self.folder_name + "flow")
        if epoch_num is not None:
            torch.save({"model_state_dict": self.flow.state_dict(),
                        "optimizer_state_dict": self.flow_optimizer.state_dict()
                        }, self.folder_name + "flow_" + str(epoch_num))
    def load(self, epoch=None):
        if epoch is None:
            flow_checkpoint = torch.load(self.folder_name + "flow", map_location=self.device)
        else:
            flow_checkpoint = torch.load(self.folder_name + "flow_" + str(epoch), map_location=self.device)
        self.flow.load_state_dict(flow_checkpoint["model_state_dict"])
        self.flow_optimizer.load_state_dict(flow_checkpoint["optimizer_state_dict"])
