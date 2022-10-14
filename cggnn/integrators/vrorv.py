import torch
from torch.autograd import grad
import math
class VRORV:
    """
    https://github.com/coarse-graining/cgnet/blob/a3e0e8ddc06f4b6a9f48f4886b73b4cf372ff481/cgnet/network/simulation.py#L450
    pos in units of A
    forces in units of kcal/A
    """
    def __init__(self, u_model, masses, dt=0.002, friction=10.,
                 device=torch.device("cuda:0"), temperature=298.15):
        self.device = device
        kB = 8.314268278  # (J/(kelvin * mole))
        friction = friction  # 1/picoseconds
        self.dt = dt  # picoseconds
        self.u_model = u_model
        self.kt = kB * temperature  # (kg/mol * m^2/s^2) = j/mol
        self.kt = self.kt * 1E-4  # (kg/mol * Å^2/ps^2)
        self.masses = masses

        self.a = math.exp(-friction * self.dt)  # Unitless
        self.b = math.sqrt((1 - self.a ** 2) * self.kt)  # (sqrt((1-a)/beta*m))
        self.b /= torch.sqrt(self.masses)  # Å/ps
        self.t_rescale = torch.sqrt((2/(friction * self.dt)) * torch.tanh(torch.tensor(friction*self.dt/2))).to(device)

    def vrorv_step(self, x, v, f_prev):
        # time-step rescaling (b in paper) is set to be 1
        # v is in Å/ps
        # x is in Å
        # f_prev is in units of kg/mol * (Å/(ps^2))
        # f is in kJ/Å
        v = v + self.dt * (f_prev / self.masses)
        x = x + 0.5 * self.dt * v * self.t_rescale
        v = self.a * v + self.b * torch.randn_like(x)
        x = x + 0.5 * self.dt * v * self.t_rescale
        f = self.u_model.get_force(self.sinfo.get_info(x), info="force_only") 
        f = f.detach()  # Units of kcal/Å
        f = f * 4184  # Units of J/Å
        f = f * 1E-4  # Units of kg/mol * (Å/(ps^2))
        x = x.detach()
        return x, v, f

    def integrate(self, init_x, init_v, steps, save_freq=1000, save_prefix=""):
        traj_x = []
        traj_v = []
        traj_f = []
        x = init_x
        v = init_v
        f_prev = self.u_model.get_force(self.sinfo.get_info(x), info="force_only") 
        f_prev = f_prev.detach()  # Units of kcal/Å
        f_prev = f_prev * 4184  # Units of J/Å
        f_prev = f_prev * 1E-4  # Units of kg/mol * (Å/(ps^2))
        x = x.detach()
        for i in range(steps):
            if (i % 10000) == 0:
                print(i)
            if (i % save_freq) == 0:
                traj_x.append(x)
                traj_v.append(v)
                traj_f.append(f_prev)
            x, v, f_prev = self.vrorv_step(x, v, f_prev)
        traj_x = torch.stack(traj_x)
        traj_v = torch.stack(traj_v)
        traj_f = torch.stack(traj_f)
        torch.save(traj_x, save_prefix + "sim_bead_x.pt")
        torch.save(traj_v, save_prefix + "sim_bead_v.pt")
        torch.save(traj_f, save_prefix + "sim_bead_f.pt")
