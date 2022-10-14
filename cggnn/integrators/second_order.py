import torch
from torch.autograd import grad
import math
class SecondOrder:
    """
    E. V.-Eijnden, and G. Ciccotti, Chem. Phys. Lett. 429, 310 (2006)
    pos in units of A
    forces in units of kcal/A
    """
    def __init__(self, u_model, masses, dt=0.002, friction=10.,
                 device=torch.device("cuda:0"), temperature=298.15):
        self.device = device
        kB = 8.314268278  # (J/(kelvin * mole))
        self.friction = friction  # 1/picoseconds
        self.dt = dt  # picoseconds
        self.u_model = u_model
        self.kt = kB * temperature  # (kg/mol * m^2/s^2) = j/mol
        self.kt = self.kt * 1E-4  # (kg/mol * Å^2/ps^2)
        self.masses = masses
        self.sigma = torch.sqrt(2 * self.kt * self.friction / self.masses)

    def second_order_step(self, x, v, f_prev):
        # v is in Å/ps
        # x is in Å
        # f_prev is in units of kg/mol * (Å/(ps^2))
        # f is in kJ/Å
        xi = torch.randn_like(x)
        eta = torch.randn_like(x)

        v = (v + (0.5 * self.dt * f_prev) - (0.5 * self.dt*self.friction*v) + (0.5 * math.sqrt(self.dt) * self.sigma * xi)
             - (1/8 * (self.dt ** 2) * self.friction * (f_prev - self.friction*v))
             - (1/4 * ((self.dt ** (1.5)) * self.friction * self.sigma * (0.5 * xi + ((1 / math.sqrt(3)) * eta)))))
        x = x + self.dt * v + ((self.dt ** (1.5)) * self.sigma * (1/(2 * math.sqrt(3)) * eta))
        f = self.u_model.get_force(self.sinfo.get_info(x), info="force_only") 
        f = f.detach()  # Units of kcal/Å
        f = f * 4184  # Units of J/Å
        f = f * 1E-4  # Units of kg/mol * (Å/(ps^2))
        x = x.detach()
        v = (v + (0.5 * self.dt * f) - (0.5 * self.dt * self.friction * v) + (0.5 * math.sqrt(self.dt) * self.sigma * xi)
        - (1/8 * (self.dt ** 2) * self.friction * (f - self.friction * v)) 
        - (1/4 * (self.dt ** (1.5)) * self.friction * self.sigma * (0.5 * xi + ((1 / math.sqrt(3)) * eta))))

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
            x, v, f_prev = self.second_order_step(x, v, f_prev)
        traj_x = torch.stack(traj_x)
        traj_v = torch.stack(traj_v)
        traj_f = torch.stack(traj_f)
        torch.save(traj_x, save_prefix + "sim_bead_x.pt")
        torch.save(traj_v, save_prefix + "sim_bead_v.pt")
        torch.save(traj_f, save_prefix + "sim_bead_f.pt")
