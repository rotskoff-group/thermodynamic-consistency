import torch
import numpy as np
import warnings

class Chain:
    def __init__(self, reconstructor, seed_angles, seed_positions,
                 omm_int, velocity_dist, chain_buffer, num_relax_steps=2, skip_metropolize=False):
        self.reconstructor = reconstructor
        self.seed_angles = seed_angles
        self.seed_positions = seed_positions
        self.omm_int = omm_int
        self.velocity_dist = velocity_dist
        self.num_relax_steps = num_relax_steps
        self.chain_buffer = chain_buffer
        self.x_curr = None
        self.pe_curr = None
        self.ke_curr = None
        self.log_px_curr = None
        self.log_pvel_curr = None
        self.skip_metropolize=skip_metropolize

    def _generate_velocity(self, num_samples=1):
        """Generates random velocities
        Arguments:
            num_samples: Number of velocities to generate
        Returns:
            prop_vel: A numpy of shape 
        """
        prop_vel = self.velocity_dist.sample((num_samples, ))
        log_pvel = self.velocity_dist.log_prob(prop_vel)
        prop_vel = prop_vel.reshape((num_samples, -1, 3)).cpu().numpy()
        prop_vel = np.zeros_like(prop_vel)
        return prop_vel, log_pvel

    def generate_batch_wo_relax(self, num_to_gen, batch_size=1000, threshold=5E6):
        """Generates batch of trajectories without relaxation
        Arguments:
            num_to_gen: An int representing the number of samples to generate
            batch_size: An int representing the batch size to use 
            threshold: A float representing the minimum energy to allow (in kcal/mol)
        Returns:
            all_init_pos: A numpy array of shape (num_generated_below_threshold, n_atoms, 3) represnting positions in Angstroms
            all_init_pe: A numpy array of shape (num_generated_below_threshold, 1) represnting potential energy in kcal/mol
        """
        all_init_pe = []
        all_init_pos = []
        num_batches = num_to_gen//batch_size
        for _ in range(num_batches):
            seed_positions_batch = self.seed_positions.clone().repeat((batch_size, 1, 1))
            if self.seed_angles is None:
                seed_angles_batch = None
            else:
                seed_angles_batch = self.seed_angles.clone().repeat((batch_size, 1))
            all_reconstructed_positions, all_log_px = self.reconstructor.generate_new_configs(batch_size, seed_positions_batch,
                                                                                            seed_angles_batch)
            all_reconstructed_positions = all_reconstructed_positions.cpu().numpy()
            all_prop_vel, _ = self._generate_velocity(num_samples=batch_size)
            all_init_pe = []
            all_init_pos = []

            for (pos, vel) in zip(all_reconstructed_positions, all_prop_vel):
                self.omm_int.update_position_and_velocities(positions=pos,
                                                            velocities=vel)
                _, _, _, init_pe, _ = self.omm_int.get_information()
                if init_pe < threshold:
                    all_init_pe.append(init_pe)
                    all_init_pos.append(pos)
        if len(all_init_pos) > 0:
            all_init_pos = np.stack(all_init_pos)
            all_init_pe = np.array(all_init_pe)
        else:
            all_init_pos = np.array([])
            all_init_pe = np.array([])



        return all_init_pos, all_init_pe



    def _generate_batch(self, num_to_gen):
        """MCMC sampling with verlet relaxation (Not Used)
        """
        seed_positions_batch = self.seed_positions.clone().repeat((num_to_gen, 1, 1))
        if self.seed_angles is None:
            seed_angles_batch = None
        else:
            seed_angles_batch = self.seed_angles.clone().repeat((num_to_gen, 1))
        all_reconstructed_positions, all_log_px = self.reconstructor.generate_new_configs(num_to_gen, seed_positions_batch,
                                                                                          seed_angles_batch)
        all_prop_vel, all_log_pvel = self._generate_velocity(
            num_samples=num_to_gen)
        all_relaxed_positions = []
        all_pe = []
        all_ke = []
        all_reconstructed_positions = all_reconstructed_positions.cpu().numpy()
        all_log_pvel = []
        for (pos, vel) in zip(all_reconstructed_positions, all_prop_vel):
            self.omm_int.update_position_and_velocities(positions=pos,
                                                        velocities=vel)
            _, _, _, init_pe, _ = self.omm_int.get_information()
            all_log_pvel.append(init_pe)
            relaxed_position, pe, ke = self.omm_int.relax_energies(
                num_relax_steps=self.num_relax_steps)
            all_relaxed_positions.append(relaxed_position)
            all_pe.append(pe)
            all_ke.append(ke)

        return all_relaxed_positions, all_pe, all_ke, all_log_px.tolist(), all_log_pvel

    def _propose_step(self):
        """MCMC sampling with verlet relaxation (Not Used)
        """
        all_reconstructed_positions, log_px = self.reconstructor.generate_new_configs(1, self.seed_positions,
                                                                                      self.seed_angles)
        prop_vel, log_pvel = self._generate_velocity()
        all_reconstructed_positions = all_reconstructed_positions.reshape(
            (-1, 3)).cpu().numpy()
        self.omm_int.update_position_and_velocities(positions=all_reconstructed_positions,
                                                    velocities=prop_vel)
        relaxed_position, pe, ke = self.omm_int.relax_energies(
            num_relax_steps=self.num_relax_steps)
        return relaxed_position, pe, ke, log_px.item(), log_pvel.item()

    def _compute_acceptance_criterion(self, log_px_new, log_pvel_new, ke_new, pe_new):
        """MCMC sampling with verlet relaxation (Not Used)
        """
        if self.skip_metropolize:
            return 1
        log_frac = ((self.log_px_curr + self.log_pvel_curr - ke_new - pe_new)
                    - (log_px_new + log_pvel_new - self.ke_curr - self.pe_curr))
        acc_prob = np.minimum(0, log_frac)
        return acc_prob

    def step_chain(self, num_steps):
        """MCMC sampling with verlet relaxation (Not Used)
        """
        with torch.no_grad():
            all_relaxed_positions, all_pe, all_ke, all_log_px, all_log_pvel = self._generate_batch(
                num_steps)
        for (relaxed_pos, pe, ke, log_px, log_pvel) in zip(all_relaxed_positions, all_pe, all_ke, all_log_px, all_log_pvel):
            if self.x_curr is None:
                assert self.pe_curr is None
                assert self.ke_curr is None
                assert self.log_px_curr is None
                assert self.log_pvel_curr is None
                self.x_curr = relaxed_pos
                self.pe_curr = pe
                self.ke_curr = ke
                self.log_px_curr = log_px
                self.log_pvel_curr = log_pvel
                self.chain_buffer.push(*[self.x_curr, self.pe_curr,
                                         self.ke_curr, self.log_px_curr,
                                         self.log_pvel_curr])
                continue
            acc_prob = self._compute_acceptance_criterion(log_px_new=log_px,
                                                          log_pvel_new=log_pvel,
                                                          ke_new=ke,
                                                          pe_new=pe)

            if np.log(np.random.rand()) < acc_prob:
                """Update Chain
                """
                self.x_curr = relaxed_pos
                self.pe_curr = pe
                self.ke_curr = ke
                self.log_px_curr = log_px
                self.log_pvel_curr = log_pvel
                self.chain_buffer.push(*[self.x_curr, self.pe_curr,
                                         self.ke_curr, self.log_px_curr,
                                         self.log_pvel_curr])
            else:
                self.chain_buffer.update_chain_position()
