import numpy as np
from openmm import *
from openmm.app import *
from openmm.unit import *
from mdtraj.reporters import HDF5Reporter


class OMMOVRVO():
    def __init__(self, init_pos, bead_masses, torch_force,
                 temperature=300.0,
                 dt=0.002, friction=0.1, save_int=10, tag=""):
        num_beads = len(bead_masses)
        temperature = temperature * kelvin
        friction /= picoseconds
        dt = dt * picoseconds
        self.beta = 1/(temperature * BOLTZMANN_CONSTANT_kB *
                       AVOGADRO_CONSTANT_NA)
        integrator = self._get_ovrvo_integrator(temperature, friction, dt)
        reporter = HDF5Reporter(tag + "sim.h5", save_int)
        system = self._init_system(bead_masses)
        system.addForce(torch_force)
        platform = Platform.getPlatformByName("CUDA")
        self.simulation = self._init_simulation(system, integrator,
                                                platform, reporter, num_beads)
        init_pos = [Vec3(*i) for i in init_pos] * angstroms
        self.simulation.context.setPositions(init_pos)
        self.simulation.context.setVelocitiesToTemperature(temperature)



    def get_information(self, as_numpy=True, enforce_periodic_box=True):
        """Gets information (positions, forces and PE of system)
        Arguments:
            as_numpy: A boolean of whether to return as a numpy array
            enforce_periodic_box: A boolean of whether to enforce periodic boundary conditions
        Returns:
            positions: A numpy array of shape (n_atoms, 3) corresponding to the positions in Angstroms
            velocities: A numpy array of shape (n_atoms, 3) corresponding to the velocities in Angstroms/ps
            forces: A numpy array of shape (n_atoms, 3) corresponding to the force in kcal/mol*Angstroms
            pe: A float coressponding to the potential energy in kcal/mol
            ke: A float coressponding to the kinetic energy in kcal/mol
        """
        state = self.simulation.context.getState(getForces=True,
                                                 getEnergy=True,
                                                 getPositions=True,
                                                 getVelocities=True,
                                                 enforcePeriodicBox=enforce_periodic_box)
        positions = state.getPositions(asNumpy=as_numpy).in_units_of(angstroms)
        forces = state.getForces(asNumpy=as_numpy).in_units_of(
            kilocalories_per_mole / angstroms)
        velocities = state.getVelocities(
            asNumpy=as_numpy).in_units_of(angstroms / picoseconds)
        positions = positions
        forces = forces
        velocities = velocities

        pe = state.getPotentialEnergy().in_units_of(kilocalories_per_mole)._value
        ke = state.getKineticEnergy().in_units_of(kilocalories_per_mole)._value

        return positions, velocities, forces, pe, ke

    def run_sim(self, steps, close_file=False):
        """Runs self.simulation for steps steps
        Arguments:
            steps: The number of steps to run the simulation for
            close_file: A bool to determine whether to close file. Necessary
            if using HDF5Reporter
        """
        self.simulation.step(steps)
        if close_file:
            self.simulation.reporters[0].close()

    def _init_system(self, masses):
        """Initializes an OpenMM system 
        Arguments:
            masses: A list of masses of the "beads" in units of kg/mol
        Returns:
            system: An OpenMM system
        """
        system = System()
        for mass in masses:
            system.addParticle(mass * kilograms/mole)
        return system

    def _init_simulation(self, system, integrator,
                         platform, reporter,
                         num_beads):
        """Initializes an OpenMM simulation 
        Arguments:
            system: An OpenMM system
            integrator: An OpenMM integrator 
            platform: An OpenMM platform specifying the device information
            reporter: An OpenMM reporter
            num_beads: An int specifying the number of beads to use
        Returns:
            simulation: An OpenMM simulation object
        """
        topology = Topology()
        element = Element.getBySymbol('H')
        chain = topology.addChain()
        residue = topology.addResidue('beads', chain)
        for _ in range(num_beads):
            topology.addAtom('cg', element, residue)
        simulation = Simulation(topology, system, integrator, platform)
        simulation.reporters.append(reporter)
        return simulation

    def generate_long_trajectory(self, num_data_points=int(1E6), save_freq=250, tag = ""):
        """Generates long trajectory of length num_data_points*save_freq time steps where information (pos, vel, forces, pe, ke)
           are saved every save_freq time steps
        Arguments:
            num_data_points: An int representing the number of data points to generate
            save_freq: An int representing the frequency for which to save data points
            tag: A string representing the prefix to add to a file
        Saves:
            tag + "_positions.npy": A numpy array of shape (num_data_points,n_atoms,3) representing the positions of the trajectory in units of Angstroms
            tag + "_velocities.npy": A numpy array of shape (num_data_points,n_atoms,3) representing the velocities of the trajectory in units of  Angstroms/picoseconds
            tag + "_forces.npy": A numpy array of shape (num_data_points,n_atoms,3) representing the forces of the trajectory in units of kcal/mol*Angstroms
            tag + "_pe.npy": A numpy array of shape (num_data_points,) representing the pe of the trajectory in units of kcal/mol
            tag + "_ke.npy": A numpy array of shape (num_data_points,) representing the ke of the trajectory in units of kcal/mol
        """
        all_positions = []
        all_velocities = []
        all_forces = []
        all_pe = []
        all_ke = []

        for i in range(num_data_points):
            self.run_sim(save_freq)
            positions, velocities, forces, pe, ke = self.get_information(enforce_periodic_box=False)
            all_positions.append(positions)
            all_velocities.append(velocities)
            all_forces.append(forces)
            all_pe.append(pe)
            all_ke.append(ke)

            if (i % 1000 == 0 or i == (num_data_points - 1)):
                np.save(tag + "positions.npy", all_positions)
                np.save(tag + "velocities.npy", all_velocities)
                np.save(tag + "forces.npy", all_forces)
                np.save(tag + "pe.npy", all_pe)
                np.save(tag + "ke.npy", all_ke)

    def _get_ovrvo_integrator(self, temperature, friction, dt):
        """Creates OpenMM integrator to carry out ovrvo integration (Sivak, Chodera, and Crooks 2014)
        Arguments:
            temperature: temperature with OpenMM units
            friction: friction coefficient with OpenMM units
            dt: time step with OpenMM units
        Returns:
            ovrvo_integrator: OpenMM Integrator
        """
        ovrvo_integrator = CustomIntegrator(dt)
        ovrvo_integrator.setConstraintTolerance(1e-8)
        ovrvo_integrator.addGlobalVariable("a", math.exp(-friction * dt/2))
        ovrvo_integrator.addGlobalVariable(
            "b", np.sqrt(1 - np.exp(-2 * friction * dt/2)))
        ovrvo_integrator.addGlobalVariable("kT", 1/self.beta)
        ovrvo_integrator.addPerDofVariable("x1", 0)

        ovrvo_integrator.addComputePerDof(
            "v", "(a * v) + (b * sqrt(kT/m) * gaussian)")
        ovrvo_integrator.addConstrainVelocities()

        ovrvo_integrator.addComputePerDof("v", "v + 0.5*dt*(f/m)")
        ovrvo_integrator.addConstrainVelocities()

        ovrvo_integrator.addComputePerDof("x", "x + dt*v")
        ovrvo_integrator.addComputePerDof("x1", "x")
        ovrvo_integrator.addConstrainPositions()
        ovrvo_integrator.addComputePerDof("v", "v + (x-x1)/dt")
        ovrvo_integrator.addConstrainVelocities()
        ovrvo_integrator.addUpdateContextState()

        ovrvo_integrator.addComputePerDof("v", "v + 0.5*dt*f/m")
        ovrvo_integrator.addConstrainVelocities()
        ovrvo_integrator.addComputePerDof(
            "v", "(a * v) + (b * sqrt(kT/m) * gaussian)")
        ovrvo_integrator.addConstrainVelocities()
        return ovrvo_integrator
