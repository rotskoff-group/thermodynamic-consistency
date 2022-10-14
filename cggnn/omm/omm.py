from openmm import *
from openmm.app import *
from openmm.unit import *
from openmmtools import integrators
from mdtraj.reporters import HDF5Reporter
import numpy as np
import math

class Protein:
    def __init__(self, topology, system, integrator_to_use, platform, positions, reporter,
                 temperature=300.0, dt=0.002, friction=0.1):
        temperature = temperature * kelvin
        friction /= picoseconds
        dt = dt * picoseconds
        self.beta = 1/(temperature * BOLTZMANN_CONSTANT_kB * AVOGADRO_CONSTANT_NA)

        if integrator_to_use == "ovrvo":
            integrator = self._get_ovrvo_integrator(temperature, friction, dt)
        elif integrator_to_use == "verlet":
            integrator = self._get_verlet_integrator(temperature, friction, dt)
        elif integrator_to_use == "omm_ovrvo":
            integrator = integrators.VVVRIntegrator(temperature, friction, dt)
        elif integrator_to_use == "overdamped":
            integrator = self._get_overdamped_integrator(temperature, friction, dt)
        else:
            raise ValueError("Incorrect integrator supplied")



        self.simulation = Simulation(topology, system, integrator, platform)
        self.simulation.context.setPositions(positions)
        self.simulation.minimizeEnergy()
        self.simulation.context.setVelocitiesToTemperature(temperature)
        self.simulation.reporters.append(reporter)
        self.target_atom_indices = self._get_target_atom_indices()
        self.pdb_to_prm_top = np.arange(len(self.target_atom_indices))
        self.prm_to_pdb_top = self.pdb_to_prm_top.argsort()
    
    def _get_overdamped_integrator(self, temperature, friction, dt):
        """Creates OpenMM integrator to carry out overdamped Brownian integration
        Arguments:
            temperature: temperature with OpenMM units
            friction: friction coefficient with OpenMM units
            dt: time step with OpenMM units
        Returns:
            overdamped_integrator: OpenMM Integrator
        """

        overdamped_integrator = CustomIntegrator(dt)
        overdamped_integrator.addGlobalVariable("kT", 1/self.beta)
        overdamped_integrator.addGlobalVariable("friction", friction)

        overdamped_integrator.addUpdateContextState()
        overdamped_integrator.addComputePerDof("x", "x+dt*f/(m*friction) + gaussian*sqrt(2*kT*dt/(m*friction))")
        return overdamped_integrator
    
    def _get_verlet_integrator(self, temperature, friction, dt):
        """Creates OpenMM integrator to carry out Verlet integration
        Arguments:
            temperature: temperature with OpenMM units
            friction: friction coefficient with OpenMM units
            dt: time step with OpenMM units
        Returns:
            verlet_integrator: OpenMM Integrator
        """

        verlet_integrator = CustomIntegrator(dt)
        verlet_integrator.addUpdateContextState()
        verlet_integrator.addComputePerDof("v", "v+0.5*dt*f/m")
        verlet_integrator.addComputePerDof("x", "x+dt*v")
        verlet_integrator.addComputePerDof("v", "v+0.5*dt*f/m")
        return verlet_integrator
    
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
        ovrvo_integrator.addGlobalVariable("b", np.sqrt(1 - np.exp(-2 * friction * dt/2)))
        ovrvo_integrator.addGlobalVariable("kT", 1/self.beta)
        ovrvo_integrator.addPerDofVariable("x1", 0)
        
        
        ovrvo_integrator.addComputePerDof("v", "(a * v) + (b * sqrt(kT/m) * gaussian)")
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
        ovrvo_integrator.addComputePerDof("v", "(a * v) + (b * sqrt(kT/m) * gaussian)")
        ovrvo_integrator.addConstrainVelocities()
        return ovrvo_integrator

    def relax_energies(self, num_relax_steps=5):
        """Carries out num_relax_steps of integration
        Arguments:
            num_relax_steps: Number of time steps of dynamics to run
        Returns:
            positions: A numpy array of shape (n_atoms, 3) corresponding to the positions in Angstroms
            pe: A float coressponding to the potential energy in kT
            ke: A float coressponding to the kinetic energy in kT
        """
        self.run_sim(num_relax_steps)
        state = self.simulation.context.getState(getPositions=True,
                                                 getEnergy=True,
                                                 enforcePeriodicBox=False)
        ke = state.getKineticEnergy() * self.beta
        pe = state.getPotentialEnergy() * self.beta
        positions = state.getPositions(asNumpy=True).in_units_of(angstroms)._value
        positions = positions[self.prm_to_pdb_top]
        return positions, pe, ke

    def update_position_and_velocities(self, positions, velocities):
        """Updates position and velocities of the system
        Arguments:
            positions: A numpy array of shape (n_atoms, 3) corresponding to the positions in Angstroms
            velocities: A numpy array of shape (n_atoms, 3) corresponding to the velocities in Angstroms/ps
        """
        positions = positions[self.pdb_to_prm_top] * angstroms
        velocities = velocities * (angstroms/picosecond)
        self.simulation.context.setPositions(positions)
        self.simulation.context.setVelocities(velocities)

        
    def _get_target_atom_indices(self):
        """Gets the indices of all non H2O atoms
        Returns:
            all_atom_indices: The indices of all non water atoms
        """
        all_atom_indices = []
        for residue in self.simulation.topology.residues():
            if residue.name != "HOH":
                for atom in residue.atoms():
                    all_atom_indices.append(atom.index)
        return all_atom_indices    

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
        forces = state.getForces(asNumpy=as_numpy).in_units_of(kilocalories_per_mole / angstroms)
        velocities = state.getVelocities(asNumpy=as_numpy).in_units_of(angstroms / picoseconds)
        positions = positions[self.target_atom_indices]
        forces = forces[self.target_atom_indices]
        velocities = velocities[self.target_atom_indices]

        pe = state.getPotentialEnergy().in_units_of(kilocalories_per_mole)._value
        ke = state.getKineticEnergy().in_units_of(kilocalories_per_mole)._value

        return positions, velocities, forces, pe, ke

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
            positions, velocities, forces, pe, ke = self.get_information()
            all_positions.append(positions)
            all_velocities.append(velocities)
            all_forces.append(forces)
            all_pe.append(pe)
            all_ke.append(ke)

            if (i % 1000 == 0 or i == (num_data_points - 1)):
                np.save(tag + "_positions.npy", all_positions)
                np.save(tag + "_velocities.npy", all_velocities)
                np.save(tag + "_forces.npy", all_forces)
                np.save(tag + "_pe.npy", all_pe)
                np.save(tag + "_ke.npy", all_ke)

   


class ADPVacuum(Protein):
    def __init__(self, integrator_to_use,
                 temperature=300.0, dt=0.002, friction=0.1, save_int=1000,
                 tag = ""):
        prmtop = AmberPrmtopFile('cggnn/omm/adp_vacuum/adp_vacuum.prmtop')
        system = prmtop.createSystem(
            implicitSolvent=None,
            constraints=None,
            nonbondedCutoff=None,
            rigidWater=True,
            hydrogenMass=None
        )
        topology = prmtop.topology
        inpcrd = AmberInpcrdFile('cggnn/omm/adp_vacuum/adp_vacuum.crd')
        positions = inpcrd.getPositions(asNumpy=True)
        platform = Platform.getPlatformByName("CUDA")
        reporter = HDF5Reporter(tag + "_adp_vacuum.h5", save_int)


        super().__init__(topology=topology, system=system, integrator_to_use=integrator_to_use,
                         platform=platform, reporter=reporter,
                         positions=positions, temperature=temperature, dt=dt,
                         friction=friction)

class ADPImplicit(Protein):
    def __init__(self, integrator_to_use,
                 temperature=300.0, dt=0.002, friction=0.1, save_int=1000,
                 tag = ""):
        prmtop = AmberPrmtopFile('cggnn/omm/adp_vacuum/adp_vacuum.prmtop')
        system = prmtop.createSystem(
            implicitSolvent=OBC1,
            constraints=None,
            nonbondedCutoff=None,
            rigidWater=True,
            hydrogenMass=None
        )
        topology = prmtop.topology
        inpcrd = AmberInpcrdFile('cggnn/omm/adp_vacuum/adp_vacuum.crd')
        positions = inpcrd.getPositions(asNumpy=True)
        platform = Platform.getPlatformByName("CUDA")
        reporter = HDF5Reporter(tag + "_adp_implicit.h5", save_int)


        super().__init__(topology=topology, system=system, integrator_to_use=integrator_to_use,
                         platform=platform, reporter=reporter,
                         positions=positions, temperature=temperature, dt=dt,
                         friction=friction)

class ADPSolvent(Protein):
    def __init__(self, integrator_to_use, temperature=300.0, dt=0.004, nonbonded_cutoff=1.0,
                 friction=0.1, ewald_tolerance=1.0E-5, switch_width=0.15, save_int=1000,
                 tag = ""):
        nonbonded_cutoff = nonbonded_cutoff * nanometer
        switch_width = switch_width * nanometer

        prmtop = AmberPrmtopFile('cggnn/omm/adp_solvent/adp_solvent.prmtop')
        system = prmtop.createSystem(
            constraints=HBonds,
            nonbondedMethod=PME,
            nonbondedCutoff=nonbonded_cutoff,
            rigidWater=True,
            hydrogenMass=None
        )

        topology = prmtop.topology
        system.addForce(MonteCarloBarostat(1*bar, temperature, 25))
        forces = {system.getForce(index).__class__.__name__: system.getForce(index)
                  for index in range(system.getNumForces())}
        forces['NonbondedForce'].setUseDispersionCorrection(True)
        forces['NonbondedForce'].setEwaldErrorTolerance(ewald_tolerance)
        forces['NonbondedForce'].setUseSwitchingFunction(True)
        forces['NonbondedForce'].setSwitchingDistance((nonbonded_cutoff
                                                       - switch_width))

        inpcrd = AmberInpcrdFile('cggnn/omm/adp_solvent/adp_solvent.crd')
        positions = inpcrd.getPositions(asNumpy=True)
        box_vectors = inpcrd.getBoxVectors(asNumpy=True)
        system.setDefaultPeriodicBoxVectors(*box_vectors)
        platform = Platform.getPlatformByName("CUDA")
        reporter = HDF5Reporter(tag + "_adp_solvent.h5", save_int)

        super().__init__(topology=topology, system=system, integrator_to_use=integrator_to_use,
                         platform=platform, reporter=reporter,
                         positions=positions, temperature=temperature, dt=dt,
                         friction=friction)
class ChignolinImplicit(Protein):
    def __init__(self, integrator_to_use,
                 temperature=350.0, dt=0.002, friction=0.1, save_int=1000, platform="CUDA",
                 tag = ""):
        prmtop = AmberPrmtopFile('cggnn/omm/chignolin_implicit/chignolin.prmtop')
        system = prmtop.createSystem(
            implicitSolvent=OBC1,
            constraints=None,
            nonbondedCutoff=None,
            rigidWater=True,
            hydrogenMass=None
        )
        topology = prmtop.topology
        inpcrd = AmberInpcrdFile('cggnn/omm/chignolin_implicit/chignolin.crd')
        positions = inpcrd.getPositions(asNumpy=True)
        platform = Platform.getPlatformByName(platform)
        reporter = HDF5Reporter(tag + "_chignolin_implicit.h5", save_int)
        super().__init__(topology=topology, system=system, integrator_to_use=integrator_to_use,
                    platform=platform, reporter=reporter,
                    positions=positions, temperature=temperature, dt=dt,
                    friction=friction)
        self.pdb_to_prm_top = np.array([1,   0,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
                                        13,  14,  15,  16,  17,  18,  19,  20,  23,  24,  21,  22,  25,
                                        26,  27,  28,  29,  30,  31,  33,  32,  34,  35,  36,  37,  38,
                                        39,  40,  41,  44,  45,  42,  43,  46,  47,  48,  49,  50,  51,
                                        52,  54,  53,  55,  56,  57,  58,  59,  60,  61,  63,  62,  69,
                                        71,  70,  66,  68,  67,  64,  65,  72,  73,  74,  75,  76,  77,
                                        78,  80,  79,  81,  83,  82,  84,  85,  86,  87,  88,  89,  90,
                                        91,  92,  93,  94,  97,  98,  99, 100,  95,  96, 101, 102, 103,
                                        104, 105, 107, 106, 108, 109, 110, 111, 112, 113, 114, 115, 118,
                                        119, 120, 121, 116, 117, 122, 123, 124, 125, 126, 127, 128, 130,
                                        129, 131, 132, 133, 134, 135, 136, 142, 143, 144, 145, 140, 141,
                                        138, 139, 137, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
                                        156, 157, 158, 159, 160, 161, 162, 165, 166, 163, 164, 167, 168,
                                        169, 170, 171, 172, 173, 174])
        self.prm_to_pdb_top = self.pdb_to_prm_top.argsort()

