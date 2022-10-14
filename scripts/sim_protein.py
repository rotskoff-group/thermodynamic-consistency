import argparse
from cggnn.omm.omm import ADPVacuum, ADPSolvent, ADPImplicit, ChignolinImplicit

parser=argparse.ArgumentParser()
parser.add_argument("--system_name", type=str, default="ADPSolvent", choices=["ADPVacuum", "ADPSolvent", "ADPImplicit", "ChignolinImplicit"])
parser.add_argument("--integrator_name", type=str, default="ovrvo", choices=["ovrvo", "omm_ovrvo"])
parser.add_argument("--temperature", type=float, default=300)

config = parser.parse_args() 
integrator_name = config.integrator_name 
system_name = config.system_name
temperature = config.temperature

tag = integrator_name + "_integrator_" + system_name + "_system_" + str(temperature) + "_temperature"

if system_name == "ADPVacuum":
    system = ADPVacuum(integrator_to_use = integrator_name, temperature=temperature, tag=tag)
elif system_name == "ADPSolvent":
    system = ADPSolvent(integrator_to_use = integrator_name, temperature=temperature, tag=tag)
elif system_name == "ADPImplicit":
    system = ADPImplicit(integrator_to_use = integrator_name, temperature=temperature, tag=tag)
elif system_name == "ChignolinImplicit":
    system = ChignolinImplicit(integrator_to_use = integrator_name, temperature=temperature, tag=tag)

system.generate_long_trajectory(num_data_points=int(1E8), save_freq=250, tag=tag)

