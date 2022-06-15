import os
import argparse
from termcolor import colored
import time

start = time.time()
# Parse arguments
parser = argparse.ArgumentParser(description='Neural network for neutrino energy reconstruction')
parser.add_argument("run_id", type=str ,help="the id of the run, eg '3.2' for run3.2")
args = parser.parse_args()
run_id = args.run_id


os.system(f"python PytorchLightning_Flow_training.py {run_id}")
# Sleep for a few seconds to free up some resources...
time.sleep(5)

# Testing and evaluate
os.system(f"python evaluator.py {run_id}")
time.sleep(5)

# Plot performance
os.system(f"python plot_performance.py {run_id}")
time.sleep(5)

# Testing and evaluate
os.system(f"python plot_samples.py {run_id}")
time.sleep(5)

# Plot resolution as a function of SNR, energy, zenith and azimuth
# os.system(f"python resolution_plotter.py {run_id}")

print(colored(f"Done training {run_id}!", "green", attrs=["bold"]))
print("The total running time is ",time.time()-start, " s")
