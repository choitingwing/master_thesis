# # GPU allocation
# from gpuutils import GpuUtils
# GpuUtils.allocate(gpu_count=1, framework='keras')

# import tensorflow as tf
# physical_devices = tf.config.list_physical_devices('GPU')
# for device in physical_devices:
#     tf.config.experimental.set_memory_growth(device, True)
# --------------
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

# Imports
import matplotlib.pyplot as plt
import numpy as np
from constants import plots_dir, dataset
from toolbox import get_pred_energy_diff_data, calculate_percentage_interval, load_file_all_properties, models_dir
import sys
import argparse
import os
import time
import pickle
from NuRadioReco.utilities import units
from scipy import stats
from termcolor import colored
from mpl_toolkits.mplot3d import Axes3D
from itertools import product, combinations
from radiotools import plthelpers as php
from tensorflow import keras
from radiotools import helper as hp
import functools
# -------

# Parse arguments
parser = argparse.ArgumentParser(description='Plot resolution as a function of different parameters')
parser.add_argument("run_id", type=str ,help="the id of the run to be analyzed, eg '3.2' for run3.2")
parser.add_argument("percentage_intervals_str", type=str, help="the percentage intervals, comma-separated, (20,50,68,80)")

args = parser.parse_args()
run_id = args.run_id
percentage_intervals_str = args.percentage_intervals_str

# Save the run name
run_name = f"run{run_id}"

# Parse percentage intervals
percentage_intervals = percentage_intervals_str.split(',')

print(colored(f"Plotting percentage plots for {run_name}...", "yellow"))

# Make sure plots folder exists
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Models folder
saved_model_dir = models_dir(run_name)

# Make sure saved_models folder exists
if not os.path.exists(saved_model_dir):
    os.makedirs(saved_model_dir)

# Load the model
model = keras.models.load_model(f'{saved_model_dir}/model.{run_name}.h5')

# Load test file data and make predictions
    # Load first file
data, nu_direction, nu_zenith, nu_azimuth, nu_energy, nu_flavor, shower_energy = load_file_all_properties(dataset.test_file_ids[0])

    # Then load rest of files
if len(dataset.test_file_ids) > 1:
    for test_file_id in dataset.test_file_ids:
        if test_file_id != dataset.test_file_ids[0]:
            data_tmp, nu_direction_tmp, nu_zenith_tmp, nu_azimuth_tmp, nu_energy_tmp, nu_flavor_tmp, shower_energy_tmp = load_file_all_properties(test_file_id)

            data = np.concatenate((data, data_tmp))
            nu_direction = np.concatenate((nu_direction, nu_direction_tmp))
            nu_zenith = np.concatenate((nu_zenith, nu_zenith_tmp))
            nu_azimuth = np.concatenate((nu_azimuth, nu_azimuth_tmp))
            nu_energy = np.concatenate((nu_energy, nu_energy_tmp))
            nu_flavor = np.concatenate((nu_flavor, nu_flavor_tmp))
            shower_energy = np.concatenate((shower_energy, shower_energy_tmp))


# Get angle difference data
energy_difference_data = get_pred_energy_diff_data(run_name)

# --------- Energy plotting ---------
# Create figure
fig_energy = plt.figure()

# Calculate binned statistics
ax = fig_energy.add_subplot(1, 1, 1)
nu_energy_bins = np.logspace(np.log10(1e17),np.log10(1e19), 30)
nu_energy_bins_with_one_extra = np.append(np.logspace(np.log10(1e17),np.log10(1e19), 30), [1e20])

binned_resolution_nu_energy = np.empty((len(percentage_intervals), 30))

for i in range(len(percentage_intervals)):
    percentage = float(f"0.{percentage_intervals[i]}")
    print(f"Binning percentage {percentage}...")
    partial_func = functools.partial(calculate_percentage_interval, percentage=percentage)
    tmp_binned_stat = stats.binned_statistic(nu_energy, energy_difference_data, bins = nu_energy_bins_with_one_extra, statistic=partial_func)[0].tolist()
    binned_resolution_nu_energy[i, :] = tmp_binned_stat

for i in range(len(percentage_intervals)):
    ax.plot(nu_energy_bins, binned_resolution_nu_energy[i,:], "o", label=f'{percentage_intervals[i]} %')
    
# ax.set_ylim(0, 0.4)
ax.set_xlabel("true nu energy (eV)")
ax.set_ylabel("energy resolution (log10 of eV)")
ax.set_xscale('log')
ax.legend()

# ax = fig_energy.add_subplot(1, 2, 2)
# ax.plot(nu_energy, angle_difference_data, 'o')
# ax.set_xscale('log')

plt.title(f"Mean energy resolution as a function of nu_energy for {run_name}")
fig_energy.tight_layout()
fig_energy.savefig(f"{plots_dir}/mean_resolution_nu_energy_{run_name}_intervals{percentage_intervals_str}.png")
# ___________________________________

print(colored(f"Done plotting percentage plots for {run_name}!", "green", attrs=["bold"]))
print("")