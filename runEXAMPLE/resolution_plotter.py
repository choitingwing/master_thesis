#!/usr/bin/env python
# coding: utf-8

# In[1]:


# GPU allocation
from gpuutils import GpuUtils
GpuUtils.allocate(gpu_count=1, framework='keras')

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
# --------------
# import tensorflow as tf
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#   try:
#     # Currently, memory growth needs to be the same across GPUs
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#     logical_gpus = tf.config.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#     print(e)

# In[2]:


# Imports
import matplotlib.pyplot as plt
import numpy as np
from constants import plots_dir, datapath, data_filename, label_filename, test_file_ids, run_version
from toolbox import get_pred_energy_diff_data, load_file_all_properties, models_dir
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
# -------


# In[3]:


# Parse arguments
parser = argparse.ArgumentParser(description='Plot energy resolution as a function of different parameters')
parser.add_argument("run_id", type=str ,help="the id of the run to be analyzed, eg '3.2' for run3.2")

args = parser.parse_args()
run_id = args.run_id

# run_id = "EXAMPLE.8"
# Save the run name
run_name = f"run{run_id}"

print(colored(f"Plotting energy resolution as function of neutrino signal properties for {run_name}...", "yellow"))

# Make sure run_name is compatible with run_version
this_run_version = run_name.split(".")[0]
this_run_id = run_name.split(".")[1]
assert this_run_version == run_version, f"run_version ({run_version}) does not match the run version for this run ({this_run_version})"

# Models folder
saved_model_dir = models_dir(run_name)

# Make sure saved_models folder exists
if not os.path.exists(saved_model_dir):
    os.makedirs(saved_model_dir)

# Make sure plots folder exists
plots_dir = f"{plots_dir}/{run_id}"
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)


# Load the model
# model = keras.models.load_model(f'{saved_model_dir}/model.{run_name}.h5',custom_objects={'obj'})


# In[4]:


# Load test file data and make predictions
    # Load first file
data, nu_direction, nu_zenith, nu_azimuth, nu_energy, nu_flavor, shower_energy = load_file_all_properties(test_file_ids[0])

    # Then load rest of files
if len(test_file_ids) > 1:
    for test_file_id in test_file_ids:
        if test_file_id != test_file_ids[0]:
            data_tmp, nu_direction_tmp, nu_zenith_tmp, nu_azimuth_tmp, nu_energy_tmp, nu_flavor_tmp, shower_energy_tmp = load_file_all_properties(test_file_id)

            data = np.concatenate((data, data_tmp))
            nu_direction = np.concatenate((nu_direction, nu_direction_tmp))
            nu_zenith = np.concatenate((nu_zenith, nu_zenith_tmp))
            nu_azimuth = np.concatenate((nu_azimuth, nu_azimuth_tmp))
            nu_energy = np.concatenate((nu_energy, nu_energy_tmp))
            nu_flavor = np.concatenate((nu_flavor, nu_flavor_tmp))
            shower_energy = np.concatenate((shower_energy, shower_energy_tmp))


# Get energy difference data
energy_difference_data = get_pred_energy_diff_data(run_name)

delta_log_E_string = r"$\Delta(\log_{10}\:E)$"
#--


# In[20]:


print(shower_energy.shape)


# In[5]:


# Supposted statistics
supported_statistics = ["mean", "std", "count", "median"]


# In[22]:


# --------- Energy plotting ---------
def plot_energy(statistics):
    assert statistics in supported_statistics

    xlabel = "true nu energy (eV)"
    xscale = 'log'
    yscale = 'linear'
    filename = f"{plots_dir}/{statistics}_log10_energy_difference_nu_energy_{run_name}.png"

    if statistics == "mean":
        ylabel = f"binned mean of {delta_log_E_string} (1)"
        title = f"Mean of {delta_log_E_string} as a function of nu_energy for {run_name}"
    elif statistics == "median":
        ylabel = f"binned median of {delta_log_E_string} (1)"
        title = f"Median of {delta_log_E_string} as a function of nu_energy for {run_name}"
    elif statistics == "std":
        ylabel = f"binned std of {delta_log_E_string} (1)"
        title = f"Standard deviation of {delta_log_E_string} as a function of nu_energy for {run_name}"
    elif statistics == "count":
        ylabel = "count"
        title = f"Count of events inside bins as a function of nu_energy for {run_name}"
    else:
        raise Exception(f"Statistics {statistics} not supported")

    # Create figure
    fig = plt.figure()

    # Calculate binned statistics
    ax = fig.add_subplot(1, 1, 1)
    nu_energy_bins = np.logspace(np.log10(1e17),np.log10(1e19), 30)
    nu_energy_bins_with_one_extra = np.append(np.logspace(np.log10(1e17),np.log10(1e19), 30), [1e20])
    binned_resolution = stats.binned_statistic(nu_energy, energy_difference_data, bins = nu_energy_bins_with_one_extra, statistic=statistics)[0]

    ax.plot(nu_energy_bins, binned_resolution, "o")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    plt.title(title)
    fig.tight_layout()
    fig.savefig(filename)

# ___________________________________


# In[23]:


for statistics in supported_statistics:
    plot_energy(statistics)


# In[24]:


# --------- Azimuth plotting ---------
def plot_azimuth(statistics):
    assert statistics in supported_statistics

    xlabel = "true neutrino direction azimuth angle (°)"
    xscale = 'linear'
    yscale = 'linear'
    filename = f"{plots_dir}/{statistics}_log10_energy_difference_nu_azimuth_{run_name}.png"

    if statistics == "mean":
        ylabel = f"binned mean of {delta_log_E_string} (1)"
        title = f"Mean of {delta_log_E_string} as a function of nu_azimuth for {run_name}"
    elif statistics == "median":
        ylabel = f"binned median of {delta_log_E_string} (1)"
        title = f"Median of {delta_log_E_string} as a function of nu_azimuth for {run_name}"
    elif statistics == "std":
        ylabel = f"binned std of {delta_log_E_string} (1)"
        title = f"Standard deviation of {delta_log_E_string} as a function of nu_azimuth for {run_name}"
    elif statistics == "count":
        ylabel = "count"
        title = f"Count of events inside bins as a function of nu_azimuth for {run_name}"
    else:
        raise Exception(f"Statistics {statistics} not supported")

    # Create figure
    fig = plt.figure()

    # Calculate binned statistics
    ax = fig.add_subplot(1, 1, 1)
    nu_azimuth_bins = np.linspace(0,2*np.pi, 30)
    nu_azimuth_bins_with_one_extra = np.append(np.linspace(0,2*np.pi, 30), 2*np.pi+1)
    binned_resolution = stats.binned_statistic(nu_azimuth, energy_difference_data, bins = nu_azimuth_bins_with_one_extra, statistic=statistics)[0]

    ax.plot(nu_azimuth_bins / units.deg, binned_resolution, "o")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    plt.title(title)
    fig.tight_layout()
    fig.savefig(filename)
# ___________________________________


# In[25]:


for statistics in supported_statistics:
    plot_azimuth(statistics)


# In[26]:


# --------- Zenith plotting ---------
def plot_zenith(statistics):
    assert statistics in supported_statistics

    xlabel = "true neutrino direction zenith angle (°)"
    xscale = 'linear'
    yscale = 'linear'
    filename = f"{plots_dir}/{statistics}_log10_energy_difference_nu_zenith_{run_name}.png"

    if statistics == "mean":
        ylabel = f"binned mean of {delta_log_E_string} (1)"
        title = f"Mean of {delta_log_E_string} as a function of nu_zenith for {run_name}"
    elif statistics == "median":
        ylabel = f"binned median of {delta_log_E_string} (1)"
        title = f"Median of {delta_log_E_string} as a function of nu_zenith for {run_name}"
    elif statistics == "std":
        ylabel = f"binned std of {delta_log_E_string} (1)"
        title = f"Standard deviation of {delta_log_E_string} as a function of nu_zenith for {run_name}"
    elif statistics == "count":
        ylabel = "count"
        title = f"Count of events inside bins as a function of nu_zenith for {run_name}"
    else:
        raise Exception(f"Statistics {statistics} not supported")

    # Create figure
    fig = plt.figure()

    # Calculate binned statistics
    ax = fig.add_subplot(1, 1, 1)
    nu_zenith_bins = np.linspace(0,np.pi, 30)
    nu_zenith_bins_with_one_extra = np.append(np.linspace(0,np.pi, 30), np.pi+1)
    binned_resolution = stats.binned_statistic(nu_zenith, energy_difference_data, bins = nu_zenith_bins_with_one_extra, statistic=statistics)[0]

    ax.plot(nu_zenith_bins / units.deg, binned_resolution, "o")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    plt.title(title)
    fig.tight_layout()
    fig.savefig(filename)
# ___________________________________


# In[27]:


for statistics in supported_statistics:
    plot_zenith(statistics)


# In[28]:


# --------- SNR plotting ---------
def plot_SNR(statistics):
    assert statistics in supported_statistics

    xlabel = "SNR"
    xscale = 'linear'
    yscale = 'linear'
    filename = f"{plots_dir}/{statistics}_log10_energy_difference_SNR_{run_name}.png"

    if statistics == "mean":
        ylabel = f"binned mean of {delta_log_E_string} (1)"
        title = f"Mean of {delta_log_E_string} as a function of SNR for {run_name}"
    elif statistics == "median":
        ylabel = f"binned median of {delta_log_E_string} (1)"
        title = f"Median of {delta_log_E_string} as a function of SNR for {run_name}"
    elif statistics == "std":
        ylabel = f"binned std of {delta_log_E_string} (1)"
        title = f"Standard deviation of {delta_log_E_string} as a function of SNR for {run_name}"
    elif statistics == "count":
        ylabel = "count"
        title = f"Count of events inside bins as a function of SNR for {run_name}"
    else:
        raise Exception(f"Statistics {statistics} not supported")

    # Create figure
    fig = plt.figure()

    # Calculate binned statistics
    ax = fig.add_subplot(1, 1, 1)

    max_LPDA = np.max(np.max(np.abs(data[:, 0:4, :]), axis=1), axis=1)
    SNR_means = np.arange(2.5, 22.5, 1)
    SNR_bins = np.append(np.arange(2, 22, 1), [23])
    binned_resolution = stats.binned_statistic(max_LPDA[:, 0] / 10., energy_difference_data, bins=SNR_bins, statistic=statistics)[0]

    ax.plot(SNR_means, binned_resolution, "o")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    plt.title(title)
    fig.tight_layout()
    fig.savefig(filename)
# ___________________________________


# In[29]:


for statistics in supported_statistics:
    plot_SNR(statistics)


# In[30]:


# for statistics in supported_statistics:
#     plot_energy(statistics)
#     plot_azimuth(statistics)
#     plot_zenith(statistics)
#     plot_SNR(statistics)

print(colored(f"Plotting angular resolution depending on properties for {run_name}!", "green", attrs=["bold"]))
print("")


# In[ ]:




