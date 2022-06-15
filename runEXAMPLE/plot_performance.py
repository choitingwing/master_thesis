#!/usr/bin/env python
# coding: utf-8


# Imports
import numpy as np
from radiotools import plthelpers as php
from matplotlib import pyplot as plt
import os
import time
import pickle
from scipy import stats
from radiotools import helper as hp
from NuRadioReco.utilities import units
from toolbox import load_file, calculate_percentage_interval, get_pred_energy_diff_data, models_dir, get_histogram2d, load_file_all_properties, get_2dhist_normalized_columns
import argparse
import argparse
from termcolor import colored
from constants import datapath, data_filename, label_filename, plots_dir, test_file_ids
# -------

# Parse arguments
parser = argparse.ArgumentParser(description='Plot energy resolution')
parser.add_argument("run_id", type=str ,help="the id of the run, eg '3.2' for run3.2")

args = parser.parse_args()
run_id = args.run_id


# Save the run name and filename
run_name = f"run{run_id}"
filename = f"model_history_log_{run_name}.csv"

print(colored(f"Plotting energy resolution for {run_name}...", "yellow"))

# Make sure plots folder exists
plots_dir = f"{plots_dir}/{run_id}"
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

plots_dir_E = f"{plots_dir}/energy"
if not os.path.exists(plots_dir_E):
    os.makedirs(plots_dir_E)

plots_dir_SNR = f"{plots_dir}/SNR"
if not os.path.exists(plots_dir_SNR):
    os.makedirs(plots_dir_SNR)

plots_dir_azimuth = f"{plots_dir}/azimuth"
if not os.path.exists(plots_dir_azimuth):
    os.makedirs(plots_dir_azimuth)

plots_dir_zenith = f"{plots_dir}/zenith"
if not os.path.exists(plots_dir_zenith):
    os.makedirs(plots_dir_zenith)
    

# Models folder
saved_model_dir = models_dir(run_name)
print(saved_model_dir)


# plot loss
loss_file = f'{saved_model_dir}/model_history_log_{run_name}.csv'
loss_data = np.loadtxt(loss_file, delimiter = ',', skiprows = 1, dtype = str)
epoch_list = loss_data[:,0].astype(np.float)
loss_list = loss_data[:,1].astype(np.float)
val_loss_list = loss_data[:,2].astype(np.float)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(epoch_list,loss_list)
ax.plot(epoch_list,val_loss_list)
ax.set_xlabel("epochs -1")
ax.set_ylabel("loss")
ax.legend(['train', 'val'], loc='upper right')
fig.savefig(f"{plots_dir}/loss_{run_name}.png")


# Make sure predicted file exists, otherwise run evaluator
prediction_file = f'{saved_model_dir}/model.{run_name}.h5_predicted.pkl'

if not os.path.isfile(prediction_file):
    print("Prediction file does not exist, running evaluator...")
    os.system(f"python evaluator.py {run_id}")

with open(prediction_file, "br") as fin:
    shower_energy_log10_predict, shower_energy_log10, shower_energy_log10_sigma_predict = pickle.load(fin)
    

print(shower_energy_log10_predict.shape, shower_energy_log10.shape, shower_energy_log10_sigma_predict.shape)

energy_difference_data = shower_energy_log10_predict - shower_energy_log10
# Redefine N
N = energy_difference_data.size
# Calculate 68 %
energy_68 = calculate_percentage_interval(energy_difference_data, 0.68)

delta_log_E_string = r"$\Delta(\log_{10}\:E)$"

print(energy_difference_data[0],energy_68, delta_log_E_string)

plt.close()
fig, ax = php.get_histogram(energy_difference_data, bins=np.linspace(-1.5, 1.5, 90),
                            xlabel=delta_log_E_string)

plt.title(f"Energy resolution with\n68 % interval of {delta_log_E_string} at {energy_68:.2f}")
fig.savefig(f"{plots_dir}/energy_resolution_{run_name}.png")
print(colored(f"Saved energy resolution for {run_name}!", "green", attrs=["bold"]))
print("")

#sigma distribution
sigma_string = r"$\sigma_{E}$"
fig, ax = php.get_histogram(shower_energy_log10_sigma_predict,  bins = 100, xlabel=sigma_string)
plt.title(f"Predicted Energy uncertainty")
fig.savefig(f"{plots_dir}/pred_energy_sigma_{run_name}.png")

# #Heat map for energy
# #####################
# plot_title = f"Heatmap of predicted and true energy for {run_name}"
# xlabel = f"true  energy"
# ylabel = f"predicted energy"
# cmap = "BuPu"
# bins = 100

# for cscale in ["linear", "log"]:
#     file_name = f"plots/scatter_2dhistogram_{run_name}_cscale{cscale}.png"
    
#     # Also plot a heatmap of the scatter plot instead of just dots
#     fig, ax, im = get_histogram2d(shower_energy_log10, shower_energy_log10_predict, fname=file_name, 
#                                   title=plot_title, xlabel=xlabel, ylabel=ylabel, bins=bins, 
#                                   cmap=cmap, cscale=cscale)

#     # Plot a black line through the middle
#     xmin = min(shower_energy_log10)
#     xmax = max(shower_energy_log10)
#     ymin = min(shower_energy_log10_predict)
#     ymax = max(shower_energy_log10_predict)

#     ax.plot([min(xmin, ymin), max(xmax, ymax)], [min(xmin, ymin), max(xmax, ymax)], 'k--')

#     plt.tight_layout()
#     plt.savefig(f"{plots_dir}/predicted_energy_vs_true_energy_{cscale}_{run_name}.png", dpi=300)


#evaluate sigma:
true_log10_sigma = np.absolute(shower_energy_log10 - shower_energy_log10_predict)
print(shower_energy_log10[0], shower_energy_log10_predict[0], true_log10_sigma[0])
print(shower_energy_log10.shape,shower_energy_log10_predict.shape,true_log10_sigma.shape)

#sigma resolution
delta_sigma_string = r"$\Delta(\sigma)$"

sigma_difference_data = np.array([ shower_energy_log10_sigma_predict[i] - true_log10_sigma[i] for i in range(len(true_log10_sigma))])
# Calculate 68 %

# fig, ax = php.get_histogram(sigma_difference_data, bins=100, xlabel=delta_sigma_string)
# plt.title(f"sigma resolution for {run_name} with\n68 % interval of {delta_sigma_string} at {sigma_68:.2f}")
# fig.savefig(f"{plots_dir}/sigma_resolution_{run_name}.png")

# print(colored(f"Saved energy resolution for {run_name}!", "green", attrs=["bold"]))
# print("")

normal_sigma_string = r"$(E_P - E_T)/\sigma_P$"

sigma_ratio_data = np.array([ energy_difference_data[i]/shower_energy_log10_sigma_predict[i] for i in range(len(energy_difference_data))])
# Calculate 68 %
sigma_68 = calculate_percentage_interval(sigma_ratio_data, 0.68)

#fitting
from scipy.optimize import curve_fit
# (count, bins) = np.histogram(sigma_ratio_data, bins=100)
# bins_middle = (bins[:-1] + bins[1:])/2
p0 = [20000, 0., 1.]
def gauss_fit_plot(sigma_ratio_data, plots_dir, energysmallbin=''):   
    (count, bins) = np.histogram(sigma_ratio_data, bins=50)#np.linspace(-40, 40, 400)
                                 #bins=np.linspace(np.min(sigma_ratio_data)-0.1, np.max(sigma_ratio_data)+0.1, 100))

    bins_middle = (bins[:-1] + bins[1:])/2
    p0 = [1000, 0., 1.]
    def gauss(x, *p):
        A, mu, sigma = p
        return A*(2 * np.pi * sigma)**-0.5 * np.exp(-0.5 * (x - mu) ** 2 * sigma ** -2)
    coeff, var_matrix = curve_fit(gauss, bins_middle, count, p0=p0)
    hist_fit = gauss(bins_middle, *coeff)
    fig, ax = php.get_histogram(sigma_ratio_data, bins=50, xlabel=normal_sigma_string)
                                #bins=np.linspace(np.min(sigma_ratio_data)-0.1, np.max(sigma_ratio_data)+0.1, 100),

    ax.plot(bins_middle,hist_fit)
    ax.legend([f"Gaussian fit with \n$\mu$ = {coeff[1]:.2f}\n$\sigma$ = {coeff[2]:.2f}"], loc='upper right')
    plt.title(f"sigma normalised Gaussian, {energysmallbin}") #with\n68 % interval of {normal_sigma_string} at {sigma_68:.2f}
    ax.set_xlim(-6, 6)
    fig.savefig(f"{plots_dir}/sigma_Normalised_Gaussian_{run_name} for {energysmallbin}.png")
    plt.close()
    print(colored(f"Saved sigma Normalised Gaussian for {run_name}!", "green", attrs=["bold"]))
    print("")
gauss_fit_plot(sigma_ratio_data, plots_dir)

#plot small energy bin and fit
#################################
def smallplots(index, index_source, smalltype, plots_dir, sigma = False):
    for i in range(len(index)):
        if i == len(index) - 1:
            print(f"End {smalltype} plots")
        else:  
            energy_range = shower_energy_log10[np.logical_and(index_source > index[i], index_source < index[i+1])]   
            energy_range_p = shower_energy_log10_predict[np.logical_and(index_source > index[i], index_source < index[i+1])]
            sigma_range_p = shower_energy_log10_sigma_predict[np.logical_and(index_source > index[i], index_source < index[i+1])]
            energysmallbin = f"{index[i]:.2f} < {smalltype} < {index[i+1]:.2f}"
        
        normal_sigma_string = r"$(E_P - E_T)/\sigma_P$"
        #sigma resolution
        energy_difference_data_small = energy_range_p - energy_range
        sigma_ratio_data = np.array([ energy_difference_data_small[i]/sigma_range_p[i] for i in range(len(energy_difference_data_small))])
        # Calculate 68 %
        sigma_68 = calculate_percentage_interval(sigma_ratio_data, 0.68)
            
        if sigma:
            fig, ax = php.get_histogram(sigma_range_p, bins = 30, xlabel=sigma_string)
            ax.set_xlim(0,0.6)
            plt.title(f"Energy uncertainty for {energysmallbin}")
            fig.savefig(f"{plots_dir}/energy_sigma_{run_name}_{energysmallbin}.png")
            plt.close()
        else:
            gauss_fit_plot(sigma_ratio_data, plots_dir, energysmallbin)

#  vs Energy
# energy_index = np.linspace(16,19,11)  
energy_index = np.linspace(np.ceil(np.min(shower_energy_log10)), np.ceil(np.max(shower_energy_log10)), 7)     #16,19       
smallplots(energy_index, shower_energy_log10, smalltype='E', plots_dir = plots_dir_E)
smallplots(energy_index, shower_energy_log10, smalltype='E', sigma =True, plots_dir = plots_dir_E)
print(colored(f"Saved sigma Normalised Gaussian for {run_name}!", "green", attrs=["bold"]))

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
            
# vs SNR  
max_LPDA = np.max(np.max(np.abs(data[:, 0:4, :]), axis=1), axis=1)
SNR_x = max_LPDA[:, 0] / 10.
SNR_index = np.linspace(np.floor(np.min(SNR_x)),23, 7)

smallplots(SNR_index, SNR_x, smalltype='SNR', plots_dir = plots_dir_SNR)
smallplots(SNR_index, SNR_x, smalltype='SNR', sigma =True, plots_dir = plots_dir_SNR)
print(colored(f"Saved sigma Normalised Gaussian for {run_name}!", "green", attrs=["bold"]))

# vs nu_azimuth
from NuRadioReco.utilities import units
nu_azimuth_deg = nu_azimuth/units.deg
nu_azimuth_index = np.linspace(np.floor(np.min(nu_azimuth_deg)),np.ceil(np.max(nu_azimuth_deg)), 7)
smallplots(nu_azimuth_index, nu_azimuth_deg, smalltype='azimuth', plots_dir = plots_dir_azimuth)
smallplots(nu_azimuth_index, nu_azimuth_deg, smalltype='azimuth', sigma = True, plots_dir = plots_dir_azimuth)
print(colored(f"Saved sigma Normalised Gaussian for {run_name}!", "green", attrs=["bold"]))

# vs zenith
nu_zenith_deg = nu_zenith/units.deg
nu_zenith_index = np.linspace(np.floor(np.min(nu_zenith_deg)),np.ceil(np.max(nu_zenith_deg)), 7)
smallplots(nu_zenith_index, nu_zenith_deg, smalltype='zenith', plots_dir = plots_dir_zenith)
smallplots(nu_zenith_index, nu_zenith_deg, smalltype='zenith', sigma = True, plots_dir = plots_dir_zenith)
print(colored(f"Saved sigma Normalised Gaussian for {run_name}!", "green", attrs=["bold"]))


#New Heat map for energy
cmap = "BuPu"
plt.close()
fig = plt.figure()
ax = fig.add_subplot(111)
get_2dhist_normalized_columns(shower_energy_log10, shower_energy_log10_predict, fig, ax, binsx=50, binsy=50, cmap = cmap, clim=(0,0.1))
ax.set_xlim(16.5,19)       # x-limits are -1 to 2
ax.set_ylim(16.5,19)
# ax.set_title(f"Heatmap for {run_name}")
ax.set_xlabel("True energy")
ax.set_ylabel("Predicted energy")

    # Plot a black line through the middle
xmin = min(shower_energy_log10)
xmax = max(shower_energy_log10)
ymin = min(shower_energy_log10_sigma_predict)
ymax = max(shower_energy_log10_sigma_predict)
ax.plot([min(xmin, ymin), max(xmax, ymax)], [min(xmin, ymin), max(xmax, ymax)], 'k--')

fig.tight_layout()
fig.savefig(f"{plots_dir}/scatter_2dhistogram_{run_name}.png")

#New Heat map for sigma
plt.close()
fig = plt.figure()
ax = fig.add_subplot(111)
get_2dhist_normalized_columns(shower_energy_log10, shower_energy_log10_sigma_predict, fig, ax, binsx=50, binsy=50, cmap = cmap, clim=(0,0.05))
ax.set_xlim(16.5,19)       
ax.set_ylim(0, 0.65)
# ax.set_title(f"Heatmap for {run_name}")
ax.set_xlabel("True energy")
ax.set_ylabel("Predicted sigma")

fig.tight_layout()
fig.savefig(f"{plots_dir}/scatter_2dhistogram_{run_name}_sigma.png")


# #Heat map for sigma 1
# #####################
# xlabel = f"true  energy"
# ylabel = f"predicted sigma"
# cmap = "BuPu"
# bins = 100

# for cscale in ["linear", "log"]:
#     file_name = f"plots/scatter_2dhistogram_{run_name}_cscale{cscale}.png"
#     plot_title = f"Heatmap in {cscale} scale for {run_name}"
#     # Also plot a heatmap of the scatter plot instead of just dots
#     fig, ax, im = get_histogram2d(shower_energy_log10, shower_energy_log10_sigma_predict, fname=file_name, 
#                                   title=plot_title, xlabel=xlabel, ylabel=ylabel, bins=bins, 
#                                   cmap=cmap, cscale=cscale)

#     # Plot a black line through the middle
#     xmin = min(shower_energy_log10)
#     xmax = max(shower_energy_log10)
#     ymin = min(shower_energy_log10_sigma_predict)
#     ymax = max(shower_energy_log10_sigma_predict)

#     # ax.plot([min(xmin, ymin), max(xmax, ymax)], [min(xmin, ymin), max(xmax, ymax)], 'k--')

#     plt.tight_layout()
#     plt.savefig(f"{plots_dir}/predicted_sigma_vs_true_energy_{cscale}_{run_name}.png", dpi=300)

# #Heat map for sigma 2
# #####################
# xlabel = f"true sigma"
# ylabel = f"predicted sigma"
# cmap = "BuPu"
# bins = 100

# for cscale in ["linear", "log"]:
#     file_name = f"plots/scatter_2dhistogram_{run_name}_cscale{cscale}.png"
#     plot_title = f"Heatmap in {cscale} scale for {run_name}"
#     # Also plot a heatmap of the scatter plot instead of just dots
#     fig, ax, im = get_histogram2d(true_log10_sigma, shower_energy_log10_sigma_predict, fname=file_name, 
#                                   title=plot_title, xlabel=xlabel, ylabel=ylabel, bins=bins, 
#                                   cmap=cmap, cscale=cscale)

#     # Plot a black line through the middle
#     xmin = min(true_log10_sigma)
#     xmax = max(true_log10_sigma)
#     ymin = min(shower_energy_log10_sigma_predict)
#     ymax = max(shower_energy_log10_sigma_predict)

#     ax.plot([min(xmin, ymin), max(xmax, ymax)], [min(xmin, ymin), max(xmax, ymax)], 'k--')

#     plt.tight_layout()
#     plt.savefig(f"{plots_dir}/sigma_Normalised_Gaussian_{cscale}_{run_name}.png", dpi=300)
