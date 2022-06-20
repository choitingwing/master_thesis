#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Imports
import numpy as np
from radiotools import plthelpers as php
from matplotlib import pyplot as plt

import os
import time
from radiotools import helper as hp
from NuRadioReco.utilities import units
import pickle
import argparse
from termcolor import colored
from toolbox import load_file, models_dir, calculate_percentage_interval
from constants import datapath, data_filename, label_filename, test_file_ids, plots_dir
# -------

import torch
#GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using",device,". GPU # is",torch.cuda.current_device())

# Parse arguments
parser = argparse.ArgumentParser(description='Evaluate energy resolution')
parser.add_argument("run_id", type=str ,help="the id of the run, eg '3.2' for run3.2")

args = parser.parse_args()
run_id = args.run_id

# Save the run name and filename
run_name = f"run{run_id}"
# filename = f"model_history_log_{run_name}.csv"

# Models folder
saved_model_dir = models_dir(run_name)
print(saved_model_dir)
print(colored(f"plotting samples for {run_name}...", "yellow"))

# Load the model
from generator import E_Model
import pytorch_lightning as pl
import jammy_flows

# mymodel = E_Model().to(device)
mymodel = E_Model()

# save_model_path=os.path.join(saved_model_dir,  f"{run_name}.pt")
# mymodel.load_state_dict(torch.load(save_model_path))

save_model_path=os.path.join(saved_model_dir,  "latest_model_checkpoint.ckpt")
mymodel = E_Model().load_from_checkpoint(save_model_path)

mymodel.eval()
mymodel.double()

# Load test file data and make predictions
from generator import Prepare_Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from constants import test_data_points, sample_numbers

list_of_file_ids_test_small = np.random.choice(test_file_ids, size=1, replace=False)
test = Prepare_Dataset(file_ids=list_of_file_ids_test_small, points = 9)
print("Picked test set ids:",list_of_file_ids_test_small)
print("Length of test dataset: ", len(test))

test_loader = DataLoader(test, batch_size=1, shuffle=False)


# In[ ]:


plots_dir = f"{plots_dir}/{run_id}/samples"
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)


# In[ ]:

binning = np.linspace(16, 20, 55)
from scipy.optimize import curve_fit
def gauss_fit_plot(sample_data, energysmallbin=''):   
    (count, bins) = np.histogram(sample_data, bins=binning)

    bins_middle = (bins[:-1] + bins[1:])/2
    p0 = [np.max(count), 18., 0.5]
    def gauss(x, *p):
        A, mu, sigma = p
        return A*(2 * np.pi * sigma)**-0.5 * np.exp(-0.5 * (x - mu) ** 2 * sigma ** -2)
    coeff, var_matrix = curve_fit(gauss, bins_middle, count, p0=p0, maxfev = 5000)
    hist_fit = gauss(bins_middle, *coeff)

    fig, ax = php.get_histogram(sample_data, bins=binning, xlabel="log(E)")

    # ax.plot(bins_middle, hist_fit, label=f"Gaussian fit with \n$\mu$ = {coeff[1]:.2f}\n$\sigma$ = {coeff[2]:.2f}")
    ax.axvline(x=y.item(), label = f"ture E = {y.item():.2f}", linewidth = 3)
    ax.legend(bbox_to_anchor=(1.0, 1), loc='upper right')
    plt.title(f"Predicted sample distribution with\n68 % interval of E at {energy_68:.2f}, ture E = {y.item():.2f}")
    fig.savefig(f"{plots_dir}/Predicted_sample_trueE_{y.item():.2f}_{run_name}.png")
    plt.close()


# In[ ]:


with torch.no_grad():
    # Iterate through test set minibatchs 
    for x, y in tqdm(test_loader):
        target_sample_list = []
        conv_out = mymodel.forward(x)
        target_sample, base_sample, target_log_pdf, base_log_pdf = mymodel.pdf.sample(samplesize=sample_numbers, conditional_input=conv_out.repeat(sample_numbers,1))
        
        sample_dist = np.squeeze(target_sample.numpy())
        energy_68 = calculate_percentage_interval(sample_dist, 0.68)
        gauss_fit_plot(sample_dist)

        # pdf vs energy
        target_pdf = np.exp(target_log_pdf.numpy())
        sorted_target_sample = np.sort(np.squeeze(target_sample.numpy()))
        index_sorted_target_sample = np.argsort(np.squeeze(target_sample.numpy()))
        sorted_target_pdf = target_pdf[index_sorted_target_sample]
        
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(sorted_target_sample, sorted_target_pdf, 'k-', label = 'Energy PDF')
        ax.axvline(x=y.item(), label = f"true E = {y.item():.2f}", linewidth = 3)
        ax.set_xlim(16, 20)
        ax.set_ylim(0)
        ax.legend(bbox_to_anchor=(1.0, 1), loc='upper right')
        # plt.title(f"Predicted sample pdf, ture E = {y.item():.2f}")
        fig.savefig(f"{plots_dir}/Predicted_sample_trueE_{y.item():.2f}_{run_name}_curve.png")
        plt.close()

        with open(f'{plots_dir}/Predicted_sample_trueE_{y.item():.2f}_{run_name}_curve.pkl', "bw") as fout:
            pickle.dump([sorted_target_sample, sorted_target_pdf], fout, protocol=4)




# In[ ]:




