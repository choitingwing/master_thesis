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
from toolbox import load_file, models_dir
from constants import datapath, data_filename, label_filename, test_file_ids
# -------
start = time.time()
#GPU
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using",device,". GPU # is",torch.cuda.current_device())

# Parse arguments
parser = argparse.ArgumentParser(description='Evaluate energy resolution')
parser.add_argument("run_id", type=str ,help="the id of the run, eg '3.2' for run3.2")

args = parser.parse_args()
run_id = args.run_id

# Save the run name and filename
run_name = f"run{run_id}"
filename = f"model_history_log_{run_name}.csv"

# Models folder
saved_model_dir = models_dir(run_name)

print(colored(f"Evaluating energy resolution for {run_name}...", "yellow"))
print(saved_model_dir)

# Load the model
from generator import E_Model
import pytorch_lightning as pl
import jammy_flows

# mymodel = E_Model().to(device)
mymodel = E_Model()
save_model_path=os.path.join(saved_model_dir,  "latest_model_checkpoint.ckpt")
mymodel = E_Model().load_from_checkpoint(save_model_path)

# save_model_path=os.path.join(saved_model_dir,  f"{run_name}.pt")
# mymodel.load_state_dict(torch.load(save_model_path))
mymodel.eval()
mymodel.double()

# Load test file data and make predictions
from generator import Prepare_Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from constants import test_data_points

list_of_file_ids_test_small = np.random.choice(test_file_ids, size=3, replace=False)
test = Prepare_Dataset(file_ids=list_of_file_ids_test_small, points = test_data_points)#test_data_points)
print("Picked test set ids:",list_of_file_ids_test_small)
print("Length of test dataset: ", len(test))

test_loader = DataLoader(test, batch_size=1, shuffle=False)

n = 10000
x_list=[]
y_list=[]
shower_energy_log10 = []

gauss_fit_sigma_list = []
true_energy_prob_list = []
from scipy.optimize import curve_fit
from scipy import interpolate
def gauss(x, *p):
    A, mu, sigma = p
    return A*(2 * np.pi * sigma)**-0.5 * np.exp(-0.5 * (x - mu) ** 2 * sigma ** -2)
    
# target_sample_list = np.zeros((1,n))
with torch.no_grad():
    # Iterate through test set minibatchs 
    for x, y in tqdm(test_loader):
        conv_out = mymodel.forward(x)
        target_sample, base_sample, target_log_pdf, base_log_pdf = mymodel.pdf.sample(samplesize=n,conditional_input=conv_out.repeat(n,1))

        x_list.append(np.mean(target_sample.numpy()))
        y_list.append(np.std(target_sample.numpy()))
        shower_energy_log10.append(y.item())
        
        # find gaussian fit sigma
        (count, bins) = np.histogram(target_sample, bins=200)
        bins_middle = (bins[:-1] + bins[1:])/2
        p0 = [np.max(count), np.mean(target_sample.numpy()), np.std(target_sample.numpy())]
        coeff, var_matrix = curve_fit(gauss, bins_middle, count, p0=p0, maxfev = 5000)
        
        gauss_fit_sigma_list.append(coeff[2]) #sigma fit
        
        # Coverage
        target_pdf = np.exp(target_log_pdf.numpy())
        sorted_target_sample = np.sort(np.squeeze(target_sample.numpy()))
        index_sorted_target_sample = np.argsort(np.squeeze(target_sample.numpy()))
        sorted_target_pdf = target_pdf[index_sorted_target_sample]

        #CDF
        cdf_dx = np.diff(sorted_target_sample)
        cdf_x = 0.5*(sorted_target_sample[1:] + sorted_target_sample[:-1])
        pdf_y = 0.5*(sorted_target_pdf[1:] + sorted_target_pdf[:-1])
        pdf_middle_y = cdf_dx*pdf_y
            
        cdf_y = np.zeros(len(pdf_middle_y))
        for i, pdf_middle_y_i in enumerate(pdf_middle_y):
            cdf_y[i] = np.sum(pdf_middle_y[:i])
            
        cdf = interpolate.interp1d(cdf_x, cdf_y)
        try:
            true_energy_prob = cdf(y.item())
            true_energy_prob_list.append(true_energy_prob)
        except: 
            true_energy_prob_list.append(0)

        
        
# Save predicted angles
shower_energy_log10 = np.array(shower_energy_log10)
shower_energy_log10_predict = np.array(x_list)
shower_energy_log10_sigma_predict = np.array(y_list)
gauss_fit_sigma_list = np.array(gauss_fit_sigma_list)
true_energy_prob_list = np.array(true_energy_prob_list)

with open(f'{saved_model_dir}/model.{run_name}.h5_predicted.pkl', "bw") as fout:
    pickle.dump([shower_energy_log10_predict, shower_energy_log10, shower_energy_log10_sigma_predict, gauss_fit_sigma_list, true_energy_prob_list], fout, protocol=4)

print(colored(f"Done evaluating energy resolution for {run_name}!", "green", attrs=["bold"]))
print("")
print("The total evalutation time is ",time.time()-start, " s")


