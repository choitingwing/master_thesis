#!/usr/bin/env python
# coding: utf-8

# In[12]:


# # GPU allocation
from gpuutils import GpuUtils
GpuUtils.allocate(gpu_count=1, framework='keras')

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
# # --------------
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
    
# Imports
import numpy as np
from radiotools import plthelpers as php
from matplotlib import pyplot as plt
from tensorflow import keras
import os
import time
from radiotools import helper as hp
from NuRadioReco.utilities import units
import pickle
import argparse
from termcolor import colored
from toolbox import load_file, models_dir
from constants import datapath, data_filename, label_filename, test_file_ids
import tensorflow.keras.backend as K
# -------

# Parse arguments
parser = argparse.ArgumentParser(description='Evaluate energy resolution')
parser.add_argument("run_id", type=str ,help="the id of the run, eg '3.2' for run3.2")

args = parser.parse_args()
run_id = args.run_id


# In[13]:

# Save the run name and filename
run_name = f"run{run_id}"
filename = f"model_history_log_{run_name}.csv"

# Models folder
saved_model_dir = models_dir(run_name)

print(colored(f"Evaluating energy resolution for {run_name}...", "yellow"))
print(saved_model_dir)


# In[14]:


# new loss function

def obj(true_e, y_pred):
    split = Lambda( lambda x: tf.split(x, num_or_size_splits=2, axis=1))(y_pred)
    pred_e = split[0]
    pred_var= split[1]
    return K.log(pred_var)+ (pred_e-true_e)**2/(pred_var + 10**(-6))


# Load the model
model = keras.models.load_model(f'{saved_model_dir}/model.{run_name}.h5', 
                                custom_objects = {'obj': obj})


# In[16]:


# Load test file data and make predictions
    # Load first file
data, shower_energy_log10 = load_file(test_file_ids[0])


# In[17]:


print(test_file_ids[0], data.shape, shower_energy_log10.shape)


# In[18]:


# Then load rest of files
if len(test_file_ids) > 1:
 for test_file_id in test_file_ids:
    if test_file_id != test_file_ids[0]:
        data_tmp, shower_energy_log10_tmp = load_file(test_file_id)

        data = np.concatenate((data, data_tmp))
        shower_energy_log10 = np.concatenate((shower_energy_log10, shower_energy_log10_tmp))
        


# In[19]:


shower_predict = model.predict(data)


# In[20]:


arrayenergy = np.asarray(shower_predict)
x_list=[]
y_list=[]
for x,y in arrayenergy:
    x_list.append(x)
    y_list.append(y)


# In[21]:


# Save predicted angles
shower_energy_log10_predict = np.array(x_list)
shower_energy_log10_sigma_predict = np.sqrt(np.array(y_list))
with open(f'{saved_model_dir}/model.{run_name}.h5_predicted.pkl', "bw") as fout:
    pickle.dump([shower_energy_log10_predict, shower_energy_log10, shower_energy_log10_sigma_predict], fout, protocol=4)

print(colored(f"Done evaluating energy resolution for {run_name}!", "green", attrs=["bold"]))
print("")


# In[19]:




