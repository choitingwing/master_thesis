#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

#import
import datasets
from toolbox import load_file, find_68_interval, models_dir
from constants import run_version, dataset_name, datapath, data_filename, label_filename, plots_dir, project_name, n_files, n_files_val, dataset_em, dataset_noise
from constants import batchSize, test_file_ids, train_files, train_data_points, val_files,val_data_points, norm, epochs, es_patience, es_min_delta
from generator import Prepare_Dataset, E_Model

import os
import numpy as np
import pickle
import argparse
from termcolor import colored
import time
from radiotools import helper as hp
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchviz import make_dot
#--------------------------------------------
# some constants
architectures_dir = "architectures"
# ---------------------
start = time.time()

# In[ ]:


# Parse arguments
parser = argparse.ArgumentParser(description='Neural network for neutrino energy reconstruction')
parser.add_argument("run_id", type=str ,help="the id of the run, eg '3.2' for run3.2")
args = parser.parse_args()
run_id = args.run_id

# Save the run name
run_name = f"run{run_id}"
# Make sure run_name is compatible with run_version
this_run_version = run_name.split(".")[0]
this_run_id = run_name.split(".")[1]
assert this_run_version == run_version, f"run_version ({run_version}) does not match the run version for this run ({this_run_version})"

# Models folder
saved_model_dir = models_dir(run_name)
# Make sure saved_models folder exists
if not os.path.exists(saved_model_dir):
    os.makedirs(saved_model_dir)
# Make sure architectures folder exists
if not os.path.exists(f"{saved_model_dir}/{architectures_dir}"):
    os.makedirs(f"{saved_model_dir}/{architectures_dir}")
# Make sure plot dir exists
plots_dir=f"{plots_dir}/{run_id}"
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)


# In[ ]:


# Info for dataset
print("\ndatapath = ",datapath)
print("data_filename = ", data_filename)
print("label_filename = ", label_filename)

n_files_test = 3
n_files_train = n_files - n_files_val - n_files_test

list_of_file_ids_train = np.arange(n_files_train, dtype=int)
list_of_file_ids_val = np.arange(n_files_train, n_files_train + n_files_val, dtype=int)#np.int?
list_of_file_ids_test = np.arange(n_files_train + n_files_val, n_files, dtype=int)
n_events_per_file = 100000

print("\nTraining files #: ",list_of_file_ids_train)
print("Val files #: ",list_of_file_ids_val)
print("Test files #: ",list_of_file_ids_test)


# In[ ]:

# dataset
print("\nPreparing datasets...")

list_of_file_ids_train_small = np.random.choice(list_of_file_ids_train, size=train_files, replace=False)
print("Picked training set ids:",list_of_file_ids_train_small)
list_of_file_ids_val_small = np.random.choice(list_of_file_ids_val, size=val_files, replace=False)
print("Picked val set ids:",list_of_file_ids_val_small)

train = Prepare_Dataset(file_ids=list_of_file_ids_train_small,points = train_data_points)
val = Prepare_Dataset(file_ids=list_of_file_ids_val_small, points = val_data_points)

# x_train, y_train = train[0]
# x_val, y_val = val[0]

# print(train, len(train), x_train.shape, y_train)
# print(val, len(val), x_val.shape, y_val)

# load data
from torch.utils.data import DataLoader
train_loader = DataLoader(train, batch_size=batchSize, shuffle=True, num_workers=64, pin_memory=True)
val_loader = DataLoader(val, batch_size=batchSize, shuffle=False, num_workers=64, pin_memory=True)
                               
# x, y = next(iter(train_loader))
# print(f"Feature batch shape: {x.size()}")
# print(f"Labels batch shape: {y.size()}")


# In[ ]:


# Create model using Pytorch Lightning 
from torch.optim.lr_scheduler import ReduceLROnPlateau 
import pytorch_lightning as pl
from torchsummary import summary
import jammy_flows

# create a model
model = E_Model()
model.double()


# In[ ]:


os.system(f"python print_model.py > model_{run_id}_structure.txt")
# Sleep for a few seconds to free up some resources...
time.sleep(1)

# def weights_init(m):
#     if isinstance(m, nn.Conv2d):
#         nn.init.xavier_uniform_(m.weight.data)
#         torch.nn.init.zeros_(m.bias.data)
#     if isinstance(m, nn.Linear):
#         nn.init.xavier_uniform_(m.weight.data)
#         torch.nn.init.zeros_(m.bias.data)
# model.apply(weights_init)


# In[ ]:


#callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import DeviceStatsMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import Callback

mc = ModelCheckpoint(dirpath=saved_model_dir, filename= "latest_model_checkpoint", monitor='val_loss', verbose=1) # save best model
es = EarlyStopping("val_loss", patience=es_patience, min_delta=es_min_delta, verbose=1)

class MyPrintingCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is starting")
#         x, y = next(iter(train_loader))
#         model.pdf.init_params(data=y)

# ck = Callback.ModelCheckpoint(
#     monitor="val_loss",
#     dirpath=saved_model_dir,
#     filename=f"{run_name}_{epoch:02d}_{val_loss:.2f}",
#     mode="min",
#     save_weights_only=True,
# )

callbacks = [es, mc, MyPrintingCallback()] # DeviceStatsMonitor()

# Configuring CSV-logger : save epoch and loss values
csv_logger = CSVLogger(saved_model_dir, version=0)


# In[ ]:


# training
trainer = pl.Trainer(
    gpus=2, 
    auto_select_gpus=True,
    ## alternative setting for gpu
    # accelerator="gpu", 
    # devices=[2],
    # num_nodes=2,

    strategy="ddp_sharded",
    precision=16,

    callbacks = callbacks, 
    max_epochs = epochs,
    logger = csv_logger,
    num_sanity_val_steps=0,
    enable_progress_bar = False,
    # profiler="advanced" # or "simple" ,how long a function takes or how much memory is used.
    ) 
    
try: # unsolved error, error occur when running trainer for the first time.
    trainer.fit(model, train_loader, val_loader)
except: 
    trainer.fit(model, train_loader, val_loader)

# In[ ]:


# Save the model (last model)
save_model_path=os.path.join(saved_model_dir, f"{run_name}.pt")
torch.save(model.state_dict(), save_model_path)


# In[ ]:

print("The total training time is ",time.time()-start, " s")



