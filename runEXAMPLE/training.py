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
################################################
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
# --------------


# In[2]:


# Imports
import os
import numpy as np
import pickle
import argparse
from termcolor import colored
import time
from toolbox import load_file, find_68_interval, models_dir
from radiotools import helper as hp
from PIL import Image

#from scipy import stats
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv1D, Flatten, Dropout
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation, Concatenate
from tensorflow.keras.layers import AveragePooling2D, AveragePooling1D, Input, Flatten
from tensorflow.keras.layers import BatchNormalization, Lambda, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence, plot_model
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau

from generator import TrainDataset, ValDataset, n_events_per_file, n_files_train, batch_size
from constants import run_version, dataset_name, datapath, data_filename, label_filename, plots_dir, project_name, n_files, n_files_val, dataset_em, dataset_noise, test_file_ids
# -------


# In[3]:
start = time.time()

# Values
feedback_freq = 20 # Only train on 1/feedback_freq of data per epoch
architectures_dir = "architectures"
learning_rate = 0.00005
epochs = 50
# loss_function = "mean_squared_error"
es_patience = 8
es_min_delta = 0.0001 # Old value: es_min_delta = 0.0001
# ------


# In[4]:


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


# In[5]:


# print(saved_model_dir)
# print(architectures_dir)


# In[6]:


# Model params
conv2D_filter_size = 5
pooling_size = 4
amount_Conv2D_layers_per_block = 3 
amount_Conv2D_blocks = 4
conv2D_filter_amount = 32
activation_function = "relu"
# ----------- Create model -----------
inputs = keras.Input(shape=(5, 512, 1))

# Conv2D block 1

# model.add(Conv2D(conv2D_filter_amount, (1, conv2D_filter_size), strides=(1, 1), 
#                  padding='same', activation=activation_function, input_shape=(5, 512, 1)))

x = Conv2D(conv2D_filter_amount, (1, conv2D_filter_size), strides=(1, 1), 
                 padding='same', activation=activation_function)(inputs)
for _ in range(amount_Conv2D_layers_per_block-1):
    x = Conv2D(conv2D_filter_amount, (1, conv2D_filter_size), strides=(1, 1), 
               padding='same', activation=activation_function)(x)

# MaxPooling to reduce size
x = AveragePooling2D(pool_size=(1, pooling_size))(x)

for i in range(amount_Conv2D_blocks-1):
    # Conv2D block
    for _ in range(amount_Conv2D_layers_per_block):
        x = Conv2D(conv2D_filter_amount*2**(i+1), (1, conv2D_filter_size), strides=(1, 1), 
                  padding='same', activation=activation_function)(x)

    # MaxPooling to reduce size
    x = AveragePooling2D(pool_size=(1, pooling_size))(x)

# Batch normalization
x = BatchNormalization()(x)

# Flatten prior to dense layers
x = Flatten()(x)

# Dense layers (fully connected)
x = Dense(1024, activation=activation_function)(x)
x = Dense(1024, activation=activation_function)(x)
x = Dense(512, activation=activation_function)(x)
x = Dense(256, activation=activation_function)(x)
x = Dense(128, activation=activation_function)(x)

# Output layer
out1 = Dense(1)(x)
out2 = Dense(1, activation = 'softplus')(x)
out = Concatenate()([out1, out2])
#out = Dense(2, activation = 'softplus')(x)
model = keras.Model(inputs=inputs, outputs=out, name="one_output") 

model.summary()



# new loss function
def obj(true_e, y_pred=out):
    split = Lambda( lambda x: tf.split(x, num_or_size_splits=2, axis=1))(y_pred)
    pred_e = split[0]
    pred_var= split[1]
    return K.log(pred_var)+ (pred_e-true_e)**2/(pred_var + 10**(-6))

model.compile(loss=obj, optimizer=Adam(lr=learning_rate))
# ------------------------------------


# In[9]:


# Save the model (for opening in eg Netron)
#model.save(f'{saved_model_dir}/{architectures_dir}/model_architecture_{run_name}.h5')
plot_model(model, to_file=f'{saved_model_dir}/{architectures_dir}/model_architecture_{run_name}.png', show_shapes=True)
model_json = model.to_json()
with open(f'{saved_model_dir}/{architectures_dir}/model_architecture_{run_name}.json', "w") as json_file:
    json_file.write(model_json)


# In[10]:


# Calculate number of params
trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])



# Configuring CSV-logger : save epoch and loss values
csv_logger = CSVLogger(os.path.join(saved_model_dir, f"model_history_log_{run_name}.csv"), append=True)

# Configuring callbacks
es = EarlyStopping(monitor="val_loss", patience=es_patience, min_delta=es_min_delta, verbose=1)

mc = ModelCheckpoint(filepath=os.path.join(saved_model_dir,  f"model.{run_name}.h5"), 
                     monitor='val_loss', verbose=1,
                     save_best_only=True, mode='auto', 
                     save_weights_only=False)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=es_patience-3, min_lr=0.000001, verbose=1)

callbacks = [es, mc, csv_logger, reduce_lr]      
# callbacks = [mc, csv_logger, reduce_lr]      

# Calculating steps per epoch and batches per file
steps_per_epoch = n_files_train // feedback_freq * (n_events_per_file // batch_size)
n_batches_per_file = n_events_per_file // batch_size
print(f"steps_per_epoch {steps_per_epoch}, n_batches_per_file {n_batches_per_file}")


# In[13]:


print(n_files_train, n_events_per_file, batch_size)
#41-5-3 = 33


# In[14]:


# Configuring training dataset
dataset_train = tf.data.Dataset.range(n_files_train).prefetch(n_batches_per_file * 10).interleave(
        TrainDataset,# return x,y
        cycle_length=2,
        num_parallel_calls=2,
        deterministic=False).repeat()

# Configuring validation dataset
dataset_val = tf.data.Dataset.range(n_files_val).prefetch(n_batches_per_file * 10).interleave(
        ValDataset,
        cycle_length=2,
        num_parallel_calls=2,
        deterministic=False)


# In[15]:


# print(n_files_val)
# print(dataset_train) #33*100000/64
# print(dataset_val) # 5*100000/64


# In[ ]:


# it = iter(dataset_train)
# print(next(it)[1])


# In[ ]:


# for elem in dataset_train.take(1):
#   print (elem[1])


# In[16]:


# Configuring history and model fit
history = model.fit(
    x=dataset_train, 
    steps_per_epoch=steps_per_epoch, 
    epochs=epochs,
    validation_data=dataset_val, 
    callbacks=callbacks)


# In[17]:


# Dump history with pickle
with open(os.path.join(saved_model_dir, f'history_{run_name}.pkl'), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

# Sleep for a few seconds to free up some resources...
time.sleep(5)


# In[18]:


# Plot loss and evaluate
os.system(f"python plot_performance.py {run_id}")


# In[19]:


# Calculate 68 % interval
energy_68 = find_68_interval(run_name)
print(energy_68)


# In[24]:


# Plot resolution as a function of SNR, energy, zenith and azimuth
os.system(f"python resolution_plotter.py {run_id}")


# In[25]:


print(colored(f"Done training {run_name}!", "green", attrs=["bold"]))
print("")


# In[ ]:
print("The total running time is ",time.time()-start, " s")



