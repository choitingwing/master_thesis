{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "GPU 1  will be allocated\n"
     ]
    }
   ],
   "source": [
    "# GPU allocation\n",
    "from gpuutils import GpuUtils\n",
    "GpuUtils.allocate(gpu_count=1, framework='keras')\n",
    "\n",
    "import tensorflow as tf\n",
    "physical_devices = tf.config.list_physical_devices('GPU')\n",
    "for device in physical_devices:\n",
    "    tf.config.experimental.set_memory_growth(device, True)\n",
    "# --------------"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "training on 33 files (80.5%), validating on 5 files (12.2%), testing on 3 files (7.3%)\n"
     ]
    }
   ],
   "source": [
    "# Imports\n",
    "import os\n",
    "import numpy as np\n",
    "import pickle\n",
    "import argparse\n",
    "from termcolor import colored\n",
    "import time\n",
    "from toolbox import load_file, find_68_interval, models_dir\n",
    "from radiotools import helper as hp\n",
    "from PIL import Image\n",
    "\n",
    "#from scipy import stats\n",
    "from tensorflow import keras\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Dense, Activation, Conv1D, Flatten, Dropout\n",
    "from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation\n",
    "from tensorflow.keras.layers import AveragePooling2D, AveragePooling1D, Input, Flatten\n",
    "from tensorflow.keras.layers import BatchNormalization, Lambda, MaxPooling2D\n",
    "from tensorflow.keras.optimizers import Adam\n",
    "from tensorflow.keras.regularizers import l2\n",
    "from tensorflow.keras.models import Model\n",
    "from tensorflow.keras.utils import Sequence, plot_model\n",
    "import tensorflow.keras.backend as K\n",
    "from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau\n",
    "\n",
    "from generator import TrainDataset, ValDataset, n_events_per_file, n_files_train, batch_size\n",
    "from constants import run_version, dataset_name, datapath, data_filename, label_filename, plots_dir, project_name, n_files, n_files_val, dataset_em, dataset_noise, test_file_ids\n",
    "# -------"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Values\n",
    "feedback_freq = 10 # Only train on 1/feedback_freq of data per epoch\n",
    "architectures_dir = \"architectures\"\n",
    "learning_rate = 0.00005\n",
    "epochs = 10\n",
    "loss_function = \"mean_squared_error\"\n",
    "es_patience = 8\n",
    "es_min_delta = 0.0001 # Old value: es_min_delta = 0.0001\n",
    "# ------"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Parse arguments\n",
    "\"\"\"\n",
    "parser = argparse.ArgumentParser(description='Neural network for neutrino energy reconstruction')\n",
    "parser.add_argument(\"run_id\", type=str ,help=\"the id of the run, eg '3.2' for run3.2\")\n",
    "\n",
    "args = parser.parse_args()\n",
    "run_id = args.run_id\n",
    "\"\"\"\n",
    "run_id = \"EXAMPLE.8\"\n",
    "# Save the run name\n",
    "run_name = f\"run{run_id}\"\n",
    "\n",
    "# Make sure run_name is compatible with run_version\n",
    "this_run_version = run_name.split(\".\")[0]\n",
    "this_run_id = run_name.split(\".\")[1]\n",
    "assert this_run_version == run_version, f\"run_version ({run_version}) does not match the run version for this run ({this_run_version})\"\n",
    "\n",
    "# Models folder\n",
    "saved_model_dir = models_dir(run_name)\n",
    "\n",
    "# Make sure saved_models folder exists\n",
    "if not os.path.exists(saved_model_dir):\n",
    "    os.makedirs(saved_model_dir)\n",
    "\n",
    "# Make sure architectures folder exists\n",
    "if not os.path.exists(f\"{saved_model_dir}/{architectures_dir}\"):\n",
    "    os.makedirs(f\"{saved_model_dir}/{architectures_dir}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "../common/models/runEXAMPLE.9\n",
      "architectures\n"
     ]
    }
   ],
   "source": [
    "# print(saved_model_dir)\n",
    "# print(architectures_dir)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"one_output\"\n",
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "input_1 (InputLayer)         [(None, 5, 512, 1)]       0         \n",
      "_________________________________________________________________\n",
      "conv2d (Conv2D)              (None, 5, 512, 32)        192       \n",
      "_________________________________________________________________\n",
      "conv2d_1 (Conv2D)            (None, 5, 512, 32)        5152      \n",
      "_________________________________________________________________\n",
      "conv2d_2 (Conv2D)            (None, 5, 512, 32)        5152      \n",
      "_________________________________________________________________\n",
      "average_pooling2d (AveragePo (None, 5, 128, 32)        0         \n",
      "_________________________________________________________________\n",
      "conv2d_3 (Conv2D)            (None, 5, 128, 64)        10304     \n",
      "_________________________________________________________________\n",
      "conv2d_4 (Conv2D)            (None, 5, 128, 64)        20544     \n",
      "_________________________________________________________________\n",
      "conv2d_5 (Conv2D)            (None, 5, 128, 64)        20544     \n",
      "_________________________________________________________________\n",
      "average_pooling2d_1 (Average (None, 5, 32, 64)         0         \n",
      "_________________________________________________________________\n",
      "conv2d_6 (Conv2D)            (None, 5, 32, 128)        41088     \n",
      "_________________________________________________________________\n",
      "conv2d_7 (Conv2D)            (None, 5, 32, 128)        82048     \n",
      "_________________________________________________________________\n",
      "conv2d_8 (Conv2D)            (None, 5, 32, 128)        82048     \n",
      "_________________________________________________________________\n",
      "average_pooling2d_2 (Average (None, 5, 8, 128)         0         \n",
      "_________________________________________________________________\n",
      "conv2d_9 (Conv2D)            (None, 5, 8, 256)         164096    \n",
      "_________________________________________________________________\n",
      "conv2d_10 (Conv2D)           (None, 5, 8, 256)         327936    \n",
      "_________________________________________________________________\n",
      "conv2d_11 (Conv2D)           (None, 5, 8, 256)         327936    \n",
      "_________________________________________________________________\n",
      "average_pooling2d_3 (Average (None, 5, 2, 256)         0         \n",
      "_________________________________________________________________\n",
      "batch_normalization (BatchNo (None, 5, 2, 256)         1024      \n",
      "_________________________________________________________________\n",
      "flatten (Flatten)            (None, 2560)              0         \n",
      "_________________________________________________________________\n",
      "dense (Dense)                (None, 1024)              2622464   \n",
      "_________________________________________________________________\n",
      "dense_1 (Dense)              (None, 1024)              1049600   \n",
      "_________________________________________________________________\n",
      "dense_2 (Dense)              (None, 512)               524800    \n",
      "_________________________________________________________________\n",
      "dense_3 (Dense)              (None, 256)               131328    \n",
      "_________________________________________________________________\n",
      "dense_4 (Dense)              (None, 128)               32896     \n",
      "_________________________________________________________________\n",
      "dense_5 (Dense)              (None, 2)                 258       \n",
      "=================================================================\n",
      "Total params: 5,449,410\n",
      "Trainable params: 5,448,898\n",
      "Non-trainable params: 512\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "# Model params\n",
    "conv2D_filter_size = 5\n",
    "pooling_size = 4\n",
    "amount_Conv2D_layers_per_block = 3 \n",
    "amount_Conv2D_blocks = 4\n",
    "conv2D_filter_amount = 32\n",
    "activation_function = \"relu\"\n",
    "# ----------- Create model -----------\n",
    "inputs = keras.Input(shape=(5, 512, 1))\n",
    "\n",
    "# Conv2D block 1\n",
    "\n",
    "# model.add(Conv2D(conv2D_filter_amount, (1, conv2D_filter_size), strides=(1, 1), \n",
    "#                  padding='same', activation=activation_function, input_shape=(5, 512, 1)))\n",
    "\n",
    "x = Conv2D(conv2D_filter_amount, (1, conv2D_filter_size), strides=(1, 1), \n",
    "                 padding='same', activation=activation_function)(inputs)\n",
    "for _ in range(amount_Conv2D_layers_per_block-1):\n",
    "    x = Conv2D(conv2D_filter_amount, (1, conv2D_filter_size), strides=(1, 1), \n",
    "               padding='same', activation=activation_function)(x)\n",
    "\n",
    "# MaxPooling to reduce size\n",
    "x = AveragePooling2D(pool_size=(1, pooling_size))(x)\n",
    "\n",
    "for i in range(amount_Conv2D_blocks-1):\n",
    "    # Conv2D block\n",
    "    for _ in range(amount_Conv2D_layers_per_block):\n",
    "        x = Conv2D(conv2D_filter_amount*2**(i+1), (1, conv2D_filter_size), strides=(1, 1), \n",
    "                  padding='same', activation=activation_function)(x)\n",
    "\n",
    "    # MaxPooling to reduce size\n",
    "    x = AveragePooling2D(pool_size=(1, pooling_size))(x)\n",
    "\n",
    "# Batch normalization\n",
    "x = BatchNormalization()(x)\n",
    "\n",
    "# Flatten prior to dense layers\n",
    "x = Flatten()(x)\n",
    "\n",
    "# Dense layers (fully connected)\n",
    "x = Dense(1024, activation=activation_function)(x)\n",
    "x = Dense(1024, activation=activation_function)(x)\n",
    "x = Dense(512, activation=activation_function)(x)\n",
    "x = Dense(256, activation=activation_function)(x)\n",
    "x = Dense(128, activation=activation_function)(x)\n",
    "\n",
    "# Output layer\n",
    "out = Dense(2)(x)\n",
    "model = keras.Model(inputs=inputs, outputs=out, name=\"one_output\") #[out1, out2]\n",
    "\n",
    "# out1 = Dense(1)(x)\n",
    "# out2 = Dense(1)(x)\n",
    "# model = keras.Model(inputs=inputs, outputs=[out1, out2], name=\"two_outputs\")\n",
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Tensor(\"dense_12/Identity:0\", shape=(None, 2), dtype=float32)\n",
      "Tensor(\"lambda_3/Identity:0\", shape=(None, 1), dtype=float32)\n",
      "Tensor(\"lambda_3/Identity_1:0\", shape=(None, 1), dtype=float32)\n"
     ]
    }
   ],
   "source": [
    "# print(out)\n",
    "# split = Lambda( lambda x: tf.split(x, num_or_size_splits=2, axis=1))(out)\n",
    "# print (split[0]) # 0: (?, 1024, 1)\n",
    "# print (split[1])  # 1: (?, 1024, 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "def Klog10(x):\n",
    "    numerator = K.log(x)\n",
    "    denominator = K.log(tf.constant(10, dtype=numerator.dtype))\n",
    "    return numerator / denominator\n",
    "\n",
    "# new loss function\n",
    "def obj(true_e, y_pred=out):\n",
    "    split = Lambda( lambda x: tf.split(x, num_or_size_splits=2, axis=1))(y_pred)\n",
    "    pred_e = split[0]\n",
    "    pred_sigma= split[1]\n",
    "    return 0.5*( (true_e-pred_e)/K.exp( Klog10( K.abs(pred_sigma) ) ) )**2 \n",
    "\n",
    "# 0.5*( (true_e-pred_e)/K.exp(pred_sigma) )**2  #working\n",
    "# 0.5*( (true_e-pred_e)/K.exp( Klog10( K.abs(pred_sigma) ) ) )**2 #working\n",
    "\n",
    "# def obj(true_e, y_pred=out):\n",
    "#     split = Lambda( lambda x: tf.split(x, num_or_size_splits=2, axis=1))(y_pred)\n",
    "#     pred_e = split[0]\n",
    "#     pred_sigma = split[1]\n",
    "    \n",
    "#     mse =  (true_e - pred_e)**2\n",
    "#     t = K.abs(pred_sigma**2 - (true_e - pred_e)**2)\n",
    "#     return mse + t\n",
    "\n",
    "#compile\n",
    "                \n",
    "# model.add_loss(obj)\n",
    "# model.compile(optimizer=Adam(lr=learning_rate))\n",
    "\n",
    "model.compile(loss=obj, \n",
    "              optimizer=Adam(lr=learning_rate))\n",
    "\n",
    "# ------------------------------------"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Save the model (for opening in eg Netron)\n",
    "#model.save(f'{saved_model_dir}/{architectures_dir}/model_architecture_{run_name}.h5')\n",
    "plot_model(model, to_file=f'{saved_model_dir}/{architectures_dir}/model_architecture_{run_name}.png', show_shapes=True)\n",
    "model_json = model.to_json()\n",
    "with open(f'{saved_model_dir}/{architectures_dir}/model_architecture_{run_name}.json', \"w\") as json_file:\n",
    "    json_file.write(model_json)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Calculate number of params\n",
    "trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])\n",
    "non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5448898 512\n"
     ]
    }
   ],
   "source": [
    "print(trainable_count, non_trainable_count)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "steps_per_epoch 4686, n_batches_per_file 1562\n"
     ]
    }
   ],
   "source": [
    "# Configuring CSV-logger : save epoch and loss values\n",
    "csv_logger = CSVLogger(os.path.join(saved_model_dir, f\"model_history_log_{run_name}.csv\"), append=True)\n",
    "\n",
    "# Configuring callbacks\n",
    "es = EarlyStopping(monitor=\"val_loss\", patience=es_patience, min_delta=es_min_delta, verbose=1)\n",
    "\n",
    "mc = ModelCheckpoint(filepath=os.path.join(saved_model_dir,  f\"model.{run_name}.h5\"), \n",
    "                     monitor='val_loss', verbose=1,\n",
    "                     save_best_only=True, mode='auto', \n",
    "                     save_weights_only=False)\n",
    "\n",
    "reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=es_patience-3, min_lr=0.000001, verbose=1)\n",
    "\n",
    "callbacks = [es, mc, csv_logger, reduce_lr]      \n",
    "\n",
    "# Calculating steps per epoch and batches per file\n",
    "steps_per_epoch = n_files_train // feedback_freq * (n_events_per_file // batch_size)\n",
    "n_batches_per_file = n_events_per_file // batch_size\n",
    "print(f\"steps_per_epoch {steps_per_epoch}, n_batches_per_file {n_batches_per_file}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "33 100000 64\n"
     ]
    }
   ],
   "source": [
    "print(n_files_train, n_events_per_file, batch_size)\n",
    "#41-5-3 = 33"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Configuring training dataset\n",
    "dataset_train = tf.data.Dataset.range(n_files_train).prefetch(n_batches_per_file * 10).interleave(\n",
    "        TrainDataset,# return x,y\n",
    "        cycle_length=2,\n",
    "        num_parallel_calls=2,\n",
    "        deterministic=False).repeat()\n",
    "\n",
    "# Configuring validation dataset\n",
    "dataset_val = tf.data.Dataset.range(n_files_val).prefetch(n_batches_per_file * 10).interleave(\n",
    "        ValDataset,\n",
    "        cycle_length=2,\n",
    "        num_parallel_calls=2,\n",
    "        deterministic=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 94,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5\n",
      "<RepeatDataset shapes: ((64, 5, 512, 1), (64,)), types: (tf.float64, tf.float64)>\n",
      "<ParallelInterleaveDataset shapes: ((64, 5, 512, 1), (64,)), types: (tf.float64, tf.float64)>\n"
     ]
    }
   ],
   "source": [
    "# print(n_files_val)\n",
    "# print(dataset_train) #33*100000/64\n",
    "# print(dataset_val) # 5*100000/64"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# it = iter(dataset_train)\n",
    "# print(next(it)[1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# for elem in dataset_train.take(1):\n",
    "#   print (elem[1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/10\n",
      "4684/4686 [============================>.] - ETA: 0s - loss: 2.5914\n",
      "Epoch 00001: val_loss improved from inf to 0.00579, saving model to ../common/models/runEXAMPLE.8/model.runEXAMPLE.8.h5\n",
      "4686/4686 [==============================] - 116s 25ms/step - loss: 2.5903 - val_loss: 0.0058 - lr: 5.0000e-05\n",
      "Epoch 2/10\n",
      "4685/4686 [============================>.] - ETA: 0s - loss: 0.0033\n",
      "Epoch 00002: val_loss improved from 0.00579 to 0.00219, saving model to ../common/models/runEXAMPLE.8/model.runEXAMPLE.8.h5\n",
      "4686/4686 [==============================] - 100s 21ms/step - loss: 0.0033 - val_loss: 0.0022 - lr: 5.0000e-05\n",
      "Epoch 3/10\n",
      "4685/4686 [============================>.] - ETA: 0s - loss: 0.0012\n",
      "Epoch 00003: val_loss improved from 0.00219 to 0.00059, saving model to ../common/models/runEXAMPLE.8/model.runEXAMPLE.8.h5\n",
      "4686/4686 [==============================] - 101s 22ms/step - loss: 0.0012 - val_loss: 5.8976e-04 - lr: 5.0000e-05\n",
      "Epoch 4/10\n",
      "4684/4686 [============================>.] - ETA: 0s - loss: 6.1692e-04\n",
      "Epoch 00004: val_loss did not improve from 0.00059\n",
      "4686/4686 [==============================] - 101s 22ms/step - loss: 6.1697e-04 - val_loss: 9.6135e-04 - lr: 5.0000e-05\n",
      "Epoch 5/10\n",
      "4685/4686 [============================>.] - ETA: 0s - loss: 3.5242e-04\n",
      "Epoch 00005: val_loss improved from 0.00059 to 0.00021, saving model to ../common/models/runEXAMPLE.8/model.runEXAMPLE.8.h5\n",
      "4686/4686 [==============================] - 101s 22ms/step - loss: 3.5238e-04 - val_loss: 2.1364e-04 - lr: 5.0000e-05\n",
      "Epoch 6/10\n",
      "4686/4686 [==============================] - ETA: 0s - loss: 2.3201e-04\n",
      "Epoch 00006: val_loss improved from 0.00021 to 0.00010, saving model to ../common/models/runEXAMPLE.8/model.runEXAMPLE.8.h5\n",
      "4686/4686 [==============================] - 102s 22ms/step - loss: 2.3201e-04 - val_loss: 9.8068e-05 - lr: 5.0000e-05\n",
      "Epoch 7/10\n",
      "4684/4686 [============================>.] - ETA: 0s - loss: 1.6564e-04\n",
      "Epoch 00007: val_loss did not improve from 0.00010\n",
      "4686/4686 [==============================] - 102s 22ms/step - loss: 1.6560e-04 - val_loss: 3.2528e-04 - lr: 5.0000e-05\n",
      "Epoch 8/10\n",
      "4684/4686 [============================>.] - ETA: 0s - loss: 9.8824e-05\n",
      "Epoch 00008: val_loss improved from 0.00010 to 0.00005, saving model to ../common/models/runEXAMPLE.8/model.runEXAMPLE.8.h5\n",
      "4686/4686 [==============================] - 102s 22ms/step - loss: 9.8804e-05 - val_loss: 4.8366e-05 - lr: 5.0000e-05\n",
      "Epoch 9/10\n",
      "4683/4686 [============================>.] - ETA: 0s - loss: 7.1818e-05\n",
      "Epoch 00009: val_loss improved from 0.00005 to 0.00004, saving model to ../common/models/runEXAMPLE.8/model.runEXAMPLE.8.h5\n",
      "4686/4686 [==============================] - 103s 22ms/step - loss: 7.1803e-05 - val_loss: 4.1261e-05 - lr: 5.0000e-05\n",
      "Epoch 10/10\n",
      "4684/4686 [============================>.] - ETA: 0s - loss: 4.9156e-05\n",
      "Epoch 00010: val_loss improved from 0.00004 to 0.00003, saving model to ../common/models/runEXAMPLE.8/model.runEXAMPLE.8.h5\n",
      "4686/4686 [==============================] - 100s 21ms/step - loss: 4.9148e-05 - val_loss: 2.6539e-05 - lr: 5.0000e-05\n"
     ]
    }
   ],
   "source": [
    "# Configuring history and model fit\n",
    "history = model.fit(\n",
    "    x=dataset_train, \n",
    "    steps_per_epoch=steps_per_epoch, \n",
    "    epochs=epochs,\n",
    "    validation_data=dataset_val, \n",
    "    callbacks=callbacks)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Dump history with pickle\n",
    "with open(os.path.join(saved_model_dir, f'history_{run_name}.pkl'), 'wb') as file_pi:\n",
    "    pickle.dump(history.history, file_pi)\n",
    "\n",
    "# Sleep for a few seconds to free up some resources...\n",
    "time.sleep(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Plot loss and evaluate\n",
    "os.system(f\"python plot_performance.py {run_id}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.4385639747728296\n"
     ]
    }
   ],
   "source": [
    "# Calculate 68 % interval\n",
    "energy_68 = find_68_interval(run_name)\n",
    "print(energy_68)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Plot resolution as a function of SNR, energy, zenith and azimuth\n",
    "os.system(f\"python resolution_plotter.py {run_id}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m\u001b[32mDone training runEXAMPLE.9!\u001b[0m\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(colored(f\"Done training {run_name}!\", \"green\", attrs=[\"bold\"]))\n",
    "print(\"\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
