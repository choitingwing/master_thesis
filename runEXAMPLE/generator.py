# Imports
import os
import numpy as np
import tensorflow as tf
import time
from toolbox import load_file
from constants import datapath, data_filename, label_filename, n_files, n_files_val
# -------

np.set_printoptions(precision=4)

# n_files and n_files_val comes from dataset in constants.py
n_files_test = 3
norm = 1e-6

n_files_train = n_files - n_files_val - n_files_test
list_of_file_ids_train = np.arange(n_files_train, dtype=np.int)
list_of_file_ids_val = np.arange(n_files_train, n_files_train + n_files_val, dtype=np.int)
list_of_file_ids_test = np.arange(n_files_train + n_files_val, n_files, dtype=np.int)
n_events_per_file = 100000
batch_size = 64

print(f"training on {n_files_train} files ({n_files_train/n_files*100:.1f}%), validating on {n_files_val} files ({n_files_val/n_files*100:.1f}%), testing on {n_files_test} files ({n_files_test/n_files*100:.1f}%)")

class TrainDataset(tf.data.Dataset):

    def _generator(file_id):
        if((file_id + 1) == n_files_train):
#             print("reshuffling")
            np.random.shuffle(list_of_file_ids_train)

        # Opening the file
        i_file = list_of_file_ids_train[file_id]
        data, shower_energy_log10 = load_file(i_file, norm)
        num_samples = data.shape[0]
        rand_ids = np.arange(num_samples, dtype=np.int)
        np.random.shuffle(rand_ids)
        n_batches = num_samples // batch_size
        for i_batch in range(n_batches):
            # Reading data (line, record) from the file
            y = shower_energy_log10[rand_ids[i_batch * batch_size:(i_batch + 1) * batch_size]]
            x = data[rand_ids[i_batch * batch_size:(i_batch + 1) * batch_size], :, :, :]
            yield x, y

    def __new__(cls, file_id):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=(tf.dtypes.float64, tf.dtypes.float64),
            output_shapes=((batch_size, 5, 512, 1), (batch_size, )),
            args=(file_id,)
        )


class ValDataset(tf.data.Dataset):

    def _generator(file_id):
        if((file_id + 1) == n_files_val):
            np.random.shuffle(list_of_file_ids_val)

        # Opening the file
        i_file = list_of_file_ids_val[file_id]
        data, shower_energy_log10 = load_file(i_file, norm)
        num_samples = data.shape[0]
        rand_ids = np.arange(num_samples, dtype=np.int)
        np.random.shuffle(rand_ids)
        n_batches = num_samples // batch_size
        for i_batch in range(n_batches):
            # Reading data (line, record) from the file
            y = shower_energy_log10[rand_ids[i_batch * batch_size:(i_batch + 1) * batch_size]]
            x = data[rand_ids[i_batch * batch_size:(i_batch + 1) * batch_size], :, :, :]
            yield x, y

    def __new__(cls, file_id):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=(tf.dtypes.float64, tf.dtypes.float64),
            output_shapes=((batch_size, 5, 512, 1), (batch_size, )),
            args=(file_id,)
        )

