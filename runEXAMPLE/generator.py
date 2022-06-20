# Imports
import os
import numpy as np

import time
from toolbox import load_file
from constants import datapath, data_filename, label_filename, n_files, n_files_val
from constants import learning_rate, es_patience, norm
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
# -------

np.set_printoptions(precision=4)

# n_files and n_files_val comes from dataset in constants.py
n_files_test = 3

n_files_train = n_files - n_files_val - n_files_test
list_of_file_ids_train = np.arange(n_files_train, dtype=np.int)
list_of_file_ids_val = np.arange(n_files_train, n_files_train + n_files_val, dtype=np.int)
list_of_file_ids_test = np.arange(n_files_train + n_files_val, n_files, dtype=np.int)
n_events_per_file = 100000

#dataset
class Prepare_Dataset(Dataset):
    def __init__(self, file_ids, points = 10000, transform=None, target_transform=None): # not using the last two inputs
        data, shower_energy_log10 = load_file(file_ids[0],norm)
        # Then load rest of files
        if len(file_ids) > 1:
            for file_id in file_ids:
                if file_id != file_ids[0]:
                    data_tmp, shower_energy_log10_tmp = load_file(file_id)
                    data = np.concatenate((data, data_tmp))
                    shower_energy_log10 = np.concatenate((shower_energy_log10, shower_energy_log10_tmp))
        
        print("Total data points: ",shower_energy_log10.shape[0])
        # randomly choose the points in a file 
        idx = np.random.choice(shower_energy_log10.shape[0], size=points, replace=False)

        # swap the axes since inputs shape in torch is (batch, channel, input dimension1, input dimension2)
        data = np.swapaxes(data,1,3)
        data = np.swapaxes(data,2,3)

        shower_energy_log10 = np.expand_dims(shower_energy_log10,1)
        
        data = data[idx,:]
        shower_energy_log10 = shower_energy_log10[idx,:]

        self.data = torch.from_numpy(data)
        self.labels = torch.from_numpy(shower_energy_log10)
                   
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_i = self.data[idx]
        label_i = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return data_i, label_i


## Model
conv2D_filter_size = 5
pooling_size = 4
amount_Conv2D_layers_per_block = 3 
amount_Conv2D_blocks = 4
conv2D_filter_amount = 32

from torch.optim.lr_scheduler import ReduceLROnPlateau 
import pytorch_lightning as pl
import jammy_flows

class E_Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.cnn0 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, conv2D_filter_size), padding='same')
        self.cnn1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, conv2D_filter_size), padding='same')
        
        self.cnn2_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, conv2D_filter_size), padding='same')
        self.cnn2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, conv2D_filter_size), padding='same')

        self.cnn3_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, conv2D_filter_size), padding='same')
        self.cnn3_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, conv2D_filter_size), padding='same')
        
        self.cnn4_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, conv2D_filter_size), padding='same')
        self.cnn4_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, conv2D_filter_size), padding='same')

        self.avgpool = nn.AvgPool2d(kernel_size=(1, pooling_size))
        
        self.bn1 = nn.BatchNorm2d(256, eps = 0.001, momentum = 0.99, affine=True)

        # self.pdf = jammy_flows.pdf("e1", "gg", conditional_input_dim=2560, hidden_mlp_dims_sub_pdfs="1024-1024-512-256-128") # high loss, bad resol
        # self.pdf = jammy_flows.pdf("e1", "gg", conditional_input_dim=2560, hidden_mlp_dims_sub_pdfs="512-128") # low loss, tf like result
        # self.pdf = jammy_flows.pdf("e1", "gg", conditional_input_dim=2560, hidden_mlp_dims_sub_pdfs="512-256-128") # bad sigma result
        # self.pdf = jammy_flows.pdf("e1", "gggg", conditional_input_dim=2560, hidden_mlp_dims_sub_pdfs="512-256-128") #  bad sigma result
        self.pdf = jammy_flows.pdf("e1", "gg", conditional_input_dim=2560, hidden_mlp_dims_sub_pdfs="128") #best

    def forward(self, x):
        x = self.cnn0(x)
        x = torch.relu(x)
        x = self.cnn1(x)
        x = torch.relu(x)
        x = self.cnn1(x)
        x = torch.relu(x)
        x = self.avgpool(x)
        
        x = self.cnn2_1(x)
        x = torch.relu(x)
        x = self.cnn2_2(x)
        x = torch.relu(x)
        x = self.cnn2_2(x)
        x = torch.relu(x)
        x = self.avgpool(x)
        
        x = self.cnn3_1(x)
        x = torch.relu(x)
        x = self.cnn3_2(x)
        x = torch.relu(x)
        x = self.cnn3_2(x)
        x = torch.relu(x)
        x = self.avgpool(x)
        
        x = self.cnn4_1(x)
        x = torch.relu(x)
        x = self.cnn4_2(x)
        x = torch.relu(x)
        x = self.cnn4_2(x)
        x = torch.relu(x)
        x = self.avgpool(x)
        
        x = self.bn1(x)
        out =  torch.flatten(x, 1) #x.view(-1,2560) # torch.flatten(x, 1) ##
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, eps=1e-7) 
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=es_patience-3, min_lr=0.000001, verbose=1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        conv_out = self.forward(x)
#         print(torch.max(conv_out),torch.min(conv_out))
        log_pdf, _,_= self.pdf(y, conditional_input=conv_out)
        loss=-log_pdf.mean()
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, logger=True, on_step=False)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        conv_out = self.forward(x)
        log_pdf, _,_= self.pdf(y, conditional_input=conv_out)
        loss=-log_pdf.mean()
        self.log('val_loss', loss, prog_bar=True, on_epoch = True, logger=True)
    

