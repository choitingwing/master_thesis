
import numpy as np
import torch
import pytorch_lightning as pl
from torchsummary import summary
from generator import E_Model
# create a model
model_print = E_Model()
print(model_print)
input_shape = (1,5,512)
summary(model_print.cuda(), input_shape)
