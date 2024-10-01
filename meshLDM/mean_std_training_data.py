import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Uncomment for ES
# encoded_output_dir = "/home/vaen/Desktop/MeshLDM_publication/data/encoded/encoded_only_ES_16"
# mean_dir = "/home/vaen/Desktop/MeshLDM_publication/data/train_data_ES_mean.npy"
# std_dir = "/home/vaen/Desktop/MeshLDM_publication/data/train_data_ES_std.npy"

# Uncomment for ED
encoded_output_dir = "/home/vaen/Desktop/MeshLDM_publication/data/encoded/encoded_only_ED_16"
mean_dir = "/home/vaen/Desktop/MeshLDM_publication/data/train_data_ED_mean.npy"
std_dir = "/home/vaen/Desktop/MeshLDM_publication/data/train_data_ED_std.npy"

with open(encoded_output_dir + '/train/z_data.npy', 'rb') as f:
    z_data_train = torch.tensor(np.load(f))

z_data_train = torch.flatten(z_data_train, start_dim=1)
train_data_mean = torch.mean(z_data_train, dim=0)
train_data_std = torch.std(z_data_train, dim=0)
print(train_data_mean)
print(train_data_std)

with open(mean_dir, 'wb') as f:
        np.save(f, train_data_mean)

with open(std_dir, 'wb') as f:
    np.save(f, train_data_std)
