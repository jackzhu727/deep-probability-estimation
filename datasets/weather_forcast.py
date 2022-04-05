import torch
import torch.nn as nn
import numpy as np


import h5py
class Weather_Dataset(torch.utils.data.Dataset):
    def __init__(self, filename,key_filename,sparse_key_index,label_key):
        super().__init__()
        self.h5f = h5py.File(filename, "r")
        self.h5f_labels=h5py.File(label_key, "r")
        self.keys=np.load(key_filename)
        #keys in data chosen for training 
        self.sparse_index=np.load(sparse_key_index)
        self.proposed_probs=None

    def __len__(self):
        return int(len(self.sparse_index))

    def __getitem__(self, index):
        ind=self.sparse_index[index]
        key = self.keys[ind]
        data = np.array(self.h5f[key])
        data_t=torch.Tensor(data[None,300:600,300:600])
        ind+=3
        for i in range(2):
            key = self.keys[ind]
            data = np.array(self.h5f[key])
            data_t=torch.cat((data_t,torch.Tensor(data[None,300:600,300:600])))
            ind+=3
        key = self.keys[ind+10]
        pred_t=torch.tensor(np.array(self.h5f_labels[key]))
        if self.proposed_probs is not None:
            probs = self.proposed_probs[index]
        else:
            probs = 0
        return data_t, pred_t[None,...],probs, index, 0*probs
