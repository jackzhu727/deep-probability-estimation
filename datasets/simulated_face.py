import torch
import h5py
from scipy.special import expit
import numpy as np


class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, root, prob_type, mode='train'):
        with h5py.File(root + mode + '_im.h5', 'r') as f:
            self.keys = list(f.keys())
        self.prob_type = prob_type
        self.root = root
        self.mode = mode
        self.proposed_probs = None
        if not hasattr(self, 'img_data'):
            self.open_data()

    def open_data(self):
        self.img_data = h5py.File(self.root + self.mode + '_im.h5', 'r')
        self.target = h5py.File(self.root + "labels_" + self.prob_type + '_' + self.mode + '.h5', 'r')
        self.demo = h5py.File(self.root + self.mode + '_label.h5', 'r')

        # call different scenarios
        if self.prob_type == 'unif':
            self.prob_sim_func = lambda x: x
        elif self.prob_type == 'sig':
            self.prob_sim_func = lambda x: 1.0 - expit((x - 0.29) * 25)
        elif self.prob_type == 'scaled':
            self.prob_sim_func = lambda x: x / 2.5
        elif self.prob_type == 'mid':
            self.prob_sim_func = lambda x: x / 3.0 + 0.35
        elif self.prob_type == 'step':
            self.prob_sim_func = lambda x: (x < 0.2) * 0.1 + ((x >= 0.2) & (x < 0.4)) * 0.3 + \
                    ((x >= 0.4) & (x < 0.6)) * 0.5 + ((x >= 0.6) & (x < 0.8)) * 0.7 + (x >= 0.8) * 0.9
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        data = (torch.tensor(self.img_data[self.keys[index]]).clone().permute(2, 0, 1)) / 255
        target = torch.tensor(np.array(self.target[self.keys[index]])).clone()
        age = torch.tensor(self.demo[self.keys[index]][0, 0])
        target_prob = self.prob_sim_func(torch.minimum(age / 100.0, torch.tensor(1.)))
        if self.proposed_probs is not None:
            probs = self.proposed_probs[index]
        else:
            probs = 0
        return data, target, probs, index, target_prob

    def __len__(self):
        return len(self.keys)
