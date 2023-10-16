# import numpy as np
import torch as th
from torch.utils.data.dataset import Dataset

class VolumeDataset(Dataset):
    def __init__(self,volume,min_x=0.,min_y=0.,min_z=0.,max_x=1.,max_y=1.,max_z=1.,oversample=16):

        self.volume = volume
        self.n_voxels = volume.shape[0]

        self.min_bb = th.tensor([min_x,min_y,min_z],dtype=th.float)
        self.max_bb = th.tensor([max_x,max_y,max_z],dtype=th.float)

        self.diag = self.max_bb-self.min_bb

        self.max_dim = th.max(self.diag)
        self.scales = self.diag/self.max_dim

        self.oversample = oversample
    #

    def __len__(self):
        return self.n_voxels
    #

    def __getitem__(self, index):
        random_positions = self.volume[th.randint(self.n_voxels,(self.oversample,))]
        normalized_positions = 2.0 * ( (random_positions - self.min_bb.unsqueeze(0).unsqueeze(0)) / \
                                      (self.max_bb-self.min_bb).unsqueeze(0).unsqueeze(0) ) - 1.0
        # normalized_positions = self.scales.unsqueeze(0).unsqueeze(0)*normalized_positions
        return random_positions, normalized_positions
    #
#
