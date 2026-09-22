import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import qmc
import numpy as np

class clwrapper(nn.Module):

    def __init__(self, ch_in):

        super(clwrapper, self).__init__()

        self.num_channels = ch_in  
        self.head_proj = nn.Sequential(
            nn.Linear(self.num_channels, self.num_channels),
            nn.BatchNorm1d(self.num_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_channels, 128)
        )

    def forward(self, feats):

        proj = self.head_proj(feats)
        proj = F.normalize(proj, dim=1) 
        return proj
