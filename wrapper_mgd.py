import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import copy
# from attention.CBAM import CBAMBlock
# from attention.SEAttention import SEAttention
# from attention.BAM import BAM
# from lightly.utils import deactivate_requires_grad, update_momentum
import numpy as np

def mask_channel(feat,r=0.15):
    N, C, H, W = feat.shape
    device = feat.device
    masked_feat = feat.clone()
    mat = torch.rand((N,C,1,1)).to(device)
    mat = torch.where(mat< r,0,1)  ##0.05
    masked_feat = torch.mul(masked_feat, mat)
    return masked_feat

def mask_spatial(feat,r=0.15):
    N, C, H, W = feat.shape
    device = feat.device
    masked_feat = feat.clone()
    mat = torch.rand((N,1,H,W)).to(device)
    mat = torch.where(mat< r,0,1)  ##0.05
    masked_feat = torch.mul(masked_feat, mat)
    return masked_feat

class View(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape     
    def forward(self, x):
        return x.view(*self.shape)

class wrapper(nn.Module):
    def __init__(self, feat_H,num_channels1,num_channels2,num_channels3,num_channels4,num_in,num_classes): 
        super(wrapper, self).__init__()
        ch_num = num_channels4
        
        self.sp_generation = nn.Sequential(
            nn.Conv2d(ch_num, ch_num, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch_num), 
            nn.GELU(), 
            nn.Conv2d(ch_num, ch_num, kernel_size=3, padding=1),
            )

        self.ch_generation = nn.Sequential(
            nn.Conv2d(ch_num, ch_num, kernel_size=1),
            nn.LayerNorm([ch_num, feat_H, feat_H]),
            nn.GELU(),  
            nn.Conv2d(ch_num, ch_num, kernel_size=1),
            )
        ##resnet k=2 s=2 others=4
        self.aligment2 =nn.Sequential(
            nn.Conv2d(num_channels2,out_channels=num_channels4, kernel_size=4,stride=4),
            )

        self.aux_head1 = nn.Sequential( 
            nn.AdaptiveAvgPool2d((1,1)),
            View(-1, num_in),
            nn.Linear(num_in, num_classes),
            )

    def forward(self,ori_feat):
        #####默认是1##########
        feat2 = ori_feat.clone()
        feat2 = self.aligment2(feat2) 
        mask_feat = mask_channel(feat2,r=0.15)
        mask_feat = self.sp_generation(mask_feat)
        mask_feat = self.ch_generation(mask_feat)
        f_out1 = self.aux_head1(mask_feat)
          
        return f_out1

   
    
    
