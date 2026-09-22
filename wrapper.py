import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import copy
# from attention.CBAM import CBAMBlock
# from attention.SEAttention import SEAttention
# from attention.BAM import BAM

def get_attention( preds, temp=1):
        """ preds: Bs*C*W*H """
        N, C, H, W= preds.shape
        value = torch.abs(preds)
        fea_map = value.mean(axis=1, keepdim=True)
        S_attention = F.sigmoid((fea_map/temp).view(N,-1)).view(N, -1,H, W)
        channel_map = value.mean(axis=2,keepdim=False).mean(axis=2,keepdim=False)
        C_attention = F.sigmoid(channel_map/temp).view(N, C,1, 1)
        return S_attention, C_attention

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
        # scale = 2
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

        # self.generation = nn.Sequential(
        #     nn.Conv2d(ch_num, ch_num//scale, kernel_size=1),
        #     nn.LayerNorm([ch_num//scale, feat_H, feat_H]),
        #     nn.GELU(), 

        #     nn.Conv2d(ch_num//scale, ch_num//scale, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(ch_num//scale), 
        #     nn.GELU(), 
        #     nn.Conv2d(ch_num//scale, ch_num//scale, kernel_size=3, padding=1),
            
        #     nn.Conv2d(ch_num//scale, ch_num, kernel_size=1),
        #     nn.LayerNorm([ch_num, feat_H, feat_H]),
        #     nn.GELU(),  
        #     )

        self.aligment2 =nn.Sequential(
            nn.Conv2d(num_channels2,out_channels=num_channels4, kernel_size=4,stride=4),
            )

        self.aligment1 =nn.Sequential(
            nn.Conv2d(num_channels1,out_channels=num_channels4, kernel_size=8,stride=8),)
        
        self.aligment3 =nn.Sequential(
            nn.Conv2d(num_channels3,out_channels=num_channels4, kernel_size=2,stride=2),)

        self.aux_head1 = nn.Sequential( 
            nn.AdaptiveAvgPool2d((1,1)),
            View(-1, num_in),
            nn.Linear(num_in, num_classes),
            )

        self.aux_head2 = nn.Sequential( 
            nn.AdaptiveAvgPool2d((1,1)),
            View(-1, num_in),
            nn.Linear(num_in, num_classes),
            )
        
        self.aux_head3 = nn.Sequential( 
            nn.AdaptiveAvgPool2d((1,1)),
            View(-1, num_in),
            nn.Linear(num_in, num_classes),
            )
    def forward(self,ori_feat):
        feat3 = ori_feat[-2].clone()
        N, C, H, W= feat3.shape
        alg_feat = self.aligment2(ori_feat[2]) 
        S_attention, C_attention = get_attention(feat3)       
        #####ch gen###### 
        # mat = torch.where(S_attention >= 0.55,0,1)  ##0.05
        mat = torch.where(C_attention >= 0.65,0,1)  ##0.05
        sp_feat = torch.mul(alg_feat, mat)
        sp_gen = self.sp_generation(sp_feat)  
        sp_gen = self.ch_generation(sp_feat) 
        f_out = self.aux_head1(sp_gen)

        return sp_gen,f_out

class SepConv(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            # nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=1, bias=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=1, bias=False),
            # nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)

class CLSD(nn.Module):
    def __init__(self, channel0, channel1, channel2,channel3,num_in,num_class):
        super(CLSD, self).__init__()
        
        self.fc = nn.Linear(num_in,num_class)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # self.scala0 = nn.Sequential(
        #     SepConv(
        #         channel_in=channel0,
        #         channel_out=channel1,
        #     ),  
        # )

        self.scala1 = nn.Sequential(
            SepConv(
                channel_in=channel1,
                channel_out=channel2,
            ),  
        )

        self.scala2 = nn.Sequential(
            SepConv(
                channel_in=channel2,
                channel_out=channel3,
            ),  
        )

    def forward(self, x0, x, y,z,z1):
        bs = x0.size(0) // 2 

        level_x0 = self.scala1(x)
        s_att,c_att = get_attention(y)
        s_feat = level_x0*s_att
        c_feat = level_x0*c_att
        level_x0 = s_feat + c_feat
        level_x = self.scala2(level_x0)
  
        level_x1 = level_x[bs:]
        out = self.avgpool(level_x1)
        out1_embd = out.view(out.size(0), -1)
        out1 = self.fc(out1_embd)
        
        level_x2 = level_x[:bs]
        out2 = self.avgpool(level_x2)
        out2_embd = out2.view(out2.size(0), -1)
        out2 = self.fc(out2_embd)
        
        return out1,out2
        
    # def forward(self, x0, x, y,z,z1):
    #     bs = x0.size(0) // 2 
    #     level_x0 = self.scala0(x0)
    #     level_x0 = self.scala1(level_x0)
    #     # s_att,c_att = get_attention(y)
    #     # s_feat = level_x0*s_att
    #     # c_feat = level_x0*c_att
    #     # level_x0 = s_feat + c_feat
    #     level_x = self.scala2(level_x0)
    #     s_att,c_att = get_attention(z)
    #     s_feat = level_x*s_att
    #     c_feat = level_x*c_att
    #     level_x = s_feat + c_feat
       
    #     level_x1 = level_x[bs:]
    #     out = self.avgpool(level_x1)
    #     out1_embd = out.view(out.size(0), -1)
    #     out1 = self.fc(out1_embd)
        
    #     level_x2 = level_x[:bs]
    #     out2 = self.avgpool(level_x2)
    #     out2_embd = out2.view(out2.size(0), -1)
    #     out2 = self.fc(out2_embd)
        
    #     return out1,out2
    

    

    
    
   

    
    
    
