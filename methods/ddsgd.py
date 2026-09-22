import os
import sys
import time
import math

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init

# def DDGSD(net, inputs, targets, criterion_cls, criterion_div):
#     loss_div = torch.tensor(0.).cuda()
#     loss_cls = torch.tensor(0.).cuda()

#     inputs = torch.cat(inputs, dim=0).cuda()
#     batch_size = inputs.size(0) // 2
#     logit, features = net(inputs, embedding=True)
#     loss_cls += criterion_cls(logit, torch.cat([targets, targets], dim=0)) / 2
#     loss_div += criterion_div(logit[:batch_size], logit[batch_size:].detach())
#     loss_div += criterion_div(logit[batch_size:], logit[:batch_size].detach())
#     loss_div += 5e-4 * (features[:batch_size].mean()-features[batch_size:].mean()) ** 2
#     logit = (logit[batch_size:] + logit[:batch_size]) / 2
#     return logit, loss_cls, loss_div


def DDGSD(net, inputs, targets, criterion_cls, criterion_div):
    loss_div = torch.tensor(0.).cuda()
    loss_cls = torch.tensor(0.).cuda()
    
    inputs = torch.cat(inputs, dim=0).cuda()
    batch_size = inputs.size(0) // 2
    logits, features = net(inputs, feature=True)
    for i in range(len(logits)):
        loss_cls += criterion_cls(logits[i], torch.cat([targets, targets], dim=0)) / 2
        if i != 0:
            loss_div += criterion_div(logits[i], logits[0].detach())
    for i in range(1, len(features)):
        if i != 1:
            loss_div += 0.5 * 0.1 * ((features[i] - features[1].detach()) ** 2).mean()
    
    logit = logits[0]   
    
    loss_div += criterion_div(logit[:batch_size], logit[batch_size:].detach())
    loss_div += criterion_div(logit[batch_size:], logit[:batch_size].detach())
    logit = (logit[batch_size:] + logit[:batch_size]) / 2
    
    return logit, loss_cls, loss_div


