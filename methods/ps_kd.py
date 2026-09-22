import os
import sys
import time
import math

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init


class Custom_CrossEntropy_PSKD(nn.Module):
    def __init__(self):
        super(Custom_CrossEntropy_PSKD, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

    def forward(self, output, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(output)
        loss = (- targets * log_probs).mean(0).sum()
        return loss     

criterion_CE_pskd = Custom_CrossEntropy_PSKD().cuda()
class KL_Loss2(nn.Module):
    def __init__(self, temperature=4):
        super(KL_Loss2, self).__init__()
        self.T = temperature

    def forward(self, output_batch, teacher_outputs):
        # output_batch  -> B X num_classes
        # teacher_outputs -> B X num_classes

        # loss_2 = -torch.sum(torch.sum(torch.mul(F.log_softmax(teacher_outputs,dim=1), F.softmax(teacher_outputs,dim=1)+10**(-7))))/teacher_outputs.size(0)
        # print('loss H:',loss_2)

        output_batch = F.log_softmax(output_batch / self.T, dim=1)
        # teacher_outputs = F.softmax(teacher_outputs / self.T, dim=1) + 10 ** (-7)
        teacher_outputs = teacher_outputs + 10 ** (-7)
        # loss = self.T * self.T * nn.KLDivLoss(reduction='batchmean')(output_batch, teacher_outputs)

        # Same result KL-loss implementation
        loss = self.T * self.T * torch.sum(torch.mul(teacher_outputs, torch.log(teacher_outputs) - output_batch))/teacher_outputs.size(0)
        return loss

# criterion_CE_pskd = KL_Loss2(temperature=1).cuda()

class KL_Loss(nn.Module):
    def __init__(self, temperature=4):
        super(KL_Loss, self).__init__()
        self.T = temperature

    def forward(self, output_batch, teacher_outputs):
        # output_batch  -> B X num_classes
        # teacher_outputs -> B X num_classes

        # loss_2 = -torch.sum(torch.sum(torch.mul(F.log_softmax(teacher_outputs,dim=1), F.softmax(teacher_outputs,dim=1)+10**(-7))))/teacher_outputs.size(0)
        # print('loss H:',loss_2)

        output_batch = F.log_softmax(output_batch / self.T, dim=1)
        teacher_outputs = F.softmax(teacher_outputs / self.T, dim=1) + 10 ** (-7)

        # loss = self.T * self.T * nn.KLDivLoss(reduction='batchmean')(output_batch, teacher_outputs)

        # Same result KL-loss implementation
        loss = self.T * self.T * torch.sum(torch.mul(teacher_outputs, torch.log(teacher_outputs) - output_batch))/teacher_outputs.size(0)
        return loss

criterion_CE_hskd = KL_Loss2(temperature=1).cuda()
criterion_KD_hskd = KL_Loss(temperature=4).cuda()

def PSKD(net, inputs, targets, input_indices, epoch, all_predictions, num_classes, args):
    alpha_t = args.alpha_T * ((epoch + 1) / args.epochs)
    alpha_t = max(0, alpha_t)

    targets_numpy = targets.cpu().detach().numpy()
    identity_matrix = torch.eye(num_classes) 
    targets_one_hot = identity_matrix[targets_numpy]

    if epoch == 0:
        all_predictions[input_indices] = targets_one_hot

    soft_targets = ((1 - alpha_t) * targets_one_hot) + (alpha_t * all_predictions[input_indices])
    soft_targets = soft_targets.cuda()

    outputs = net(inputs)
    softmax_output = F.softmax(outputs, dim=1) 
    loss = criterion_CE_pskd(outputs, soft_targets)

    all_predictions[input_indices] = softmax_output.cpu().detach()

    return outputs, loss

def HSD(logit,targets_a, targets_b, epoch, last_out, num_classes,index,lam,args):
    alpha_t = args.alpha_T * ((epoch + 1) / args.epochs)
    alpha_t = max(0, alpha_t)
    ######last_out#####
    last_out_a = last_out
    last_out_b = last_out[index]
    last_out_a = F.softmax(last_out_a, dim=1) 
    last_out_b = F.softmax(last_out_b, dim=1) 
    ######last_out#####

    targets_a_numpy = targets_a.cpu().detach().numpy()
    identity_matrix = torch.eye(num_classes) 
    targets_one_hot_a = identity_matrix[targets_a_numpy]
    soft_targets_a = ((1 - alpha_t) * targets_one_hot_a) + (alpha_t * last_out_a)
    soft_targets_a = soft_targets_a.cuda()

    targets_b_numpy = targets_b.cpu().detach().numpy()
    identity_matrix_b = torch.eye(num_classes) 
    targets_one_hot_b = identity_matrix_b[targets_b_numpy]
    soft_targets_b = ((1 - alpha_t) * targets_one_hot_b) + (alpha_t * last_out_b)
    soft_targets_b = soft_targets_b.cuda()
    
    # print(logit,soft_targets_a)
    loss = lam * criterion_CE_pskd(logit, soft_targets_a) + (1 - lam) * criterion_CE_pskd(logit,soft_targets_b)
   
    return loss


#vgg16,res18,shv2
cos_max = 0.9  #0.9
cos_min = 0.0  #0.0
PI = math.acos(-1.0)

def PSDTSKD(net, inputs, targets, input_indices, epoch, all_predictions,b1_predictions,b2_predictions,b3_predictions,num_classes,criterion_div,args):
    
    targets_numpy = targets.cpu().detach().numpy()
    identity_matrix = torch.eye(num_classes) 
    targets_one_hot = identity_matrix[targets_numpy]

    if epoch == 0:
        all_predictions[input_indices] = targets_one_hot
        b1_predictions[input_indices] = targets_one_hot
        b2_predictions[input_indices] = targets_one_hot
        b3_predictions[input_indices] = targets_one_hot

    ########calculate beta############
    ratio = 1.0 * epoch / args.epochs
    scale = (math.cos(ratio * PI) + 1.) / 2
    momentum_label_final = cos_min
    momentum_label_range = cos_max - cos_min
    beta_t = scale * momentum_label_range + momentum_label_final
    #################################
 
    soft_targets = (beta_t * targets_one_hot) + ((1-beta_t) * all_predictions[input_indices])
    soft_targets = soft_targets.cuda()

    soft_b1_targets = (beta_t * targets_one_hot) + ((1-beta_t) * b1_predictions[input_indices])
    soft_b1_targets = soft_b1_targets.cuda()

    soft_b2_targets = (beta_t * targets_one_hot) + ((1-beta_t) * b2_predictions[input_indices])
    soft_b2_targets = soft_b2_targets.cuda()

    soft_b3_targets = (beta_t * targets_one_hot) + ((1-beta_t) * b3_predictions[input_indices])
    soft_b3_targets = soft_b3_targets.cuda()
   
    outputs,b1_output,b2_output,b3_output = net(inputs)
    softmax_output = F.softmax(outputs, dim=1) 
    b1_softmax_out = F.softmax(b1_output, dim=1)
    b2_softmax_out = F.softmax(b2_output, dim=1)
    b3_softmax_out = F.softmax(b3_output, dim=1)
    
    loss_ce =  3.0 * criterion_CE_hskd(outputs, soft_targets)  #3.0
    loss_ce += 1.0 * criterion_CE_hskd(b1_output, soft_b1_targets)
    loss_ce += 1.0 * criterion_CE_hskd(b2_output, soft_b2_targets)
    loss_ce += 1.0 * criterion_CE_hskd(b3_output, soft_b3_targets)
    
    loss_kd = criterion_KD_hskd(b1_output, outputs.detach())
    loss_kd += criterion_KD_hskd(b2_output, outputs.detach())
    loss_kd += criterion_KD_hskd(b3_output, outputs.detach())
    
    all_predictions[input_indices] = softmax_output.cpu().detach()
    b1_predictions[input_indices] = b1_softmax_out.cpu().detach()
    b2_predictions[input_indices] = b2_softmax_out.cpu().detach()
    b3_predictions[input_indices] = b3_softmax_out.cpu().detach()

    return outputs,loss_ce,loss_kd