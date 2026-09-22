from __future__ import print_function, division
from cProfile import label

import sys
import time
import torch
import torch.nn as nn
from .util import AverageMeter, accuracy, reduce_tensor
import math
from scipy.stats import qmc
import numpy as np
import torch.nn.functional as F
import lightly.loss as ligtloss
from distiller_zoo.PDC import PDCLoss
from models.util import Normalize
import torchvision.transforms as T

def Generate_persuaded(num,dim,l_bounds=[-0.01],u_bounds=[0.01]):
  sampler = qmc.Halton(d=dim, scramble=False)
  sample = sampler.random(num)
  halton_data = qmc.scale(sample, l_bounds, u_bounds)
  return halton_data

def freeze_model(model):
    for name, param in model.named_parameters():
        param.requires_grad = False  
        if name[0] == 'f':
            param.requires_grad = True
    return model

def train_distill(epoch, train_loader, mlp_net, cos_value,module_list, criterion_list, optimizer,opt):
    """one epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    
    if opt.have_mlp:
        mlp_net.train()

    # set teacher as eval()
    module_list[-1].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]
    criterion_kd2 = criterion_list[-1]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    n_batch = len(train_loader) if opt.dali is None else (train_loader._size + opt.batch_size - 1) // opt.batch_size  ##782
    end = time.time()
    dclw_loss = ligtloss.DCLWLoss(temperature=0.07)
    l2_norm =  Normalize(2)
    ###len(train_loader) 782
    for idx, data in enumerate(train_loader):
        if opt.dali is None:
            if opt.distill in ['crd']:
                images, labels, index, contrast_idx = data
            else:
                images, labels = data
        else:
            images, labels = data[0]['data'], data[0]['label'].squeeze().long()
        
        if opt.distill == 'semckd' and images.shape[0] < opt.batch_size:
            continue
    
        if opt.gpu is not None:
            images = images.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
        if torch.cuda.is_available():
            labels = labels.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
            if opt.distill in ['crd']:
                index = index.cuda()
                contrast_idx = contrast_idx.cuda()

        # ===================forward=====================
        feat_s, logit_s = model_s(images, is_feat=True)
        with torch.no_grad():
            feat_t, logit_t = model_t(images, is_feat=True)
            feat_t = [f.detach() for f in feat_t]

        if opt.have_mlp:
            temp = mlp_net(logit_t, logit_s, cos_value)  # (teacher_output, student_output)
            temp = opt.t_start + opt.t_end * torch.sigmoid(temp)
            temp = temp.cuda()
        else:
            temp = (opt.kd_T * torch.ones(1)).cuda()


        cls_t = model_t.module.get_feat_modules()[-1] if opt.multiprocessing_distributed else model_t.get_feat_modules()[-1]
        cls_s = model_s.module.get_feat_modules()[-1] if opt.multiprocessing_distributed else model_s.get_feat_modules()[-1]

        # cls + kl div
        loss_cls = criterion_cls(logit_s, labels)
        loss_div = criterion_div(logit_s, logit_t)  ##lsrrl 没有这一项损失
       
        ############
        # other kd loss
        if opt.distill == 'kd':
            ####输入作扰动#######  
            loss_kd = loss_div
            loss_kd2 = 0.0  
            # loss_kd = criterion_kd(logit_s,logit_t)
            # loss_kd2 = 0.            
        elif opt.distill == 'qkd':
            #########################################
            s3 = feat_t[-1].clone()
            halton_data3 = Generate_persuaded(s3.size(0),s3.size(-1))  ##针对feature作扰动 8,8
            halton_data3 = torch.tensor(halton_data3).type(torch.cuda.FloatTensor).cuda()
            s3 = halton_data3 + s3
            logit_t1 = cls_t(s3)
            ###random perturbations##
            # p_data = torch.empty((s3.size(0), s3.size(-1)), dtype=torch.float32).uniform_(-0.01,0.01)
            # s3 = p_data.cuda() + s3
            # logit_t1 = cls_t(s3)
            tran_s,tran_t,pred_feat_s = module_list[1](feat_s[-1],s3,cls_t)      
            loss_kd = criterion_kd(pred_feat_s,logit_t1)
            loss_kd2 = dclw_loss(tran_s,tran_t)

        elif opt.distill == 'crd':
           ###最后一层增加扰动75.77  only_t:75.7
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = loss_div
            loss_kd2 = criterion_kd(f_s, f_t, index, contrast_idx) # criterion_kd(f_s, f_t, index, contrast_idx)
            
        elif opt.distill == 'hint':
            f_s, f_t = module_list[1](feat_s[opt.hint_layer], feat_t[opt .hint_layer])
            loss_kd =   loss_div
            loss_kd2 =  criterion_kd(f_s, f_t)
        elif opt.distill == 'attention': 
            # include 1, exclude -1.
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = loss_div
            loss_kd2 = sum(loss_group)  #1000
        elif opt.distill == 'similarity':
            g_s = [feat_s[-2]]
            g_t = [feat_t[-2]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = loss_div
            loss_kd2 = sum(loss_group)
        elif opt.distill == 'vid':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
            loss_kd = loss_div 
            loss_kd2 = sum(loss_group)
        elif opt.distill == 'semckd':
            s_value, f_target, weight = module_list[1](feat_s[1:-1], feat_t[1:-1])
            loss_kd = loss_div
            loss_kd2 = criterion_kd(s_value, f_target, weight)                                                 
        elif opt.distill == 'srrl':
            ##########srrl#############################
            s3 = feat_t[-1].clone()
            halton_data3 = Generate_persuaded(s3.size(0),s3.size(-1))  ##针对feature作扰动 8,8
            halton_data3 = torch.tensor(halton_data3).type(torch.cuda.FloatTensor).cuda()
            s3 = halton_data3 + s3
            halton_logit_t = cls_t(s3)
            trans_feat_s, pred_feat_s = module_list[1](feat_s[-1], cls_t)
            loss_kd = criterion_kd(logit_s, logit_t)
            loss_kd2 = criterion_kd(trans_feat_s, feat_t[-1]) + criterion_kd(pred_feat_s, halton_logit_t)

        elif opt.distill == 'simkd':
            trans_feat_s, trans_feat_t, pred_feat_s = module_list[1](feat_s[-2], feat_t[-2], cls_t)
            logit_s = pred_feat_s
            loss_kd = criterion_kd(trans_feat_s, trans_feat_t)  
            loss_kd2 = 0.0
        elif opt.distill == 'dkd': ##ce kd 1:1
            loss_kd = criterion_kd(logit_s, logit_t, labels)
        
        elif opt.distill == 'dist': 
            loss_kd = criterion_kd(logit_s, logit_t)
        ##0.1cls 0.9kd
        elif opt.distill == 'ctkd ': 
            loss_kd = criterion_kd(logit_s, logit_t,temp)
            
        else:
            raise NotImplementedError(opt.distill)

        # loss = 4.0 * loss_cls + 8.0 * loss_kd + 7.0 * loss_kd2 
        # loss = 4.0 * loss_cls + 8.0 * loss_kd  + 8.0 * loss_kd2
        # loss = 4.0 * loss_cls + 8.0 * loss_kd
        # loss = opt.cls * loss_cls + opt.div * loss_kd + opt.beta * loss_kd2 
        loss = 0.1*loss_cls + 0.9*loss_kd
        
        losses.update(loss.item(), images.size(0)) 
        
        # ===================Metrics=====================
        metrics = accuracy(logit_s, labels, topk=(1, 5))
        top1.update(metrics[0].item(), images.size(0))
        top5.update(metrics[1].item(), images.size(0))
        batch_time.update(time.time() - end)
        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  
        ###########################
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'GPU {3}\t'
                  'Time: {batch_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f}\t'
                  'Acc@1 {top1.avg:.3f}\t'
                  'Acc@5 {top5.avg:.3f}'.format(
                epoch, idx, n_batch, opt.gpu, loss=losses, top1=top1, top5=top5,
                batch_time=batch_time))
            sys.stdout.flush()
    return top1.avg, top5.avg, losses.avg

def validate_distill(val_loader, module_list, criterion_list, opt):
    """validation""" 
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    # switch to evaluate mode
    for module in module_list:
        module.eval()
    
    model_s = module_list[0]
    model_t = module_list[-1]
    criterion = criterion_list[0]
    n_batch = len(val_loader) if opt.dali is None else (val_loader._size + opt.batch_size - 1) // opt.batch_size

    with torch.no_grad():
        end = time.time()
        # feat_t, logit_t = model_t(images, is_feat=True)
        # feat_t = [f.detach() for f in feat_t]
        for idx, batch_data in enumerate(val_loader):
            
            if opt.dali is None:
                images, labels = batch_data
            else:
                images, labels = batch_data[0]['data'], batch_data[0]['label'].squeeze().long()

            if opt.gpu is not None:
                images = images.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
            if torch.cuda.is_available():
                labels = labels.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)

            # compute output
            if opt.distill == 'simkd':
                feat_s, _ = model_s(images, is_feat=True)
                feat_t, _ = model_t(images, is_feat=True)
                feat_t = [f.detach() for f in feat_t]
                cls_t = model_t.module.get_feat_modules()[-1] if opt.multiprocessing_distributed else model_t.get_feat_modules()[-1]
                _, _, output = module_list[1](feat_s[-2], feat_t[-2], cls_t)
            else:
                output = model_s(images)
            loss = criterion(output, labels)
            losses.update(loss.item(), images.size(0))

            # ===================Metrics=====================
            metrics = accuracy(output, labels, topk=(1, 5))
            top1.update(metrics[0].item(), images.size(0))
            top5.update(metrics[1].item(), images.size(0))
            batch_time.update(time.time() - end)
            
            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'GPU: {2}\t'
                      'Time: {batch_time.avg:.3f}\t'
                      'Loss {loss.avg:.4f}\t'
                      'Acc@1 {top1.avg:.3f}\t'
                      'Acc@5 {top5.avg:.3f}'.format(
                       idx, n_batch, opt.gpu, batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))
                
    if opt.multiprocessing_distributed:
        # Batch size may not be equal across multiple gpus
        total_metrics = torch.tensor([top1.sum, top5.sum, losses.sum]).to(opt.gpu)
        count_metrics = torch.tensor([top1.count, top5.count, losses.count]).to(opt.gpu)
        total_metrics = reduce_tensor(total_metrics, 1) # here world_size=1, because they should be summed up
        count_metrics = reduce_tensor(count_metrics, 1)
        ret = []
        for s, n in zip(total_metrics.tolist(), count_metrics.tolist()):
            ret.append(s / (1.0 * n))
        return ret

    return top1.avg, top5.avg, losses.avg

def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    n_batch = len(train_loader) if opt.dali is None else (train_loader._size + opt.batch_size - 1) // opt.batch_size

    end = time.time()
    for idx, batch_data in enumerate(train_loader):
        if opt.dali is None:
            images, labels = batch_data
        else:
            images, labels = batch_data[0]['data'], batch_data[0]['label'].squeeze().long()
        
        if opt.gpu is not None:
            images = images.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
        if torch.cuda.is_available():
            labels = labels.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)

        # ===================forward=====================
        output = model(images)
        loss = criterion(output, labels)
        losses.update(loss.item(), images.size(0))

        # ===================Metrics=====================
        metrics = accuracy(output, labels, topk=(1, 5))
        top1.update(metrics[0].item(), images.size(0))
        top5.update(metrics[1].item(), images.size(0))
        batch_time.update(time.time() - end)

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'GPU {3}\t'
                  'Time: {batch_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f}\t'
                  'Acc@1 {top1.avg:.3f}\t'
                  'Acc@5 {top5.avg:.3f}'.format(
                   epoch, idx, n_batch, opt.gpu, batch_time=batch_time,
                   loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()
            
    return top1.avg, top5.avg, losses.avg

def validate_vanilla(val_loader, model, criterion, opt):
    """validation"""
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    n_batch = len(val_loader) if opt.dali is None else (val_loader._size + opt.batch_size - 1) // opt.batch_size

    with torch.no_grad():
        end = time.time()
        for idx, batch_data in enumerate(val_loader):
            
            if opt.dali is None:
                images, labels = batch_data
            else:
                images, labels = batch_data[0]['data'], batch_data[0]['label'].squeeze().long()

            if opt.gpu is not None:
                images = images.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
            if torch.cuda.is_available():
                labels = labels.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
            
            # compute output
            output = model(images)

            loss = criterion(output, labels)
            losses.update(loss.item(), images.size(0))
    
            # ===================Metrics=====================
            metrics = accuracy(output, labels, topk=(1, 5))
            top1.update(metrics[0].item(), images.size(0))
            top5.update(metrics[1].item(), images.size(0))
            batch_time.update(time.time() - end)

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'GPU: {2}\t'
                      'Time: {batch_time.avg:.3f}\t'
                      'Loss {loss.avg:.4f}\t'
                      'Acc@1 {top1.avg:.3f}\t'
                      'Acc@5 {top5.avg:.3f}'.format(
                       idx, n_batch, opt.gpu, batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))
    
    if opt.multiprocessing_distributed:
        # Batch size may not be equal across multiple gpus
        total_metrics = torch.tensor([top1.sum, top5.sum, losses.sum]).to(opt.gpu)
        count_metrics = torch.tensor([top1.count, top5.count, losses.count]).to(opt.gpu)
        total_metrics = reduce_tensor(total_metrics, 1) # here world_size=1, because they should be summed up
        count_metrics = reduce_tensor(count_metrics, 1)
        ret = []
        for s, n in zip(total_metrics.tolist(), count_metrics.tolist()):
            ret.append(s / (1.0 * n))
        return ret

    return top1.avg, top5.avg, losses.avg

def train_binary(epoch, train_loader, module_list, criterion_list, optimizer,opt):
    """one epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    n_batch = len(train_loader) if opt.dali is None else (train_loader._size + opt.batch_size - 1) // opt.batch_size  ##782
    end = time.time()
    dclw_loss = ligtloss.DCLWLoss(temperature=0.1)
    dclw_loss2 = ligtloss.DCLWLoss2(temperature=0.1)
    ###len(train_loader) 782
    for idx, data in enumerate(train_loader):
        if opt.dali is None:
            if opt.distill in ['crd']:
                images, labels, index, contrast_idx = data
            else:
                images, labels = data
        else:
            images, labels = data[0]['data'], data[0]['label'].squeeze().long()
        
        if opt.distill == 'semckd' and images.shape[0] < opt.batch_size:
            continue

        if opt.gpu is not None:
            images = images.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
        if torch.cuda.is_available():
            labels = labels.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
            if opt.distill in ['crd']:
                index = index.cuda()
                contrast_idx = contrast_idx.cuda()

        # ===================forward=====================
        logit_s = model_s(images)
        ############get the feature###################
        backbone = nn.Sequential(*list(model_s.children())[:-1])
        feat_s = backbone(images).flatten(start_dim=1)
        ###############################
        with torch.no_grad():
            logit_t = model_t(images)
            # feat_t = [f.detach() for f in feat_t]
            t_backbone = nn.Sequential(*list(model_t.children())[:-1])
            feat_t = t_backbone(images).flatten(start_dim=1).detach()
          
        cls_t = model_t.module.get_feat_modules()[-1] if opt.multiprocessing_distributed else model_t.get_feat_modules()[-1]
        
        # cls + kl div
        loss_cls = criterion_cls(logit_s, labels)
        loss_div = criterion_div(logit_s, logit_t)  ##lsrrl 没有这一项损失
        # print(logit_s.shape,logit_t.shape)
        ############
        # other kd loss
        if opt.distill == 'kd':
            loss_kd = 0.0
            loss_kd2 = 0.0  
        ####在channel维度进行匹配对其######
        elif opt.distill == 'autokd':
            s3 = feat_t.clone()
            halton_data3 = Generate_persuaded(s3.size(0),s3.size(-1))  ##针对feature作扰动 8,8
            halton_data3 = torch.tensor(halton_data3).type(torch.cuda.FloatTensor).cuda()
            s4 = halton_data3 + s3
            halton_logit_t = cls_t(s4)
            tran_s,tran_t,tran_s2,tran_t2,pred_feat_s = module_list[1](feat_s,s3,s4,cls_t)  
            loss_kd = criterion_kd(pred_feat_s,halton_logit_t)
            loss_kd2 = 0.5*(dclw_loss(tran_s,tran_t)+dclw_loss(tran_s2,tran_t2))
            # loss_kd2 = 0.5*(dclw_loss2(tran_s,tran_t)+dclw_loss2(tran_t2,tran_s2)) #sym_trans
           
        elif opt.distill == 'tkd':
            s3 = feat_t.clone()
            halton_data3 = Generate_persuaded(s3.size(0),s3.size(-1))  ##针对feature作扰动 8,8
            halton_data3 = torch.tensor(halton_data3).type(torch.cuda.FloatTensor).cuda()
            s3 = halton_data3 + s3
            logit_t1 = cls_t(s3)
            tran_s,tran_t,pred_feat_s = module_list[1](feat_s,s3,cls_t)   
            loss_kd = criterion_kd(pred_feat_s,logit_t1)  
            loss_kd2 = dclw_loss(tran_s,tran_t)
        else:
            raise NotImplementedError(opt.distill)

        # loss = opt.cls * loss_cls + opt.div * loss_div + opt.beta * loss_kd
        loss = opt.cls * loss_cls + opt.div * loss_kd + opt.beta * loss_kd2
        losses.update(loss.item(), images.size(0)) 
        
        # ===================Metrics=====================
        metrics = accuracy(logit_s, labels, topk=(1, 5))
        top1.update(metrics[0].item(), images.size(0))
        top5.update(metrics[1].item(), images.size(0))
        batch_time.update(time.time() - end)
        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  
 
        ###########################
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'GPU {3}\t'
                  'Time: {batch_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f}\t'
                  'Acc@1 {top1.avg:.3f}\t'
                  'Acc@5 {top5.avg:.3f}'.format(
                epoch, idx, n_batch, opt.gpu, loss=losses, top1=top1, top5=top5,
                batch_time=batch_time))
            sys.stdout.flush()
    return top1.avg, top5.avg, losses.avg

def validate_binary(val_loader, module_list, criterion_list, opt):
    """validation""" 
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    # switch to evaluate mode
    for module in module_list:
        module.eval()
    
    model_s = module_list[0]
    criterion = criterion_list[0]
    n_batch = len(val_loader) if opt.dali is None else (val_loader._size + opt.batch_size - 1) // opt.batch_size

    with torch.no_grad():
        end = time.time()
        for idx, batch_data in enumerate(val_loader):
            
            if opt.dali is None:
                images, labels = batch_data
            else:
                images, labels = batch_data[0]['data'], batch_data[0]['label'].squeeze().long()

            if opt.gpu is not None:
                images = images.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
            if torch.cuda.is_available():
                labels = labels.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)

            output = model_s(images)
            loss = criterion(output, labels)
            losses.update(loss.item(), images.size(0))

            # ===================Metrics=====================
            metrics = accuracy(output, labels, topk=(1, 5))
            top1.update(metrics[0].item(), images.size(0))
            top5.update(metrics[1].item(), images.size(0))
            batch_time.update(time.time() - end)
            
            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'GPU: {2}\t'
                      'Time: {batch_time.avg:.3f}\t'
                      'Loss {loss.avg:.4f}\t'
                      'Acc@1 {top1.avg:.3f}\t'
                      'Acc@5 {top5.avg:.3f}'.format(
                       idx, n_batch, opt.gpu, batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))
                
    return top1.avg, top5.avg, losses.avg