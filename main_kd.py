import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import os
import shutil
import argparse
import numpy as np

from methods import *
import models
import torchvision
import torchvision.transforms as transforms
from utils import cal_param_size, cal_multi_adds, correct_num, adjust_lr, DistillKL, AverageMeter,accuracy
from dataloader.dataloaders import load_dataset
from dataloader.my_dataloader import get_dataloader

import time
import math
from wrapper import wrapper,CLSD

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--dataset', default='CIFAR-100', type=str, help='Dataset')
parser.add_argument('--data', default='/home/devuser/Downloads/data/', type=str, help='Dataset directory')
parser.add_argument('--arch', default='CIFAR_ResNet18', type=str, help='network architecture')
parser.add_argument('--init_lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--warmup-epoch', default=5, type=int, help='warmup epoch')
parser.add_argument('--lr_type', default='multistep', type=str, help='learning rate strategy')
parser.add_argument('--data-aug', default='None', type=str, help='extra data augmentation')
parser.add_argument('--milestones', default=[100, 150], type=list, help='milestones for lr-multistep')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--num_workers', type=int, default=8, help='number of workers')
parser.add_argument('--method', default='cross_entropy', type=str, help='method')
parser.add_argument('--gpu-id', type=str, default='0')
parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--weight-cls', type=float, default=1, help='weight for cross-entropy loss')
parser.add_argument('--weight-kd', type=float, default=1, help='weight for KD loss')
parser.add_argument('--T', type=float, default=4, help='temperature for KD distillation')
parser.add_argument('--omega', default=0.5, type=float, help='ensembling weight in BAKE')
parser.add_argument('--intra-imgs', '-m', default=3, type=int, help='intra-class images, M in BAKE')
parser.add_argument('--alpha-T', default=0.8, type=float, help='alpha T in PS-KD')
parser.add_argument('--manual_seed', type=int, default=0)
parser.add_argument('--checkpoint_dir', default='./checkpoint/', type=str, help='saved checkpoint directory')
parser.add_argument('--eval-checkpoint', default='./checkpoint/resnet18_best.pth', type=str, help='evaluate checkpoint directory')
parser.add_argument('--resume-checkpoint', default='./checkpoint/resnet18.pth', type=str, help='resume checkpoint directory')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--evaluate', '-e', action='store_true', help='evaluate model')
parser.add_argument('--kl_T', type=float, default=4.0, help='temperature for KD distillation')
parser.add_argument('--dl_T', type=float, default=8.0, help='temperature for KD distillation')
parser.add_argument('-t', '--trial', type=str, default='0', help='the experiment id')

args = parser.parse_args()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

info = str(os.path.basename(__file__).split('.')[0]) \
          + '_dataset_' + args.dataset \
          + '_arch_' + args.arch \
          + '_method_' + args.method \
          + '_data_aug_' + args.data_aug \
          + '_' + str(args.manual_seed) \
          + '_' + str(args.trial)

args.checkpoint_dir = os.path.join(args.checkpoint_dir, info)
if not os.path.isdir(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)
args.log_txt = os.path.join(args.checkpoint_dir, info + '.txt')

print('dir for checkpoint:', args.checkpoint_dir)
with open(args.log_txt, 'a+') as f:
    f.write("==========\nArgs:{}\n==========".format(args) + '\n')
        

np.random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)

trainloader, valloader = load_dataset(args=args)

if args.method == 'dlb':
    trainloader, valloader = get_dataloader(args=args)

print('Dataset: '+ args.dataset)
if args.dataset.startswith('CIFAR'):
    num_classes = len(set(trainloader.dataset.targets))
elif args.dataset.startswith('tinyimagenet'):
    num_classes = 200
else:
    num_classes = len(set(trainloader.dataset.classes))

print('Number of train dataset: ' ,len(trainloader.dataset))
print('Number of validation dataset: ' ,len(valloader.dataset))
print('Number of classes: ' , num_classes)
if args.method == 'dlb' or args.method == 'PSKD' or args.method == 'psmgf':
    if args.dataset == "cifar100" or args.dataset == 'tinyimagenet':
        C, H, W =  3,32,32
    else:
        C, H, W =  3,224,224
else:    
    C, H, W =  trainloader.dataset[0][0][0].size() if isinstance(trainloader.dataset[0][0], list) is True  else trainloader.dataset[0][0].size()
   
# --------------------------------------------------------------------------------------------
# Model
print('==> Building model..')
model = getattr(models, args.arch)

if args.method == 'virtual_softmax':
    net = model(num_classes=num_classes, is_bias=False).cuda()
else:
    net = model(num_classes=num_classes).cuda()
 
cudnn.benchmark = True

class ConvReg(nn.Module):
    """Convolutional regression for FitNet (feature-map layer)"""
    def __init__(self, s_shape, t_shape):
        super(ConvReg, self).__init__()
        _, s_C, s_H, s_W = s_shape
        _, t_C, t_H, t_W = t_shape
        self.s_H = s_H
        self.t_H = t_H
        if s_H == 2 * t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, stride=2, padding=1)
        elif s_H * 2 == t_H:
            self.conv = nn.ConvTranspose2d(s_C, t_C, kernel_size=4, stride=2, padding=1)
        elif s_H >= t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=(1+s_H-t_H, 1+s_W-t_W))
        else:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, padding=1, stride=1)
        self.bn = nn.BatchNorm2d(t_C)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, t):
        x = self.conv(x)
        if self.s_H == 2 * self.t_H or self.s_H * 2 == self.t_H or self.s_H >= self.t_H:
            return self.relu(self.bn(x)), t
        else:
            return self.relu(self.bn(x)), F.adaptive_avg_pool2d(t, (self.s_H, self.s_H))

class SmiLoss(nn.Module):
    """
    KL-Divergence symmetric loss between two distributions
    Used in here for knowledge distillation
    """
    def __init__(self):
        super(SmiLoss, self).__init__()
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def forward(self, zxs, zys, zxt, zyt, temperature=0.1):
        sim_s = self.similarity_f(zxs.unsqueeze(1), zys.unsqueeze(0)) / temperature
        sim_s = F.softmax(sim_s, dim=1)
        sim_t = self.similarity_f(zxt.unsqueeze(1), zyt.unsqueeze(0)) / temperature
        sim_t = F.softmax(sim_t, dim=1)
        loss_s = F.kl_div(sim_s.log(), sim_t.detach())
        return loss_s

class MYKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self):
        super(MYKL, self).__init__()

    def forward(self, y_s, y_t,temperature=4):
        p_s = F.log_softmax(y_s / temperature, dim=1)
        p_t = F.softmax(y_t / temperature, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (temperature ** 2)
        return loss
    
# Training
def train(epoch, criterion_list, optimizer):
    train_loss = AverageMeter('train_loss', ':.4e')
    train_loss_cls = AverageMeter('train_loss_cls', ':.4e')
    train_loss_div = AverageMeter('train_loss_div', ':.4e')
    top1_num = 0
    top5_num = 0
    total = 0

    if epoch >= args.warmup_epoch:
        lr = adjust_lr(optimizer, epoch, args)
    start_time = time.time()
    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]

    net.train()
    model_t.eval()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        batch_start_time = time.time()
 
        if isinstance(inputs, list) is False:
            inputs = inputs.cuda()
            batch_size = inputs.size(0)
        else:
            batch_size = inputs[0].size(0)
        
        if isinstance(targets, list) is False:
            targets = targets.cuda()
        else:
            input_indices = targets[1].cuda()
            targets = targets[0].cuda()

        if epoch < args.warmup_epoch:
            lr = adjust_lr(optimizer, epoch, args, batch_idx, len(trainloader))

        loss_div = torch.tensor(0.).cuda()
        loss_cls = torch.tensor(0.).cuda()
        ####teacher_eval###
        with torch.no_grad():
            logit_t, feat_t = model_t(inputs, feature=True)
            # logit_t, feat_t = model_t(inputs, feature_emd=True)
            feat_t = [f.detach() for f in feat_t]
            # logit_t1 = model_t(inputs[0].cuda())
            # logit_t2 = model_t(inputs[1].cuda())

        if args.method == 'cross_entropy':
            logit = net(inputs)
            loss_cls += criterion_cls(logit, targets)
        elif args.method == 'kd':  
            logit = net(inputs)
            loss_cls += criterion_cls(logit, targets)
            loss_div += criterion_div(logit,logit_t)
        elif args.method == 'hint':  
            logit,feat_s = net(inputs,feature=True)
            loss_cls += criterion_cls(logit, targets)
            loss_div += criterion_div(logit,logit_t)
            f_s, f_t = regress_s(feat_s[1],feat_t[1])
            loss_div += 100*criterion_hint(f_s, f_t)
         
        elif args.method == 'AT':  
            logit,feat_s = net(inputs,feature=True)
            loss_cls += criterion_cls(logit, targets)
            loss_div += criterion_div(logit,logit_t)
            loss_div += 1000*sum(criterion_at(feat_s,feat_t))

        elif args.method == 'SP':  
            logit,feat_s = net(inputs,feature=True)
            loss_cls += criterion_cls(logit, targets)
            loss_div += criterion_div(logit,logit_t)
            g_s = feat_s[-1]
            g_t = feat_t[-1]
            loss_group = criterion_sp(g_s,g_t)
            loss_div += 3000*sum(loss_group)
    
        elif args.method == 'PKT':  
            logit,feat_s = net(inputs,feature_emd=True)
            loss_cls += criterion_cls(logit, targets)
            loss_div += criterion_div(logit,logit_t)
            g_s = feat_s[-1]
            g_t = feat_t[-1]
            loss_div += 30000*criterion_pkt(g_s,g_t)
        
        elif args.method.startswith('clsd'):
            inputs = torch.cat(inputs, dim=0).cuda()
            batch_size = inputs.size(0) // 2
            ####1layer##########
            logit,feature = net(inputs,feature_emd=True)
            out1,out2= clsd(feature[1],feature[-4],feature[-3],feature[-2],feature[-1])
            ################he_loss##################
            aux_loss1 = criterion_div(out1,logit[:batch_size],args.kl_T)
            aux_loss2 = criterion_div(out2,logit[batch_size:],args.kl_T)
            loss_cls += 0.8*criterion_cls(logit, torch.cat([targets, targets], dim=0))
            he_kl_loss = (aux_loss1 + aux_loss2)
            ####
            ho_loss = (criterion_div(out1,out2.detach(),args.dl_T) + criterion_div(out2,out1.detach(),args.dl_T)) /2.0
            ###teacher_kd###
            kl_loss = (criterion_div(logit[batch_size:],logit_t2) + criterion_div(logit[:batch_size],logit_t1)) / 2.0
            loss_div += 0.2*he_kl_loss + 0.5*ho_loss + kl_loss
            #####################
            logit = (logit[batch_size:] + logit[:batch_size]) / 2
        
        else:
            raise ValueError('Unknown method: {}'.format(args.method))
        loss = loss_cls + loss_div
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), batch_size)
        train_loss_cls.update(loss_cls.item(), batch_size)
        train_loss_div.update(loss_div.item(), batch_size)

        top1, top5 = correct_num(logit, targets, topk=(1, 5))
        top1_num += top1
        top5_num += top5
        total += targets.size(0)
        
        print('Epoch:{}, batch_idx:{}/{}, lr:{:.5f}, Acc:{:.4f}, Duration:{:.2f}'.format(epoch, batch_idx, len(trainloader), lr, top1_num.item() / total, time.time()-batch_start_time))
        
    
    train_info = 'Epoch:{}\t lr:{:.5f}\t duration:{:.3f}'\
                 '\ntrain_loss:{:.5f}\t train_loss_cls:{:.5f}'\
                 '\t train_loss_div:{:.5f}' \
                 '\ntrain top1_acc: {:.4f} \t train top5_acc:{:.4f}' \
                .format(epoch, lr, time.time() - start_time,
                        train_loss.avg, train_loss_cls.avg,
                        train_loss_div.avg, (top1_num/total).item(), (top5_num/total).item())
    print(train_info)
    with open(args.log_txt, 'a+') as f:
        f.write(train_info+'\n')
    
def test(epoch, criterion_list):
    test_loss = AverageMeter('test_loss', ':.4e')
    top1_num = 0
    top5_num = 0
    total = 0

    criterion_cls = criterion_list[0]
    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs = inputs.cuda()
            targets = targets.cuda()

            logit = net(inputs)
            
            if isinstance(logit, list) or isinstance(logit, tuple):
                logit = logit[0]
            loss_cls = criterion_cls(logit, targets)
            

            test_loss.update(loss_cls.item(), inputs.size(0))

            top1, top5 = correct_num(logit, targets, topk=(1, 5))
            top1_num += top1
            top5_num += top5
            total += targets.size(0)
            print('Epoch:{}, batch_idx:{}/{}, Acc:{:.4f}'.format(epoch, batch_idx, len(trainloader), top1_num.item() / total))
            ############
         
    test_info = 'test_loss:{:.5f}\t test top1_acc:{:.4f} \t test top5_acc:{:.4f} \n' \
                .format(test_loss.avg, (top1_num/total).item(), (top5_num/total).item())
    with open(args.log_txt, 'a+') as f:
        f.write(test_info)
    print(test_info)

    return (top1_num/total).item()



if __name__ == '__main__':
    best_acc = 0.  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    criterion_cls = nn.CrossEntropyLoss()
    # criterion_div = DistillKL(args.T)
    criterion_div = MYKL()
    criterion_at = Attention()
    criterion_sp = Similarity()
    criterion_hint = HintLoss()
    criterion_pkt = PKT()
    criterion_smi = SmiLoss()
    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # classification loss
    criterion_list.append(criterion_div)
    criterion_list.append(criterion_hint)
    criterion_list.append(criterion_at)
    criterion_list.append(criterion_sp)
    criterion_list.append(criterion_pkt)
    criterion_list.append(criterion_smi)
    criterion_list.cuda()

    #####load_teacher_model
    teacher_arch = "CIFAR_ResNet50"
    model_t = getattr(models, teacher_arch)
    model_t = model_t(num_classes=num_classes).cuda()
    # teacher_dir = "/home/devuser/Downloads/baseline_skd/Self-KD-Lib/checkpoint/baseline/ResNet101/main_dataset_CIFAR-100_arch_CIFAR_ResNet101_method_cross_entropy_data_aug_None_0_test2/CIFAR_ResNet101_best.pth.tar"
    teacher_dir = "/home/devuser/Downloads/baseline_skd/Self-KD-Lib/checkpoint/baseline/ResNet50/main_dataset_CIFAR-100_arch_CIFAR_ResNet50_method_cross_entropy_data_aug_None_0_test2/CIFAR_ResNet50_best.pth.tar"
    # teacher_dir = "./checkpoint/baseline/ResNet50/main_dataset_CIFAR-100_arch_CIFAR_ResNet50_method_cross_entropy_data_aug_None_0_test2/CIFAR_ResNet50_best.pth.tar"
    # teacher_dir = "/home/devuser/Downloads/baseline_skd/Self-KD-Lib/checkpoint/baseline/ResNet101/main_dataset_CIFAR-100_arch_CIFAR_ResNet101_method_cross_entropy_data_aug_None_0_test2/CIFAR_ResNet101_best.pth.tar"
    teacher_model = torch.load(teacher_dir,
                            map_location=torch.device('cpu'))
    model_t.load_state_dict(teacher_model['net'])
    teacher_acc = teacher_model['acc']
    print(teacher_acc)

    if args.evaluate:
        print('load trained weights from '+ args.eval_checkpoint)
        checkpoint = torch.load(args.eval_checkpoint,
                                map_location=torch.device('cpu'))
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
        top1_acc = test(start_epoch, criterion_list)     
    else:
        if args.method ==  'clsd':
            trainable_list = nn.ModuleList([])
            trainable_list.append(net)
            if args.dataset == 'CIFAR-100' or args.dataset == 'tinyimagenet':
                data = torch.randn(2, 3, 32, 32) 
                data = data.cuda()
            else:
                data = torch.randn(2, 3, 224, 224) 
                data = data.cuda() 
            net.eval()
            _,feat = net(data, feature_emd=True)
            num_channels0 = feat[1].shape[1] 
            num_channels1 = feat[-4].shape[1] ##128
            num_channels2 = feat[-3].shape[1] ##256
            num_channels3 = feat[-2].shape[1] ##512
            clsd = CLSD(num_channels0,num_channels1,num_channels2,num_channels3,feat[-1].shape[1],num_classes)
            trainable_list.append(clsd)
            clsd.cuda()
            optimizer = optim.SGD(trainable_list.parameters(), lr=0.1, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
        else:
            if args.dataset == 'CIFAR-100':
                data = torch.randn(2, 3, 32, 32) 
                data = data.cuda()
            net.eval()
            model_t.eval()
            _,feat_s = net(data, feature=True)
            _,feat_t = model_t(data, feature=True)
            regress_s = ConvReg(feat_s[1].shape, feat_t[1].shape).cuda()
            trainable_list = nn.ModuleList([])
            trainable_list.append(net)
            trainable_list.append(regress_s)
            optimizer = optim.SGD(trainable_list.parameters(), lr=0.1, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)

            # trainable_list = nn.ModuleList([])
            # trainable_list.append(net)
            # optimizer = optim.SGD(trainable_list.parameters(), lr=0.1, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)

        if args.resume:
            print('Resume from '+ args.resume_checkpoint)

            checkpoint = torch.load(args.resume_checkpoint,
                                    map_location=torch.device('cpu'))
            net.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch']+1

        for epoch in range(start_epoch, args.epochs):
            train(epoch, criterion_list, optimizer)
            acc = test(epoch, criterion_list)
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, os.path.join(args.checkpoint_dir, model.__name__ + '.pth.tar'))

            is_best = False
            if best_acc < acc:
                best_acc = acc
                is_best = True

            if is_best:
                shutil.copyfile(os.path.join(args.checkpoint_dir, model.__name__ + '.pth.tar'),
                                os.path.join(args.checkpoint_dir, model.__name__ + '_best.pth.tar'))
            
        print('Evaluate the best model:')
        print('best_acc:',best_acc)
        args.evaluate = True
        checkpoint = torch.load(args.checkpoint_dir + '/' +  model.__name__ + '_best.pth.tar',
                                map_location=torch.device('cpu'))
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        top1_acc = test(start_epoch, criterion_list)

        with open(args.log_txt, 'a+') as f:
            f.write('best_accuracy: {} \n'.format(best_acc))
        print('best_accuracy: {} \n'.format(best_acc))

