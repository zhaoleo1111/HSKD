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
from wrapper import wrapper

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
parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=str, help='saved checkpoint directory')
parser.add_argument('--eval-checkpoint', default='./checkpoint/resnet18_best.pth', type=str, help='evaluate checkpoint directory')
parser.add_argument('--resume-checkpoint', default='/home/devuser/zhaolei_skd/Self-KD-Lib/checkpoint/main_dataset_CIFAR-100_arch_shufflenetv2_method_mixup_data_aug_None_0_test3/shufflenetv2.pth.tar', type=str, help='resume checkpoint directory')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--evaluate', '-e', action='store_true', help='evaluate model')
parser.add_argument('--kl_T', type=float, default=4.0, help='temperature for KD distillation')
parser.add_argument('--dl_T', type=float, default=3.0, help='temperature for KD distillation')
parser.add_argument('-t', '--trial', type=str, default='0', help='the experiment id')

# global hyperparameter set
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
if args.method == 'dlb' or args.method == 'PSKD' or args.method == 'psmgf' or args.method == 'dtskd':
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

# if args.method == 'virtual_softmax':
#     net = model(num_classes=num_classes, is_bias=False).eval()
# else:
#     net = model(num_classes=num_classes).eval()

# print('Arch: %s, Params: %.2fM, Multi-adds: %.2fG'
#     % (args.arch, cal_param_size(net) / 1e6, cal_multi_adds(net, (1, C, H, W)) / 1e9))
# del (net)

if args.method == 'virtual_softmax':
    net = model(num_classes=num_classes, is_bias=False).cuda()
else:
    net = model(num_classes=num_classes).cuda()
 
cudnn.benchmark = True

if args.method.startswith('dtskd'):  
    all_predictions = torch.zeros(len(trainloader.dataset), num_classes, dtype=torch.float32)
    b1_predictions = torch.zeros(len(trainloader.dataset), num_classes, dtype=torch.float32)
    b2_predictions = torch.zeros(len(trainloader.dataset), num_classes, dtype=torch.float32)
    b3_predictions = torch.zeros(len(trainloader.dataset), num_classes, dtype=torch.float32)

# Training
def train(epoch, criterion_list, optimizer,all_predictions, b1_predictions,b2_predictions,b3_predictions):
    train_loss = AverageMeter('train_loss', ':.4e')
    train_loss_cls = AverageMeter('train_loss_cls', ':.4e')
    train_loss_div = AverageMeter('train_loss_div', ':.4e')
    train_loss_mixcls = AverageMeter('train_loss_mixcls', ':.4e')
    top1_num = 0
    top5_num = 0
    total = 0
    if epoch >= args.warmup_epoch:
        lr = adjust_lr(optimizer, epoch, args)
    start_time = time.time()
    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]


    net.train()

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
        loss_mixcls = torch.tensor(0.).cuda()

        if args.method == 'cross_entropy':
            logit = net(inputs)
            loss_cls += criterion_cls(logit, targets)

        elif args.method.startswith('dtskd'):
            logit,hist_loss,stru_loss = PSDTSKD(net, inputs, targets, input_indices, epoch, all_predictions,b1_predictions,b2_predictions,b3_predictions,num_classes,criterion_div,args)
            loss_cls += 0.8*hist_loss
            loss_div += 0.2*stru_loss
 
        else:
            raise ValueError('Unknown method: {}'.format(args.method))
        loss = loss_cls + loss_div
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), batch_size)
        train_loss_cls.update(loss_cls.item(), batch_size)
        train_loss_div.update(loss_div.item(), batch_size)
        train_loss_mixcls.update(loss_mixcls.item(), batch_size)

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

   
    
    return all_predictions,b1_predictions, b2_predictions,b3_predictions

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
    criterion_div = DistillKL(args.kl_T)
    criterion_mse = nn.MSELoss(reduction='sum')
    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # classification loss
    criterion_list.append(criterion_div)
    criterion_list.append(criterion_mse)
    criterion_list.cuda()

    if args.evaluate:
        print('load trained weights from '+ args.eval_checkpoint)
        checkpoint = torch.load(args.eval_checkpoint,
                                map_location=torch.device('cpu'))
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
        top1_acc = test(start_epoch, criterion_list)
    else:
       
        if args.method ==  'mgf' or args.method ==  'psmgf':
            trainable_list = nn.ModuleList([])
            trainable_list.append(net)
            
            if args.dataset == 'CIFAR-100' or args.dataset == 'tinyimagenet':
                data = torch.randn(2, 3, 32, 32) 
                data = data.cuda()
            else:
                data = torch.randn(2, 3, 224, 224) 
                data = data.cuda()
                
            net.eval()
            _,feat = net(data, feature=True)
            num_channels1 = feat[0].shape[1]
            num_channels2 = feat[1].shape[1]
            num_channels3 = feat[2].shape[1]
            num_channels4 = feat[3].shape[1]
            num_in = feat[-1].shape[1]
            feat_H = feat[3].shape[2]
            decoder = wrapper(feat_H,num_channels1,num_channels2,num_channels3,num_channels4,num_in,num_classes)
            trainable_list.append(decoder)
            decoder.cuda()
            optimizer = optim.SGD(trainable_list.parameters(), lr=0.1, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)

        elif args.method ==  'uskd':
            trainable_list = nn.ModuleList([])
            trainable_list.append(net)
            if args.dataset == 'CIFAR-100'  or args.dataset == 'tinyimagenet':
                data = torch.randn(2, 3, 32, 32) 
                data = data.cuda()
            else:
                data = torch.randn(2, 3, 224, 224) 
                data = data.cuda()    
            net.eval()
            _,feat = net(data, feature=True)  
            ch  = feat[1].shape[1]
            uskd = USKDLoss(channel=ch,num_classes=num_classes)
            trainable_list.append(uskd)
            optimizer = optim.SGD(trainable_list.parameters(), lr=0.1, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
            trainable_list.cuda()
        else:
            trainable_list = nn.ModuleList([])
            trainable_list.append(net)
            optimizer = optim.SGD(trainable_list.parameters(), lr=0.1, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)

        if args.resume:
            print('Resume from '+ args.resume_checkpoint)
            checkpoint = torch.load(args.resume_checkpoint,
                                    map_location=torch.device('cpu'))
            net.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch']+1
    
        for epoch in range(start_epoch, args.epochs):
            t0_time = time.time()
            all_predictions, b1_predictions, b2_predictions,b3_predictions = train(epoch, criterion_list, optimizer,all_predictions, b1_predictions, b2_predictions,b3_predictions)
            t1_time = time.time()
            epcoh_time = t1_time-t0_time
            print('Evaluate the total time:',t1_time-t0_time)
            test_info2 = 'epcoh_time:{:.3f}\n' \
                .format(epcoh_time)
            with open(args.log_txt, 'a+') as f:
                f.write(test_info2)

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

