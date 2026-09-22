'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out

def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        if stride != 1 or inplanes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )
        else:
            self.downsample = None
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class Aux_DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=100, bias=True, branch_layers=[]):
        super(Aux_DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes, bias=bias)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)
        
class Aux2_DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=100, bias=True, branch_layers=[]):
        super(Aux2_DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes, bias=bias)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

class Aux3_DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=100, bias=True, branch_layers=[]):
        super(Aux3_DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes, bias=bias)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

class CIFAR_DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=100, bias=True, branch_layers=[]):
        super(CIFAR_DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.in_planes = num_planes

        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes, bias=bias)

        self.branch_layers = branch_layers

        if len(self.branch_layers) != 0:
            # aux1 = Aux_DenseNet(block,nblocks,growth_rate=growth_rate,num_classes=num_classes)
            # self.aux1_dense2 = aux1.dense2
            # self.aux1_trans2 = aux1.trans2
            # self.aux1_dense3 = aux1.dense3
            # self.aux1_trans3 = aux1.trans3
            # self.aux1_dense4 = aux1.dense4
            # self.aux1_bn = aux1.bn
            # self.aux1_linear = aux1.linear

            # aux2 = Aux2_DenseNet(block,nblocks,growth_rate=growth_rate,num_classes=num_classes)
            # self.aux2_dense3 = aux2.dense3
            # self.aux2_trans3 = aux2.trans3
            # self.aux2_dense4 = aux2.dense4
            # self.aux2_bn = aux2.bn
            # self.aux2_linear = aux2.linear

            # aux3 = Aux3_DenseNet(block,nblocks,growth_rate=growth_rate,num_classes=num_classes)
            # self.aux3_dense4 = aux3.dense4
            # self.aux3_bn = aux3.bn
            # self.aux3_linear = aux3.linear

            self.in_planes = 128
            self.layer2_head1 = self._make_layer(BasicBlock, 256, 1, stride=2)
            self.layer3_head1 = self._make_layer(BasicBlock, 512, 1, stride=2)
            self.layer4_head1 = self._make_layer(BasicBlock, 1024, 1, stride=1)
            self.fc_head1 = nn.Linear(1024, num_classes)

            self.in_planes = 256
            self.layer3_head2 = self._make_layer(BasicBlock, 512, 1, stride=2)
            self.layer4_head2 = self._make_layer(BasicBlock, 1024, 1, stride=1)
            self.fc_head2 = nn.Linear(1024, num_classes)

            self.in_planes = 512
            self.layer4_head3 = self._make_layer(BasicBlock, 1024, 1, stride=1)
            self.fc_head3 = nn.Linear(1024, num_classes)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)
    

    def forward(self, x, feature=False,feature_emd=False, embedding=False):
        out = self.conv1(x)
        out0 = out
        out = self.trans1(self.dense1(out))
        out1 = out #32, 128, 16, 16
        out = self.trans2(self.dense2(out))
        out2 = out #32, 256, 8, 8
        out = self.trans3(self.dense3(out))
        out3 = out #32, 512, 4, 4
        out = self.dense4(out)
        out4 = out #32, 1024, 4, 4
        # print(out4.shape)
        f0 = out

        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        embedding0 = out
        out = self.linear(out)

        if len(self.branch_layers) != 0:
            # x = self.aux1_trans2(self.aux1_dense2(out1))
            # x = self.aux1_trans3(self.aux1_dense3(x))
            # x = self.aux1_dense4(x)
            # f1 =x
            # x = F.avg_pool2d(F.relu(self.aux1_bn(x)), 4)
            # x = x.view(x.size(0), -1)
            # x1 =self.aux1_linear(x)

            # x = self.aux2_trans3(self.aux2_dense3(out2))
            # x = self.aux2_dense4(x)
            # f2 =x
            # x = F.avg_pool2d(F.relu(self.aux2_bn(x)), 4)
            # x = x.view(x.size(0), -1)
            # x2 =self.aux2_linear(x)

            # x = self.aux3_dense4(out3)
            # f3 =x
            # x = F.avg_pool2d(F.relu(self.aux3_bn(x)), 4)
            # x = x.view(x.size(0), -1)
            # x3 =self.aux3_linear(x)
            ####采用Res作为辅助网络####
            x = self.layer2_head1(out1)
            x = self.layer3_head1(x)
            x = self.layer4_head1(x)
            f1 = x
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x1 = self.fc_head1(x)

            x = self.layer3_head2(out2)
            x = self.layer4_head2(x)
            f2 = x
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x2 = self.fc_head2(x)

            x = self.layer4_head3(out3)
            f3 = x
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x3 = self.fc_head3(x)
            ####采用Res作为辅助网络####
            if feature:
                return [out, x1, x2, x3], [embedding0, f0, f1, f2,f3]
            else:
                return [out, x1, x2, x3]
        else:
            if feature:
                return out,[out1, out2, out3, out4,embedding0]
            elif embedding:
                return out, embedding0
            elif feature_emd:
                return out, [out0, out1, out2, out3, out4, embedding0]
            else:
                return out

def CIFAR_DenseNet121(pretrained=False, num_classes=100, bias=True, **kwargs):
    return CIFAR_DenseNet(Bottleneck, [6,12,24,16], growth_rate=32, num_classes=num_classes, bias=bias)

def CIFAR_DenseNet121_byot(pretrained=False, num_classes=100, bias=True, **kwargs):
    return CIFAR_DenseNet(Bottleneck, [6,12,24,16], growth_rate=32, num_classes=num_classes, bias=bias,branch_layers=[[1, 2], [2]])
