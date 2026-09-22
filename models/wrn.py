import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        # self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(x)
        else:
            out = self.relu1(x)
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes=100,widen_factor=1, dropRate=0.0,branch_layers=[],is_bias=True):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        self.block1bn = nn.BatchNorm2d(nChannels[1])
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        self.block2bn = nn.BatchNorm2d(nChannels[2])
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        self.block3bn = nn.BatchNorm2d(nChannels[3])
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.network_channels = [16*widen_factor, 32*widen_factor, 64*widen_factor]

        self.branch_layers = branch_layers
        if len(self.branch_layers) != 0:
            self.layer2_head2 =  NetworkBlock(branch_layers[0][0], nChannels[1], nChannels[2], block, 2, dropRate)
            self.blockl2bn = nn.BatchNorm2d(nChannels[2])

            self.layer3_head2 = NetworkBlock(branch_layers[0][1], nChannels[2], nChannels[3], block, 2, dropRate)
            self.blockl3bn = nn.BatchNorm2d(nChannels[3])
            self.fc_head2 = nn.Linear(nChannels[3], num_classes)

            self.layer3_head1 = NetworkBlock(branch_layers[1][0], nChannels[2], nChannels[3], block, 2, dropRate)
            self.blockl4bn = nn.BatchNorm2d(nChannels[3])
            self.fc_head1 = nn.Linear(nChannels[3], num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, loss_type='cross_entropy', feature=False, feature_emd=False, embedding=False):
        out = self.conv1(x)
        out0 = out

        out = self.block1(out)
        f1 = self.block1bn(out)
        out1 = out  #256 32 32 32
        
        out = self.block2(f1)
        f2 = self.block2bn(out)
        out2 = out #256 64 16 16

        out = self.block3(f2)
        f3 = self.block3bn(out)
        out = self.relu(f3)
        f0 = out
        out3 = out

        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.network_channels[-1])
        embedding0 = out

        out = self.fc(embedding0)

        if len(self.branch_layers) != 0:
            x = self.layer2_head2(out1)
            x = self.blockl2bn(x)
            x = self.layer3_head2(x)
            x = self.blockl3bn(x)
            x = self.relu(x)
            f2 = x
            x = F.avg_pool2d(x, 8)
            x = x.view(x.size(0), -1)

            x2 = self.fc_head2(x)

            x = self.layer3_head1(out2)
            x = self.blockl4bn(x)
            x = self.relu(x)
            f1 = x
            x =  F.avg_pool2d(x, 8)
            x = x.view(x.size(0), -1)

            x1 = self.fc_head1(x)
            if feature:
                return [out, x1, x2], [embedding0, f0, f1, f2]
            else:
                return [out, x1, x2]

        else:
            if loss_type == 'cross_entropy':
                if feature:
                    return out, [out1, out2, out3]    
                elif feature_emd:
                   return out, [out0, out1, out2, out3, embedding0]
                if embedding:
                    return out, embedding0
                else:
                    return out
           
    def get_bn_before_relu(self):
        bn1 = self.block2.layer[0].bn1
        bn2 = self.block3.layer[0].bn1
        bn3 = self.bn1

        return [bn1, bn2, bn3]


def wrn(**kwargs):
    """
    Constructs a Wide Residual Networks.
    """
    model = WideResNet(**kwargs)
    return model


def wrn16x2(pretrained=False,**kwargs):
    model = WideResNet(depth=16, widen_factor=2, **kwargs)
    return model

def wrn16x8(pretrained=False,**kwargs):
    model = WideResNet(depth=16, widen_factor=8, **kwargs)
    return model

def wrn16x8_byot(pretrained=False,**kwargs):
    model = WideResNet(depth=16, widen_factor=8, branch_layers=[[1, 1], [1]], **kwargs)
    return model

def wrn16x2_byot(pretrained=False,**kwargs):
    model = WideResNet(depth=16, widen_factor=2, branch_layers=[[1, 1], [1]], **kwargs)
    return model

def wrn20x8(pretrained=False,**kwargs):
    model = WideResNet(depth=20, widen_factor=8, **kwargs)
    return model

def wrn20x8_byot(pretrained=False,**kwargs):
    model = WideResNet(depth=20, widen_factor=8, branch_layers=[[1, 1], [1]], **kwargs)
    return model

def wrn40x2(pretrained=False,**kwargs):
    model = WideResNet(depth=40, widen_factor=2, **kwargs)
    return model

def wrn40x2_byot(pretrained=False,**kwargs):
    model = WideResNet(depth=40, widen_factor=2, branch_layers=[[1, 1], [1]], **kwargs)
    return model