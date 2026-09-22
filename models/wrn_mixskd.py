import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

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
    def __init__(self, depth, widen_factor=1,num_classes=100, dropRate=0.0,branch_layers=[],is_bias=True):
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
        self.network_channels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        features = []
        out = self.conv1(x)
        out0 = out

        out = self.block1(out)
        f1 = self.block1bn(out)
        features.append(f1)
        out = self.block2(f1)
        f2 = self.block2bn(out)
        features.append(f2)
        out = self.block3(f2)
        f3 = self.block3bn(out)
        out = self.relu(f3)
        features.append(out)

        out = F.avg_pool2d(out, 8)
        embedding0 = out.view(-1, self.network_channels[-1])
        logits = self.fc(embedding0)
        # print(features[0].shape,features[1].shape,features[-1].shape)
        return logits, features

class Auxiliary_Classifier(nn.Module):
    def __init__(self,depth,widen_factor=1,num_classes=100,):
        super(Auxiliary_Classifier, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        n = (depth - 4) / 6
        block = BasicBlock
        self.in_planes = 32
        self.block_extractor1 = nn.Sequential(*[NetworkBlock(n, nChannels[1], nChannels[2], block, 2),
                                                nn.BatchNorm2d(nChannels[2]),
                                                NetworkBlock(n, nChannels[2], nChannels[3], block, 2),
                                                nn.BatchNorm2d(nChannels[3]),
                                                nn.ReLU(inplace=True)
                                                ])
        self.in_planes = 64
        self.block_extractor2 = nn.Sequential(*[
                                                NetworkBlock(n, nChannels[2], nChannels[3], block, 2),
                                                nn.BatchNorm2d(nChannels[3]),
                                                nn.ReLU(inplace=True)
                                                ])

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(nChannels[3], num_classes)
        self.fc2 = nn.Linear(nChannels[3], num_classes)

    def forward(self, x):
        aux_logits = []
        aux_feats = []
        for i in range(len(x)):
            idx = i + 1
            out = getattr(self, 'block_extractor'+str(idx))(x[i])
            aux_feats.append(out)
            out = self.avg_pool(out)
            out = out.view(out.size(0), -1)

            out = getattr(self, 'fc'+str(idx))(out)
            aux_logits.append(out)
                   
        return aux_logits, aux_feats

class WRN_Final_Auxiliary_Classifer(nn.Module):
    def __init__(self, widen_factor=1, num_classes=100):
        super(WRN_Final_Auxiliary_Classifer, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        self.conv = conv1x1(nChannels[3] * 3, nChannels[3])
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(nChannels[3], num_classes)
    
    def forward(self, x):
        sum_fea = torch.cat(x, dim=1)
        out = self.conv(sum_fea)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class WRN_Auxiliary(nn.Module):
    def __init__(self, depth, widen_factor, num_classes=100):
        super(WRN_Auxiliary, self).__init__()
        self.backbone = WideResNet(depth, widen_factor,num_classes)
        self.auxiliary_classifier = Auxiliary_Classifier(depth, widen_factor, num_classes)
        self.final_aux_classifier = WRN_Final_Auxiliary_Classifer(widen_factor, num_classes) 
        
    def forward(self, x, lam=0.5, index=None):
        logits, features = self.backbone(x)
        aux_logits, aux_feats = self.auxiliary_classifier(features[:-1])
        aux_feats.append(features[-1])
        bs = features[0].size(0)

        aux_logits.append(logits)

        if self.training is False:
            return aux_logits, aux_feats

        ensemble_features = [lam * (fea[:bs//2]) + (1 - lam) * (fea[index]) for fea in aux_feats]
        ensemble_mixup_features = [fea[bs//2:] for fea in aux_feats]

        ensemle_logits = self.final_aux_classifier(ensemble_features)
        ensemble_mixup_logits = self.final_aux_classifier(ensemble_mixup_features)

        return aux_logits, aux_feats, ensemle_logits, ensemble_mixup_logits

def wrn(**kwargs):
    """
    Constructs a Wide Residual Networks.
    """
    model = WideResNet(**kwargs)
    return model

def MixSKD_wrn16x2(pretrained=False,**kwargs):
    model = WRN_Auxiliary(depth=16, widen_factor=2, **kwargs)
    return model

def MixSKD_wrn40x2(pretrained=False,**kwargs):
    model = WRN_Auxiliary(depth=40, widen_factor=2, **kwargs)
    return model
