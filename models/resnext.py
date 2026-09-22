import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torchvision.models import resnet18
__all__ = ['ResNet', 'resnet18', 'resnext50_32x4d']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


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
        self.downsample = downsample
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=100, zero_init_residual=False,
                 groups=1, width_per_group=64, branch_layers=[], norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        inplanes_head1 = self.inplanes
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        inplanes_head2 = self.inplanes
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
        inplanes_head3 = self.inplanes
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.branch_layers = branch_layers
        
        if len(self.branch_layers) != 0:
            self.inplanes = inplanes_head1
            self.layer2_head1 = self._make_layer(block, 128, branch_layers[0][0], stride=2)
            self.layer3_head1 = self._make_layer(block, 256, branch_layers[0][1], stride=2)
            self.layer4_head1 = self._make_layer(block, 512, branch_layers[0][2], stride=2)
            self.fc_head1 = nn.Linear(512 * block.expansion, num_classes)


            self.inplanes = inplanes_head2
            self.layer3_head2 = self._make_layer(block, 256, branch_layers[0][0], stride=2,norm_layer=norm_layer)
            self.layer4_head2 = self._make_layer(block, 512, branch_layers[0][1], stride=2,norm_layer=norm_layer)
            self.fc_head2 = nn.Linear(512 * block.expansion, num_classes)
            
            self.inplanes = inplanes_head3
            self.layer4_head3 = self._make_layer(block, 512, branch_layers[1][0], stride=2,norm_layer=norm_layer)
            self.fc_head3 = nn.Linear(512 * block.expansion, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
        

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, y=None, loss_type='cross_entropy', feature=False, feature_emd=False,embedding=False):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out0 = out
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out = out4 = self.layer4(out3)
        f0 = out4
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        embedding0 = out
        out = self.fc(out)

        if len(self.branch_layers) != 0:
            # x = self.layer3_head2(out2)
            # x = self.layer4_head2(x)
            # f2 = x
            # x = self.avgpool(x)
            # x = x.view(x.size(0), -1)
            # x2 = self.fc_head2(x)

            # x = self.layer4_head1(out3)
            # f1 = x
            # x = self.avgpool(x)
            # x = x.view(x.size(0), -1)

            # x1 = self.fc_head1(x)
            # if feature:
            #     return [out, x1, x2], [embedding0, f0, f1, f2]
            # else:
            #     return [out, x1, x2]
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

            if feature:
                return [out, x1, x2, x3], [embedding0, f0, f1, f2,f3]
            else:
                return [out, x1, x2, x3]
        else:
            if loss_type == 'cross_entropy':
                if feature:
                    return out, [out1, out2, out3, out4,embedding0]
                elif feature_emd:
                    return out,[out0,out1, out2, out3, out4,embedding0] 
                elif embedding:
                    return out, embedding0
                else:
                    return out
            elif loss_type == 'virtual_softmax':
                target_w = self.fc.weight[y]
                L2_target_w = target_w.pow(2).sum(1, keepdim=True).pow(1. / 2.)
                x_target_w = embedding0.pow(2).sum(1, keepdim=True).pow(1. / 2.)
                out = torch.cat([out, L2_target_w * x_target_w], dim=1)
                return out


def resnext50_32x4d(pretrained=False, **kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], groups=32, width_per_group=4, branch_layers=[],**kwargs)


def resnext50_32x4d_byot(pretrained=False, **kwargs):
    # return ResNet(Bottleneck, [3, 4, 6, 3], groups=32, width_per_group=4, branch_layers=[[1, 1], [1]],**kwargs)
    return ResNet(Bottleneck, [3, 4, 6, 3], groups=32, width_per_group=4, branch_layers=[[1, 1, 1], [1]],**kwargs)