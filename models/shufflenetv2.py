"""shufflenetv2 in pytorch
[1] Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, Jian Sun
    ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design
    https://arxiv.org/abs/1807.11164
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def channel_split(x, split):
    """split a tensor into two pieces along channel dimension
    Args:
        x: input tensor
        split:(int) channel size for each pieces
    """
    assert x.size(1) == split * 2
    return torch.split(x, split, dim=1)


def channel_shuffle(x, groups):
    """channel shuffle operation
    Args:
        x: input tensor
        groups: input branch number
    """

    batch_size, channels, height, width = x.size()
    channels_per_group = int(channels // groups)

    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = x.transpose(1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)

    return x


class ShuffleUnit(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, int(out_channels / 2), 1),
                nn.BatchNorm2d(int(out_channels / 2)),
                nn.ReLU(inplace=True)
            )

            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, int(out_channels / 2), 1),
                nn.BatchNorm2d(int(out_channels / 2)),
                nn.ReLU(inplace=True)
            )
        else:
            self.shortcut = nn.Sequential()

            in_channels = int(in_channels / 2)
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )


    def forward(self, x):

        if self.stride == 1 and self.out_channels == self.in_channels:
            shortcut, residual = channel_split(x, int(self.in_channels / 2))
        else:
            shortcut = x
            residual = x

        shortcut = self.shortcut(shortcut)
        residual = self.residual(residual)
        x = torch.cat([shortcut, residual], dim=1)
        x = channel_shuffle(x, 2)

        return x


class ShuffleNetV2(nn.Module):

    def __init__(self, ratio=1, num_classes=100,branch_layers=[]):
        super().__init__()
        if ratio == 0.5:
            out_channels = [48, 96, 192, 1024]
        elif ratio == 1:
            out_channels = [116, 232, 464, 1024]
        elif ratio == 1.5:
            out_channels = [176, 352, 704, 1024]
        elif ratio == 2:
            out_channels = [244, 488, 976, 2048]
        else:
            ValueError('unsupported ratio number')

        self.pre = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1),
            nn.BatchNorm2d(24)
        )

        self.stage2 = self._make_stage(24, out_channels[0], 3)
        self.stage3 = self._make_stage(out_channels[0], out_channels[1], 7)
        self.stage4 = self._make_stage(out_channels[1], out_channels[2], 3)
        self.conv5 = nn.Sequential(
            nn.Conv2d(out_channels[2], out_channels[3], 1),
            nn.BatchNorm2d(out_channels[3]),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Linear(out_channels[3], num_classes)

        self.branch_layers = branch_layers

        if len(self.branch_layers) != 0:
            self.aux_stage31 = self._make_stage(out_channels[0], out_channels[1], 1)
            self.aux_stage41 = self._make_stage(out_channels[1], out_channels[2], 1)
            self.aux_conv51 = nn.Sequential(
            nn.Conv2d(out_channels[2], out_channels[3], 1),
            nn.BatchNorm2d(out_channels[3]),
            nn.ReLU(inplace=True))
            self.fc_head1 = nn.Linear(out_channels[3], num_classes)

            self.aux_stage42 = self._make_stage(out_channels[1], out_channels[2], 1)
            self.aux_conv52 = nn.Sequential(
            nn.Conv2d(out_channels[2], out_channels[3], 1),
            nn.BatchNorm2d(out_channels[3]),
            nn.ReLU(inplace=True))
            self.fc_head2 = nn.Linear(out_channels[3], num_classes)

            self.aux_conv53 = nn.Sequential(
            nn.Conv2d(out_channels[2], out_channels[3], 1),
            nn.BatchNorm2d(out_channels[3]),
            nn.ReLU(inplace=True))
            self.fc_head3 = nn.Linear(out_channels[3], num_classes)


    def forward(self, x,feature=False):
        x = self.pre(x)
        s_out1 = self.stage2(x)  # 116,16,16
        s_out2 = self.stage3(s_out1)  # 232,8,8
        s_out3 = self.stage4(s_out2)  # 464,4,4
        s_out4 = self.conv5(s_out3)  # 1024,4,4

        out = F.adaptive_avg_pool2d(s_out4, (1,1))
        logits_out = out 
        out = out.view(out.size(0), -1)
        embedding0 = out
        out = self.fc(out)
        
        if len(self.branch_layers) != 0:
            aux_out1 = self.aux_stage31(s_out1)
            aux_out1 = self.aux_stage41(aux_out1)
            aux_out1 = self.aux_conv51(aux_out1)
            f1 =  aux_out1
            aux_out1 = F.adaptive_avg_pool2d(aux_out1, (1,1))
            aux_out1 = aux_out1.view(aux_out1.size(0), -1)
            aux_out1 = self.fc_head1(aux_out1)

            aux_out2 = self.aux_stage41(s_out2)
            aux_out2 = self.aux_conv51(aux_out2)
            f2 =  aux_out2
            aux_out2 = F.adaptive_avg_pool2d(aux_out2, (1,1))
            aux_out2 = aux_out2.view(aux_out2.size(0), -1) 
            aux_out2 = self.fc_head1(aux_out2)

            aux_out3 = self.aux_conv51(s_out3) 
            f3 =  aux_out3
            aux_out3 = F.adaptive_avg_pool2d(aux_out3, (1,1))
            aux_out3 = aux_out3.view(aux_out3.size(0), -1) 
            aux_out3 = self.fc_head1(aux_out3)
            
            if feature:
                return [out, aux_out1, aux_out2, aux_out3], [embedding0, s_out4, f1, f2,f3]
            else:
                return [out, aux_out1, aux_out2, aux_out3]
        else:
            if feature:
                    return out,[s_out1, s_out2, s_out3,s_out4,embedding0]
            else:
                return out

    def _make_stage(self, in_channels, out_channels, repeat):
        layers = []
        layers.append(ShuffleUnit(in_channels, out_channels, 2))

        while repeat:
            layers.append(ShuffleUnit(out_channels, out_channels, 1))
            repeat -= 1

        return nn.Sequential(*layers)


def shufflenetv2(**kwargs):
    return ShuffleNetV2(**kwargs)

def shufflenetv2_byot(**kwargs):
    return ShuffleNetV2(branch_layers=[[1, 1], [1]],**kwargs)


if __name__ == '__main__':
    input = torch.ones([128, 3, 32, 32])
    model = shufflenetv2()
    output = model(input)
    print("test")
