"""
VGG16 for CIFAR-10/100 Dataset.

Reference:
1. https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py

"""

import torch
import torch.nn as nn

__all__ = ["vgg16", "vgg19"]

# cfg = {
#    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
# }

class VGG(nn.Module):
    def __init__(self, num_classes=100, depth=16, dropout=0.0, branch_layers=[], is_bias=True):
        super(VGG, self).__init__()
        self.inplances = 64
        self.conv1 = nn.Conv2d(3, self.inplances, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.inplances)
        self.conv2 = nn.Conv2d(self.inplances, self.inplances, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(self.inplances)
        self.relu = nn.ReLU(True)
        self.layer1 = self._make_layers(128, 2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.branch_layers = branch_layers

        if depth == 16:
            num_layer = 3
        elif depth == 19:
            num_layer = 4

        inplanes_head1 = self.inplances
        self.layer2 = self._make_layers(256, num_layer)
        inplanes_head2 = self.inplances
        self.layer3 = self._make_layers(512, num_layer)
        inplanes_head3 = self.inplances
        self.layer4 = self._make_layers(512, num_layer)

        if len(self.branch_layers) != 0:
            self.inplances = inplanes_head1
            self.layer2_head1 = self._make_layers(256, 1)
            self.layer3_head1 = self._make_layers(512,1)
            self.layer4_head1 = self._make_layers(512,1)
            self.fc_head1 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
            )

            self.inplances = inplanes_head2
            self.layer3_head2 = self._make_layers(512, 1)
            self.layer4_head2 = self._make_layers(512, 1)
            self.fc_head2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
            )

            self.inplances = inplanes_head3
            self.layer4_head3 =self._make_layers(512, 1)
            self.fc_head3 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),)
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, input, num_layer):
        layers = []
        for i in range(num_layer):
            conv2d = nn.Conv2d(self.inplances, input, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(input), nn.ReLU(inplace=True)]
            self.inplances = input
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)

    def forward(self, x,loss_type='cross_entropy', feature=False,feature_emd=False,embedding=False):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
     
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        out0 = x
        x = self.layer1(x)
        out1 = x
        x = self.layer2(x)
        out2 = x
        x = self.layer3(x)
        out3 = x
        xs = self.layer4(x)
        f0 = xs
        out4 = xs

        x_f = xs.view(xs.size(0), -1)
        embedding0 = x_f
        out = self.classifier(x_f)

        if len(self.branch_layers) != 0:
            # x = self.layer3_head2(out2)
            # x = self.layer4_head2(x)
            # f2 = x
            # x = x.view(x.size(0), -1)

            # x2 = self.fc_head2(x)

            # x = self.layer4_head1(out3)
            # f1 = x
            # x = x.view(x.size(0), -1)

            # x1 = self.fc_head1(x)
            x = self.layer2_head1(out1)
            x = self.layer3_head1(x)
            x = self.layer4_head1(x)
            f1 = x
            x = x.view(x.size(0), -1)
            x1 = self.fc_head1(x)

            x = self.layer3_head2(out2)
            x = self.layer4_head2(x)
            f2 = x
            x = x.view(x.size(0), -1)
            x2 = self.fc_head2(x)

            x = self.layer4_head3(out3)
            f3 = x
            x = x.view(x.size(0), -1)
            x3 = self.fc_head3(x)

            if feature:
                return [out, x1, x2,x3], [embedding0, f0, f1, f2,f3]
            else:
                # return [out, x1, x2,x3]
                return out
        else:
            if loss_type == 'cross_entropy':
                if feature:
                    return out, [out1, out2, out3, out4,embedding0]
                elif feature_emd:
                    return out, [out0,out1, out2, out3, out4,embedding0]
                elif embedding:
                    return out, embedding0 
                else:
                    return out

def vgg16(pretrained=False, path=None, **kwargs):
    """
    Constructs a VGG16 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """
    model = VGG(depth=16, **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))["state_dict"])
    return model


def vgg16_byot(pretrained=False, path=None, **kwargs):
    """
    Constructs a VGG16 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """
    model = VGG(depth=16, branch_layers=[[1, 1], [1]], **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))["state_dict"])
    return model

def vgg19(pretrained=False, path=None, **kwargs):
    """
    Constructs a VGG19 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """
    model = VGG(depth=19, **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))["state_dict"])
    return model
