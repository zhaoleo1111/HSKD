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
def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class VGG(nn.Module):
    def __init__(self, depth=16, num_classes=100,  dropout=0.0, is_bias=True):
        super(VGG, self).__init__()
        self.inplances = 64
        self.conv1 = nn.Conv2d(3, self.inplances, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.inplances)
        self.conv2 = nn.Conv2d(self.inplances, self.inplances, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(self.inplances)
        self.relu = nn.ReLU(True)
        self.layer1 = self._make_layers(128, 2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        num_layer = 3
        if depth == 16:
            num_layer = 3
        elif depth == 19:
            num_layer = 4

        self.layer2 = self._make_layers(256, num_layer)
        self.layer3 = self._make_layers(512, num_layer)
        self.layer4 = self._make_layers(512, num_layer)

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

    def forward(self, x):
        features = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
     
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        xs = self.layer4(x)
        features.append(xs)
       
        x_f = xs.view(xs.size(0), -1)
        logit = self.classifier(x_f)
        return logit,features

class Auxiliary_Classifier(nn.Module):
    def __init__(self, depth=16, num_classes=100,dropout=0.0):
        super(Auxiliary_Classifier, self).__init__()
        self.inplances = 128
        num_layer = 3
        if depth == 16:
            num_layer = 3
        elif depth == 19:
            num_layer = 4

        self.block_extractor1 = nn.Sequential(*[self._make_layers(256, num_layer),
                                                self._make_layers(512, num_layer),
                                                self._make_layers(512, num_layer)])
        self.inplances = 256
        self.block_extractor2 = nn.Sequential(*[
                                                self._make_layers(512, num_layer),
                                                self._make_layers(512, num_layer)])
        self.inplances = 512
        self.block_extractor3 = nn.Sequential(*[
                                                self._make_layers(512, num_layer)])

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )
        self.fc3 =  nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

    def _make_layers(self, input, num_layer):
        layers = []
        for i in range(num_layer):
            conv2d = nn.Conv2d(self.inplances, input, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(input), nn.ReLU(inplace=True)]
            self.inplances = input
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)

    def forward(self, x):
        aux_logits = []
        aux_feats = []
        for i in range(len(x)):
            idx = i + 1
            out = getattr(self, 'block_extractor'+str(idx))(x[i])
            # print(out.shape)
            aux_feats.append(out)
            out = self.avg_pool(out)
            out = out.view(out.size(0), -1)
            
            out = getattr(self, 'fc'+str(idx))(out)
            aux_logits.append(out)     
        return aux_logits, aux_feats

class VGG_Final_Auxiliary_Classifer(nn.Module):
    def __init__(self, num_classes=100):
        super(VGG_Final_Auxiliary_Classifer, self).__init__()
        self.conv = conv1x1(512 * 4, 512 )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        sum_fea = torch.cat(x, dim=1)
        out = self.conv(sum_fea)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class VGG_Auxiliary(nn.Module):
    def __init__(self, depth, num_classes=100):
        super(VGG_Auxiliary, self).__init__()
        self.backbone = VGG(depth, num_classes)
        self.auxiliary_classifier = Auxiliary_Classifier(depth, num_classes)
        self.final_aux_classifier = VGG_Final_Auxiliary_Classifer(num_classes) 
        
    def forward(self, x, lam=0.5, index=None):
        logits, features = self.backbone(x)
        aux_logits, aux_feats = self.auxiliary_classifier(features[:-1])
        aux_feats.append(features[-1])
        bs = features[0].size(0)
        # print(logits.shape)  
        aux_logits.append(logits)

        if self.training is False:
            return aux_logits, aux_feats

        ensemble_features = [lam * (fea[:bs//2]) + (1 - lam) * (fea[index]) for fea in aux_feats]
        ensemble_mixup_features = [fea[bs//2:] for fea in aux_feats]

        ensemle_logits = self.final_aux_classifier(ensemble_features)
        ensemble_mixup_logits = self.final_aux_classifier(ensemble_mixup_features)
        
        return aux_logits, aux_feats, ensemle_logits, ensemble_mixup_logits
     
def MixSKD_vgg16(pretrained=False, path=None, **kwargs):
    """
    Constructs a VGG16 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """
    model = VGG_Auxiliary(depth=16, **kwargs)
    
    return model
