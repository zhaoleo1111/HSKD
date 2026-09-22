from .resnet import *
from .bifpnc import *
from .resnet_mixskd import *
from .resnet_imagenet import *
from .resnet_imagenet_mixskd import *
from .wrn import wrn16x2_byot,wrn16x2,wrn40x2,wrn40x2_byot,wrn20x8,wrn20x8_byot
from .resnext import resnext50_32x4d,resnext50_32x4d_byot
from .resnet import CIFAR_ResNet101,CIFAR_ResNet101_byot,CIFAR_SeResNet50,CIFAR_SeResNet50_byot
from .vgg import vgg16,vgg16_byot,vgg19
# from .resnet_mgf import *
from .wrn_mixskd import MixSKD_wrn40x2,MixSKD_wrn16x2
from .vgg_mixskd import MixSKD_vgg16
from .PyramidNet import PyramidNet110,PyramidNet110_byot
from .DenseNet121 import CIFAR_DenseNet121,CIFAR_DenseNet121_byot
from .resnet_mstc import *
from .shufflenetv2 import shufflenetv2,shufflenetv2_byot
from .shufflenetv2_dtskd import shufflenetv2_dtskd
from .resnet_dtskd import resnet18_dtskd,resnet50_dtskd
from .resnet_imagenet import resnet18_imagenet
# from .our_resnet_imagenet import resnet18_imagenet
from .vgg_dtskd import vgg16_dtskd