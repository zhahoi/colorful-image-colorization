import torch
import torch.nn as nn
from torchvision import models
from models.uafm import UAFM, SPPM
from models.blocks import *

# extract vgg features
vgg_conv1_2 = vgg_conv2_2 = vgg_conv3_3 = None

def conv_1_2_hook(module, input, output):
    global vgg_conv1_2
    vgg_conv1_2 = output
    return None

def conv_2_2_hook(module, input, output):
    global vgg_conv2_2
    vgg_conv2_2 = output
    return None

def conv_3_3_hook(module, input, output):
    global vgg_conv3_3
    vgg_conv3_3 = output
    return None

class ColorNet(nn.Module):
    def __init__(self, dim=64):
        super(ColorNet, self).__init__()
        # 1, 512, 28, 28
        vggnet = models.vgg16(pretrained=True).features
        self.vggnet = vggnet[:-8]

        # Extract and register intermediate features of VGG-16
        # 224, 224, 64
        self.vggnet[3].register_forward_hook(conv_1_2_hook)
        # 112, 112, 128
        self.vggnet[8].register_forward_hook(conv_2_2_hook)
        # 56, 56, 256
        self.vggnet[15].register_forward_hook(conv_3_3_hook)
       
        # local_feature 1, 512, 28, 28
        self.local_features = nn.Sequential(
            ConvBlock(dim * 8, dim * 8, kernel_size=3, stride=1),
            ConvBlock(dim * 8, dim * 8, kernel_size=3, stride=1),
            ConvBlock(dim * 8, dim * 8, kernel_size=3, stride=1)
        )

        # mid_scale features  1, 512, 14, 14
        self.mid_features_1 = ConvBlock(dim * 8, dim * 8, kernel_size=3, stride=2)
        self.mid_features_2 = nn.Sequential(
            ConvBlock(dim * 8, dim * 8, kernel_size=3, stride=1),
            ConvBlock(dim * 8, dim * 8, kernel_size=3, stride=1)
        )

        # global features 1, 512, 7, 7
        self.global_feature = nn.Sequential(
            ConvBlock(dim * 8, dim * 8, kernel_size=3, stride=2),
            ConvBlock(dim * 8, dim * 8, kernel_size=3, stride=1)
        )

        # SPPM 1, 256, 7, 7
        self.sppm = SPPM(in_channels=dim * 8, inter_channels=dim * 4, out_channels=dim * 4, bin_sizes=[1, 2, 4])

        # UAFM 1, 128, 14, 14
        self.uafm1 = UAFM(low_chan=dim * 8, hight_chan=dim * 4, out_chan= dim * 2)
        
        # UAFM 1, 64, 28, 28
        self.uafm2 = UAFM(low_chan=dim * 8, hight_chan=dim * 2, out_chan= dim * 1)

        # output 1, 313, 56, 56
        self.q = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(dim * 1, 313, kernel_size=1, stride=1, padding=0, bias=False)
        )
        

    def forward(self, x):
        global vgg_conv1_2, vgg_conv2_2, vgg_conv3_3

        x_repeat = x.expand(-1, 3, -1, -1)
        # 1, 512, 28, 28
        backbone = self.vggnet(x_repeat)

        # local_feature 1, 512, 28, 28
        local_feature = self.local_features(backbone)
        # mid_features 1, 512, 14, 14
        mid_feature_1 = self.mid_features_1(backbone)
        mid_feature = self.mid_features_2(mid_feature_1)
        # global_features 1, 512, 7, 7
        global_feature = self.global_feature(mid_feature_1)
        
        # sppm 1, 256, 7, 7
        sppm = self.sppm(global_feature)

        # uafm1 1, 128, 14, 14
        uafm1 = self.uafm1(sppm, mid_feature)
        
        # uafm2 1, 64, 28, 28
        uafm2 = self.uafm2(uafm1, local_feature)
        
        # output 1, 313, 56, 56
        q = self.q(uafm2) / 0.38  # 0.38 is Softmax temperature T. Paper Eq.(5)

        return q
