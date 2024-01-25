import torch
import torch.nn as nn
from torchvision import models
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

        # downsampling
        self.compress = ConvBlock(dim, dim, kernel_size=3, stride=2)
       
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

        # CAJPU 1, 256, 28, 28
        self.cajpu = CAJPU(in_channels=(dim * 8, dim * 8, dim * 8), width=dim * 4)

        # ASFF 1, 256, 56, 56
        self.asff = ASFF(inter_dim=dim, out_dim=dim * 4)

        # output
        self.q = nn.Conv2d(dim * 4, 313, kernel_size=1, stride=1, padding=0, bias=False)

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
        # jpu 1, 256, 28, 28
        cajpu = self.cajpu(local_feature, mid_feature, global_feature)

        # downsample conv1_2: 1, 64, 224, 224 -> 1, 64, 112, 112
        compress = self.compress(vgg_conv1_2)
        # 1, 64, 112, 112 + 1, 128, 112, 112 -> 1, 192, 112, 112
        concat = torch.cat([vgg_conv2_2, compress], dim=1)

        # asff 1, 256, 56, 56
        asff = self.asff(cajpu, vgg_conv3_3, concat)

        # output 1, 313, 56, 56
        q = self.q(asff) / 0.38  # 0.38 is Softmax temperature T. Paper Eq.(5)

        return q
