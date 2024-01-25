import torch
import torch.nn as nn
import torch.nn.functional as F

# define Conv-BN-RELU block
def ConvBlock(in_channels, out_channels, kernel_size, stride):
    pad = (kernel_size - 1) // 2
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=pad, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


# define ASFF Module
class ASFF(nn.Module):
    def __init__(self, inter_dim=128, out_dim=256, vis=False):
        super(ASFF, self).__init__()
        self.inter_dim = inter_dim
        self.out_dim = out_dim

        # 28, 28, 256 -> 28, 28, 128
        self.level_0 = ConvBlock(256, self.inter_dim, 1, 1)
        # 56, 56, 256 -> 56, 56, 128
        self.level_1 = ConvBlock(256, self.inter_dim, 1, 1)
        # 112, 112, 192 -> 56, 56, 128
        self.level_2 = ConvBlock(128 + 64, self.inter_dim, 3, 2)
        self.expand = ConvBlock(self.inter_dim, self.out_dim, 3, 1)

        # compress ratio
        compress_c = 16

        self.weight_level_0 = ConvBlock(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = ConvBlock(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = ConvBlock(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)
        self.vis = vis
    
    def forward(self, x_level_0, x_level_1, x_level_2):
        # 28, 28, 512 -> 28, 28, 128
        level_0_compressed = self.level_0(x_level_0)
        # 28, 28, 512 -> 56, 56, 128
        level_0_resized = F.interpolate(level_0_compressed, scale_factor=2, mode='bilinear', align_corners=False)
        # 56, 56, 256 -> 56, 56, 128
        level_1_resized = self.level_1(x_level_1)
        # 112, 112, 192 -> 56, 56, 128
        level_2_resized =self.level_2(x_level_2)

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)

        levels_weight_v = torch.cat([level_0_weight_v, level_1_weight_v, level_2_weight_v], dim=1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :]+\
                            level_1_resized * levels_weight[:, 1:2, :, :]+\
                            level_2_resized * levels_weight[:, 2:, :, :]
        # 56, 56, 256
        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out


class ChannelwiseAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelwiseAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )
       
    def forward(self, x):
        n_b, n_c, _, _ = x.size()
        y = self.avg_pool(x).view((n_b, n_c))
        y = self.fc(y).view((n_b, n_c, 1, 1))

        return x * y.expand_as(x)


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(
            inplanes, inplanes, kernel_size,
            stride, padding, dilation, groups=inplanes, bias=bias)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.activ(self.bn1(self.conv1(x)))
        x = self.activ(self.bn2(self.pointwise(x)))
        return x


class CAJPU(nn.Module):
    def __init__(self, in_channels, width=256):
        super(CAJPU, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels[0], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels[1], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels[2], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(4 * width, width, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )

        self.cha_attn = ChannelwiseAttention(4 * width)

        self.dilation0 = SeparableConv2d(3 * width, width, kernel_size=1, padding=0, dilation=1, bias=False)
        self.dilation1 = SeparableConv2d(3 * width, width, kernel_size=3, padding=1, dilation=1, bias=False)
        self.dilation2 = SeparableConv2d(3 * width, width, kernel_size=3, padding=2, dilation=2, bias=False)
        self.dilation3 = SeparableConv2d(3 * width, width, kernel_size=3, padding=3, dilation=3, bias=False)
        
    def forward(self, *inputs):
        feats = [self.conv0(inputs[0]), self.conv1(inputs[1]), self.conv2(inputs[2])]

        _, _, h, w = feats[0].shape

        for i in range(1, len(feats)):
            feats[i] = F.interpolate(
                feats[i], size=(h, w), mode='bilinear', align_corners=False
            )

        feat = torch.cat(feats, dim=1)
        feat = torch.cat([self.dilation0(feat), self.dilation1(feat), self.dilation2(feat), self.dilation3(feat)], dim=1)
        feat = self.cha_attn(feat)
        output = self.conv3(feat)

        return output