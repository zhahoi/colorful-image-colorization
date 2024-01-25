import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvX(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False, act='relu') -> None:
        super().__init__()
        self.act = act

        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, 
                              padding=kernel_size // 2, bias=bias)
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        if self.act == 'leaky':
            x = F.leaky_relu(x)
        else:
            x = F.relu(x)

        return x
    
    
class SPPM(nn.Module):
    """Simple Pyramid Pooling Module
    """
    def __init__(self, in_channels, inter_channels, out_channels, bin_sizes) -> None:
        """
        :param in_channels: int, channels of input feature
        :param inter_channels: int, chennels of mid conv
        :param out_channels: int, channels of output feature
        :param bin_sizes: list, avg pool size of 3 features
        """
        super().__init__()

        self.stage1_pool = nn.AdaptiveAvgPool2d(output_size=bin_sizes[0])
        self.stage1_conv = ConvX(in_channels, inter_channels, kernel_size=1)

        self.stage2_pool = nn.AdaptiveAvgPool2d(output_size=bin_sizes[1])
        self.stage2_conv = ConvX(in_channels, inter_channels, kernel_size=1)

        self.stage3_pool = nn.AdaptiveAvgPool2d(output_size=bin_sizes[2])
        self.stage3_conv = ConvX(in_channels, inter_channels, kernel_size=1)

        self.conv_out = ConvX(inter_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        h, w =  x.size()[2:]

        f1 = self.stage1_pool(x)
        f1 = self.stage1_conv(f1)
        f1 =  F.interpolate(f1, (h, w), mode='bilinear', align_corners=False)
    
        f2 = self.stage2_pool(x)
        f2 = self.stage2_conv(f2)
        f2 =  F.interpolate(f2, (h, w), mode='bilinear', align_corners=False)

        f3 = self.stage3_pool(x)
        f3 = self.stage3_conv(f3)
        f3 =  F.interpolate(f3, (h, w), mode='bilinear', align_corners=False)

        x = self.conv_out(f1 + f2 + f3)

        return x


class UAFM(nn.Module):
    """Unified Attention Fusion Modul
    """
    def __init__(self, low_chan, hight_chan, out_chan, u_type='sp') -> None:
        """
        :param low_chan: int, channels of input low-level feature
        :param hight_chan: int, channels of input high-level feature
        :param out_chan: int, channels of output faeture
        :param u_type: string, attention type, sp: spatial attention, ch: channel attention
        """
        super().__init__()
        self.u_type = u_type

        if u_type == 'sp':
            self.conv_atten = nn.Sequential(
                ConvX(4, 2, kernel_size=3),
                nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(1),
                )
        else:
            self.conv_atten = nn.Sequential(
                ConvX(4 * hight_chan, hight_chan // 2,  kernel_size=1, bias=False, act="leaky"),
                nn.Conv2d(hight_chan // 2, hight_chan, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(hight_chan),
            )

        self.conv_low = ConvX(low_chan, hight_chan, kernel_size=3, padding=1, bias=False)
        self.conv_out = ConvX(hight_chan, out_chan, kernel_size=3, padding=1, bias=False)

    def _spatial_attention(self, x):
        """
        :param x: tensor, feature
        :return x: tensor, fused feature
        """
        mean_value = torch.max(x, dim=1, keepdim=True)[0]
        max_value = torch.mean(x, dim=1, keepdim=True)

        value = torch.cat([mean_value, max_value], dim=1)

        return value

    def _channel_attention(self, x):
        """
        :param x: tensor, feature
        :return x: tensor, fused feature
        """
        avg_value = F.adaptive_avg_pool2d(x, 1)
        max_value = torch.max(torch.max(x, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]
        value = torch.cat([avg_value, max_value], dim=1)

        return value


    def forward(self, x_high, x_low):
        """
        :param x_high: tensor, high-level feature
        :param x_low: tensor, low-level feature
        :return x: tensor, fused feature
        """
        h, w =  x_low.size()[2:]

        x_low = self.conv_low(x_low)
        x_high = F.interpolate(x_high, (h, w), mode='bilinear', align_corners=False)

        if self.u_type == 'sp':
            atten_high = self._spatial_attention(x_high)
            atten_low = self._spatial_attention(x_low)
        else:
            atten_high = self._channel_attention(x_high)
            atten_low = self._channel_attention(x_low)

        atten = torch.cat([atten_high, atten_low], dim=1)
        atten = torch.sigmoid(self.conv_atten(atten))

        x = x_high * atten + x_low * (1 - atten)
        x = self.conv_out(x)

        return x
