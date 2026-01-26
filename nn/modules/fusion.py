import torch
import torch.nn as nn
import torch.nn.functional as F

class SplitInput(nn.Module):
    def __init__(self, c1=None):
        super().__init__()
    
    def forward(self, x):
        # x: [Batch, 6, H, W]
        c = x.shape[1] // 2
        return [x[:, :c, ...], x[:, c:, ...]]

class SelectItem(nn.Module):
    def __init__(self, index=0):
        super().__init__()
        self.index = index

    def forward(self, x):
        return x[self.index]

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        inter_channels = max(1, in_planes // ratio)
        self.f1 = nn.Conv2d(in_planes, inter_channels, 1, bias=False)
        self.relu = nn.ReLU()
        self.f2 = nn.Conv2d(inter_channels, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.f2(self.relu(self.f1(self.avg_pool(x))))
        max_out = self.f2(self.relu(self.f1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x))

class CGAFusion(nn.Module):
    def __init__(self, c1, reduction=8):
        super().__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(c1, ratio=reduction)
        self.pa = nn.Sequential(
            nn.Conv2d(2 * c1, c1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        rgb, depth = x[0], x[1]
        if rgb.shape != depth.shape:
            depth = F.interpolate(depth, size=rgb.shape[2:], mode='bilinear', align_corners=False)
        initial = rgb + depth
        sa_out = self.sa(initial)
        ca_out = self.ca(initial)
        mask = self.pa(torch.cat([rgb, depth], dim=1))
        return initial + (rgb * mask + depth * (1 - mask))
