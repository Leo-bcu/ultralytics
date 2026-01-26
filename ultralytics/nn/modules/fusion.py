import torch
import torch.nn as nn
import torch.nn.functional as F

# =======================================================
# 1. 核心切分模块 (修复了 __init__ 报错)
# =======================================================
class SplitInput(nn.Module):
    # === 关键修改：将 c1 设为默认参数 None，防止报错 ===
    def __init__(self, c1=None):
        super().__init__()
    
    def forward(self, x):
        # x: [Batch, 6, H, W] -> (Batch, 3, H, W), (Batch, 3, H, W)
        # 动态获取通道数并对半切分
        c = x.shape[1] // 2
        return [x[:, :c, ...], x[:, c:, ...]]

# =======================================================
# 2. 选择模块
# =======================================================
class SelectItem(nn.Module):
    def __init__(self, index=0):
        super().__init__()
        self.index = index # 0=RGB, 1=Depth

    def forward(self, x):
        return x[self.index]

# =======================================================
# 3. 基础注意力组件
# =======================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 确保中间通道至少为1
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
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x))

# =======================================================
# 4. 融合模块 CGAFusion
# =======================================================
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
        # x 是列表 [rgb_feat, depth_feat]
        rgb, depth = x[0], x[1]
        
        # 尺寸对齐保护
        if rgb.shape != depth.shape:
            depth = F.interpolate(depth, size=rgb.shape[2:], mode='bilinear', align_corners=False)

        initial = rgb + depth
        sa_out = self.sa(initial)
        ca_out = self.ca(initial)
        mask = self.pa(torch.cat([rgb, depth], dim=1))
        
        fused = initial + (rgb * mask + depth * (1 - mask))
        return fused