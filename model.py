import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv3D(nn.Module):
    """(Conv3d => BatchNorm => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNetFireEmulator3D(nn.Module):
    def __init__(self, in_channels=5, out_channels=1):
        super(UNetFireEmulator3D, self).__init__()
        
        # Encoder
        self.inc = DoubleConv3D(in_channels, 32)       # Reduced filters to save VRAM
        self.down1 = DoubleConv3D(32, 64)
        self.down2 = DoubleConv3D(64, 128)
        
        # Bottleneck
        self.bot = DoubleConv3D(128, 256)
        
        # Decoder
        # Using trilinear interpolation for upsampling (lighter than TransposeConv3d)
        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv1 = DoubleConv3D(256 + 128, 128)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv2 = DoubleConv3D(128 + 64, 64)
        
        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv3 = DoubleConv3D(64 + 32, 32)
        
        self.outc = nn.Conv3d(32, out_channels, kernel_size=1)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        # x shape: (Batch, 5, 30, 100, 100)
        
        # Down 1
        x1 = self.inc(x)
        p1 = self.pool(x1)
        
        # Down 2
        x2 = self.down1(p1)
        p2 = self.pool(x2)
        
        # Down 3
        x3 = self.down2(p2)
        p3 = self.pool(x3)
        
        # Bottom
        x4 = self.bot(p3)
        
        # Up 1
        x = self.up1(x4)
        x = torch.cat([x3, x], dim=1)
        x = self.conv1(x)
        
        # Up 2
        x = self.up2(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv2(x)
        
        # Up 3
        x = self.up3(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv3(x)
        
        logits = self.outc(x)
        return F.relu(logits)