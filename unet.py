import torch
import torch.nn as nn
import torch.nn.functional as F
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.time_embed = nn.Linear(time_channels, out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
    def forward(self, x, t_emb):
        residual = self.shortcut(x)
        x = self.conv1(x)
        t_emb = self.time_embed(t_emb).unsqueeze(-1).unsqueeze(-1)
        x = x + t_emb
        x = self.conv2(x)
        return x + residual
    
    
class UNet(nn.Module):
    def __init__(self, in_channels=5, out_channels=4, h_dim=64, time_dim=256):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.down1 = ResidualBlock(in_channels, h_dim, time_dim)
        self.down2 = ResidualBlock(h_dim, h_dim*2, time_dim)
        self.down3 = ResidualBlock(h_dim*2, h_dim*4, time_dim)
        self.bottleneck = ResidualBlock(h_dim*4, h_dim*4, time_dim)
        self.up3 = ResidualBlock(h_dim*8, h_dim*2, time_dim)
        self.up2 = ResidualBlock(h_dim*4, h_dim, time_dim)
        self.up1 = ResidualBlock(h_dim*2, h_dim, time_dim)
        self.out_conv = nn.Conv2d(h_dim, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
    def forward(self, x, t):
        t_emb = self.time_embed(t)
        # Downsample
        d1 = self.down1(x, t_emb)
        d2 = self.down2(self.pool(d1), t_emb)
        d3 = self.down3(self.pool(d2), t_emb)
        # Bottleneck
        b = self.bottleneck(self.pool(d3), t_emb)
        # Upsample
        u3 = self.upsample(b)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.up3(u3, t_emb)
        u2 = self.upsample(u3)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.up2(u2, t_emb)
        u1 = self.upsample(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.up1(u1, t_emb)
        return self.out_conv(u1)




