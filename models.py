import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels, num_blocks, base_channels=64, kernel_size=3, dropout=0.0):
        super(Encoder, self).__init__()
        layers = []
        channels = in_channels
        for i in range(num_blocks):
            layers.append(ConvBlock(channels, base_channels * (2 ** i), kernel_size, dropout))
            channels = base_channels * (2 ** i)
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, out_channels, num_blocks, base_channels=64, kernel_size=3, dropout=0.0):
        super(Decoder, self).__init__()
        layers = []
        for i in range(num_blocks - 1, -1, -1):
            layers.append(nn.ConvTranspose2d(base_channels * (2 ** (i+1)), base_channels * (2 ** i), kernel_size=2, stride=2))
            layers.append(ConvBlock(base_channels * (2 ** i), base_channels * (2 ** i), kernel_size, dropout))
        layers.append(nn.ConvTranspose2d(base_channels, out_channels, kernel_size=2, stride=2))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)

class FCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, base_channels=64, kernel_size=3, dropout=0.0):
        super(FCN, self).__init__()
        self.encoder = Encoder(in_channels, num_blocks, base_channels, kernel_size, dropout)
        self.decoder = Decoder(out_channels, num_blocks, base_channels, kernel_size, dropout)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x