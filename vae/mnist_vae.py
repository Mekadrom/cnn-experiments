import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import nn_utils

class MaxAvgPool(nn.Module):
    def __init__(self):
        super(MaxAvgPool, self).__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        max_pool = self.max_pool(x[:, :x.size(1) // 2, :, :])
        avg_pool = self.avg_pool(x[:, x.size(1) // 2:, :, :])

        return torch.cat((max_pool, avg_pool), dim=1)

class Encoder(nn.Module):
    def __init__(self, activation_function, latent_size, dropout):
        super(Encoder, self).__init__()

        self.activation_function = nn_utils.create_activation_function(activation_function)

        # self.pooling_layer = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pooling_layer = MaxAvgPool()

        def conv_block(in_channels, out_channels, stride=2, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=padding),
                self.activation_function,
            )

        self.conv1 = conv_block(1, 8) # 32x32 -> 16x16
        self.conv2 = conv_block(8, 16) # 16x16 -> 8x8
        self.conv3 = conv_block(16, 32) # 8x8 -> 4x4
        self.conv4 = conv_block(32, 64, padding=0) # 4x4 -> 1x1

        self.fc1 = nn.Linear(64, 128)
        self.mu = nn.Linear(128, latent_size)
        self.log_var = nn.Linear(128, latent_size)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).reshape(x.size(0), -1)
        hidden = self.fc1(x)

        mu = self.mu(hidden)
        log_var = self.log_var(hidden)
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, activation_function, latent_size, dropout):
        super(Decoder, self).__init__()

        self.activation_function = nn_utils.create_activation_function(activation_function)

        def deconv_block(in_channels, out_channels, stride=2, padding=1, output_padding=0):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=padding, output_padding=output_padding),
                self.activation_function,
            )

        self.latent = nn.Linear(latent_size, 64)

        self.deconv1 = deconv_block(64, 32, stride=1, padding=0) # 1x1 -> 4x4
        self.deconv2 = deconv_block(32, 16)
        self.deconv3 = deconv_block(16, 8)
        self.deconv4 = nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, z: torch.Tensor):
        z = self.latent(z)
        z = z.view(-1, 64, 1, 1)
        x = self.deconv1(z)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        return x
        