import torch
import torch.nn as nn
import vae.utils as utils

class Encoder(nn.Module):
    def __init__(self, activation_function, latent_size, dropout):
        super(Encoder, self).__init__()

        self.activation_function = utils.create_activation_function(activation_function)

        def conv_block(in_channels, out_channels, stride=2, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=padding),
                self.activation_function,
                nn.BatchNorm2d(out_channels),
                nn.Dropout2d(dropout)
            )

        self.conv1 = conv_block(3, 16) # 256x256 -> 128x128
        self.conv2 = conv_block(16, 32) # 128x128 -> 64x64
        self.conv3 = conv_block(32, 64) # 64x64 -> 32x32
        self.conv4 = conv_block(64, 128) # 32x32 -> 16x16
        self.conv5 = conv_block(128, 256) # 16x16 -> 8x8
        self.conv6 = conv_block(256, 512) # 8x8 -> 4x4
        self.conv7 = conv_block(512, 1024) # 4x4 -> 2x2
        self.conv8 = conv_block(1024, 2048) # 2x2 -> 1x1

        self.flatten = nn.Flatten()

        flatten_dim = 2048
        interp1 = (flatten_dim + (latent_size * 2)) // 2

        self.latent = nn.Sequential(
            nn.Linear(flatten_dim, interp1),
            self.activation_function,
            nn.LayerNorm(interp1),
            nn.Dropout(dropout),
            nn.Linear(interp1, latent_size * 2)
        )

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.flatten(x)
        x = self.latent(x)
        mu, log_var = torch.split(x, x.size(1) // 2, dim=1)
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, activation_function, latent_size, dropout):
        super(Decoder, self).__init__()

        self.activation_function = utils.create_activation_function(activation_function)

        def deconv_block(in_channels, out_channels, stride=2, padding=1, output_padding=0):
            layers = [
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=padding, output_padding=output_padding),
                self.activation_function,
                nn.BatchNorm2d(out_channels),
                nn.Dropout2d(dropout)
            ]
            return nn.Sequential(*layers)

        flatten_dim = 2048
        interp1 = flatten_dim

        self.latent = nn.Sequential(
            nn.Linear(latent_size, interp1),
            self.activation_function,
            nn.LayerNorm(interp1),
            nn.Dropout(dropout),
            nn.Linear(interp1, flatten_dim),
            self.activation_function,
            nn.LayerNorm(flatten_dim),
            nn.Dropout(dropout)
        )

        self.unflatten = nn.Unflatten(1, (2048, 1, 1))

        self.deconv1 = deconv_block(2048, 1024) # 1x1 -> 2x2
        self.deconv2 = deconv_block(1024, 512) # 2x2 -> 4x4
        self.deconv3 = deconv_block(512, 256) # 4x4 -> 8x8
        self.deconv4 = deconv_block(256, 128) # 8x8 -> 16x16
        self.deconv5 = deconv_block(128, 64) # 16x16 -> 32x32
        self.deconv6 = deconv_block(64, 32) # 32x32 -> 64x64
        self.deconv7 = deconv_block(32, 16) # 64x64 -> 128x128
        self.deconv8 = nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1) # 128x128 -> 256x256

    def forward(self, x: torch.Tensor):
        x = self.latent(x)

        # reshape to be feature maps (essentially 512 images of size 3x4)
        x = self.unflatten(x)

        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        x = self.deconv6(x)
        x = self.deconv7(x)
        x = self.deconv8(x)
        x = torch.sigmoid(x) # sigmoid is applied in the loss function for BCE only
        return x
        