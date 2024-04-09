from einops import einsum, rearrange
from reverse_swin import Swinv2EncoderReverse
from transformers import Swinv2Config, Swinv2Model

import torch
import torch.nn as nn

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import nn_utils

class Encoder(nn.Module):
    def __init__(self, activation_function, latent_size, dropout):
        super(Encoder, self).__init__()

        self.activation_function = nn_utils.create_activation_function(activation_function)

        def conv_block(in_channels, out_channels, stride=2, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=padding),
                self.activation_function,
                nn.BatchNorm2d(out_channels),
                nn.Dropout2d(dropout)
            )

        self.conv1 = conv_block(3, 16) # 224x224 -> 112x112
        self.conv2 = conv_block(16, 32) # 112x112 -> 56x56
        self.conv3 = conv_block(32, 64) # 56x56 -> 28x28
        self.conv4 = conv_block(64, 128) # 28x28 -> 14x14
        self.conv5 = conv_block(128, 256) # 14x14 -> 7x7
        self.conv6 = conv_block(256, 512, padding=2) # 7x7 -> 4x4
        self.conv7 = conv_block(512, 1024) # 4x4 -> 2x2
        self.conv8 = conv_block(1024, 2048) # 2x2 -> 1x1

        self.flatten = nn.Flatten()

        flatten_dim = 2048
        interp1 = (flatten_dim + (latent_size * 2)) // 2

        self.latent = nn.Sequential(
            # nn.Linear(flatten_dim, interp1),
            # self.activation_function,
            # nn.LayerNorm(interp1),
            # nn.Dropout(dropout),
            # nn.Linear(interp1, latent_size * 2)
            nn.Linear(flatten_dim, latent_size * 2)
        )

    def forward(self, x: torch.Tensor):
        # x is of shape (batch_size, 3, 224, 224) eg (16, 3, 224, 224)
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

        self.activation_function = nn_utils.create_activation_function(activation_function)

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
            # nn.Linear(latent_size, interp1),
            # self.activation_function,
            # nn.LayerNorm(interp1),
            # nn.Dropout(dropout),
            # nn.Linear(interp1, flatten_dim),
            # self.activation_function,
            # nn.LayerNorm(flatten_dim),
            # nn.Dropout(dropout)
            nn.Linear(latent_size, flatten_dim),
            self.activation_function,
            nn.LayerNorm(flatten_dim),
            nn.Dropout(dropout)
        )

        self.unflatten = nn.Unflatten(1, (flatten_dim, 1, 1))

        self.deconv1 = deconv_block(2048, 1024) # 1x1 -> 2x2
        self.deconv2 = deconv_block(1024, 512) # 2x2 -> 4x4
        self.deconv3 = deconv_block(512, 256) # 4x4 -> 8x8
        self.deconv4 = deconv_block(256, 128, padding=2) # 8x8 -> 14x14
        self.deconv5 = deconv_block(128, 64) # 14x14 -> 28x28
        self.deconv6 = deconv_block(64, 32) # 28x28 -> 56x56
        self.deconv7 = deconv_block(32, 16) # 56x56 -> 112x112
        self.deconv8 = nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1) # 112x112 -> 224x224

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
    
class Decoder224(nn.Module):
    def __init__(self, activation_function, latent_size, dropout):
        super(Decoder224, self).__init__()

        self.activation_function = nn_utils.create_activation_function(activation_function)

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
        self.deconv4 = deconv_block(256, 128, padding=2) # 8x8 -> 14x14
        self.deconv5 = deconv_block(128, 64) # 14x14 -> 28x28
        self.deconv6 = deconv_block(64, 32) # 28x28 -> 56x56
        self.deconv7 = deconv_block(32, 16) # 56x56 -> 112x112
        self.deconv8 = nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1) # 112x112 -> 224x224

    def forward(self, x: torch.Tensor):
        x = self.latent(x)

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
        
class SwinEncoder(nn.Module):
    def __init__(self, latent_size, dropout):
        super(SwinEncoder, self).__init__()

        self.config = Swinv2Config(
            img_size=224,
            patch_size=4,
            in_chans=3,
            num_classes=0,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=dropout,
            attn_drop_rate=dropout,
            drop_path_rate=0.2,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=False
        )

        self.swin_model = Swinv2Model(self.config, add_pooling_layer=True)

        self.latent_expansion = nn.Linear(self.swin_model.num_features, latent_size * 2)

    def forward(self, x: torch.Tensor):
        x = self.swin_model(x)[1]
        x = self.latent_expansion(x)
        mu, log_var = torch.split(x, x.size(-1) // 2, dim=-1)
        return mu, log_var
    
class SwinDecoder(nn.Module):
    def __init__(self, orig_embed_dim, patch_grid, latent_size, dropout=0.0):
        super(SwinDecoder, self).__init__()

        self.config = Swinv2Config(
            img_size=224,
            patch_size=4,
            in_chans=3,
            num_classes=0,
            embed_dim=768,
            depths=[2, 6, 2, 2],
            num_heads=[24, 12, 6, 3],
            window_size=7,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=dropout,
            attn_drop_rate=dropout,
            drop_path_rate=0.2,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=False
        )

        self.num_layers = len(self.config.depths)
        self.num_features = int(orig_embed_dim * 2 ** (self.num_layers - 1))

        self.latent_repr = nn.Linear(latent_size, self.num_features * 49)

        self.decoder = Swinv2EncoderReverse(self.config, patch_grid)

        self.deconv = nn.ConvTranspose2d(self.config.embed_dim // (2**(self.num_layers-1)), 3, kernel_size=4, stride=4, padding=0)

        self.layernorm = nn.LayerNorm(3, eps=self.config.layer_norm_eps)

    def forward(self, z):
        z = self.latent_repr(z) # (batch_size, latent_size)
        z = z.view(-1, 49, z.size(-1) // 49)
        reconstruction = self.decoder(z, (7, 7))[0] # (batch_size, 3, H, W)

        reconstruction = rearrange(reconstruction, 'b (h w) c -> b c h w', h=self.config.image_size // 4, w=self.config.image_size // 4)
        reconstruction = self.deconv(reconstruction)
        reconstruction = reconstruction.permute(0, 2, 3, 1)
        reconstruction = self.layernorm(reconstruction)
        reconstruction = torch.sigmoid(reconstruction)
        return reconstruction.permute(0, 3, 1, 2)
