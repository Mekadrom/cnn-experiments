import sys
import torch.nn as nn

sys.path.append("..")

import nn_utils

def apply_residual_block(x, conv_block):
    residual = x
    for i, sub_conv_block in enumerate(conv_block):
        if i % 2 == 1:
            x = residual + sub_conv_block(x)
            residual = x
        else:
            x = sub_conv_block(x)
    return x
        
class CNNClassifier(nn.Module):
    def __init__(self, args, num_classes):
        super(CNNClassifier, self).__init__()

        self.args = args

        self.activation_function = nn_utils.create_activation_function(args.activation_function)

        def create_conv_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm=True):
            return nn.Sequential(*[
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                self.activation_function,
                nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
                nn.Dropout2d(args.dropout)
            ])

        self.initial_conv = nn.Sequential(
            create_conv_block(3, 16, kernel_size=7, stride=1, padding=(7-1)//2), # 192x256 -> 192x256
            nn.MaxPool2d(kernel_size=2, stride=2), # 192x256 -> 96x128
        )

        self.convs = nn.ModuleList([
            nn.ModuleList([create_conv_block(16, 16, kernel_size=3, stride=1, padding=(3-1)//2) for _ in range(6)]), # 96x128 -> 96x128
            create_conv_block(16, 32), # 96x128 -> 48x64
            nn.ModuleList([create_conv_block(32, 32, kernel_size=3, stride=1, padding=(3-1)//2) for _ in range(7)]), # 48x64 -> 48x64
            create_conv_block(32, 64), # 48x64 -> 24x32
            nn.ModuleList([create_conv_block(64, 64, kernel_size=3, stride=1, padding=(3-1)//2) for _ in range(11)]), # 24x32 -> 24x32
            create_conv_block(64, 128), # 24x32 -> 12x16
            nn.ModuleList([create_conv_block(128, 128, kernel_size=3, stride=1, padding=(3-1)//2) for _ in range(7)]), # 12x16 -> 12x16
            create_conv_block(128, 256), # 12x16 -> 6x8
            nn.ModuleList([create_conv_block(256, 256, kernel_size=3, stride=1, padding=(3-1)//2) for _ in range(6)]), # 6x8 -> 6x8
            create_conv_block(256, 512), # 6x8 -> 3x4
            nn.ModuleList([create_conv_block(512, 512, kernel_size=3, stride=1, padding=(3-1)//2) for _ in range(5)]), # 3x4 -> 3x4
        ])

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        flatten_dim = 512 * 1 * 1
        interp1 = flatten_dim // 2
        interp2 = interp1

        self.flatten = nn.Flatten()

        self.sequential = nn.Sequential(
            nn.Linear(flatten_dim, interp1),
            self.activation_function,
            nn.LayerNorm(interp1),
            nn.Dropout(args.dropout),
            nn.Linear(interp1, interp2),
            self.activation_function,
            nn.LayerNorm(interp2),
            nn.Dropout(args.dropout),
        )

        self.classifier = nn.Sequential(
            nn.Linear(interp2, num_classes),
            nn.Softmax(dim=1)
        )

        self.box_regressor = nn.Sequential(
            nn.Linear(interp2, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.initial_conv(x)
        # residual = x
        for conv_block in self.convs:
            if isinstance(conv_block, nn.ModuleList):
                x = apply_residual_block(x, conv_block)
                # residual = x
            else:
                # residual may need to be casted here
                # x = residual + conv_block(x)
                x = conv_block(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.sequential(x)
        return self.classifier(x), self.box_regressor(x)
