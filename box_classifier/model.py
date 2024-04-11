import sys
import torch
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
        
class ResNet34(nn.Module):
    def __init__(self, args, num_classes):
        super(ResNet34, self).__init__()

        self.args = args

        self.activation_function = nn_utils.create_activation_function(args.activation_function)

        def create_conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=None, batch_norm=True, pooling=False, dropout=True):
            if padding is None:
                padding = (kernel_size - 1) // 2

            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                self.activation_function,
                nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
                nn.MaxPool2d(kernel_size=2, stride=2) if pooling else nn.Identity(),
                nn.Dropout(args.dropout) if dropout else nn.Identity()
            ]

            return nn.Sequential(*layers)
        
        # sobel = nn.Conv2d(3, 2, 3, 1, 1, bias=False)
        # sobel.weight.data = nn.Parameter(torch.FloatTensor([[
        #     [[-1,  0,  1], 
        #      [-2,  0,  2],
        #      [-1,  0,  1]], 
        #     [[-1, -2, -1], 
        #      [ 0,  0,  0], 
        #      [ 1,  2,  1]]
        # ]]))
        # sobel.weight.requires_grad = False

        # scharr = nn.Conv2d(3, 2, 3, 1, 1, bias=False)
        # scharr.weight.data = nn.Parameter(torch.FloatTensor([[
        #     [[-3,  0,  3], 
        #      [-10, 0, 10],
        #      [-3,  0,  3]], 
        #     [[-3, -10, -3], 
        #      [ 0,  0,  0], 
        #      [ 3,  10,  3]]
        # ]]))
        # scharr.weight.requires_grad = False

        # prewitt = nn.Conv2d(3, 2, 3, 1, 1, bias=False)
        # prewitt.weight.data = nn.Parameter(torch.FloatTensor([[
        #     [[-1,  0,  1], 
        #      [-1,  0,  1],
        #      [-1,  0,  1]], 
        #     [[-1, -1, -1], 
        #      [ 0,  0,  0], 
        #      [ 1,  1,  1]]
        # ]]))
        # prewitt.weight.requires_grad = False

        # laplacian = nn.Conv2d(3, 2, 3, 1, 1, bias=False)
        # laplacian.weight.data = nn.Parameter(torch.FloatTensor([[
        #     [[ 0,  1,  0], 
        #      [ 1, -4,  1],
        #      [ 0,  1,  0]],
        #     [[-1, -1, -1],
        #      [-1,  8, -1],
        #      [-1, -1, -1]]
        # ]]))
        # laplacian.weight.requires_grad = False

        # gaussian = nn.Conv2d(3, 1, 3, 1, 1, bias=False)
        # gaussian.weight.data = nn.Parameter(torch.FloatTensor([[
        #     [[1, 2, 1], 
        #      [2, 4, 2],
        #      [1, 2, 1]]
        # ]]))
        # gaussian.weight.requires_grad = False

        # self.hardcoded_convs = nn.ModuleList([
        #     sobel,
        #     scharr,
        #     prewitt,
        #     laplacian,
        #     gaussian
        # ])

        # self.initial_conv = nn.Conv2d(3, 64, 7, 2, (7-1) // 2) # 112 -> 112

        self.initial_conv = nn.Sequential(*[
            nn.Conv2d(3, 64, 7, 2, (7-1) // 2), # 224 -> 112
            self.activation_function,
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # 112 -> 56
        ])

        self.convs = nn.ModuleList([
            nn.ModuleList([create_conv_block(64, 64, kernel_size=3, stride=1, padding=1) for _ in range(3)]), # 56 -> 56
            create_conv_block(64, 128, stride=2), # 56 -> 28
            nn.ModuleList([create_conv_block(128, 128, kernel_size=3, stride=1, padding=1) for _ in range(3)]), # 28 -> 28
            create_conv_block(128, 256, stride=2), # 28 -> 14
            nn.ModuleList([create_conv_block(256, 256, kernel_size=3, stride=1, padding=1) for _ in range(5)]), # 14 -> 14
            create_conv_block(256, 512, stride=2), # 14 -> 7
            nn.ModuleList([create_conv_block(512, 512, kernel_size=3, stride=1, padding=1) for _ in range(2)]), # 7 -> 7
        ])

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        flatten_dim = 512 * 1 * 1
        interp1 = flatten_dim

        self.flatten = nn.Flatten()

        self.sequential = nn.Sequential(
            nn.Linear(flatten_dim, interp1),
            self.activation_function,
            nn.LayerNorm(interp1),
            nn.Dropout(args.dropout),
        )

        self.classifier = nn.Sequential(
            nn.Linear(interp1, num_classes),
            nn.Softmax(dim=1)
        )

        self.box_regressor = nn.Sequential(
            nn.Linear(interp1, 4),
            nn.Sigmoid()
        )

        self.init_weights(self)

    def init_weights(self, m):
        print(f"Initializing weights for {m}")
        if isinstance(m, nn.ModuleList) or isinstance(m, nn.Sequential):
            for sub_m in m:
                self.init_weights(sub_m)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Module):
            for child in m.children():
                self.init_weights(child)

    def forward(self, x):
        # hardcoded_features = torch.stack([conv(x) for conv in self.hardcoded_convs], dim=1)
        x = self.initial_conv(x)

        # x = torch.cat([x, hardcoded_features], dim=1)

        # x = self.initial_conv(x) # expects 64 features so goes after concat

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
