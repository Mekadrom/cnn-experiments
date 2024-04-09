from torch.utils.tensorboard import SummaryWriter
from typing import Optional
from torchmetrics import LogCoshError

import os
import random
import torch
import torch.nn as nn

class Trainer:
    def __init__(self, args, train_dataloader_provider, test_dataloader_provider, vae):
        self.args = args
        self.train_dataloader_provider = train_dataloader_provider
        self.test_dataloader_provider = test_dataloader_provider
        self.vae = vae

        self.run_dir = os.path.join("runs", args.run_name)
        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)

        self.summary_writer = SummaryWriter(log_dir=self.run_dir)

        print(f"model: {self.vae}")
        print(f"number of encoder params: {sum(p.numel() for p in self.vae.encoder.parameters()):,}")
        print(f"number of decoder params: {sum(p.numel() for p in self.vae.decoder.parameters()):,}")
        print(f"total number of model params: {sum(p.numel() for p in self.vae.parameters()):,}")

        print(f"hi cindy: {self.vae.encoder.latent[0].weight.shape}")
        print(f"hi cindy: {self.vae.encoder.latent[0].bias.shape}")

        self.vae_criterion: Optional[nn.Module] = None
        if args.criterion == 'bce':
            self.vae_criterion = nn.BCEWithLogitsLoss(reduction=args.criterion_reduction)
        elif args.criterion == 'mse':
            self.vae_criterion = nn.MSELoss(reduction=args.criterion_reduction)
        elif args.criterion == 'se':
            self.vae_criterion = nn.SmoothL1Loss(reduction=args.criterion_reduction)
        elif args.criterion == 'log_cosh':
            self.vae_criterion = LogCoshError()
        elif args.criterion == 'l1':
            self.vae_criterion = nn.L1Loss(reduction=args.criterion_reduction)
        else:
            raise ValueError(f"Unknown criterion: {args.criterion}")

        print(f"VAE Criterion: {self.vae_criterion}")

        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=args.lr)

        self.train_steps = 0

    def get_data_loaders(self, split):
        return self.train_dataloader_provider() if split == 'train' else self.test_dataloader_provider()

    def get_image_example(self, dataloader, test_image=None):
        self.vae.eval()

        if test_image is None:
            if hasattr(dataloader, 'dataset'):
                print('grabbing dataset example')
                test_image = random.choice(dataloader.dataset)[0].to(self.args.device)
            elif hasattr(dataloader, 'data'):
                print('grabbing data example')
                test_image = random.choice(dataloader.data).to(self.args.device)
            else:
                raise ValueError(f"Unknown dataloader type: {dataloader}")

        reconstructed_image, _, _ = self.vae(test_image.unsqueeze(0))

        # combine original and reconstructed image into a single image for side-by-side comparison
        combined_image = torch.cat([test_image, reconstructed_image[0]], dim=2)

        return torch.nn.functional.interpolate(combined_image.unsqueeze(0), scale_factor=3, mode='bilinear', align_corners=False).squeeze(0)

    def train(self):
        raise NotImplementedError
