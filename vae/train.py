from model import VAEModel
from normal_trainer import NormalTrainer
from trainer import Trainer
from typing import Optional

import dataloader
import mnist_vae
import numpy as np
import oid_vae
import random
import torch
import torchvision
import torchvision.transforms as transforms
import vae.utils as utils

def get_mnist_dataloaders():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.MNIST(root='../data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root='../data', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, test_loader

if __name__ == "__main__":
    args, unk = utils.get_args()

    # set seeds
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(args.seed)
        np.random.seed(args.seed)

    encoder: Optional[torch.nn.Module] = None
    decoder: Optional[torch.nn.Module] = None

    if args.dataset == 'mnist':
        train_dataloader, test_dataloader = get_mnist_dataloaders()

        train_dataloader_provider = lambda: train_dataloader
        test_dataloader_provider = lambda: test_dataloader

        encoder = mnist_vae.Encoder(args.activation_function, args.latent_size, args.dropout)
        decoder = mnist_vae.Decoder(args.activation_function, args.latent_size, args.dropout)
    elif args.dataset == 'oidv6':
        train_dataloader_provider = lambda: dataloader.DataLoader(args, args.data_path, 'train', args.batch_size)
        test_dataloader_provider = lambda: dataloader.DataLoader(args, args.data_path, 'test', args.batch_size)

        encoder = oid_vae.Encoder(args.activation_function, args.latent_size, args.dropout)
        decoder = oid_vae.Decoder(args.activation_function, args.latent_size, args.dropout)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    vae = VAEModel(encoder, decoder).to(args.device)

    trainer: Optional[Trainer] = None
    if args.trainer == 'normal':
        trainer = NormalTrainer(args, train_dataloader_provider, test_dataloader_provider, vae)
    else:
        raise ValueError(f"Unknown trainer: {args.trainer}")

    torch.autograd.set_detect_anomaly(args.detect_nans) # evil, remove before actual runs

    trainer.train()
