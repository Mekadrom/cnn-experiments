from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms import AutoAugmentPolicy
from transformers import Swinv2Config, Swinv2Model
from timm.scheduler.cosine_lr import CosineLRScheduler
from vae.reverse_swin import Swinv2EncoderReverse

import argparse
import os
import torch
import torch.nn as nn

def train(args):
    run_dir = os.path.join('..', 'runs', 'swin', args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    summary_writer = SummaryWriter(run_dir)

    config = Swinv2Config(
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=args.dropout,
        attn_drop_rate=args.dropout,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False
    )

    model = Swinv2Model(config)

    def crop_or_pad_to_multiple_of(img_size):
        def crop_or_pad(x):
            h, w = x.shape[-2:]
            h_pad = (h // img_size + 1) * img_size - h
            w_pad = (w // img_size + 1) * img_size - w
            x = nn.functional.pad(x, (0, w_pad, 0, h_pad))
            return x
        return crop_or_pad

    # dataset images are various sizes like 512x1024 and 768x1024, so crop/pad/resize to 224x224
    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
        transforms.Lambda(crop_or_pad_to_multiple_of(224)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # dataset_augmentations = [
    #     transforms.RandomHorizontalFlip(1.0),
    #     transforms.RandomVerticalFlip(1.0),
    #     transforms.Compose([
    #         transforms.RandomHorizontalFlip(1.0),
    #         transforms.RandomVerticalFlip(1.0),
    #     ]),
    #     transforms.RandomRotation(180),
    #     transforms.ElasticTransform(0.1),
    #     transforms.Grayscale(3),
    #     transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    #     transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)),

    # ]

    dataset_augmentations = [
        transforms.AutoAugment(policy=AutoAugmentPolicy.SVHN),
        transforms.AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
    ]

    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr)

    criterion = nn.CrossEntropyLoss()

    n_iter_per_epoch = len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS
    num_steps = int(config.TRAIN.EPOCHS * n_iter_per_epoch)
    warmup_steps = int(config.TRAIN.WARMUP_EPOCHS * n_iter_per_epoch)
    decay_steps = int(config.TRAIN.LR_SCHEDULER.DECAY_EPOCHS * n_iter_per_epoch)
    multi_steps = [i * n_iter_per_epoch for i in config.TRAIN.LR_SCHEDULER.MULTISTEPS]
    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=(num_steps - warmup_steps) if config.TRAIN.LR_SCHEDULER.WARMUP_PREFIX else num_steps,
        t_mul=1.,
        lr_min=config.TRAIN.MIN_LR,
        warmup_lr_init=config.TRAIN.WARMUP_LR,
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
        warmup_prefix=config.TRAIN.LR_SCHEDULER.WARMUP_PREFIX,
    )

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--data_path', type=str, default=os.path.join('..', 'data', 'multidata'), help='Path to the data')
    argparser.add_argument('--run_name', type=str, required=True, help='Name of the run')

    argparser.add_argument('--n_epochs', type=int, default=2000, help='Number of epochs to train for')
    argparser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    argparser.add_argument('--activation_function', type=str, default='gelu', help='Activation function to use', choices=['relu', 'gelu', 'elu', 'selu', 'prelu', 'leaky_relu', 'tanh', 'sigmoid'])
    argparser.add_argument('--dropout', type=float, default=0.20, help='Dropout rate')
    argparser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    argparser.add_argument('--batches_to_accumulate', type=int, default=2, help='Number of batches to accumulate before performing an optimization step')

    argparser.add_argument('--test_image_steps', type=int, default=100, help='Number of steps between test image generation')
    argparser.add_argument('--device', type=str, default='cuda', help='Device to train on')
    argparser.add_argument('--detect_nans', action='store_true', help='Detect NaNs in the model')
    argparser.add_argument('--seed', type=int, default=None, help='Seed for reproducibility')

    args, unk = argparser.parse_known_args()

    train(args)
