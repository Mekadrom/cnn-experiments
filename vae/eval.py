from dataloader import DataLoader
from model import VAEModel
from torch.utils.tensorboard import SummaryWriter

import argparse
import os
import random
import torch

argparser = argparse.ArgumentParser()

argparser.add_argument('--data_path', type=str, default=os.path.join('data', 'multidata'), help='Path to the data')
argparser.add_argument('--run_name', type=str, required=True, help='Name of the run')
argparser.add_argument('--model_name', type=str, default='model_0.pt', help='Name of the model to load')

argparser.add_argument('--device', type=str, default='cuda', help='Device to train on')

args, unk = argparser.parse_known_args()

run_dir = os.path.join("runs", args.run_name)
if not os.path.exists(run_dir):
    os.makedirs(run_dir)

summary_writer = SummaryWriter(log_dir=run_dir)

model = VAEModel(None, None).to(args.device)

model.load_state_dict(torch.load(os.path.join(run_dir, args.model_name)))

dataloader = DataLoader(args, args.data_path, "validation", 1)

test_image = random.choice(random.choice(dataloader)).to(args.device)
reconstructed_image, _, _ = model(test_image.unsqueeze(0))

# combine original and reconstructed image into a single image for side-by-side comparison
combined_image = torch.cat([test_image, reconstructed_image[0]], dim=2)

summary_writer.add_image("reconstruction", combined_image, 0)