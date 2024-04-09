from PIL import Image, ImageDraw, ImageFont
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import argparse
import model
import os
import random
import torch
import torchvision
import torchvision.transforms as transforms

def pad_to_size(target_size):
    def pad(img):
        width, height = img.size
        pad_width = target_size[1] - width
        pad_height = target_size[0] - height
        padding = (pad_width // 2, pad_height // 2, (pad_width + 1) // 2, (pad_height + 1) // 2)
        return torchvision.transforms.functional.pad(img, padding, fill=0, padding_mode='constant'), pad_width // 2, pad_height // 2
    return pad

transform = transforms.Compose([
    transforms.Resize((192, 256)),
    # makes sure img is RGB
    transforms.Lambda(lambda img: img.convert('RGB')),
    transforms.ToTensor(),
])

with open(os.path.join('..', 'classes.txt'), 'r') as f:
    class_names = [x.strip().lower().replace(" ", "_") for x in f.readlines()]

def process_split(data_path, output_data_path, split, img_file_names):
    for img_file_name in tqdm(img_file_names, desc=f"Processing '{split}' data"):
        label_file = os.path.join(data_path, split, 'labels', img_file_name.replace('.jpg', '.txt'))
        with open(label_file, 'r') as f:
            # read only first line
            label = f.readline().strip()
            # label line consists of (class, x1, y1, x1, y1); the bounding box needs to be scaled down by 4x
            label = label.split(' ')
            class_name = label[0]
            class_id = class_names.index(class_name)

            label_x1 = float(label[1])
            label_y1 = float(label[2])
            label_x2 = float(label[3])
            label_y2 = float(label[4])

            img = Image.open(os.path.join(data_path, split, img_file_name))

            img, pad_width, pad_height = pad_to_size((768, 1024))(img)

            pad_height = min(768, max(0, pad_height))
            pad_width = min(1024, max(0, pad_width))

            label_x1 += pad_width
            label_y1 += pad_height
            label_x2 += pad_width
            label_y2 += pad_height

            img = transform(img)

            label_x1 /= 4.0
            label_y1 /= 4.0
            label_x2 /= 4.0
            label_y2 /= 4.0

            label_x1 /= 256.0
            label_y1 /= 192.0
            label_x2 /= 256.0
            label_y2 /= 192.0

            label = torch.FloatTensor([label_x1, label_y1, label_x2, label_y2])
            label = torch.clamp(label, 0.0, 1.0) # clamp to [0, 1]

            with open(os.path.join(output_data_path, split, img_file_name), 'w') as f:
                img = (img.permute(1, 2, 0) * 255.0).to(torch.uint8).numpy()
                pil_image = Image.fromarray(img)
                pil_image.save(f, 'JPEG')

            with open(os.path.join(output_data_path, split, 'labels', img_file_name.replace("jpg", "txt")), 'w') as f:
                f.write(f"{class_id},{label[0]},{label[1]},{label[2]},{label[3]}\n")

            # print(f"Processed {img_file_name}")

def preprocess_set(data_path, output_data_path, split):
    input_img_paths = [f for f in os.listdir(os.path.join(data_path, split)) if f.endswith('.jpg')]

    os.makedirs(os.path.join(args.output_data_path, split), exist_ok=True)
    os.makedirs(os.path.join(args.output_data_path, split, 'labels'), exist_ok=True)

    process_split(data_path, output_data_path, split, input_img_paths)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--data_path', type=str, default=os.path.join('..', 'data', 'multidata'), help='Path to the data')
    argparser.add_argument('--output_data_path', type=str, default=os.path.join('..', 'data', 'preprocessed'), help='Path to the output data')

    args, unk = argparser.parse_known_args()

    os.makedirs(args.output_data_path, exist_ok=True)

    preprocess_set(args.data_path, args.output_data_path, 'train')
    preprocess_set(args.data_path, args.output_data_path, 'test')
    preprocess_set(args.data_path, args.output_data_path, 'validation')
