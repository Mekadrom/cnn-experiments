from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

import os
import random
import torch
import torchvision.transforms.functional as F

def convert_to_rgb(img):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img

def pad_to_size(target_size):
    def pad(img):
        width, height = img.size
        pad_width = target_size[1] - width
        pad_height = target_size[0] - height
        padding = (pad_width // 2, pad_height // 2, (pad_width + 1) // 2, (pad_height + 1) // 2)
        return F.pad(img, padding, fill=0, padding_mode='constant')
    return pad

class DataLoader(Dataset):
    def __init__(self, args, data_path, split, batch_size, do_load=True):
        self.args = args
        self.data_path = data_path
        self.split = split
        self.batch_size = batch_size        

        if do_load:
            self.data = []
            self.load_data()
            self.create_batches()

    def load_data(self):
        data_path = os.path.join(self.data_path, self.split)
        print(f"Loading data from {data_path}...")

        transform = transforms.Compose([
            # print shape
            # transforms.Lambda(lambda img: print(img.size) or img),
            pad_to_size((768, 1024)),
            transforms.CenterCrop((768, 768)),
            transforms.Resize((224, 224)),
            transforms.Lambda(convert_to_rgb),
            transforms.ToTensor()
        ])

        data_paths = os.listdir(data_path)[:100]

        # load images from data_path/split and simply store them as a list of tensors
        for image in tqdm(data_paths, desc=f"Loading {self.split} data"):
            if image.endswith(".jpg"):
                img = Image.open(os.path.join(data_path, image))
                if img is not None:
                    self.data.append(transform(img))

        # add horizontal flips of the images
        print("Adding horizontal flips...")
        self.data += [torch.flip(img, [2]) for img in self.data]

        # add vertical flips of the images
        print("Adding vertical flips...")
        self.data += [torch.flip(img, [1]) for img in self.data]

        # add vertical/horizontal flips of the images
        print("Adding vertical/horizontal flips...")
        self.data += [torch.flip(img, [1, 2]) for img in self.data]

    def create_batches(self):
        random.shuffle(self.data)

        self.all_batches = []
        for i in range(0, len(self.data), self.batch_size):
            self.all_batches.append(self.data[i:i+self.batch_size])

        random.shuffle(self.all_batches)
        self.n_batches = len(self.all_batches)
        self.current_batch = -1

    def __iter__(self):
        return self
    
    def __next__(self):
        self.current_batch += 1
        try:
            return torch.stack(self.all_batches[self.current_batch]), None
        except IndexError:
            raise StopIteration

    def __len__(self):
        return self.n_batches
    
    def __getitem__(self, idx):
        return self.all_batches[idx], None
    
    def clear(self):
        self.data = []
        self.all_batches = []
        self.n_batches = 0
        self.current_batch = -1
