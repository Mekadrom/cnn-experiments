from PIL import Image, ImageDraw, ImageFont
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import argparse
import model
import os
import random
import sys
import torch
import torchvision
import torchvision.transforms as transforms

sys.path.append("..")

import nn_utils

class_names = set()

with open(os.path.join('..', 'classes.txt'), 'r') as f:
    class_names = [x.strip() for x in f.readlines()]

def load_split(data_path, split):
    global class_names

    img_dir = os.path.join(data_path, split)
    label_dir = os.path.join(data_path, split, 'labels')

    img_file_names = [x for x in os.listdir(img_dir) if x.endswith('.jpg')]

    data = []
    for img_file_name in tqdm(img_file_names, desc=f"Loading {split} data"):
        label_file = os.path.join(label_dir, img_file_name.replace('.jpg', '.txt'))
        with open(label_file, 'r') as f:
            # read only first line
            label = f.readline().strip()
            # label line consists of (class_id, x1, y1, x1, y1)
            label = label.split(',')

            class_id = int(label[0])
            label_x1 = float(label[1])
            label_y1 = float(label[2])
            label_x2 = float(label[3])
            label_y2 = float(label[4])

            img = Image.open(os.path.join(img_dir, img_file_name))
            img = transforms.ToTensor()(img)

            data.append((img, (torch.LongTensor(class_id), torch.tensor([label_x1, label_y1, label_x2, label_y2]))))
    return data

try:
    fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 11)
except:
    fnt = ImageFont.load_default()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--data_path', type=str, default=os.path.join('..', 'data', 'preprocessed'), help='Path to the data')
    argparser.add_argument('--run_name', type=str, required=True, help='Name of the run')

    argparser.add_argument('--n_epochs', type=int, default=2000, help='Number of epochs to train for')
    argparser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    argparser.add_argument('--activation_function', type=str, default='relu', help='Activation function to use', choices=['relu', 'gelu', 'elu', 'selu', 'prelu', 'leaky_relu', 'tanh', 'sigmoid'])
    argparser.add_argument('--latent_size', type=int, default=1200, help='Size of the latent vector')
    argparser.add_argument('--dropout', type=float, default=0.20, help='Dropout rate')
    argparser.add_argument('--batch_size', type=int, default=32, help='Batch size')

    argparser.add_argument('--device', type=str, default='cuda', help='Device to train on')
    argparser.add_argument('--detect_nans', action='store_true', help='Detect NaNs in the model')
    argparser.add_argument('--seed', type=int, default=None, help='Seed for reproducibility')

    args, unk = argparser.parse_known_args()

    run_dir = os.path.join("runs", args.run_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    summary_writer = SummaryWriter(log_dir=run_dir)

    torch.autograd.set_detect_anomaly(args.detect_nans) # evil, remove before actual runs

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    train_data = load_split(args.data_path, 'train')
    val_data = load_split(args.data_path, 'validation')

    train_data = [(img, (class_names.index(class_name), label)) for img, (class_name, label) in tqdm(train_data, desc='Processing train data')]
    val_data = [(img, (class_names.index(class_name), label)) for img, (class_name, label) in tqdm(val_data, desc='Processing val data')]

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    classifier_regressor = model.CNNClassifier(args, num_classes=len(class_names)).to(args.device)
    optimizer = torch.optim.Adam(classifier_regressor.parameters(), lr=args.lr)

    classifier_criterion = torch.nn.CrossEntropyLoss()
    regression_criterion = torch.nn.MSELoss()

    print(f"model: {classifier_regressor}")
    print(f"total number of model params: {sum(p.numel() for p in classifier_regressor.parameters()):,}")

    train_steps = 0
    for epoch in range(args.n_epochs):
        classifier_regressor.train()
        
        losses = nn_utils.AverageMeter()
        for i, (img, (class_idx, label)) in enumerate(tqdm(train_loader, desc='Training')):
            img = img.to(args.device)
            class_idx = class_idx.to(args.device).long()

            label_x1, label_y1, label_x2, label_y2 = label.split(1, dim=1)
            label_x1 = label_x1.to(args.device).squeeze(1)
            label_y1 = label_y1.to(args.device).squeeze(1)
            label_x2 = label_x2.to(args.device).squeeze(1)
            label_y2 = label_y2.to(args.device).squeeze(1)

            optimizer.zero_grad()

            classification, regression = classifier_regressor(img)
            
            classification_loss = classifier_criterion(classification, class_idx)
            regression_loss_x1 = regression_criterion(regression[:, 0], label_x1)
            regression_loss_y1 = regression_criterion(regression[:, 1], label_y1)
            regression_loss_x2 = regression_criterion(regression[:, 2], label_x2)
            regression_loss_y2 = regression_criterion(regression[:, 3], label_y2)

            regression_loss = regression_loss_x1 + regression_loss_y1 + regression_loss_x2 + regression_loss_y2


            loss = classification_loss + regression_loss

            losses.update(loss.item(), img.size(0))

            loss.backward()

            optimizer.step()

            summary_writer.add_scalar('train_loss', loss.item(), train_steps)
            train_steps += 1

        print(f'Epoch {epoch}, Step {i}, Loss: {losses.avg}')

        classifier_regressor.eval()

        with torch.no_grad():
            losses = nn_utils.AverageMeter()
            for i, (img, (class_idx, label)) in enumerate(tqdm(val_loader, desc='Validation')):
                img = img.to(args.device)
                class_idx = class_idx.to(args.device).long()

                label_x1, label_y1, label_x2, label_y2 = label.split(1, dim=1)
                label_x1 = label_x1.to(args.device).squeeze(1)
                label_y1 = label_y1.to(args.device).squeeze(1)
                label_x2 = label_x2.to(args.device).squeeze(1)
                label_y2 = label_y2.to(args.device).squeeze(1)

                classification, regression = classifier_regressor(img)

                classification_loss = classifier_criterion(classification, class_idx)
                regression_loss_x1 = regression_criterion(regression[:, 0], label_x1)
                regression_loss_y1 = regression_criterion(regression[:, 1], label_y1)
                regression_loss_x2 = regression_criterion(regression[:, 2], label_x2)
                regression_loss_y2 = regression_criterion(regression[:, 3], label_y2)

                regression_loss = regression_loss_x1 + regression_loss_y1 + regression_loss_x2 + regression_loss_y2

                loss = classification_loss + regression_loss

                losses.update(loss.item(), img.size(0))

            summary_writer.add_scalar('val_loss', losses.avg, epoch)

        torch.save({'model': classifier_regressor.state_dict(), 'optim': optimizer}, os.path.join(run_dir, f"classifier_regressor.pth"))

        # get random example image from val_loader
        img, (class_idx, label) = random.choice(val_data)

        label_x1, label_y1, label_x2, label_y2 = label.split(1, dim=0)
        label_x1 = label_x1.squeeze(0).item()
        label_y1 = label_y1.squeeze(0).item()
        label_x2 = label_x2.squeeze(0).item()
        label_y2 = label_y2.squeeze(0).item()

        img = img.to(args.device).unsqueeze(0)
        classification, regression = classifier_regressor(img)
        classification = torch.argmax(classification, dim=1).cpu().detach()
        classification = class_names[classification[0].item()]

        # draw correct bounding box and label on image in green
        # draw predicted bounding box and label on image in red

        # save image to tensorboard
        img_from_tensor = transforms.ToPILImage()(img[0])
        ImageDraw.Draw(img_from_tensor).rectangle([(int(label_x1*256), int(label_y1*256)), (int(label_x2*256), int(label_y2*256))], outline='green', width=1)
        ImageDraw.Draw(img_from_tensor).rectangle([(int(regression[0][0]*256), int(regression[0][1]*256)), (int(regression[0][2]*256), int(regression[0][3]*256))], outline='red', width=1)

        ImageDraw.Draw(img_from_tensor).text((int(label_x1*256), int(label_y2*256)), class_names[class_idx], font=fnt, fill='green')
        ImageDraw.Draw(img_from_tensor).text((int(regression[0][0]*256), int(regression[0][1]*256)), classification, font=fnt, fill='red')

        summary_writer.add_image('example_image', transforms.ToTensor()(img_from_tensor), epoch)

    summary_writer.close()