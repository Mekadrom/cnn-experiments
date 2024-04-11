from PIL import Image, ImageDraw, ImageFont
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import argparse
import model
import os
import random
import sys
import torch
import torch.nn as nn
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

            data.append((img, (torch.LongTensor([class_id]).squeeze(-1), torch.FloatTensor([label_x1, label_y1, label_x2, label_y2]))))
    return data

try:
    fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 11)
except:
    fnt = ImageFont.load_default()

def save_image_example(epoch, actual_img, actual_class_id, actual_box_label, model, summary_writer):
    actual_x1, actual_y1, actual_x2, actual_y2 = actual_box_label.split(1, dim=0)
    actual_x1 = actual_x1.squeeze(0).item()
    actual_y1 = actual_y1.squeeze(0).item()
    actual_x2 = actual_x2.squeeze(0).item()
    actual_y2 = actual_y2.squeeze(0).item()

    actual_img = actual_img.to(args.device).unsqueeze(0)
    pred_class_id, pred_regr = model(actual_img)
    pred_class_id = torch.argmax(pred_class_id, dim=1).cpu().detach()
    pred_class_id = class_names[pred_class_id[0].item()]

    pred_regr = pred_regr.squeeze(0)

    # draw correct bounding box and label on image in green
    # draw predicted bounding box and label on image in red

    actual_x1 = int(actual_x1 * 224)
    actual_y1 = int(actual_y1 * 224)
    actual_x2 = int(actual_x2 * 224)
    actual_y2 = int(actual_y2 * 224)

    pred_regr[0] = int(pred_regr[0] * 224)
    pred_regr[1] = int(pred_regr[1] * 224)
    pred_regr[2] = int(pred_regr[2] * 224)
    pred_regr[3] = int(pred_regr[3] * 224)

    pred_x1 = int(min(pred_regr[0], pred_regr[2]))
    pred_y1 = int(min(pred_regr[1], pred_regr[3]))
    pred_x2 = int(max(pred_regr[0], pred_regr[2]))
    pred_y2 = int(max(pred_regr[1], pred_regr[3]))

    # save image to tensorboard
    img_from_tensor = transforms.ToPILImage()(actual_img[0])
    ImageDraw.Draw(img_from_tensor).rectangle([(actual_x1, actual_y1), (actual_x2, actual_y2)], outline='green', width=1)
    ImageDraw.Draw(img_from_tensor).rectangle([(pred_x1, pred_y1), (pred_x2, pred_y2)], outline='red', width=1)

    ImageDraw.Draw(img_from_tensor).text((actual_x1, actual_y2), class_names[actual_class_id], font=fnt, fill='green')
    ImageDraw.Draw(img_from_tensor).text((pred_x1, pred_y1), pred_class_id, font=fnt, fill='red')

    img = transforms.ToTensor()(img_from_tensor).unsqueeze(0)
    img = nn.functional.interpolate(img, scale_factor=2, mode='nearest').squeeze(0)

    summary_writer.add_image('example_image', img, epoch)

def save_conv_filters(epoch, model, summary_writer):
    initial_layer_filters = model.initial_conv[0].weight.data.view(-1, 1, 7, 7)
    block_1_filters = torch.stack([conv[0].weight.data for conv in model.convs[0]], dim=0).view(-1, 1, 3, 3)
    conv_pool_2_filters = model.convs[1][0].weight.data.view(-1, 1, 3, 3)
    block_2_filters = torch.stack([conv[0].weight.data for conv in model.convs[2]], dim=0).view(-1, 1, 3, 3)
    conv_pool_3_filters = model.convs[3][0].weight.data.view(-1, 1, 3, 3)
    block_3_filters = torch.stack([conv[0].weight.data for conv in model.convs[4]], dim=0).view(-1, 1, 3, 3)
    conv_pool_4_filters = model.convs[5][0].weight.data.view(-1, 1, 3, 3)
    block_4_filters = torch.stack([conv[0].weight.data for conv in model.convs[6]], dim=0).view(-1, 1, 3, 3)

    initial_layer_filters = torchvision.utils.make_grid(initial_layer_filters, nrow=3, padding=2, normalize=True).unsqueeze(0)
    block_1_filters = torchvision.utils.make_grid(block_1_filters, nrow=64, padding=2, normalize=True).unsqueeze(0)
    conv_pool_2_filters = torchvision.utils.make_grid(conv_pool_2_filters, nrow=128, padding=2, normalize=True).unsqueeze(0)
    block_2_filters = torchvision.utils.make_grid(block_2_filters, nrow=128, padding=2, normalize=True).unsqueeze(0)
    conv_pool_3_filters = torchvision.utils.make_grid(conv_pool_3_filters, nrow=256, padding=2, normalize=True).unsqueeze(0)
    block_3_filters = torchvision.utils.make_grid(block_3_filters, nrow=256, padding=2, normalize=True).unsqueeze(0)
    conv_pool_4_filters = torchvision.utils.make_grid(conv_pool_4_filters, nrow=512, padding=2, normalize=True).unsqueeze(0)
    block_4_filters = torchvision.utils.make_grid(block_4_filters, nrow=512, padding=2, normalize=True).unsqueeze(0)

    initial_layer_filters = nn.functional.interpolate(initial_layer_filters, scale_factor=4, mode='nearest')
    block_1_filters = nn.functional.interpolate(block_1_filters, scale_factor=4, mode='nearest')
    conv_pool_2_filters = nn.functional.interpolate(conv_pool_2_filters, scale_factor=4, mode='nearest')
    block_2_filters = nn.functional.interpolate(block_2_filters, scale_factor=4, mode='nearest')
    conv_pool_3_filters = nn.functional.interpolate(conv_pool_3_filters, scale_factor=4, mode='nearest')
    block_3_filters = nn.functional.interpolate(block_3_filters, scale_factor=4, mode='nearest')
    conv_pool_4_filters = nn.functional.interpolate(conv_pool_4_filters, scale_factor=4, mode='nearest')
    block_4_filters = nn.functional.interpolate(block_4_filters, scale_factor=4, mode='nearest')

    summary_writer.add_images('initial_layer_filters', initial_layer_filters, epoch)
    summary_writer.add_images('block_1_filters', block_1_filters, epoch)
    summary_writer.add_images('conv_pool_2_filters', conv_pool_2_filters, epoch)
    summary_writer.add_images('block_2_filters', block_2_filters, epoch)
    summary_writer.add_images('conv_pool_3_filters', conv_pool_3_filters, epoch)
    summary_writer.add_images('block_3_filters', block_3_filters, epoch)
    summary_writer.add_images('conv_pool_4_filters', conv_pool_4_filters, epoch)
    summary_writer.add_images('block_4_filters', block_4_filters, epoch)

def save_conv_outputs(epoch, actual_img, model, summary_writer):
    actual_img = actual_img.to(args.device).unsqueeze(0)

    x = model.initial_conv[0](actual_img)

    img = x.view(-1, x.size(-2), x.size(-1)).unsqueeze(1)
    img = nn.functional.interpolate(img, scale_factor=3, mode='nearest')
    summary_writer.add_images('initial_conv', img, epoch)

    for i, conv_block in enumerate(model.convs):
        if type(conv_block) == nn.ModuleList:
            block_out_imgs = []
            residual = x
            for j, sub_conv_block in enumerate(conv_block):
                if j % 2 == 1:
                    conv_out = sub_conv_block(x)

                    block_out_imgs.append(conv_out)

                    x = residual + conv_out
                    residual = x
                else:
                    x = sub_conv_block(x)

                    block_out_imgs.append(x)

            stacked = torch.stack(block_out_imgs, dim=0)
            img = stacked.view(-1, stacked.size(-2), stacked.size(-1)).unsqueeze(1)
            img = nn.functional.interpolate(img, scale_factor=3, mode='nearest')
            summary_writer.add_images(f'block_{i}_out', img, epoch)
        else:
            for j, sub_conv_block in enumerate(conv_block):
                x = sub_conv_block(x)

            img = x.view(-1, x.size(-2), x.size(-1)).unsqueeze(1)
            img = nn.functional.interpolate(img, scale_factor=3, mode='nearest')
            summary_writer.add_images(f'block_{i}_out', img, epoch)

def save_visualizations(epoch, actual_img, actual_class_id, actual_box_label, model, summary_writer):
    print(f"Saving visualizations for epoch {epoch}...")

    save_image_example(epoch, actual_img, actual_class_id, actual_box_label, model, summary_writer)
    save_conv_filters(epoch, model, summary_writer)
    save_conv_outputs(epoch, actual_img, model, summary_writer)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--data_path', type=str, default=os.path.join('..', 'data', 'preprocessed'), help='Path to the data')
    argparser.add_argument('--run_name', type=str, required=True, help='Name of the run')

    argparser.add_argument('--n_epochs', type=int, default=20, help='Number of epochs to train for')
    argparser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    argparser.add_argument('--activation_function', type=str, default='relu', help='Activation function to use', choices=['relu', 'gelu', 'elu', 'selu', 'prelu', 'leaky_relu', 'tanh', 'sigmoid'])
    argparser.add_argument('--dropout', type=float, default=0.20, help='Dropout rate')
    argparser.add_argument('--batch_size', type=int, default=32, help='Batch size')

    argparser.add_argument('--device', type=str, default='cuda', help='Device to train on')
    argparser.add_argument('--detect_nans', action='store_true', help='Detect NaNs in the model')
    argparser.add_argument('--seed', type=int, default=None, help='Seed for reproducibility')
    argparser.add_argument('--visualize_epochs', type=int, default=10, help='Visualize every n epochs (-1 means no visualization)')

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

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    classifier_regressor = model.ResNet34(args, num_classes=len(class_names)).to(args.device)
    optimizer = torch.optim.Adam(classifier_regressor.parameters(), lr=args.lr)

    classifier_criterion = torch.nn.CrossEntropyLoss()
    regression_criterion = torch.nn.MSELoss()

    print(f"model: {classifier_regressor}")
    print(f"total number of model params: {sum(p.numel() for p in classifier_regressor.parameters()):,}")

    actual_img, (actual_class_id, actual_box_label) = random.choice(val_data)
    save_visualizations(0, actual_img, actual_class_id, actual_box_label, classifier_regressor, summary_writer)

    train_steps = 0
    for epoch in range(args.n_epochs):
        classifier_regressor.train()
        
        losses = nn_utils.AverageMeter()
        for i, (img, (class_id, label)) in enumerate(tqdm(train_loader, desc='Training')):
            img = img.to(args.device)
            class_id = class_id.to(args.device).long()

            label_x1, label_y1, label_x2, label_y2 = label.split(1, dim=1)
            label_x1 = label_x1.to(args.device).squeeze(1)
            label_y1 = label_y1.to(args.device).squeeze(1)
            label_x2 = label_x2.to(args.device).squeeze(1)
            label_y2 = label_y2.to(args.device).squeeze(1)

            optimizer.zero_grad()

            classification, regression = classifier_regressor(img)

            # print(f"classification: {classification.shape}, regression: {regression.shape}")
            # print(f"class_idx: {class_idx.shape}, label_x1: {label_x1.shape}, label_y1: {label_y1.shape}, label_x2: {label_x2.shape}, label_y2: {label_y2.shape}")
            
            classification_loss = classifier_criterion(classification, class_id)
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

            img = img.cpu()
            class_id = class_id.cpu()
            label_x1 = label_x1.cpu()
            label_y1 = label_y1.cpu()
            label_x2 = label_x2.cpu()
            label_y2 = label_y2.cpu()

        print(f'Epoch {epoch+1} Loss: {losses.avg}')

        classifier_regressor.eval()

        with torch.no_grad():
            losses = nn_utils.AverageMeter()
            for i, (img, (class_id, label)) in enumerate(tqdm(val_loader, desc='Validation')):
                img = img.to(args.device)
                class_id = class_id.to(args.device).long()

                label_x1, label_y1, label_x2, label_y2 = label.split(1, dim=1)
                label_x1 = label_x1.to(args.device).squeeze(1)
                label_y1 = label_y1.to(args.device).squeeze(1)
                label_x2 = label_x2.to(args.device).squeeze(1)
                label_y2 = label_y2.to(args.device).squeeze(1)

                classification, regression = classifier_regressor(img)

                classification_loss = classifier_criterion(classification, class_id)
                regression_loss_x1 = regression_criterion(regression[:, 0], label_x1)
                regression_loss_y1 = regression_criterion(regression[:, 1], label_y1)
                regression_loss_x2 = regression_criterion(regression[:, 2], label_x2)
                regression_loss_y2 = regression_criterion(regression[:, 3], label_y2)

                regression_loss = regression_loss_x1 + regression_loss_y1 + regression_loss_x2 + regression_loss_y2

                loss = classification_loss + regression_loss

                losses.update(loss.item(), img.size(0))

                img = img.cpu()
                class_id = class_id.cpu()
                label_x1 = label_x1.cpu()
                label_y1 = label_y1.cpu()
                label_x2 = label_x2.cpu()
                label_y2 = label_y2.cpu()

            summary_writer.add_scalar('val_loss', losses.avg, epoch+1)

            if args.visualize_epochs != -1 and (epoch + 1) % args.visualize_epochs == 0:
                actual_img, (actual_class_id, actual_box_label) = random.choice(val_data)
                save_visualizations(epoch + 1, actual_img, actual_class_id, actual_box_label, classifier_regressor, summary_writer)

        torch.save({'model': classifier_regressor.state_dict(), 'optim': optimizer}, os.path.join(run_dir, f"classifier_regressor.pth"))

    summary_writer.close()