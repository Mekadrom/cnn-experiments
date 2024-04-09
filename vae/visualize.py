import dataloader
import model
import oid_vae
import os
import torch
import torchvision
import utils

args, unk = utils.get_args()

run_dir = os.path.join("runs", args.run_name)
if not os.path.exists(run_dir):
    raise ValueError(f"Run directory {run_dir} does not exist")

model_file = os.path.join(run_dir, "model.pt")
if not os.path.exists(model_file):
    raise ValueError(f"Model file {model_file} does not exist")

encoder = oid_vae.Encoder(args.activation_function, args.latent_size, args.dropout)
decoder = oid_vae.Decoder(args.activation_function, args.latent_size, args.dropout)

vae = model.VAEModel(encoder, decoder).to(args.device)

vae.load_state_dict(torch.load(model_file)['model'])

train_dataloader_provider = dataloader.DataLoader(args, args.data_path, 'train', args.batch_size)

train_dataloader_provider.create_batches()

test_image = train_dataloader_provider.data[-1].to(args.device)

def save_convs(tensors, name):
    tensors = torch.sigmoid(tensors)
    file_dir = os.path.join(run_dir, name)
    os.makedirs(file_dir, exist_ok=True)
    for i in range(tensors.shape[0]): # for each channel
        torchvision.utils.save_image(tensors[i], os.path.join(file_dir, f"test_channel_{i}.png"))

def visualize(image):
    vae.eval()

    if len(image.shape) == 3:
        image = image.unsqueeze(0)
        
    image = image.to(args.device)

    original_image = image
    os.makedirs(os.path.join(run_dir, 'preconv'), exist_ok=True)
    torchvision.utils.save_image(original_image, os.path.join(run_dir, 'preconv', "original_image.png"))

    with torch.no_grad():
        conv1_outputs = vae.encoder.conv1(image)
        save_convs(conv1_outputs.squeeze(0), "conv1")

        conv2_outputs = vae.encoder.conv2(conv1_outputs)
        save_convs(conv2_outputs.squeeze(0), "conv2")

        conv3_outputs = vae.encoder.conv3(conv2_outputs)
        save_convs(conv3_outputs.squeeze(0), "conv3")

        conv4_outputs = vae.encoder.conv4(conv3_outputs)
        save_convs(conv4_outputs.squeeze(0), "conv4")

        conv5_outputs = vae.encoder.conv5(conv4_outputs)
        save_convs(conv5_outputs.squeeze(0), "conv5")

        conv6_outputs = vae.encoder.conv6(conv5_outputs)
        save_convs(conv6_outputs.squeeze(0), "conv6")

        conv7_outputs = vae.encoder.conv7(conv6_outputs)
        save_convs(conv7_outputs.squeeze(0), "conv7")

        conv8_outputs = vae.encoder.conv8(conv7_outputs)
        save_convs(conv8_outputs.squeeze(0), "conv8")

        reconstructed_image, _, _ = vae(image)        
        torchvision.utils.save_image(reconstructed_image, os.path.join(run_dir, "reconstructed_image.png"))

visualize(test_image)
