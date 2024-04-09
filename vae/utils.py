import argparse
import os

def get_args():
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--data_path', type=str, default=os.path.join('..', 'data', 'multidata'), help='Path to the data')
    argparser.add_argument('--run_name', type=str, required=True, help='Name of the run')

    argparser.add_argument('--start_epoch', type=int, default=None, help='Epoch to start training from')
    argparser.add_argument('--n_epochs', type=int, default=2000, help='Number of epochs to train for')
    argparser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    argparser.add_argument('--activation_function', type=str, default='relu', help='Activation function to use', choices=['relu', 'gelu', 'elu', 'selu', 'prelu', 'leaky_relu', 'tanh', 'sigmoid'])
    argparser.add_argument('--latent_size', type=int, default=20000, help='Size of the latent vector')
    argparser.add_argument('--dropout', type=float, default=0.20, help='Dropout rate')
    argparser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    argparser.add_argument('--batches_to_accumulate', type=int, default=8, help='Number of batches to accumulate before performing an optimization step')
    argparser.add_argument('--trainer', type=str, default='normal', help='Which trainer to use', choices=['normal', 'gan'])
    argparser.add_argument('--architecture', type=str, default='old', help='Which model architecture to use', choices=['swin', 'old'])
    argparser.add_argument('--dataset', type=str, default='oidv6', help='Which dataset to use', choices=['mnist', 'oidv6'])
    argparser.add_argument('--criterion', type=str, default='mse', help='Loss function to use', choices=['mse', 'bce', 'se', 'l1', 'log_cosh'])
    argparser.add_argument('--criterion_reduction', type=str, default='mean', help='Reduction method for the loss function', choices=['mean', 'sum'])
    argparser.add_argument('--use_deconv', action='store_true', help='Use deconvolutional layers instead of upsampling')

    argparser.add_argument('--test_image_steps', type=int, default=100, help='Number of steps between test image generation')
    argparser.add_argument('--device', type=str, default='cuda', help='Device to train on')
    argparser.add_argument('--detect_nans', action='store_true', help='Detect NaNs in the model')
    argparser.add_argument('--seed', type=int, default=None, help='Seed for reproducibility')

    return argparser.parse_known_args()
