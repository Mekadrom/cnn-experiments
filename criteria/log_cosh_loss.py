import torch
import torch.nn as nn

class LogCoshLoss(nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, x_predicted, x):
        return torch.mean(torch.log(torch.cosh(x_predicted - x)))
