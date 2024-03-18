import torch.nn as nn

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def create_activation_function(activation_function_name):
    if activation_function_name == 'relu':
        return nn.ReLU()
    elif activation_function_name == 'gelu':
        return nn.GELU()
    elif activation_function_name == 'elu':
        return nn.ELU()
    elif activation_function_name == 'selu':
        return nn.SELU()
    elif activation_function_name == 'prelu':
        return nn.PReLU()
    elif activation_function_name == 'leaky_relu':
        return nn.LeakyReLU()
    elif activation_function_name == 'tanh':
        return nn.Tanh()
    elif activation_function_name == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise Exception(f"Unknown activation function {activation_function_name}")
