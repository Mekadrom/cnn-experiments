
import torch
import torch.nn as nn

class VAEModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAEModel, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        mu, log_var = self.encoder(x)

        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        sample = mu + eps*std

        decoded = self.decoder(sample)
        
        return decoded, mu, log_var
