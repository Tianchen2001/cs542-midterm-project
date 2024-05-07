import torch
import torch.nn as nn


def vae_loss(recon_x, x, mu, logvar):
    recon_x_flat = recon_x.view(recon_x.size(0), -1)
    x_flat = x.view(x.size(0), -1)
    
    BCE = nn.functional.binary_cross_entropy(recon_x_flat, x_flat, reduction='sum')
    KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KL
