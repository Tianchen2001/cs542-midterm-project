import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        N, *_ = x.size()
        return x.view(N, -1)


class LNN_VAE(nn.Module):
    def __init__(self, input_channels=3, image_size=32):
        super(LNN_VAE, self).__init__()
        
        self.encoder = nn.Sequential(
            Flatten(),
            nn.Linear(input_channels * image_size**2, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256), 
            nn.LeakyReLU()
        )
        
        self.mu_proj = nn.Linear(256, 20)
        self.logvar_proj = nn.Linear(256, 20)

        self.decoder = nn.Sequential(
            nn.Linear(20, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, input_channels * image_size**2)
        )

    def encode(self, x):
        x = self.encoder(x)
        return self.mu_proj(x), self.logvar_proj(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.decoder(z)
        return torch.sigmoid(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class CNN_VAE(nn.Module):
    def __init__(self, input_channels=3, image_size=32):
        super(CNN_VAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Flatten(), # Flatten for the dense layers
            nn.Linear(128 * (image_size // 4) * (image_size // 4), 256),
            nn.LeakyReLU(),
        )

        self.mu_proj = nn.Linear(256, 20)
        self.logvar_proj = nn.Linear(256, 20)
        
        self.decoder = nn.Sequential(
            nn.Linear(20, 128 * (image_size // 4) * (image_size // 4)),
            nn.LeakyReLU(),
            nn.Unflatten(1, (128, image_size // 4, image_size // 4)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, input_channels, kernel_size=4, stride=2, padding=1)
        )

    def encode(self, x):
        x = self.encoder(x)
        return self.mu_proj(x), self.logvar_proj(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.decoder(z)
        return torch.sigmoid(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
