import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        N, *_ = x.size()
        return x.view(N, -1)


class LNN_Discriminator(nn.Module):
    def __init__(self, input_channels=3, image_size=32):
        super(LNN_Discriminator, self).__init__()

        self.model = nn.Sequential(
            Flatten(),
            nn.Linear(input_channels * image_size**2, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256), 
            nn.LeakyReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        x = self.model(x)
        return x


class LNN_Generator(nn.Module):
    def __init__(self, noise_dim):
        super(LNN_Generator, self).__init__()

        self.model = nn.Sequential(
            Flatten(),
            nn.Linear(noise_dim, 1024), 
            nn.LeakyReLU(),
            nn.Linear(1024, 1024), 
            nn.LeakyReLU(),
            nn.Linear(1024, 784),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.model(x)
        return x


class CNN_Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(CNN_Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(1024, 1, kernel_size=4, stride=1),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class CNN_Generator(nn.Module):
    def __init__(self, noise_dim, output_channels=3):
        super(CNN_Generator, self).__init__()
        self.noise_dim = noise_dim
        
        self.model = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, 1024, kernel_size=4, stride=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.model(x)
        return x
