import torch
import torch.nn as nn

class LNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LNN, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.model(x)
        return out