import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights

class ResNet(nn.Module):
    def __init__(self, num_classes=10, pretrained=False):
        super(ResNet, self).__init__()

        if pretrained:
            self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.model = models.resnet18(weights=None)
            
        self.model.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        out = self.model(x)
        return out