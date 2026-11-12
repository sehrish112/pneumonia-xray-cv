import torch.nn as nn
from torchvision import models


def build_model(pretrained=True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    # one output: probability of pneumonia
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model
