import torch
from torchvision.models import resnet18
import torch.nn as nn


class ModifiedResnet(nn.Module):
    def __init__(self, n_actions, Dropout=None):
        super(ModifiedResnet, self).__init__()
        self.model = resnet18(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self.fc1 = nn.Sequential(nn.Linear(num_ftrs, n_actions), nn.Softmax(dim=1))
        if Dropout is not None:
            self.fc2 = nn.Sequential(nn.Linear(num_ftrs, 256), nn.ReLU(), torch.nn.Dropout(Dropout), nn.Linear(256, 1),
                                     nn.Sigmoid())
        else:
            self.fc2 = nn.Sequential(nn.Linear(num_ftrs, 256), nn.ReLU(), nn.Linear(256, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.model(x)
        out1 = self.fc1(x)
        out2 = self.fc2(x)
        return out1, out2

