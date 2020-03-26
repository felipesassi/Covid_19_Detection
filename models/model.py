import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
from efficientnet_pytorch import EfficientNet

class Dense_Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.D_Net = models.densenet121(pretrained = True)
    for parameter in self.D_Net.parameters():
      parameter.requires_grad = True
    self.Linear = nn.Linear(self.D_Net.classifier.in_features, 1)
    self.D_Net.classifier = self.Linear

  def forward(self, x):
    x = self.D_Net(x)
    return x

class EF_Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.EF_Net = EfficientNet.from_pretrained('efficientnet-b7')
    for parameter in self.EF_Net.parameters():
      parameter.requires_grad = True
    self.Linear = nn.Linear(self.EF_Net._fc.in_features, 1)
    self.EF_Net._fc = self.Linear

  def forward(self, x):
    x = self.EF_Net(x)
    return x

if __name__ == "__main__":
    pass