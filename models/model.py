import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
from efficientnet_pytorch import EfficientNet

class EF_Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.EF_Net = EfficientNet.from_pretrained('efficientnet-b7')
    for parameter in self.EF_Net.parameters():
      parameter.requires_grad = True
    self.Linear = nn.Linear(self.EF_Net._fc.in_features, 3)
    self.EF_Net._fc = self.Linear

  def forward(self, x):
    x = self.EF_Net(x)
    return x

if __name__ == "__main__":
    pass