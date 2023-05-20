import torch.nn as nn
import torch.nn.functional as F

#Nerf network
class Nerf(nn.Module):
  def __init__(self):
    super(Nerf, self).__init__()

    self.layer1 = nn.Linear(3+3*2*6, 256)
    self.layer2 = nn.Linear(256, 256)
    self.layer3 = nn.Linear(256, 256)
    self.layer4= nn.Linear(256, 4)
  
  def forward(self, x):
    x = F.relu(self.layer1(x))
    x = F.relu(self.layer2(x))
    x = F.relu(self.layer3(x))
    x = self.layer4(x)
     
    return x