import torch
from torch import nn

from deepproblog.network import ClassificationNetworkModule

class SymbolEncoder(nn.Module):
  def __init__(self):
    super(SymbolEncoder, self).__init__()
    self.convolutions = nn.Sequential(
      nn.Conv2d(1, 6, 3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(2),
      nn.Conv2d(6, 16, 3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(2),
      nn.Dropout2d(0.4),
    )

    self.mlp = nn.Sequential(
      nn.Linear(16 * 11 * 11, 128),
      nn.ReLU(),
      # nn.Dropout2d(0.8)
    )

  def forward(self, x):
    x = self.convolutions(x)
    x = torch.flatten(x, 1)
    x = self.mlp(x)
    return x

class SymbolClassifier(ClassificationNetworkModule):
  def __init__(self, encoder, N=10):
    super(SymbolClassifier, self).__init__()
    self.encoder = encoder
    self.fc2 = nn.Linear(128, N)
    self.softmax = nn.Softmax(dim=-1)
    self._N = N

  def forward(self, x):
    x = self.encoder(x)
    x = self.fc2(x)
    x = self.softmax(x)
    return x

  def get_output_logits(self, input):
    is0 = input.size(0)
    x = torch.empty(0, self._N)
    for i, j in zip(range(0, is0, 2), range(2, is0 + 1 + (is0 % 2), 2)):
      y = self.fc2(self.encoder(input[i:j]))
      x = torch.cat((x, y), dim = 0)
    return x
