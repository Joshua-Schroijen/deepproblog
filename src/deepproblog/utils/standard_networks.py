import torch
import torch.nn as nn
from typing import Dict, Union
from problog.logic import Term

from deepproblog.network import ClassificationNetworkModule

class MLP(ClassificationNetworkModule):
    def __init__(self, *sizes, activation=nn.ReLU, softmax=True, batch=True):
        super(MLP, self).__init__()
        layers = []
        self.sizes = sizes
        self.softmax = softmax
        self.batch = batch
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(activation())
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        if softmax:
            layers.append(nn.Softmax(-1))
        self.nn = nn.Sequential(*layers)

    def forward(self, x):
        if not self.batch:
            x = x.unsqueeze(0)
        x = self.nn(x)
        return x

    def get_output_logits(self, input):
        if type(input) == list:
            logits = torch.empty(0, self.sizes[-1])
            for i in input:
                j = i.unsqueeze(0)
                k = self.nn[:-1](j) if self.softmax else self.nn(j)
                logits = torch.cat((logits, k), dim = 0)
            return logits
        else:
            if not self.batch:
                input = input.unsqueeze(0)
            return self.nn[:-1](input) if self.softmax else self.nn(input)

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

model_urls = {
    "alexnet": "https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth",
}

class DummyNet(nn.Module):
    def __init__(self, values: Dict[Term, Union[list, torch.Tensor]]):
        super().__init__()
        self.values = values

    def forward(self, x):
        output = self.values[x]
        return torch.tensor(output, requires_grad=True)

class SmallNet(ClassificationNetworkModule):
    def __init__(self, num_classes=1000, size=None):
        super(SmallNet, self).__init__()
        self.final = nn.Sigmoid() if num_classes == 1 else nn.Softmax(1)
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.N = 2304
        if size is not None:
            input = torch.empty(1, 3, *size)
            out = self.features(input)
            self.N = torch.numel(out)
        self.classifier = nn.Sequential(
            nn.Linear(self.N, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, self.num_classes),
            self.final,
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.N) 
        x = self.classifier(x)
        return x

    def get_output_logits(self, input):
        logits = torch.empty(0, self.num_classes)

        is0 = input.size(0)
        for i, j in zip(range(0, is0), range(1, is0 + 1)):
            x = self.features(input[i:j])
            x = x.view(-1, self.N)
            x = self.classifier[:-1](x)
            x = x[0:1]
            logits = torch.cat((logits, x), dim = 0)

        return logits

def smallnet(pretrained=False, model=None, **kwargs):
    if model is None:
        model = SmallNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["alexnet"]), strict=False)
    return model
