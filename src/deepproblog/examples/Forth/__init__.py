import torch
import torch.nn as nn

from deepproblog.utils.standard_networks import MLP

class EncodeModule(nn.Module):
    def __init__(self, in_size, mid_size, out_size, activation=None):
        super().__init__()
        if activation == "tanh":
            self.mlp = MLP(in_size, mid_size, out_size, activation=nn.Tanh)
        else:
            self.mlp = MLP(in_size, mid_size, out_size)
        self.in_size = in_size
        self.out_size = out_size

    def forward(self, *x):
        input = torch.zeros(self.in_size)
        for j, i in enumerate(x):
            i = int(i)
            input[j * 10 + i] = 1
        return self.mlp(input)

    def __call__(self, *x):
        return self.forward(*x)

    def get_output_logits(self, *x):
        if len(x) == 1:
            x = x[0]
            logits = torch.empty(0, self.out_size)
            for y in x:
                input = torch.zeros(self.in_size)
                for j, i in enumerate(y):
                    i = int(i)
                    input[j * 10 + i] = 1
                logits = torch.cat((logits, self.mlp.get_output_logits(input).unsqueeze(0)), dim = 0)
            return logits
        else:
            input = torch.zeros(self.in_size)
            for j, i in enumerate(x):
                i = int(i)
                input[j * 10 + i] = 1
            return self.mlp.get_output_logits(input)