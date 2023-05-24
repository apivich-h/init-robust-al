import torch.nn as nn


class LambdaLayer(nn.Module):
    def __init__(self, fn):
        super(LambdaLayer, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)
