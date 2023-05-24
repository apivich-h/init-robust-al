"""
Module adapted from https://github.com/xternalz/WideResNet-pytorch/blob/master/wideresnet.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn_model import TorchNNModel

    
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(x)
        else:
            out = self.relu1(x)
        out = self.relu2(self.conv1(out if self.equalInOut else x))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)
    

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)
    
class FCView(nn.Module):
    """ To follow https://github.com/pytorch/pytorch/issues/2486 """
    def __init__(self, n_channels):
        super(FCView, self).__init__()
        self.n_channels = n_channels

    # noinspection PyMethodMayBeStatic
    def forward(self, x):
        x = x.view(-1, self.n_channels)
        return x

    def __repr__(self):
        return f'view(-1, {self.n_channels})'


class WideResNetTorch(TorchNNModel):
        
    def __init__(self, in_dim=(3, 32, 32), out_dim=10, depth=1, widen_factor=1, dropout_p=0.0,
                 ntk_compute_method='jac_con', kernel_batch_sz=256, rand_idxs=-1, use_cuda=True):
        assert in_dim[0] == 3
        super().__init__(in_dim=in_dim, out_dim=out_dim, ntk_compute_method=ntk_compute_method,
                             kernel_batch_sz=kernel_batch_sz, use_cuda=use_cuda, rand_idxs=rand_idxs)
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropout_p)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropout_p)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropout_p)
        # global average pooling and classifier
        # self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], out_dim)
        self.nChannels = nChannels[3]
        
        self.model = nn.Sequential(
            self.conv1,
            self.block1,
            self.block2,
            self.block3,
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=8),
            FCView(n_channels=self.nChannels),
            self.fc
        )
        if self.use_cuda:
            # if torch.cuda.device_count() > 1:
            #     print(f'Running model on {torch.cuda.device_count()} devices')
            #     self.model = nn.DataParallel(self.model)
            self.model.cuda()
        
        self.init_weights()

    def init_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def get_penultimate_and_final_output(self, xs):
        penultimate = self.model[:-1](xs)
        final = self.model[-1:](penultimate)
        return penultimate, final