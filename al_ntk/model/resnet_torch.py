from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.resnet import conv1x1, conv3x3

from .nn_model import TorchNNModel


""" Adapted from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py """


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        add_batch_norm: bool = False,
    ) -> None:
        super().__init__()
        # if norm_layer is None:
        #     norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        if add_batch_norm:
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)
        else:
            self.bn1 = None
            self.bn2 = None
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        if self.bn1 is not None:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.bn2 is not None:
            out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class _ResNet(nn.Module):
    def __init__(
        self,
        layers: List[int] = [2, 2, 2, 2],
        num_classes: int = 1000,
        # zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        add_batch_norm: bool = False,
    ) -> None:
        super().__init__()
        self.add_batch_norm = add_batch_norm

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        if self.add_batch_norm:
            self.bn1 = nn.BatchNorm2d(self.inplanes)
        else:
            self.bn1 = None
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0], add_batch_norm=add_batch_norm)
        self.layer2 = self._make_layer(128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], add_batch_norm=add_batch_norm)
        self.layer3 = self._make_layer(256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], add_batch_norm=add_batch_norm)
        self.layer4 = self._make_layer(512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], add_batch_norm=add_batch_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
        
        # self.init_weights()

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        add_batch_norm: bool = False
    ) -> nn.Sequential:
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
            if add_batch_norm:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * BasicBlock.expansion, stride),
                    nn.BatchNorm2d(planes * BasicBlock.expansion),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * BasicBlock.expansion, stride),
                )

        layers = []
        layers.append(
            BasicBlock(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, add_batch_norm
            )
        )
        self.inplanes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(
                BasicBlock(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    add_batch_norm=add_batch_norm,
                )
            )

        return nn.Sequential(*layers)
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # # Zero-initialize the last BN in each residual branch,
        # # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #     for m in self.modules():
        #         if m.bn2.weight is not None:
        #             nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def get_penultimate_and_final_output(self, x):
        x = self.conv1(x)
        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        y = torch.flatten(x, 1)
        z = self.fc(y)

        return y, z
    
    def forward(self, x):
        _, z = self.get_penultimate_and_final_output(x)
        return z


class ResNetTorch(TorchNNModel):
    
    def __init__(self, in_dim, out_dim, layers: List[int] = [2, 2, 2, 2],
                 groups: int = 1, width_per_group: int = 64, 
                 replace_stride_with_dilation: Optional[List[bool]] = None, add_batch_norm: bool = False,
                 ntk_compute_method='ntk_vps', kernel_batch_sz=256, use_cuda=True, rand_idxs=1):
        super().__init__(in_dim, out_dim, ntk_compute_method, kernel_batch_sz, use_cuda, rand_idxs)
        
        self.model = _ResNet(layers=layers, num_classes=out_dim, groups=groups, width_per_group=width_per_group,
                             replace_stride_with_dilation=replace_stride_with_dilation, add_batch_norm=add_batch_norm)
        if use_cuda:
            self.model = self.model.cuda()
        
    def init_weights(self):
        self.model.init_weights()
        
    def get_penultimate_and_final_output(self, xs):
        return self.model.get_penultimate_and_final_output(xs)
