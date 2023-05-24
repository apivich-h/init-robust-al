import torch
from torch import nn as nn
from torch.nn import functional as F
from torch import optim

from .nn_model import TorchNNModel


class CNNTorch(TorchNNModel):

    def __init__(self, in_dim=(1, 28, 28), out_dim=10, 
                 conv_layers=(32, 64), hidden_layers=(512,), dropout_p=0.5, 
                 conv_kernel_size=(5, 5), conv_stride=(1, 1), pool_kernel_size=(2, 2),
                 ntk_compute_method='jac_con', kernel_batch_sz=256, rand_idxs=-1, use_cuda=True):
        super().__init__(in_dim=in_dim, out_dim=out_dim, use_cuda=use_cuda,
                         ntk_compute_method=ntk_compute_method, kernel_batch_sz=kernel_batch_sz, rand_idxs=rand_idxs)
        layers = []
        conv_in = in_dim[0]
        for c in conv_layers:
            layers.extend([
                nn.Conv2d(in_channels=conv_in, out_channels=c, kernel_size=conv_kernel_size, stride=conv_stride),
                nn.Dropout2d(p=dropout_p),
                nn.MaxPool2d(kernel_size=pool_kernel_size),
                nn.ReLU()
            ])
            conv_in = c
        layers.append(nn.Flatten())
        for h in hidden_layers:
            layers.extend([
                nn.LazyLinear(out_features=h, bias=True),
                nn.Dropout(p=dropout_p),
                nn.ReLU(),
            ])
        layers.append(nn.LazyLinear(out_features=out_dim))
        self.model = nn.Sequential(*layers)
        # to convert LazyLinear to fixed sizes
        in_data = torch.rand(1, *in_dim)
        _ = self.model(in_data)
        # use cuda
        if self.use_cuda:
            # if torch.cuda.device_count() > 1:
            #     print(f'Running model on {torch.cuda.device_count()} devices')
            #     self.model = nn.DataParallel(self.model)
            self.model.cuda()
        
        # self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 5))
        # self.conv1_drop = nn.Dropout2d(p=dropout_p)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 5))
        # self.conv2_drop = nn.Dropout2d(p=dropout_p)
        # self.fc1 = nn.Linear(1024, hidden_sz)
        # self.fc1_drop = nn.Dropout(p=dropout_p)
        # self.fc2 = nn.Linear(hidden_sz, out_dim)
        # self.model = nn.Sequential(
        #     self.conv1,
        #     self.conv1_drop,
        #     nn.MaxPool2d(kernel_size=2),
        #     nn.ReLU(),
        #     self.conv2,
        #     self.conv2_drop,
        #     nn.MaxPool2d(kernel_size=2),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     self.fc1,
        #     self.fc1_drop,
        #     nn.ReLU(),
        #     self.fc2
        # )
        # if self.use_cuda:
        #     self.model.cuda()

    def get_penultimate_and_final_output(self, xs):
        penultimate = self.model[:-3](xs)
        final = self.model[-3:](penultimate)
        return penultimate, final
