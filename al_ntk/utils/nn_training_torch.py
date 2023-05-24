import os
import tqdm
import pickle

import torch
from torch.utils.data import DataLoader
from torch.nn.modules.loss import MSELoss, BCEWithLogitsLoss, CrossEntropyLoss

from al_ntk.model import TorchNNModel


def train_torch(model: TorchNNModel, train_loader: DataLoader, optimizer, criterion: str = 'mse', epochs: int = 50, progbar: bool = False, clip_value: float = None):

    if criterion == 'mse':
        criterion_fn = MSELoss()
    elif criterion == 'bce':
        criterion_fn = BCEWithLogitsLoss()
    elif criterion == 'ce':
        criterion_fn = CrossEntropyLoss()
    else:
        raise ValueError

    for epoch in (tqdm.trange(epochs) if progbar else range(epochs)):

        # train for one epoch, collect the labels, predictions and losses, and log them
        model.train()
        for train_idx, x_train, train_label in train_loader:

            optimizer.zero_grad()
            # forward propogation
            train_pred = model(x_train)
            # calculate loss
            train_loss = criterion_fn(train_pred, train_label)
            # back prop
            train_loss.backward()
            if clip_value is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()