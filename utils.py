import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import torchvision.transforms as transforms

import pdb


def local_train(e, model, optimizer, train_loader):
    model.train()
    for i in range(e):
        correct = 0
        train_loss = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(x)
            loss = F.nll_loss(output, y)
            train_loss += loss.item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(y.data.view_as(pred)).sum()
            loss.backward()
            optimizer.step()
        acc = float(100. * correct / len(train_loader.dataset))
    return acc, train_loss


def global_test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            output = model(x)
            test_loss += F.nll_loss(output, y, reduction=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(y.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    acc = float(100. * correct / len(test_loader.dataset))
    return acc, test_loss
