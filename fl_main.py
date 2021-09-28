# https://towardsdatascience.com/preserving-data-privacy-in-deep-learning-part-1-a04894f78029
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

from models import Simple
from utils import local_train, global_test
from aggregation import weight_avg

import pdb

# experiment set-up
DATASET = 'mnist'
#DATASET = 'cifar'
MODEL = 'simple'

# hyperparameters
n = 3
batch_size = 128
lr = .01
# global rounds
rounds = 10
# local epochs
epochs = 3

# importing the data
if DATASET == 'mnist':
    print('Using MNIST data')
    train_data = torchvision.datasets.MNIST(root='./mnist_data', train=True, download=True,
                                   transform=transforms.ToTensor())#, batch_size=batch_size, shuffle=True)
    test_data = torchvision.datasets.MNIST(root='./mnist_data', train=False, download=True,
                                   transform=transforms.ToTensor())#, batch_size=batch_size, shuffle=False)

elif DATASET == 'cifar':
    print('Using CIFAR-10 data')
    train_data = torchvision.datasets.CIFAR10(root='./cifar_data', train=True, download=True,
                                                transform=transforms.ToTensor())#, batch_size=batch_size, shuffle=True)
    test_data = torchvision.datasets.CIFAR10(root='./cifar_data', train=False, download=True,
                                        transform=transforms.ToTensor())#, batch_size=batch_size, shuffle=False)

# Dividing into clients
split_train = torch.utils.data.random_split(train_data, [int(train_data.data.shape[0] / n) for _ in range(n)])

# creating a data loader for all of the splits
train_loader = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True) for x in split_train]

# creating the test loader for the global model
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

if MODEL == 'simple':
    # initialize server model
    global_model = Simple()
    # initialize list of client models
    clients = [Simple() for _ in range(n)]

# Use the same optimizer for all models
optimizers = [optim.SGD(model.parameters(), lr=lr) for model in clients]

loss_tr = []
loss_tst = []
acc_tr = []
acc_tst = []

for r in range(rounds):
    print('Global round', r, '/', rounds)
    for i in range(n):
        #client = clients[i]
        #print(np.array((list(client.parameters())[1]).data)[0])
        #loader = train_loader[i]
        train_acc, train_loss = local_train(epochs, clients[i], optimizers[i], train_loader[i])
        #test_acc, test_loss = test(model, test_loader)
        print('model', i, ': train accuracy', train_acc)#, ', test accuracy:', test_acc)
    #w1_tr.append(np.array((list(model.parameters())[0]).data))
    #train_acc, train_loss = train(epoch, model, optimizer, train_loader)
    #test_acc, test_loss = test(model, test_loader)
    #print('train accuracy', train_acc, 'test accuracy:', test_acc)
    # aggregate
    global_model = weight_avg(global_model, clients)
    # test global model
    test_acc, test_loss = global_test(global_model, test_loader)
    print('Global test accuracy', test_acc)
    # sync clients with global model
    for client in clients:
        client.load_state_dict(global_model.state_dict())

"""
model.train()
    for e in range(epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
"""

# server aggregation function takes global model and client models and averages and then updates the clients with the global at the end

# have a normal test function for global model
