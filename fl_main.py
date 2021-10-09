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
import csv
from datetime import datetime
from models import mnist_FFNN, mnist_CNN, cifar_FFNN, cifar_CNN
from utils import local_train, global_test, graph_metric, record_global, record_local
from attack_utils import make_gaussian, make_foe
from aggregation import weight_avg, proposed_power, proposed_0_1, proposed_leaveout_0_1
import pdb

# experiment set-up
date = str(datetime.now())

# datasets
DATASET = 'mnist'
# DATASET = 'cifar'

# models
# MODEL = 'ffnn'
MODEL = 'mnist_cnn'
# MODEL = 'cifar_ffnn'
# MODEL = 'cifar_cnn'

# aggregation algorithms
aggregation = 'average'
# aggregation = 'proposed_power'
# aggregation = 'proposed_0_1'
# aggregation = 'proposed_leaveout_0_1'


# attacks
n_honest = 13
# attack = 'none'
attack = 'foe'
# attack = 'gaussian'
sd = 20

# hyperparameters
n = 25
batch_size = 128
lr = .01

# sync local models to global model after each global round

syncing = False
# global rounds
rounds = 10
# local epochs
epochs = 1

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

if MODEL == 'mnist_ffnn':
    # initialize server model
    global_model = mnist_FFNN()
    # initialize list of client models
    clients = [mnist_FFNN() for _ in range(n)]

if MODEL == 'mnist_cnn':
    # initialize server model
    global_model = mnist_CNN()
    # initialize list of client models
    clients = [mnist_CNN() for _ in range(n)]

if MODEL == 'cifar_ffnn':
    # initialize server model
    global_model = cifar_FFNN()
    # initialize list of client models
    clients = [cifar_FFNN() for _ in range(n)]

if MODEL == 'cifar_cnn':
    # initialize server model
    global_model = cifar_CNN()
    # initialize list of client models
    clients = [cifar_CNN() for _ in range(n)]

# Use the same optimizer for all models
optimizers = [optim.Adam(model.parameters(), lr=lr) for model in clients]

global_acc = []
for r in range(rounds):
    print('Global round', r, '/', rounds)

    local_acc = []
    # local training
    for i in range(n):
        train_acc, train_loss = local_train(epochs, clients[i], optimizers[i], train_loader[i])
        print('client', i, ': train accuracy', train_acc)
        local_acc.append(train_acc)

    # possibly formulate an attack
    if attack == 'gaussian':
        print('Executing a Gaussian attack')
        clients = clients[:n_honest] + make_gaussian(clients[n_honest:])
    if attack == 'foe':
        print('Executing a Fall of Empires attack')
        clients = clients[:n_honest] + make_foe(clients[:n_honest], clients[n_honest:])

    # aggregate
    if aggregation == 'proposed_0_1':
        global_model = proposed_0_1(global_model, clients, test_loader, r, date, DATASET)
    if aggregation == 'proposed_power':
        global_model = proposed_power(global_model, clients, test_loader, r, date, DATASET)
    if aggregation == 'average':
        global_model = weight_avg(global_model, clients)
    if aggregation == 'proposed_leaveout_0_1':
        global_model = proposed_leaveout_0_1(global_model, clients, test_loader, r, date, DATASET)

    # test global model
    test_acc, test_loss = global_test(global_model, test_loader)
    print('Global test accuracy', test_acc)

    # write accuracy to file
    fglobal = 'results/' + DATASET + '/global-acc-' + date + '.csv'
    flocal = 'results/' + DATASET + '/local-acc-' + date + '.csv'
    record_global(fglobal, r, test_acc)
    record_local(flocal, r, local_acc)
    global_acc.append(test_acc)

    # sync clients with global model
    if syncing:
        for client in clients:
            client.load_state_dict(global_model.state_dict())

# graph accuracy of global model across rounds
graph_metric(global_acc, 'Accuracy')
