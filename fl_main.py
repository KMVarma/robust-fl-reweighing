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

# hyperparameters
n = 25
batch_size = 128
lr = .01

# sync local models to global model after each global round
syncing = False
# global rounds
rounds = 50
# local epochs
epochs = 3

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
#aggregation = 'proposed_leaveout_0_1'

# attacks
n_honest = 13
#attack = 'none'
attack = 'foe'
#attack = 'gaussian'
sd = 20
# attack = 'label-flipping'
flip_label = [False] * n
flip_tuple = (1, 4)

if attack == 'label-flipping':
    print('Executing a Label-flipping attack')
    for i in range(len(flip_label)):
        if i >= n_honest:
            flip_label[i] = flip_tuple

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

testset0 = torchvision.datasets.MNIST(root='./mnist_data', train=False, download=True,
                                   transform=transforms.ToTensor())
testset1 = torchvision.datasets.MNIST(root='./mnist_data', train=False, download=True,
                                   transform=transforms.ToTensor())
testset2 = torchvision.datasets.MNIST(root='./mnist_data', train=False, download=True,
                                   transform=transforms.ToTensor())
testset3 = torchvision.datasets.MNIST(root='./mnist_data', train=False, download=True,
                                   transform=transforms.ToTensor())
testset4 = torchvision.datasets.MNIST(root='./mnist_data', train=False, download=True,
                                   transform=transforms.ToTensor())
testset5 = torchvision.datasets.MNIST(root='./mnist_data', train=False, download=True,
                                   transform=transforms.ToTensor())
testset6 = torchvision.datasets.MNIST(root='./mnist_data', train=False, download=True,
                                   transform=transforms.ToTensor())
testset7 = torchvision.datasets.MNIST(root='./mnist_data', train=False, download=True,
                                   transform=transforms.ToTensor())
testset8 = torchvision.datasets.MNIST(root='./mnist_data', train=False, download=True,
                                   transform=transforms.ToTensor())
testset9 = torchvision.datasets.MNIST(root='./mnist_data', train=False, download=True,
                                   transform=transforms.ToTensor())

# there will be 1,000 per class with MNIST test
idx0 = testset0.test_labels==0
idx1 = testset1.test_labels==1
idx2 = testset2.test_labels==2
idx3 = testset3.test_labels==3
idx4 = testset4.test_labels==4
idx5 = testset5.test_labels==5
idx6 = testset6.test_labels==6
idx7 = testset7.test_labels==7
idx8 = testset8.test_labels==8
idx9 = testset9.test_labels==9
#pdb.set_trace()

n_perclass = 1000

testset0.targets = testset0.targets[idx0]#[:n_perclass]
testset1.targets = testset1.targets[idx1]#[:n_perclass]
testset2.targets = testset2.targets[idx2]#[:n_perclass]
testset3.targets = testset3.targets[idx3]#[:n_perclass]
testset4.targets = testset4.targets[idx4]#[:n_perclass]
testset5.targets = testset5.targets[idx5]#[:n_perclass]
testset6.targets = testset6.targets[idx6]#[:n_perclass]
testset7.targets = testset7.targets[idx7]#[:n_perclass]
testset8.targets = testset8.targets[idx8]#[:n_perclass]
testset9.targets = testset9.targets[idx9]#[:n_perclass]

testset0.data = testset0.data[idx0]#[:n_perclass]
testset1.data = testset1.data[idx1]#[:n_perclass]
testset2.data = testset2.data[idx2]#[:n_perclass]
testset3.data = testset3.data[idx3]#[:n_perclass]
testset4.data = testset4.data[idx4]#[:n_perclass]
testset5.data = testset5.data[idx5]#[:n_perclass]
testset6.data = testset6.data[idx6]#[:n_perclass]
testset7.data = testset7.data[idx7]#[:n_perclass]
testset8.data = testset8.data[idx8]#[:n_perclass]
testset9.data = testset9.data[idx9]#[:n_perclass]

test_loader0 = torch.utils.data.DataLoader(testset0, batch_size=batch_size, shuffle=True)
test_loader1 = torch.utils.data.DataLoader(testset1, batch_size=batch_size, shuffle=True)
test_loader2 = torch.utils.data.DataLoader(testset2, batch_size=batch_size, shuffle=True)
test_loader3 = torch.utils.data.DataLoader(testset3, batch_size=batch_size, shuffle=True)
test_loader4 = torch.utils.data.DataLoader(testset4, batch_size=batch_size, shuffle=True)
test_loader5 = torch.utils.data.DataLoader(testset5, batch_size=batch_size, shuffle=True)
test_loader6 = torch.utils.data.DataLoader(testset6, batch_size=batch_size, shuffle=True)
test_loader7 = torch.utils.data.DataLoader(testset7, batch_size=batch_size, shuffle=True)
test_loader8 = torch.utils.data.DataLoader(testset8, batch_size=batch_size, shuffle=True)
test_loader9 = torch.utils.data.DataLoader(testset9, batch_size=batch_size, shuffle=True)

class_test_loaders = [test_loader0,
                      test_loader1,
                      test_loader2,
                      test_loader3,
                      test_loader4,
                      test_loader5,
                      test_loader6,
                      test_loader7,
                      test_loader8,
                      test_loader9]

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
        train_acc, train_loss = local_train(epochs, clients[i], optimizers[i], train_loader[i], flip_label[i])
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
        global_model = proposed_leaveout_0_1(global_model, clients, test_loader, r, date, DATASET, class_test_loaders)

    # test global model
    test_acc, test_loss, acc_s, test_loss_s, acc_b, test_loss_b, acc_o, test_loss_o =\
        global_test(global_model, test_loader, (attack == 'label-flipping'))
    print('Global test accuracy', test_acc)
    if attack == 'label-flipping':
        print('Global source accuracy', acc_s)
        print('Global base accuracy', acc_b)

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
