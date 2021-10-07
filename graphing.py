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
import os
import pdb

def graph_weights(filename, n_clients=25, n_rounds=20):
    #todo: maybe write better so you can read easier
    all_clients = []
    for i in range(n_clients):
        all_clients.append(list())
    with open(filename) as f:
        reader = csv.reader(f, delimiter=' ', quotechar='|')
        for row in reader:
            split_row = str(row).split(',')[1:]
            split_row[24] = split_row[24][:-2]
            for i, c in enumerate(split_row):
                all_clients[i].append(float(c))

    fig, ax1 = plt.subplots()

    # axis for accuracy
    ax1.set_xlabel('rounds')
    ax1.set_ylabel('weight')
    rounds = np.arange(1, n_rounds + 1)
    for n, c_list in enumerate(all_clients):
        color = 'r'
        if n < 13:
            color = 'b'
        ax1.plot(rounds, c_list, c=color)
    ax1.tick_params(axis='y')
    ax1.legend(['honest'] * 13 + ['adversarial'] * 12, loc='upper right')

    fig.tight_layout()
    plt.show()

def graph_accuracy(n_rounds=20):
    all_acc = []
    """all_files = ['results/mnist/1stpower/globalacc.csv',
                 'results/mnist/10thpower/globalacc.csv',
                 'results/mnist/30thpower/globalacc.csv',
                 'results/mnist/40thpower/globalacc.csv',
                 'results/mnist/50thpower/globalacc.csv',
                 'results/mnist/75thpower/globalacc.csv',
                 'results/mnist/100thpower/globalacc.csv']"""
    all_files = ['results/mnist/cleanbase.csv',
                 'results/mnist/gaussianbase.csv',
                 'results/mnist/cleanproposed.csv',
                 'results/mnist/gaussianproposed.csv']
    for filename in all_files:
        acc = []
        with open(filename) as f:
            reader = csv.reader(f, delimiter=' ', quotechar='|')
            for row in reader:
                acc.append(float(row[0].split(',')[1]))
        all_acc.append(acc)

    fig, ax1 = plt.subplots()

    # axis for accuracy
    ax1.set_xlabel('rounds')
    ax1.set_ylabel('accuracy')
    rounds = np.arange(1, n_rounds + 1)
    for a1 in all_acc:
        ax1.plot(rounds, a1[:n_rounds])
    ax1.tick_params(axis='y')
    ax1.legend(['FedAvg (baseline) - no attack',
                'FedAvg (baseline) - Gaussian attack',
                'Proposed algorithm - no attack',
                'Proposed algorithm - Gaussian attack'], loc='best')

    fig.tight_layout()
    plt.show()

graph_weights('results/mnist/01/weights.csv')
#graph_accuracy()
