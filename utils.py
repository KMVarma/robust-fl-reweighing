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

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import pdb


def local_train(epoch, model, optimizer, train_loader, flip_label=False):
    model.train()
    for i in range(epoch):
        correct = 0
        train_loss = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            if flip_label:
                y_lst = y.tolist()
                flipped_lst = [flip_label[1] if lbl is flip_label[0] else lbl for lbl in y_lst]
                y = torch.tensor(flipped_lst)
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


def global_test(model, test_loader, flip_label=False, full_metrics=True):
    # todo: try to write helper functions to eliminate the redundancy
    # todo: change all base class stuff to target class
    model.eval()

    # for full test set
    test_loss_full = 0
    correct_full = 0
    acc = 0

    # for label-flipping source class
    test_loss_s = 0
    correct_s = 0
    n_s = 0
    acc_s = 0

    # for label-flipping base class
    test_loss_b = 0
    correct_b = 0
    n_b = 0
    acc_b = 0

    # for label-flipping other classes
    test_loss_o = 0
    correct_o = 0
    n_o = 0
    acc_o = 0

    with torch.no_grad():
        all_y = []
        all_pred = []
        for x, y in test_loader:
            if flip_label:

                # label-flipping source class
                output = model(x)
                test_loss_s += F.nll_loss(output, y).item()
                pred_s = output.data.max(1, keepdim=True)[1]
                correct_s += pred_s.eq(y.data.view_as(pred_s)).sum()

                # label-flipping base class
                output = model(x)
                test_loss_b += F.nll_loss(output, y).item()
                pred_b = output.data.max(1, keepdim=True)[1]
                correct_b += pred_b.eq(y.data.view_as(pred_b)).sum()

                # all other classes
                output = model(x)
                test_loss_b += F.nll_loss(output, y).item()
                pred_o = output.data.max(1, keepdim=True)[1]
                correct_o += pred_o.eq(y.data.view_as(pred_o)).sum()

            if full_metrics:
                # full test set
                all_y.append(y)
                output = model(x)
                test_loss_full += F.nll_loss(output, y).item()
                pred = output.data.max(1, keepdim=True)[1]
                all_pred.append(pred)
                correct_full += pred.eq(y.data.view_as(pred)).sum()

    if flip_label:
        # label-flipping source class
        test_loss_s /= len(test_loader.dataset)
        acc_s = float(100. * correct_s / len(test_loader.dataset))

        # label-flipping base class
        test_loss_b /= len(test_loader.dataset)
        acc_b = float(100. * correct_b / len(test_loader.dataset))

        # label-flipping other classes
        test_loss_o /= len(test_loader.dataset)
        acc_o = float(100. * correct_o / len(test_loader.dataset))

    if full_metrics:
        # full test set
        test_loss_full /= len(test_loader.dataset)
        acc = float(100. * correct_full / len(test_loader.dataset))
    #print('ALL Y')
    #print(all_y)
    #print('ALL PRED')
    #print(all_pred)
    #print('acc', acc)
    return acc, test_loss_full, acc_s, test_loss_s, acc_b, test_loss_b, acc_o, test_loss_o


def graph_metric(metric_lst, metric_name):
    fig, ax1 = plt.subplots()
    labels = [metric_name]
    ax1.set_xlabel('Rounds')
    ax1.set_ylabel(metric_name)
    rounds = np.arange(1, len(metric_lst) + 1)
    ax1.plot(rounds, metric_lst)
    ax1.tick_params(axis='y')
    ax1.legend(labels, loc='best')
    fig.tight_layout()
    plt.show()


def record_global(fname, r, test_acc):
    with open(fname, 'a', newline='') as csvfile1:
        writer = csv.DictWriter(csvfile1, fieldnames=['round', 'acc'])
        writer.writerow({'round': r,
                         'acc': test_acc})


def record_local(fname, r, local_acc):
    with open(fname, 'a', newline='') as csvfile1:
        writer = csv.DictWriter(csvfile1, fieldnames=['round',
                                                      'c1', 'c2', 'c3', 'c4', 'c5',
                                                      'c6', 'c7', 'c8', 'c9', 'c10',
                                                      'c11', 'c12', 'c13', 'c14', 'c15',
                                                      'c16', 'c17', 'c18', 'c19', 'c20',
                                                      'c21', 'c22', 'c23', 'c24', 'c25'])
        writer.writerow({'round': r,
                         'c1': local_acc[0], 'c2': local_acc[1], 'c3': local_acc[2],
                         'c4': local_acc[3], 'c5': local_acc[4], 'c6': local_acc[5],
                         'c7': local_acc[6], 'c8': local_acc[7], 'c9': local_acc[8],
                         'c10': local_acc[9], 'c11': local_acc[10], 'c12': local_acc[11],
                         'c13': local_acc[12], 'c14': local_acc[13], 'c15': local_acc[14],
                         'c16': local_acc[15], 'c17': local_acc[16], 'c18': local_acc[17],
                         'c19': local_acc[18], 'c20': local_acc[19], 'c21': local_acc[20],
                         'c22': local_acc[21], 'c23': local_acc[22], 'c24': local_acc[23],
                         'c25': local_acc[24]})


def normalize_by_range(x, new_max=1, new_min=0):
    return [((((val - min(x)) * (new_max - new_min)) / (max(x) - min(x))) + new_min) for val in x]
