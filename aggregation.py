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
from utils import global_test, normalize_by_range, record_local
import math

import pdb


def weight_avg(global_model, clients):
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([clients[i].state_dict()[k].float() for i in range(len(clients))],
                                     0).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model


def proposed_power(global_model, clients, test_loader, r, date, dataset, power=100):
    global_dict = global_model.state_dict()
    accuracies = []
    for c in clients:
        acc, _, _, _, _, _, _, _ = global_test(c, test_loader)
        accuracies.append(acc)
    weight = []
    for a in accuracies:
        weight.append(a**power)
    weight_sum = sum(weight)
    weight = [w/weight_sum for w in weight]
    record_local('results/' + dataset + '/accuracies-' + str(date) + '.csv', r, accuracies)
    record_local('results/' + dataset + '/weights-' + str(date) + '.csv', r, weight)
    for k in global_dict.keys():
        global_dict[k] = torch.stack([weight[i] * clients[i].state_dict()[k].float() for i in range(len(clients))],
                                     0).sum(0)
    global_model.load_state_dict(global_dict)
    return global_model


def proposed_0_1(global_model, clients, test_loader, r, date, dataset):
    global_dict = global_model.state_dict()
    accuracies = []
    for c in clients:
        acc, _, _, _, _, _, _, _ = global_test(c, test_loader)
        accuracies.append(acc)

    # normalize accuracies to [0,1] range
    acc_min = min(accuracies)
    acc_max = max(accuracies)
    normalized_acc = [((x - acc_min) / (acc_max - acc_min)) for x in accuracies]
    print(normalized_acc)
    # compute weights as the closest integer to the accuracy (0 or 1)
    weight = [round(x) for x in normalized_acc]
    print(weight)

    record_local('results/' + dataset + '/accuracies-' + date + '.csv', r, accuracies)
    record_local('results/' + dataset + '/weights-' + date + '.csv', r, weight)
    for k in global_dict.keys():
        global_dict[k] = torch.stack([weight[i] * clients[i].state_dict()[k].float() for i in range(len(clients))],
                                     0).sum(0)
    global_model.load_state_dict(global_dict)
    return global_model


def proposed_leaveout_0_1(global_model, clients_orig, test_loader, r, date, dataset, class_test_loaders):
    global_dict = global_model.state_dict()
    weights = []


    for c in range(len(clients_orig)):
        clients = clients_orig.copy()
        clients = clients + (len(clients_orig) * [clients[c]])
        all_by_class = []
        for k in global_dict.keys():
            global_dict[k] = torch.stack(
                [clients[i].state_dict()[k].float() for i in range(len(clients))],
                0).sum(0)
        global_model.load_state_dict(global_dict)
        acc_all, _, _, _, _, _, _, _ = global_test(global_model, test_loader)

        exclusive_clients = clients_orig.copy()
        exclusive_clients.pop(c)
        exclusive_clients = exclusive_clients
        for k in global_dict.keys():
            global_dict[k] = torch.stack(
                [exclusive_clients[i].state_dict()[k].float() for i in range(len(exclusive_clients))],
                0).sum(0)
        global_model.load_state_dict(global_dict)

        acc, _, _, _, _, _, _, _ = global_test(global_model, test_loader)
        diff = acc_all - acc
        weights.append(diff)
    print('***')
    print('***')
    print('***')
    print(weights)
    print('***')
    print('***')
    print('***')
    w_min = min(weights)
    w_max = max(weights)

    # todo: change this back
    if -acc_all == w_max:
        w_01 = [1.] * len(clients_orig)
    else:
        normalized_w = [((w - w_min) / (w_max - w_min)) for w in weights]
        print(normalized_w)
        # compute weights as the closest integer to the accuracy (0 or 1)
        w_01 = [round(x) for x in normalized_w]
    print('***')
    print('***')
    print('***')
    print(w_01)

    #record_local('results/' + dataset + '/accuracies-' + date + '.csv', r, weights)
    #record_local('results/' + dataset + '/weights-' + date + '.csv', r, w_01)
    for k in global_dict.keys():
        global_dict[k] = torch.stack([w_01[i] * clients_orig[i].state_dict()[k].float() for i in range(len(clients_orig))],
                                     0).sum(0)
    global_model.load_state_dict(global_dict)
    return global_model

def sigmoid(z):
    return 1 / (1+math.exp(-z))
