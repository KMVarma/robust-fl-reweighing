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
        acc, _ = global_test(c, test_loader)
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
        acc, _ = global_test(c, test_loader)
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

def sigmoid(z):
    return 1 / (1+math.exp(-z))
