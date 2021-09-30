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

def weight_avg(global_model, clients):
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([clients[i].state_dict()[k].float() for i in range(len(clients))],
                                     0).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model

def proposed(global_model, clients, test_loader, r, date, dataset):
    global_dict = global_model.state_dict()
    losses = []
    accuracies = []
    for c in clients:
        acc, loss = global_test(c, test_loader)
        accuracies.append(acc)
    weight = []
    for a in accuracies:
        weight.append(a**10)
    weight_sum = sum(weight)
    weight = [w/weight_sum for w in weight]
    # normalize so that some are pruned
    #weight = normalize_by_range(weight)
    record_local('results/' + dataset + '/accuracies-' + str(date) + '.csv', r, accuracies)
    record_local('results/' + dataset + '/weights-' + str(date) + '.csv', r, weight)
    for k in global_dict.keys():
        global_dict[k] = torch.stack([weight[i] * clients[i].state_dict()[k].float() for i in range(len(clients))],
                                     0).sum(0)
    global_model.load_state_dict(global_dict)
    return global_model