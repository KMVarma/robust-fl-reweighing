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

def weight_avg(global_model, clients):
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([clients[i].state_dict()[k].float() for i in range(len(clients))],
                                     0).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model