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
import collections

import pdb


def make_gaussian(orig, sd=20):
    gaussian_p = []
    for client in orig:
        new_dict = collections.OrderedDict()
        all_parameters = client.state_dict().keys()
        for k in all_parameters:
            p = client.state_dict()[k].detach().numpy()
            p_shape = np.shape(p)
            len_dims = len(p_shape)
            N = p_shape[len_dims - 1]
            cov = sd * np.identity(N)
            mean = np.zeros(N)
            if len_dims <= 1:
                new_p = np.random.multivariate_normal(mean, cov)
            else:
                new_p = np.random.multivariate_normal(mean, cov, p_shape[:(len_dims - 1)])
            new_dict[k] = torch.from_numpy(new_p)
        client.load_state_dict(new_dict)
        # todo: is this necessary or can you just return orig
        gaussian_p.append(client)
    return gaussian_p


def make_foe(orig_h, orig_b, epsilon=1):
    p_foe = []
    # just use any client for the state dict, they're all the same architecture
    avg_h = orig_h[0].state_dict()
    byzantine_p = collections.OrderedDict()

    for k in avg_h.keys():
        # get the average of the honest clients
        avg_h[k] = torch.stack(
            [orig_h[i].state_dict()[k].float() for i in range(len(orig_h))],
            0).sum(0)
        # set the current key value in the byzantine state dict
        byzantine_p[k] = -epsilon * avg_h[k]

    # change the state dict in all byzantine models
    for b in orig_b:
        b.load_state_dict(byzantine_p)
        # todo: maybe this is redundant and you can just return orig_b
        p_foe.append(b)
    # todo: change "weights" to "parameters"
    return p_foe
