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