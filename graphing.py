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

"""filename
with open(filename) as f:
    reader = csv.reader(f, delimiter=' ', quotechar='|')
    row_list = []
    for row in reader:
        # the values are read as strings, not keeping the layer name
        row = [float(x) for x in row if (x not in layers)]
        # not keeping the party number
        row_list.append(row[1:])
return row_list"""