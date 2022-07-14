import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def nll_loss(output, target):
    return F.nll_loss(output, target)