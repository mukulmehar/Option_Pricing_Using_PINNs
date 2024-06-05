import sys
print(sys.executable)
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy

from model.pinn import PINNs
from model.pinnsformer import PINNsformer
from pyhessian import hessian
from util import get_data

dev = torch.device('cpu')