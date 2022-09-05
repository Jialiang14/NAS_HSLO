import math
import torch
import torch.nn as nn
from utils import drop_path
import torch.nn.functional as F
from utils import *
from torchstat import stat
from model_search import *
from model import *
import os
import numpy as np
import torch
import time
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from model_search import SuperNetwork
import scipy.io as scio#从fpn文件中model导入fpn类
from segmentation_evaluation import validate
from mat2pic import GeneralDataset, TestDataset, trans_separate
from model_init import weights_init, weights_init_without_kaiming
from scheduler import WarmupMultiStepLR
import copy
save_path = '/mnt/sunjialiang'
device = torch.device('cuda')

import math
import random
import matplotlib.pyplot as plt

#Main program starts here
pop_size = 20
max_gen = 1
kernel = [3, 5, 7, 9]
m_path = 4
layers = 12
solution = []
for i in range(pop_size):
    choice = random_choice(path_num=len(kernel), m=m_path, layers=layers)
    solution.append(choice)

