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

def single_evluation(path,model,choice):
    datadict = scio.loadmat(path)
    layout_map = np.zeros((50, 50))
    location = datadict['list'][0]
    for i in location:
      i = i - 1
      layout_map[i // 10 * 5:i // 10 * 5 + 5, (i % 10) * 5:(i % 10) * 5 + 5] = np.ones((5, 5))
    heat_map = (datadict['u'] - 260) / 100

    layout_map, heat_map = trans_separate(layout_map, heat_map, (50, 50))
    layout_map = layout_map.unsqueeze(0).to(device)
    preds = model(layout_map,choice)
    sequence = preds.flatten().cpu().detach().numpy()
    sequence_y = heat_map.flatten().cpu().detach().numpy()
    index = np.argmax(sequence)
    max_error = abs(sequence[index] - sequence_y[index])
    pred_numpy = (preds.cpu().detach().numpy()[0, 0, :, :]) * 100 + 260
    loss = F.l1_loss(preds, heat_map.unsqueeze(0).to(device), reduction='mean').item()
    return loss, pred_numpy, max_error


import math
import random
import matplotlib.pyplot as plt

def function1(x):
    sub_choice = copy.deepcopy(x)
    for i in range(12):
        sub_choice[i]['rate'] = [sub_choice[i]['rate']]
    sub_model = Sub_SuperNetwork(sub_choice, shadow_bn=False, layers=12, classes=10, final_upsampling=2)
    value = count_parameters_in_MB(sub_model)
    return value

# def function2(x):
#     dataset_test = TestDataset(trans_separate, resize_shape=(50, 50))
#     test_queue = torch.utils.data.DataLoader(
#         dataset_test, batch_size=8, shuffle=False, pin_memory=True, num_workers=2)
#     model = SuperNetwork(shadow_bn=False, layers=12, classes=10, final_upsampling=2)
#     model.cuda()
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     mae, var, max_value = validate(model, test_queue, x, max_iter=-1, flag_detail=False)
#     return mae
def function2(x):
    device = torch.device('cuda')
    # model_path = '/mnt/sunjialiang/AutoDeeplab-master/mixpath_supernet_2/data/fpn.pth.598'
    model_path='mixpath_supernet_kw/data/fpn.pth.590'
    path = 'train/144.mat'
    # path = ''
    model = SuperNetwork(shadow_bn=False, layers=12, classes=10, final_upsampling=2)
    model.eval()
    model.cuda()
    model.load_state_dict(torch.load(model_path, map_location=device))
    loss, pred_numpy,max_error=single_evluation(path, model, x)
    return loss

#Function to find index of list
def index_of(a,list):
    for i in range(0,len(list)):
        if list[i] == a:
            return i
    return -1

def sort_by_values(list1, values):
    sorted_list = []
    while(len(sorted_list)!=len(list1)):
        if index_of(min(values),values) in list1:
            sorted_list.append(index_of(min(values),values))
        values[index_of(min(values),values)] = math.inf
    return sorted_list

def fast_non_dominated_sort(values1, values2):
    S=[[] for i in range(0,len(values1))]
    front = [[]]
    n=[0 for i in range(0,len(values1))]
    rank = [0 for i in range(0, len(values1))]

    for p in range(0,len(values1)):
        S[p]=[]
        n[p]=0
        for q in range(0, len(values1)):
            if (values1[p] < values1[q] and values2[p] < values2[q]) or (values1[p] <= values1[q] and values2[p] > values2[q]) or (values1[p] > values1[q] and values2[p] <= values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] < values1[p] and values2[q] < values2[p]) or (values1[q] <= values1[p] and values2[q] < values2[p]) or (values1[q] < values1[p] and values2[q] <= values2[p]):
                n[p] = n[p] + 1
        if n[p]==0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while(front[i] != []):
        Q=[]
        for p in front[i]:
            for q in S[p]:
                n[q] =n[q] - 1
                if( n[q]==0):
                    rank[q]=i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        front.append(Q)

    del front[len(front)-1]
    return front

#Function to calculate crowding distance
def crowding_distance(values1, values2, front):
    distance = [0 for i in range(0,len(front))]
    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])
    distance[0] = 4444444444444444
    distance[len(front) - 1] = 4444444444444444
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted1[k+1]] - values2[sorted1[k-1]])/(max(values1)-min(values1))
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted2[k+1]] - values2[sorted2[k-1]])/(max(values2)-min(values2))
    return distance

#Function to carry out the crossover
def crossover(a,b):
    choice_large = {
     0: {'conv': [0,1,2,3], 'rate': 1},
     1: {'conv': [0,1,2,3], 'rate': 1},
     2: {'conv': [0,1,2,3], 'rate': 1},
     3: {'conv': [0,1,2,3], 'rate': 1},
     4: {'conv': [0,1,2,3], 'rate': 1},
     5: {'conv': [0,1,2,3], 'rate': 1},
     6: {'conv': [0,1,2,3], 'rate': 1},
     7: {'conv': [0,1,2,3], 'rate': 1},
     8: {'conv': [0,1,2,3], 'rate': 1},
     9: {'conv': [0,1,2,3], 'rate': 1},
     10: {'conv': [0,1,2,3], 'rate': 1},
     11: {'conv': [0,1,2,3], 'rate': 1}}
    choice_small = {
            0: {'conv': [0], 'rate': 0},
            1: {'conv': [0], 'rate': 0},
            2: {'conv': [0], 'rate': 0},
            3: {'conv': [0], 'rate': 0},
            4: {'conv': [0], 'rate': 0},
            5: {'conv': [0], 'rate': 0},
            6: {'conv': [0], 'rate': 0},
            7: {'conv': [0], 'rate': 0},
            8: {'conv': [0], 'rate': 0},
            9: {'conv': [0], 'rate': 0},
            10: {'conv': [0], 'rate': 0},
            11: {'conv': [0], 'rate': 0}}
    for i in range(len(choice)):
      r=random.random()
      if r<0.5:             
          temp = choice_large.copy()
          a[i] = temp[i]
      else:
          temp = choice_small.copy()
          a[i] = temp[i]
    return a

#Function to carry out the mutation operator
def mutation(b):
    mutation_prob = random.random()
    if mutation_prob <1:
        r = random.sample(range(1,5),1)
        r = np.array(r)
        b = random_choice(path_num=4, m=r[0], layers=12)
    return b

#Main program starts here
pop_size = 50
max_gen = 1
kernel = [3, 5, 7, 9]
m_path = 4
layers = 12
solution = []
for i in range(pop_size):
    choice = random_choice(path_num=len(kernel), m=m_path, layers=layers)
    if i==0:
       choice = {
               0: {'conv': [0], 'rate': 0},
               1: {'conv': [0], 'rate': 0},
               2: {'conv': [0], 'rate': 0},
               3: {'conv': [0], 'rate': 0},
               4: {'conv': [0], 'rate': 0},
               5: {'conv': [0], 'rate': 0},
               6: {'conv': [0], 'rate': 0},
               7: {'conv': [0], 'rate': 0},
               8: {'conv': [0], 'rate': 0},
               9: {'conv': [0], 'rate': 0},
               10: {'conv': [0], 'rate': 0},
               11: {'conv': [0], 'rate': 0}}
    if i==29:
       choice = {
           0: {'conv': [0,1,2,3], 'rate': 1},
           1: {'conv': [0,1,2,3], 'rate': 1},
           2: {'conv': [0,1,2,3], 'rate': 1},
           3: {'conv': [0,1,2,3], 'rate': 1},
           4: {'conv': [0,1,2,3], 'rate': 1},
           5: {'conv': [0,1,2,3], 'rate': 1},
           6: {'conv': [0,1,2,3], 'rate': 1},
           7: {'conv': [0,1,2,3], 'rate': 1},
           8: {'conv': [0,1,2,3], 'rate': 1},
           9: {'conv': [0,1,2,3], 'rate': 1},
           10: {'conv': [0,1,2,3], 'rate': 1},
           11: {'conv': [0,1,2,3], 'rate': 1}}
    # if i < 16:
    #     choice = random_choice(path_num=len(kernel), m=m_path, layers=layers)
    # if i > 15:
    #     choice = random_choice(path_num=len(kernel), m=2, layers=layers)
    solution.append(choice)


gen_no=0
while(gen_no<max_gen):
    print("gen_no:",gen_no)
    print("\n")
    function1_values = [function1(solution[i]) for i in range(0,pop_size)]
    print(function1_values)
    function2_values = [function2(solution[i]) for i in range(0,pop_size)]
    # print(function2_values)
    non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:],function2_values[:])
#     print("The best front for Generation number ",gen_no, " is")
#     for valuez in non_dominated_sorted_solution[0]:
#         print(round(solution[valuez],3),end=" ")
#     print("\n")
    crowding_distance_values=[]
    for i in range(0,len(non_dominated_sorted_solution)):
        crowding_distance_values.append(crowding_distance(function1_values[:],function2_values[:],non_dominated_sorted_solution[i][:]))
    solution2 = solution[:]
    #Generating offsprings
    while(len(solution2)!=2*pop_size):
        a1 = random.randint(0,pop_size-1)
        b1 = random.randint(0,pop_size-1)
        solution2.append(crossover(solution[a1],solution[b1]))
        solution2.append(mutation(solution[b1]))
    function1_values2 = [function1(solution2[i]) for i in range(0,2*pop_size)]
    function2_values2 = [function2(solution2[i]) for i in range(0,2*pop_size)]

    if gen_no==0:
        plt.xlabel('Params', fontsize=15)
        plt.ylabel('Accuracy', fontsize=15)
        plt.scatter(function1_values, function2_values, c='red')
        scio.savemat(save_path + '/random.mat', {'function1_values': function1_values,
                                                                'function2_values': function2_values})
    non_dominated_sorted_solution2 = fast_non_dominated_sort(function1_values2[:],function2_values2[:])
    crowding_distance_values2=[]
    for i in range(0,len(non_dominated_sorted_solution2)):
        crowding_distance_values2.append(crowding_distance(function1_values2[:],function2_values2[:],non_dominated_sorted_solution2[i][:]))
    new_solution= []
    for i in range(0,len(non_dominated_sorted_solution2)):
        non_dominated_sorted_solution2_1 = [index_of(non_dominated_sorted_solution2[i][j],non_dominated_sorted_solution2[i] ) for j in range(0,len(non_dominated_sorted_solution2[i]))]
        front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])
        front = [non_dominated_sorted_solution2[i][front22[j]] for j in range(0,len(non_dominated_sorted_solution2[i]))]
        front.reverse()
        for value in front:
            new_solution.append(value)
            if(len(new_solution)==pop_size):
                break
        if (len(new_solution) == pop_size):
            break
    solution = [solution2[i] for i in new_solution]
    gen_no = gen_no + 1

#Lets plot the final front now
function1 = [i for i in function1_values]
function2 = [j for j in function2_values]
plt.xlabel('Params', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.scatter(function1, function2, c = 'green')
plt.savefig('nsga.jpg')
plt.show()
scio.savemat(save_path + '/nsga.mat', {'function1_values': function1_values,
                                                        'function2_values': function2_values})