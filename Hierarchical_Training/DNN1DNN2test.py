# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 11:00:30 2022

@author: awesomezhao
"""

from pyomo.environ import *
import pandas as pd
#import gurobipy
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from operator import itemgetter
from itertools import groupby

count5 = 0
count10 = 0
count15 = 0
count20 = 0
count30 = 0
count40 = 0
count50 = 0
    
def BatteryDegradation(Capacity, Temp, DISC, SOCL, SOCH, Type):
    
    input_x = np.zeros((1,6))
    input_x[0,0] = Capacity
    input_x[0,1] = Temp
    input_x[0,2] = DISC
    input_x[0,3] = SOCL
    input_x[0,4] = SOCH
    input_x[0,5] = Type
    
    x_tensor = torch.Tensor(input_x).unsqueeze(0).cuda()  # unsqueeze gives a 1, batch_size dimension
    
    hidden = None
    #test_out, test_h = load_rnn(x_tensor, hidden)          # use trained_rnn not load_rnn
    test_out = load_rnn(x_tensor)
    Degradation = test_out.cpu().data.numpy().flatten()
    return Degradation


load_DNN1 = torch.load('DNN1_resistance.pt')
load_DNN2 = torch.load('DNN1_cyclenumber.pt')
load_DNN3 = torch.load('DNN2_10_0301.pt')
load_DNN4 = torch.load('DNN1_2output.pt')
YOutput = []

X = np.load('2DNN_Input.npy')
Y = np.load('2DNN_Output.npy')

for i in range(len(X)):
    input_x1 = np.zeros((1,5))
    input_x1[0,0] = X[i,0]
    input_x1[0,1] = X[i,1]
    input_x1[0,2] = X[i,2]
    input_x1[0,3] = X[i,3]
    input_x1[0,4] = X[i,4]
    x_tensor1 = torch.Tensor(input_x1).unsqueeze(0).cuda()
    
    IR = load_DNN1(x_tensor1).cpu().data.numpy().flatten()
    
    input_x2 = np.zeros((1,5))
    input_x2[0,0] = X[i,0]
    input_x2[0,1] = X[i,1]
    input_x2[0,2] = X[i,2]
    input_x2[0,3] = X[i,3]
    input_x2[0,4] = X[i,4]
    x_tensor2 = torch.Tensor(input_x1).unsqueeze(0).cuda()
    
    Cycle = load_DNN2(x_tensor2).cpu().data.numpy().flatten()
    
    # input_x1 = np.zeros((1,5))
    # input_x1[0,0] = X[i,0]
    # input_x1[0,1] = X[i,1]
    # input_x1[0,2] = X[i,2]
    # input_x1[0,3] = X[i,3]
    # input_x1[0,4] = X[i,4]
    # x_tensor1 = torch.Tensor(input_x1).unsqueeze(0).cuda()
    # Results = load_DNN4(x_tensor1).cpu().data.numpy().flatten()
    

    input_x3 = np.zeros((1,7))
    input_x3[0,0] = IR
    input_x3[0,1] = X[i,1]
    input_x3[0,2] = X[i,2]
    input_x3[0,3] = X[i,3]
    input_x3[0,4] = X[i,4]
    input_x3[0,5] = X[i,0]
    input_x3[0,6] = Cycle
    x_tensor3 = torch.Tensor(input_x3).unsqueeze(0).cuda()
    
    Degradation = load_DNN3(x_tensor3)
    Degradation = Degradation.cpu().data.numpy().flatten()
    error_percentage = np.abs((Degradation-Y[i])/Degradation)
    YOutput.append(Degradation)
    
    if 0< error_percentage <= 0.05:
        count5 = count5 + 1
    elif 0.05 < error_percentage <= 0.1:
        count10 = count10 + 1     
    elif 0.1 < error_percentage <= 0.15:
        count15 = count15 + 1    
    elif 0.15 < error_percentage <= 0.20:
        count20 = count20 + 1
    elif 0.20 < error_percentage <= 0.30:
        count30 = count30 + 1
    elif 0.30 < error_percentage <= 0.40:
        count40 = count40 + 1    
    else:
        count50 = count50 +1
            
print('The total number of predictions error 0-5% is',count5, 'the ratio is: ',count5/len(Y))
print('The total number of predictions error 5-10% is',count10, 'the ratio is: ',count10/len(Y))
print('The total number of predictions error 10-15% is',count15, 'the ratio is: ',count15/len(Y))
print('The total number of predictions error 15-20% is',count20, 'the ratio is: ',count20/len(Y))
print('The total number of predictions error over 20% is',count30, 'the ratio is: ',count30/len(Y))  