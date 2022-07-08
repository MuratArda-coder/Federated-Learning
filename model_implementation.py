import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import random

import copy

def set_seed(seedNum):
    torch.manual_seed(seedNum)
    np.random.seed(seedNum)
    random.seed(seedNum)

class Model(nn.Module):
    def __init__(self,input_dim,num_channel,output_class,dropout_rate=None):
        super(Model, self).__init__()
        self.input_dim = copy.deepcopy(input_dim)
        self.filters = [32,16,8]
        self.linear = [64,32]
        
        self.intitial_layer = nn.Conv2d(num_channel,self.filters[0],kernel_size=5,stride=1,padding='same')
        self.batchnorm_initial_layer = nn.BatchNorm2d(self.filters[0])
        self.initial_pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.input_dim[0],self.input_dim[1] = self.input_dim[0]//2,self.input_dim[1]//2
        
        self.conv2d_1_layer = nn.Conv2d(self.filters[0],self.filters[1],kernel_size=5,stride=1,padding='same')
        self.batchnorm_1_layer = nn.BatchNorm2d(self.filters[1])
        self.layer_1_pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.input_dim[0],self.input_dim[1] = self.input_dim[0]//2,self.input_dim[1]//2
        
        self.conv2d_2_layer = nn.Conv2d(self.filters[1],self.filters[2],kernel_size=5,stride=1,padding='same')
        self.batchnorm_2_layer = nn.BatchNorm2d(self.filters[2])
        self.layer_2_pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.input_dim[0],self.input_dim[1] = self.input_dim[0]//2,self.input_dim[1]//2
        
        self.first_fcc = nn.Linear(self.input_dim[0]*self.input_dim[1]*self.filters[-1],self.linear[0])
        self.fcc_1_layer = nn.Linear(self.linear[0],self.linear[1])
        self.last_layer = nn.Linear(self.linear[-1],output_class)
        
    def forward(self,x):
        x = F.leaky_relu(self.batchnorm_initial_layer(self.intitial_layer(x)))
        x = self.initial_pool(x)
        
        x = F.leaky_relu(self.batchnorm_1_layer(self.conv2d_1_layer(x)))
        x = self.layer_1_pool(x)
        x = F.leaky_relu(self.batchnorm_2_layer(self.conv2d_2_layer(x)))
        x = self.layer_2_pool(x)
        
        x = x.view(-1, self.input_dim[0]*self.input_dim[1]*self.filters[-1])
        x = torch.tanh(self.first_fcc(x))
        x = torch.tanh(self.fcc_1_layer(x))
        
        x = self.last_layer(x)
        x = x.view(len(x))
        return x
    
class Advance_Model(nn.Module):
    def __init__(self,input_dim,num_channel,output_class,dropout_rate=0.25):
        super(Advance_Model, self).__init__()
        self.input_dim = copy.deepcopy(input_dim)
        self.filters = [64,32,32,16]
        self.linear = [128,64]
        
        self.intitial_layer = nn.Conv2d(num_channel,self.filters[0],kernel_size=5,stride=1,padding='same')
        self.batchnorm_initial_layer = nn.BatchNorm2d(self.filters[0])
        self.initial_pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.input_dim[0],self.input_dim[1] = self.input_dim[0]//2,self.input_dim[1]//2
        
        self.conv2d_1_layer = nn.Conv2d(self.filters[0],self.filters[1],kernel_size=5,stride=1,padding='same')
        self.batchnorm_1_layer = nn.BatchNorm2d(self.filters[1])
        self.layer_1_pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.input_dim[0],self.input_dim[1] = self.input_dim[0]//2,self.input_dim[1]//2
        
        self.conv2d_2_layer = nn.Conv2d(self.filters[1],self.filters[2],kernel_size=5,stride=1,padding='same')
        self.batchnorm_2_layer = nn.BatchNorm2d(self.filters[2])
        self.layer_2_pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.input_dim[0],self.input_dim[1] = self.input_dim[0]//2,self.input_dim[1]//2
        
        self.dropout_1 = nn.Dropout(dropout_rate)
        
        self.conv2d_3_layer = nn.Conv2d(self.filters[2],self.filters[3],kernel_size=5,stride=1,padding='same')
        self.batchnorm_3_layer = nn.BatchNorm2d(self.filters[3])
        self.layer_3_pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.input_dim[0],self.input_dim[1] = self.input_dim[0]//2,self.input_dim[1]//2
        
        
        self.first_fcc = nn.Linear(self.input_dim[0]*self.input_dim[1]*self.filters[-1],self.linear[0])
        
        self.dropout_2 = nn.Dropout(dropout_rate)
        
        self.fcc_1_layer = nn.Linear(self.linear[0],self.linear[1])
        self.last_layer = nn.Linear(self.linear[-1],output_class)
        
    def forward(self,x):
        x = F.gelu(self.batchnorm_initial_layer(self.intitial_layer(x)))
        x = self.initial_pool(x)
        
        x = F.gelu(self.batchnorm_1_layer(self.conv2d_1_layer(x)))
        x = self.layer_1_pool(x)
        
        x = self.dropout_1(x)
        
        x = F.gelu(self.batchnorm_2_layer(self.conv2d_2_layer(x)))
        x = self.layer_2_pool(x)
        
        x = F.gelu(self.batchnorm_3_layer(self.conv2d_3_layer(x)))
        x = self.layer_3_pool(x)
        
        x = x.view(-1, self.input_dim[0]*self.input_dim[1]*self.filters[-1])
        x = torch.tanh_(self.first_fcc(x))
        
        x = self.dropout_2(x)
        
        x =  torch.tanh_(self.fcc_1_layer(x))
        
        x = self.last_layer(x)
        
        ##########################################
        x = x.view(len(x))
        return x

class Vanilla_Model(nn.Module):
    def __init__(self,input_dim,num_channel,output_class):
        super(Vanilla_Model, self).__init__()
        self.input_dim = copy.deepcopy(input_dim)
        self.filters = [32,16,8]
        self.linear = [64,32]
        
        self.intitial_layer = nn.Conv2d(num_channel,self.filters[0],kernel_size=5,stride=1,padding='same')
        self.initial_pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.input_dim[0],self.input_dim[1] = self.input_dim[0]//2,self.input_dim[1]//2
        
        self.conv2d_1_layer = nn.Conv2d(self.filters[0],self.filters[1],kernel_size=5,stride=1,padding='same')
        self.layer_1_pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.input_dim[0],self.input_dim[1] = self.input_dim[0]//2,self.input_dim[1]//2
        
        self.conv2d_2_layer = nn.Conv2d(self.filters[1],self.filters[2],kernel_size=5,stride=1,padding='same')
        self.layer_2_pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.input_dim[0],self.input_dim[1] = self.input_dim[0]//2,self.input_dim[1]//2
        
        self.first_fcc = nn.Linear(self.input_dim[0]*self.input_dim[1]*self.filters[-1],self.linear[0])
        self.fcc_1_layer = nn.Linear(self.linear[0],self.linear[1])
        self.last_layer = nn.Linear(self.linear[-1],output_class)
        
    def forward(self,x):
        x = F.relu(self.intitial_layer(x))
        x = self.initial_pool(x)
        
        x = F.relu(self.conv2d_1_layer(x))
        x = self.layer_1_pool(x)
        x = F.relu(self.conv2d_2_layer(x))
        x = self.layer_2_pool(x)
        
        x = x.view(-1, self.input_dim[0]*self.input_dim[1]*self.filters[-1])
        x = F.relu(self.first_fcc(x))
        x = F.relu(self.fcc_1_layer(x))

        x = self.last_layer(x)
        x = x.view(len(x))
        return x




def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)
 

















