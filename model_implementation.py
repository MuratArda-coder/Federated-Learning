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
        self.filters = [64,32,16]
        self.linear = [128,32]
        
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
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.first_fcc = nn.Linear(self.input_dim[0]*self.input_dim[1]*self.filters[-1],self.linear[0])
        self.fcc_1_layer = nn.Linear(self.linear[0],self.linear[1])
        self.last_layer = nn.Linear(self.linear[-1],output_class)
        
    def forward(self,x):
        x = F.leaky_relu(self.batchnorm_initial_layer(self.intitial_layer(x)))
        x = self.initial_pool(x)
        
        x = F.leaky_relu(self.batchnorm_1_layer(self.conv2d_1_layer(x)))
        x = self.layer_1_pool(x)
        
        x = self.dropout(x)
        
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

######################################################################
class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features, size=512, out=10):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            #nn.Dropout(),
            nn.Linear(size, size),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(size, size),
            nn.ReLU(True),
            nn.Linear(size, out),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = x.view(len(x))
        return x

def make_layers(cfg,channel):
    layers = []
    in_channels = channel
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def vgg11s(channel,out):
    return VGG(make_layers([32, 'M', 64, 'M', 128, 128, 'M', 128, 128, 'M', 128, 128, 'M'],channel), size=128, out=out)

def vgg11(channel,out):
    return VGG(make_layers([64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],channel), out=out)
######################################################################

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
 

class MulticlassModel(nn.Module):
    def __init__(self,input_dim,num_channel,output_class,dropout_rate=None):
        super(MulticlassModel, self).__init__()
        self.input_dim = copy.deepcopy(input_dim)
        self.filters = [64,32,16]
        self.linear = [128,32]
        
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
        x = F.elu(self.batchnorm_initial_layer(self.intitial_layer(x)))
        x = self.initial_pool(x)
        
        x = F.elu(self.batchnorm_1_layer(self.conv2d_1_layer(x)))
        x = self.layer_1_pool(x)
        x = F.elu(self.batchnorm_2_layer(self.conv2d_2_layer(x)))
        x = self.layer_2_pool(x)
        
        x = x.view(-1, self.input_dim[0]*self.input_dim[1]*self.filters[-1])
        x = torch.tanh(self.first_fcc(x))
        x = torch.tanh(self.fcc_1_layer(x))
        
        x = self.last_layer(x)
        return x











