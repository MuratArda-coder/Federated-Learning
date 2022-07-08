import data_utils
import model_implementation 
import training 
import model_helper
import client_server

import numpy as np
import math
import copy
import torch
import torch.nn as nn
from torch.optim import Adam

import pickle
import matplotlib.pyplot as plt



seedNum = 1
def set_all_seed(seedNum):
    data_utils.set_seed(seedNum)
    model_implementation.set_seed(seedNum)
    client_server.set_seed(seedNum)
    training.set_seed(seedNum)
    torch.manual_seed(seedNum)
    np.random.seed(seedNum)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_server_clone = 5
client_per_server = 20
#learning_rate = 0.000075

set_all_seed(seedNum)
learning_rate = 0.05
name_dataset = 'fashionmnist'
classes, x_train, y_train,x_test, y_test = data_utils.get_fashionMnist_data()
#classes, x_train, y_train,x_test, y_test = data_utils.get_mnist_data()
dim = [32,32]
data_size = 3000


sel_label = -6
specific_ratio = 0.25
other_labels={0:0.20, 
              1:0.10, 
              2:0.10, 
              3:0.10, 
              4:0.10, 
              5:0.10, 
              6:0.0, 
              7:0.10, 
              8:0.10, 
              9:0.10}

client_list, data_dict_split = data_utils.specific_split_data(x_train,y_train,10,True,name_dataset, 
                                                             dim,sel_label,other_labels,data_size,specific_ratio)

#client_list, data_dict_split = data_utils.split_data(x_train,y_train,10,True,name_dataset,dim,data_size)

model = model_implementation.Model(input_dim=dim,num_channel=1,output_class=1,dropout_rate=0.20)
model.apply(model_implementation.initialize_weights)

client_name = 'client_6'
epoch = 30
model.train()
set_all_seed(seedNum)
model, all_train_losses_per_epoch = training.train_local(model=model,
                                                         data=data_dict_split[client_name][0],
                                                         label=data_dict_split[client_name][1],
                                                         client_name=client_name,
                                                         epoch=epoch,
                                                         batch_size=16,
                                                         learning_rate=learning_rate)
model.eval()
overall_accuracy, specific_label, predictions = training.validation(model=model,
                                                                    test_data=x_test,
                                                                    test_label=y_test,
                                                                    client_name=client_name,
                                                                    size_per_class=100,
                                                                    dim=dim,
                                                                    name_dataset=name_dataset)

plt.plot(all_train_losses_per_epoch)
plt.title("all train losses per epoch ")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

import seaborn as sns
from sklearn.metrics import confusion_matrix
cf_matrix = confusion_matrix(specific_label, predictions)
ax = sns.heatmap(cf_matrix, annot=True, fmt='')
ax.set_title('Seaborn Confusion Matrix with labels\n\n')
ax.set_xlabel("predict_label", weight='bold').set_fontsize('18')
ax.set_ylabel("true_label", weight='bold').set_fontsize('18')
plt.show()

print("Validation Score is: {}".format(overall_accuracy))

print("**************************************************************")
acc = 0
for i in range(10):
    print("class: {} accuracy: {}".format(i,sum(specific_label[i*100:i*100+100]==predictions[i*100:i*100+100])))
    acc = acc+sum(specific_label[i*100:i*100+100]==predictions[i*100:i*100+100])
print("Overall accuracy points: {}".format(acc))








