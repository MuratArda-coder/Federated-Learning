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
learning_rate = 0.005

each_client_data = 600

seedNum = 1 
set_all_seed(seedNum)
    
classes, x_train, y_train,x_test, y_test = data_utils.get_fashionMnist_data()    

val_size_per_class = 1000
num_class = len(set(y_train))
(num_clients, dim, channel) = (num_server_clone*client_per_server, [32,32], x_train.shape[1])

name_dataset = "fashionmnist"
#client_list, data_dict_split = data_utils.split_data(x_train,y_train,num_clients,True,name_dataset,dim)
data_size = 600
client_list, data_dict_split = data_utils.specific_split_data(x_train,y_train,num_clients,True,name_dataset,dim,data_size,0.35)

"""    
for k,v in data_dict_split.items():
    data_dict_split[k][0] = v[0][:each_client_data]
    data_dict_split[k][1] = v[1][:each_client_data]
"""

cnn_models = []
for i in range(num_clients):
    set_all_seed(seedNum)
    model = model_implementation.Advance_Model(input_dim=[32,32],num_channel=1,output_class=1,dropout_rate=0.0)
    model.apply(model_implementation.initialize_weights)
    if torch.cuda.is_available():
        model = model.cuda()
    cnn_models.append(model)

    
client_size = [data.shape[0] for data,label in data_dict_split.values()]
        
#client_list_mine = ["client_"+str(i) for i in range(40)]
all_clients = [client_server.Client(data_dict_split[name],cnn_model,name,dim,name_dataset,i) 
               for i,(name,cnn_model) in enumerate(zip(client_list,cnn_models))]

server_names,server_id,num_of_voters = "server",1,num_class 
initial_voter_models = []
for i in range(num_of_voters): 
    set_all_seed(seedNum)
    model = model_implementation.Advance_Model(input_dim=[32,32],num_channel=1,output_class=1,dropout_rate=0.0)
    model.apply(model_implementation.initialize_weights)
    if torch.cuda.is_available():
        model = model.cuda()
    initial_voter_models.append(model)   
    
server = client_server.Server((x_test,y_test),initial_voter_models,server_names,
                               dim,name_dataset,server_id,num_of_voters,client_size,num_class)

round_server = 10
epoch = 15
batch_size = 16
all_client_losses = []
server_val_scores = []
server_evaluation = []
for r in range(round_server):
    client_loss = []
        
    for client in all_clients:
        print("Client Num:{} \t numRound {}".format(num_clients,r+1))
        set_all_seed(seedNum)
        client.syncronize_with_server_voter(server)
        loss = client.compute_weight_update(epochs = epoch, batch_size = batch_size, learning_rate = learning_rate )
        client_loss.append(loss) 
            
    all_client_losses.append(client_loss)
    server.aggregate_weight_updates(clients=all_clients, aggregation="mean")
        
    val_x , val_y = training.arrange_validation_data(x_test,y_test,val_size_per_class//10,dim)
    score, pred_result, _ = server.server_evaluation(data = val_x, label = val_y)
        
    server_val_scores.append(server.val_scores)
    server_evaluation.append([score,pred_result,val_y])
    print("Server_score: {}".format(score))
    print(server.val_scores)
    print("************************************************************")


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

val_x , val_y = training.arrange_validation_data(x_test,y_test,100,dim)

#score, pred_result, _ = server.server_evaluation(data = val_x, label = val_y)
pred_result, voting_list = training.server_voting(server.voter_model_dict,val_x,server.dim,server.name_dataset,server.val_scores)
score = np.mean(pred_result==val_y)
results = [score,pred_result,val_y]

classes, _, _,_, _ = data_utils.get_fashionMnist_data()
labels = [label+"\n("+str(i)+")" for i,label in classes.items()]

plt.figure(figsize=(12, 12))
cf_matrix = confusion_matrix(val_y, pred_result)
ax = sns.heatmap(cf_matrix, annot=True, fmt='')

ax.set_title('Seaborn Confusion Matrix with labels\n\n')
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)

ax.set_xlabel("predict_label", weight='bold').set_fontsize('18')
ax.set_ylabel("true_label", weight='bold').set_fontsize('18')
plt.show()

"""
plt.plot(all_client_losses[-1][6])
plt.title("all train losses per epoch ")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()
"""










