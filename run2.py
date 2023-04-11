import time
import data_utils
import model_implementation 
import training 
import model_helper
import client_server_multiclass

import numpy as np
import math
import copy
import torch
import torch.nn as nn
from torch.optim import Adam

import pickle
import json

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import random

from skimage.util import random_noise

import os



mempool_key = 1234                                          
mem_size = 4096                                             
memblock_key = 1294

def set_all_seed(seedNum):
    data_utils.set_seed(seedNum)
    model_implementation.set_seed(seedNum)
    client_server_multiclass.set_seed(seedNum)
    training.set_seed(seedNum)
    torch.manual_seed(seedNum)
    np.random.seed(seedNum)
    random.seed(seedNum)

#*Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#*Hyperparameters
num_server = 1
num_clients = 100
learning_rate = 0.001
data_each_client = 600
val_size_per_class = 1000
img_dim = [32,32]
name_dataset = "fashionmnist"
specific_ratio=0.10

global_model_directory = "Global_Model/"

ROUNDS = 45
LOCAL_EPOCHS = 20
BATCH_SIZE = 64

#*Create Directory for Voter Models
if not os.path.exists(global_model_directory):
    os.makedirs(global_model_directory)

#*Set Seed
seedNum = 1 
set_all_seed(seedNum)

#*Collect Data
classes, x_train, y_train,x_test, y_test = data_utils.get_fashionMnist_data()
img_channel,num_class = x_train.shape[1],len(set(y_train))

#*Preprocess and Distribute Data

"""
client_list, data_dict_split = data_utils.clean_split_data(images=x_train,
                                                           labels=y_train,
                                                           num_clients=num_clients,
                                                           shuffle=True,
                                                           name_dataset=name_dataset,
                                                           dim=img_dim,
                                                           data_size=data_each_client,
                                                           specific_ratio=specific_ratio)
"""
DATASET = pickle.load(open("DATASET_IID.pickle", "rb"))
client_list = DATASET["client_list"]
data_dict_split = DATASET["data_dict_split"]

# Make Corrupter Clients
CORRUPTED_CLIENTS_NUM = 50
CORRUPTED_CLIENTS = random.sample(range(0, len(client_list)), CORRUPTED_CLIENTS_NUM )


for c in CORRUPTED_CLIENTS:
    label = data_dict_split['client_'+str(c)][1].numpy()
    half_shuffle_index = np.linspace(0,len(label)-1,len(label)).astype(int)
    h1 = half_shuffle_index[:int(len(half_shuffle_index)*0.80)]
    h2 = half_shuffle_index[int(len(half_shuffle_index)*0.80):]

    np.random.shuffle(h1)
    half_shuffle_index = np.concatenate((h1,h2),axis=0)
    label = label[half_shuffle_index]

    data_dict_split['client_'+str(c)][1] = torch.from_numpy(label)

#*Create Client's model
cnn_models = []
for i in range(num_clients):
    set_all_seed(seedNum)
    model = model_implementation.MulticlassModel(input_dim=img_dim ,num_channel=1, output_class=10)
    model.apply(model_implementation.initialize_weights)
    if torch.cuda.is_available():
        model = model.cuda()
    cnn_models.append(model)

#*Store Each Client's Data Size
client_size = [data.shape[0] for data,label in data_dict_split.values()]

#*Create Validation Data
val_data = [x_test,y_test,name_dataset]

#*Create Clients and Distribute AI Models to Each Client
all_clients = [client_server_multiclass.Client_Multiclass(data_dict_split[name],cnn_model,name,img_dim,name_dataset,val_data,i) 
               for i,(name,cnn_model) in enumerate(zip(client_list,cnn_models))]

#*Create Server Global Model
server_names,server_id,num_of_voters = "server",1,num_class     

#*Create Main Global Server
global_model = model_implementation.MulticlassModel(input_dim=img_dim ,num_channel=img_channel, output_class=num_class)
server = client_server_multiclass.Server_Multiclass((x_test,y_test),global_model,server_names,img_dim,name_dataset,1,client_size,num_class)

######################################################
#torch.save(global_model.state_dict(), global_model_directory+"global_model_mc.pt")
#global_model.load_state_dict(torch.load(global_model_directory+"global_model_mc.pt"))
######################################################



#*Empty Lists
all_client_losses = []
server_val_scores = []
server_evaluation = []

print("PYTHON :: Start Distributed Training..\n")
t1 = time.time()
participating_clients = []

#*History Records: RECORDS = {"round_1":...,"round_2":...}
RECORDS = {}
RECORDS["Corrupted_Clients_Id"] = CORRUPTED_CLIENTS

INTERACTION_RATE = 0.5 # 0.25 <--> sender 0.25 receiver
data_change = int(len(x_train)//num_clients*0.1)

try:
    for round in range(ROUNDS):
        
        print("PYTHON :: ROUND {}!!".format(round+1))
        print("PYTHON :: Corrupted Clients Id At the Beginning {}!!".format(CORRUPTED_CLIENTS))
        
        participating_clients = []
        round_record, clients = {}, {}
        
        client_loss = []
        
        # Data Sharing
        random.seed(10+round)
        while True:
            flag = True
            interactive_clients = random.sample(range(0, len(client_list)), int(INTERACTION_RATE*len(client_list)) )
            sender = interactive_clients[:len(interactive_clients)//2]
            receiver = interactive_clients[len(interactive_clients)//2:]
            for c in sender:
                l = data_dict_split['client_'+str(c)][1]
                if len(l) < 200:
                    flag = False
            if flag:
                break
         
        for s,r in zip(sender,receiver):
            label_list = torch.reshape(data_dict_split['client_'+str(r)][1],(-1,1))
            new_label = torch.reshape(data_dict_split['client_'+str(s)][1][:data_change],(-1,1))
            data_dict_split['client_'+str(r)][1] = torch.flatten(torch.cat((label_list,new_label),dim=0))
            
            image = data_dict_split['client_'+str(s)][0][0]
            new_data = torch.reshape(data_dict_split['client_'+str(s)][0][:data_change],(-1,image.size()[0],image.size()[1],image.size()[2]))
            data_dict_split['client_'+str(r)][0] = torch.cat((data_dict_split['client_'+str(r)][0],new_data),dim=0)
            
            data_dict_split['client_'+str(s)][0] = data_dict_split['client_'+str(s)][0][data_change:]
            data_dict_split['client_'+str(s)][1] = data_dict_split['client_'+str(s)][1][data_change:]
            
            all_clients[s].data = data_dict_split['client_'+str(s)]
            all_clients[r].data = data_dict_split['client_'+str(r)]
        ########################################################################################################################## 
        print("Data is Shared!!!")
        
        round_record["sender client id"] = sender
        round_record["receiver client id"] = receiver
        round_record["data_change_rate"] = data_change
        
        for client in all_clients:

            participating_clients.append(client)
            
            client.syncronize_with_server_voter(server)
            client_update_start = time.time()
            loss,val_accuracy= client.compute_weight_update(epochs = LOCAL_EPOCHS,batch_size = BATCH_SIZE,learning_rate = learning_rate)
            client_update_duration = time.time() - client_update_start
                    
            client_loss.append(loss)

            print("PYTHON :: ClientName:{}\tAccuracies:{}".format(client.name,val_accuracy))
            print("PYTHON :: Client Update Duration: {}".format(client_update_duration))
            print("PYTHON :: Client Data Size: {}".format(len(client.data[0])))
                    
            #val_accuracy_types = ["accuracy","precision","recall","f1","macro_f1"]

                    
            client_record = {}
            client_record["client_id"] = client.id_num
            client_record["validation_scores"] = val_accuracy
            client_record["update_durations"] = client_update_duration
            client_record["last_loss"] = loss
            client_record["loss_function"] = "CrossEntropyLoss"
            client_record["learning_rate"] = learning_rate
            client_record["optimizer"] = "SGD"
            client_record["data_size"] = len(data_dict_split[client.name][1])
            client_record["batch_size"] = BATCH_SIZE
            client_record["local_epoch"] = LOCAL_EPOCHS 

            clients[client.name] = client_record
                            
            #print("PYTHON :: Client Update: all_clients:\n{}".format([c.name for c in participating_clients]))                     

           
        all_client_losses.append(client_loss)
        print("PYTHON :: all_clients:{}".format([c.name for c in participating_clients]))
        #aggregation_types = ["mean","score_based"]
                    
        server_aggregate_start = time.time()
                    
        aggregation_type = "score_averaging_cluster"
        server.aggregate_weight_updates(clients=participating_clients, aggregation=aggregation_type)
        val_x , val_y = training.arrange_validation_data(x_test,y_test,val_size_per_class//num_class,img_dim)
        score, pred_result, real_label = server.server_evaluation(data = val_x, label = val_y, dim = img_dim)
                    
        server_aggregate_test_duration = time.time() - server_aggregate_start

        print("PYTHON :: Server aggregation and test duration: {}".format(server_aggregate_test_duration))
        print("PYTHON :: Server validation: {}".format(score))
                    
        round_record["clients"] = clients
        round_record["server_id"] = server_id
        #round_record["num_voters"] = num_of_voters
        round_record["num_participating_clients"] = len(participating_clients)
        round_record["aggregation_type"] = aggregation_type
        round_record["num_clients"] = num_clients
        round_record["client_list"] = [c.name for c in participating_clients]
        round_record["server_update&test_duration"] = server_aggregate_test_duration
        round_record["server_validation_score"] = score
        #round_record["server_voter_scores"] = server.val_scores

        val_scores_for_each_class = {}
        for i in range(num_class) :
            pred_specific_class = [1 if p==i else 0 for p in pred_result]
            real_specific_class = [1 if r==i else 0 for r in real_label]

            accuracy = accuracy_score(real_specific_class, pred_specific_class)
            precision = precision_score(real_specific_class, pred_specific_class)
            recall = recall_score(real_specific_class, pred_specific_class)
            f1 = f1_score(real_specific_class, pred_specific_class)
            Accuracies = {"accuracy":accuracy,"precision":precision,"recall":recall,"f1":f1, "class_count":sum(real_specific_class)}

            val_scores_for_each_class["class_"+str(i)] = Accuracies
        round_record["val_scores_for_each_class"] = val_scores_for_each_class

        RECORDS["round_"+str(round+1)] = round_record
        #*Serialize data into file: Store Records
        json.dump(RECORDS, open( "HISTORICAL_RECORDS_STUDY_SCORED_COL_"+str(CORRUPTED_CLIENTS_NUM)+".json", 'w' ))

        #*Save Global Model
        torch.save(global_model.state_dict(), global_model_directory+"global_model_col_"+str(CORRUPTED_CLIENTS_NUM)+".pt")
        
    
except Exception as e:
    print('Something wrong')
    print(e)


# Read data from file:
# data = json.load( open( "file_name.json" ) )
