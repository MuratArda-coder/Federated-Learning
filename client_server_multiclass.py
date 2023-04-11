import training 
import model_helper
import data_utils

import numpy as np
import math
import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seed(seedNum):
    torch.manual_seed(seedNum)
    np.random.seed(seedNum)

class DistributedDevices(object):
    
    def __init__(self, data, name, dim, name_dataset, id_num = -1):
        self.data = data
        self.name = name
        self.dim = dim
        self.id_num = id_num
        self.name_dataset = name_dataset
        
        self.loss_fn = nn.CrossEntropyLoss()
        
class Client_Multiclass(DistributedDevices):
    
    def __init__(self, data, model, name, dim, name_dataset, val_data, id_num):
        super().__init__(data, name, dim, name_dataset, id_num)  
        
        self.model = model  
        self.W = {name:value.to(device) for name,value in self.model.named_parameters()}
        self.W_old = {name: torch.zeros(value.shape).to(device) for name, value in self.W.items()}
        
        self.dW = {name: torch.zeros(value.shape).to(device) for name, value in self.W.items()}
        self.A = {name: torch.zeros(value.shape).to(device) for name, value in self.W.items()}  
        self.n_params = sum([T.numel() for T in self.W.values()])
            
        self.train_loss = -1
        self.val_accuracy = -1
        self.bits_sent = []
        
        self.val_data = val_data
        self.val_scores = 0
        
    def syncronize_with_server_voter(self,server):
        model_helper.copy(target=self.W, source=server.W)
        
    def compute_weight_update(self,epochs,batch_size,learning_rate):
        self.model.train()
        
        # W_old = W
        model_helper.copy(target=self.W_old, source=self.W)
        
        self.model,self.train_loss,self.val_accuracy = training.train_local_multiclass(self.model,self.data[0],self.data[1],
                                                                                       self.name,epochs,batch_size,learning_rate,
                                                                                       self.val_data)    
        self.model.eval()
    
        # dW = W - W_old

        # model_helper.subtract_(target=self.dW, minuend=self.W, subtrahend=self.W_old)

        # new W
        self.W = {name:value.to(device) for name,value in self.model.named_parameters()}

        update_size = sum([model_helper.count_bits(T) for T in self.dW.values()])
        update_size = math.ceil(update_size / 8)
        self.bits_sent.append(update_size)

        self.val_scores = self.val_accuracy

        return self.train_loss,self.val_accuracy
    
    def validation(self,val_data,size_per_class):
        accuracy, target, predictions = training.validation_multiclass(self.model,val_data[0],val_data[1],
                                                                       self.name,size_per_class,self.dim,self.name_dataset)
        
        print("Client {} is evaluated on {} sample and accuracy score is {}".format(self.id_num,10*size_per_class,accuracy))
        
        return accuracy

    
class Server_Multiclass(DistributedDevices):
    
    def __init__(self,data, global_model, name, dim, name_dataset, id_num, client_sizes,num_class):
        super().__init__(data, name, dim, name_dataset, id_num)
        self.model = global_model
        
        self.W = {name:value for name,value in self.model.named_parameters()}
        self.dW = {name: torch.zeros(value.shape).to(device) for name, value in self.W.items()}
        self.A = {name: torch.zeros(value.shape).to(device) for name, value in self.W.items()}
        self.n_params = sum([T.numel() for T in self.W.values()])

        
        self.bits_sent = []
        self.client_sizes = torch.Tensor(client_sizes)
        self.val_scores = 0
        
    def aggregate_weight_updates(self, clients, aggregation="mean",size_per_class=200,client_scores=None):
        
        # dW = aggregate(dW_i, i=1,..,n)
        client_count = 0
        if aggregation == "mean":
            model_helper.average(target=self.W, sources=[client.W for client in clients])
            client_count = client_count + len(clients)
            
        elif aggregation == "score_based":
            model_helper.score_averaging(target=self.W, 
                                         clients=clients,
                                         client_scores={c.name:c.val_scores for c in clients})
            client_count = client_count + len(clients)

        elif aggregation == "score_averaging_cluster":
            model_helper.score_averaging_cluster(target=self.W,
                                                 clients_dict={c.name: c for c in clients},
                                                 client_scores={c.name: c.val_scores for c in clients})
            client_count = client_count + len(clients)
            
        elif aggregation == "majority":
            model_helper.majority_vote(target=self.W, sources=[client.W for client in clients], lr=0.000075)
            
            
        # take evaluation score for each voter                                  
        self.val_scores = training.validation_multiclass(self.model,
                                                         self.data[0],
                                                         self.data[1],
                                                         "Global_Model",
                                                         size_per_class,
                                                         self.dim,
                                                         self.name_dataset)[0]

    def server_evaluation(self,data,label,dim):
        data = data_utils.image_reshape(data,dim)
        data, _ = data_utils.transform_image_array(data,np.zeros(0),self.name_dataset)

        if torch.cuda.is_available():
            data = data.cuda()

        with torch.no_grad():
            outputs = self.model(data)
        
        predictions = torch.max(outputs.cpu(),dim=1)[1].numpy()
            
        score = np.sum(predictions==label).item() / len(label)
        return score, predictions, label
        