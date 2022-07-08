import training 
import model_helper

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
        
class Client(DistributedDevices):
    
    def __init__(self, data, model, name, dim, name_dataset, id_num):
        super().__init__(data, name, dim, name_dataset, id_num)  
        
        self.model = model  
        self.W = {name:value.to(device) for name,value in self.model.named_parameters()}
        self.W_old = {name: torch.zeros(value.shape).to(device) for name, value in self.W.items()}
        
        self.dW = {name: torch.zeros(value.shape).to(device) for name, value in self.W.items()}
        self.A = {name: torch.zeros(value.shape).to(device) for name, value in self.W.items()}  
        self.n_params = sum([T.numel() for T in self.W.values()])
            
        self.train_loss = -1
        self.bits_sent = []
        
    def syncronize_with_server_voter(self,server):
        dest_voter = str(self.id_num)[-1]
        voter_name = "voter_"+dest_voter
        model_helper.copy(target=self.W, source=server.W_list[voter_name])
        
    def compute_weight_update(self,epochs,batch_size,learning_rate):
        self.model.train()
        
        # W_old = W
        model_helper.copy(target=self.W_old, source=self.W)
        
        self.model.train()
        self.model,self.train_loss = training.train_local(self.model,self.data[0],self.data[1],self.name,epochs,batch_size,learning_rate)    
        self.model.eval()
    
        # dW = W - W_old
        model_helper.subtract_(target=self.dW, minuend=self.W, subtrahend=self.W_old)
        
        # new W
        self.W = {name:value.to(device) for name,value in self.model.named_parameters()}
    
        update_size = sum([model_helper.count_bits(T) for T in self.dW.values()])
        update_size = math.ceil(update_size / 8)
        self.bits_sent.append(update_size)
        
        return self.train_loss
    
    def validation(self,val_data,size_per_class):
        accuracy, target, predictions = training.validation(self.model,val_data[0],val_data[1],
                                                            self.name,size_per_class,self.dim,self.name_dataset)
        
        print("Client {} is evaluated on {} sample and accuracy score is {}".format(self.id_num,10*size_per_class,accuracy))
        
        return accuracy

    
class Server(DistributedDevices):
    
    def __init__(self,data,voter_models, name, dim, name_dataset, id_num, num_voters,client_sizes,num_class):
        super().__init__(data, name, dim, name_dataset, id_num)
        
        self.voter_list = ["voter_"+str(i) for i in range(num_class)]
        self.voter_model_dict = {voter:model for voter,model in zip(self.voter_list,voter_models)}

        
        self.W_list, self.dW_list, self.A_list, self.n_params_list = {}, {}, {}, {}
        for voter,model in self.voter_model_dict.items():
            W = {name:value for name,value in model.named_parameters()}
            dW = {name: torch.zeros(value.shape).to(device) for name, value in W.items()}
            A = {name: torch.zeros(value.shape).to(device) for name, value in W.items()}
            n_params = sum([T.numel() for T in W.values()])
            
            self.W_list[voter] = W
            self.dW_list[voter] = dW
            self.A_list[voter] = A
            self.n_params_list[voter] = n_params
        
        self.bits_sent = []
        self.client_sizes = torch.Tensor(client_sizes)
        self.val_scores = []
        
    def aggregate_weight_updates(self, clients, aggregation="mean",size_per_class=200):
        voter_aggregate_dict = {voter:[] for voter in self.voter_list}
        
        for c in clients:
            dest_voter = "voter_"+str(c.id_num)[-1]
            voter_aggregate_dict[dest_voter].append(c)
        
        """
        print("voter aggregation selection is:")
        for voter_name,clients in voter_aggregate_dict.items():
            print("{}:{}".format(voter_name,[client.name for client in clients]))
        """
        
        # dW = aggregate(dW_i, i=1,..,n)
        client_count = 0
        for voter_name,clients in voter_aggregate_dict.items():
            if aggregation == "mean":
                model_helper.average(target=self.W_list[voter_name], sources=[client.W for client in clients])
                client_count = client_count + len(clients)
                 
            elif aggregation == "weighted_mean":
                model_helper.weighted_average(target=self.W_list[voter_name], sources=[client.W for client in clients],
                                              weights=torch.stack([self.client_sizes[client_count+i] for i in range(len(clients))]))
                client_count = client_count + len(clients)
            
            elif aggregation == "majority":
                model_helper.majority_vote(target=self.W_list[voter_name], sources=[client.W for client in clients], lr=0.000075)
            
            
        # take evaluation score for each voter
        self.val_scores = training.collect_server_model_validation_scores(self.voter_model_dict,
                                                                          self.data[0],
                                                                          self.data[1],
                                                                          size_per_class,
                                                                          self.dim,
                                                                          self.name_dataset)

    def server_evaluation(self,data,label):
        pred_result, voting_list = training.server_voting(self.voter_model_dict,
                                                          data,
                                                          self.dim,
                                                          self.name_dataset,
                                                          self.val_scores)
        score = np.mean(pred_result==label)
        return score, pred_result, voting_list
        