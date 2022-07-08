import data_utils
import model_implementation 
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

def set_seed(seedNum):
    torch.manual_seed(seedNum)
    np.random.seed(seedNum)

def train_local(model,data,label,client_name,epoch,batch_size,learning_rate):
    specific_int = int(client_name[-1])
    specific_label = data_utils.make_specific_label(label,specific_int)
    
    #optimizer = Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.0)
    
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        data = data.cuda()
        specific_label = specific_label.cuda()
        print("Cuda Activated")
    
    extra = len(data)-(len(data)//batch_size)*batch_size
    batch_count = len(data)//batch_size if extra == 0 else len(data)//batch_size+1
    print("{} {} training starts!!".format(client_name.split("_")[0],client_name.split("_")[1]))
    
    all_train_losses_per_epoch = []
    for e in range(epoch):

        training_loss = []
        for b in range(batch_count):
            batch_data,batch_label = data_utils.collect_batch(data,specific_label,b,batch_size)

            #batch_label = batch_label.type(torch.LongTensor)
            if torch.cuda.is_available():
                batch_data, batch_label = batch_data.cuda(), batch_label.cuda()
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs,batch_label)
            
            training_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            outputs = model(data)
        #accuracy = torch.sum(torch.max(outputs,dim=1)[1]==specific_label).item() / len(specific_label)
        accuracy = binary_acc(outputs, specific_label)
        
        training_loss = np.average(training_loss)
        all_train_losses_per_epoch.append(training_loss)
        print("*",end=" ")    
    print('\nepoch: {} \t training loss: {} \t accuracy:{}'.format(e+1,training_loss,accuracy))
    print("{} {} training ends!!".format(client_name.split("_")[0],client_name.split("_")[1]))
    print("################################################")
    
    return model, all_train_losses_per_epoch

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

def prediction(model,data, initial_index,last_index,dim,name_dataset):
    if (initial_index > last_index) or (last_index > len(data)):
        print("no such prediction")
        return -1
    
    data = data_utils.image_reshape(data,dim)
    data, _ = data_utils.transform_image_array(data,np.zeros(0),name_dataset)
    
    if torch.cuda.is_available():
        data = data.cuda()
        
    with torch.no_grad():
        output = model(data[initial_index:last_index])
    
    # softmax = torch.exp(output).cpu()
    # prob = list(softmax.numpy())
    # predictions = np.argmax(prob, axis=1)
    # predictions_ration = np.max(prob, axis=1)
    
    predictions_ration = torch.sigmoid(output)
    predictions = torch.round(predictions_ration)
    
    return predictions,predictions_ration

def arrange_validation_data(test_data,test_label,size_per_class,dim,shuffle=False):
    if shuffle:
        test_data,test_label = data_utils.shuffle_image_array(test_data,test_label)
    
    test_data_reshaped = data_utils.image_reshape(test_data,dim)
    num_class = len(set(test_label))

    test_data_dict = {}
    for i in range(num_class):
        test_data_dict[i] = test_data_reshaped[test_label==i][:size_per_class]

    validation_data = np.zeros((size_per_class*num_class,test_data_reshaped.shape[1],test_data_reshaped.shape[2],test_data_reshaped.shape[3]))
    validation_label = np.zeros(size_per_class*num_class)

    count = 0
    for label,image_array in test_data_dict.items():
        for image in image_array:
            validation_data[count] = image
            validation_label[count] = label
            count = count+1
            
    return validation_data,validation_label

def validation(model,test_data,test_label,client_name,size_per_class,dim,name_dataset):
    val_x,val_y = arrange_validation_data(test_data,test_label,size_per_class,dim)
    val_x, val_y = data_utils.transform_image_array(val_x,val_y,name_dataset)

    specific_int = int(client_name[-1])
    specific_label = data_utils.make_specific_label(val_y,specific_int)
    
    if torch.cuda.is_available():
        val_x = val_x.cuda()
        model = model.cuda()

    with torch.no_grad():
        outputs = model(val_x)
        
    #overall_accuracy = torch.sum(torch.max(outputs.cpu(),dim=1)[1]==specific_label).item() / len(specific_label)
    
    overall_accuracy = binary_acc(outputs.cpu(), specific_label)
    
    predictions = torch.sigmoid(outputs)
    predictions = torch.round(predictions)
    
    
    # softmax = torch.exp(outputs).cpu()
    # prob = list(softmax.numpy())
    
    # predictions = np.argmax(prob, axis=1)
    specific_label = specific_label.cpu().numpy()
    predictions = predictions.cpu().numpy()
    
    return overall_accuracy, specific_label, predictions

def collect_server_model_validation_scores(model_dictionary,x_test,y_test,size_per_class,dim,name_dataset):
    dict_model_val_score = {}

    for name,model in model_dictionary.items():
        dict_model_val_score[name] = validation(model_dictionary[name],x_test,y_test,name,size_per_class,dim,name_dataset)[0]
    
    return dict_model_val_score

def server_voting(server_dict,data,dim,name_dataset,server_val_scores):
    
    predictions_for_input = {}
    predictions_ratio_for_input = {}
    for server_name,server_model in server_dict.items():
        pred,pred_ratio = prediction(server_model,data, 0,len(data),dim,name_dataset)
        predictions_for_input[server_name] = pred
        predictions_ratio_for_input[server_name] = pred_ratio
    
    list_of_result = []
    voting_list = []
    for i in range(len(data)):
        voting = [predictions_for_input[name][i] for name in predictions_for_input.keys()]
        vote_ratio = [predictions_ratio_for_input[name][i] for name in predictions_for_input.keys()]
        if sum(voting)==0:
            #voting_result = np.argmin([val for i,(name,val) in enumerate(server_val_scores.items())])
            #voting_result = -1
            voting_result = np.argmax([vote_ratio[i].cpu() for i,ratio in enumerate(server_val_scores.keys())])
        else:
            voting_result = np.argmax([val.cpu() if voting[i]==1 else 0 for i,(name,val) in enumerate(server_val_scores.items())])
        
        list_of_result.append(voting_result)
        voting_list.append(voting)
    
    return np.array(list_of_result), voting_list

