import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as im
from PIL import ImageOps
import re

def get_mnist_data():
    data_train = torchvision.datasets.MNIST(root='./data/mnist', train=True, download=True)
    data_test = torchvision.datasets.MNIST(root='./data/mnist', train=False, download=True)
    
    x_train, y_train = data_train.train_data.numpy().reshape(-1, 1, 28, 28),  np.array(data_train.train_labels)
    x_test, y_test = data_test.test_data.numpy().reshape(-1, 1, 28, 28), np.array(data_test.test_labels)
    
    #classes = {int(re.sub(' ','',c.split('-')[0])):re.sub(' ','',c.split('-')[1]) for c in data_train.classes}
    classes = {i:c for i,c in enumerate(data_train.classes)}
    return (classes,x_train, y_train,x_test, y_test)

def get_fashionMnist_data():
    data_train = torchvision.datasets.FashionMNIST(root='./data/fashionMNIST', train=True, download=True)
    data_test = torchvision.datasets.FashionMNIST(root='./data/fashionMNIST', train=False, download=True)
    
    x_train, y_train = data_train.train_data.numpy().reshape(-1, 1, 28, 28),  np.array(data_train.train_labels)
    x_test, y_test = data_test.test_data.numpy().reshape(-1, 1, 28, 28), np.array(data_test.test_labels)
    
    classes = {i:c for i,c in enumerate(data_train.classes)}
    return (classes,x_train, y_train,x_test, y_test)

def get_cifar10_data():
    data_train = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=True)
    data_test = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=True)
    
    x_train, y_train = data_train.data.transpose((0, 3, 1, 2)),  np.array(data_train.targets)
    x_test, y_test = data_test.data.transpose((0, 3, 1, 2)), np.array(data_test.targets)
    
    classes = {i:c for i,c in enumerate(data_train.classes)}
    return (classes,x_train, y_train,x_test, y_test)

def set_seed(seedNum):
    torch.manual_seed(seedNum)
    np.random.seed(seedNum)
    print("In data utils seed is : ", input)

def imshow_without_label(img):
    if torch.is_tensor(img):
        img.numpy()
    
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()
    
def imshow_with_label(img,label):
    if torch.is_tensor(img):
        img.numpy()
    
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.title(label, fontname="Times New Roman", size=28,fontweight="bold")
    plt.show()
    

def get_transform_function(name_dataset):
    transforms_func = {
            'mnist': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.06078,), (0.1957,))
            ]),
            'fashionmnist': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]),
            'cifar10': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        }
    return transforms_func[name_dataset]

def transform_image_array(images,labels,name_dataset):
    transform = get_transform_function(name_dataset)
    
    normalized_images = torch.zeros(images.shape)
    for i,image in enumerate(images):
        image = image.astype(float)
        normalized_images[i] = transform(np.transpose(image, (1, 2, 0)))
    
    return normalized_images,torch.from_numpy(labels)

def shuffle_image_array(image,label):
    np.random.seed(1)
    shuffle_index = np.linspace(0,len(image)-1,len(image)).astype(int)
    np.random.shuffle(shuffle_index)
    
    return image[shuffle_index], label[shuffle_index]

def image_reshape(image_array,dim):
    new_dim = [image_array.shape[1]]+dim
    new_image_array = np.zeros([len(image_array)]+new_dim)
    for index,image in enumerate(image_array):
        image_resized = np.transpose(image,(2,1,0))
        if new_dim[0] == 1:
            image_resized = im.fromarray(np.reshape(image_resized,image_resized.shape[0:2]))
            image_resized = ImageOps.mirror(image_resized.rotate(270))
            image_resized = image_resized.resize(new_dim[1:])
            image_resized = np.reshape(np.array(image_resized),new_dim)
        else:
            image_resized = im.fromarray(image_resized)
            image_resized = image_resized.resize(new_dim[1:])
            image_resized = np.transpose(np.array(image_resized),(2,1,0))
        new_image_array[index] = image_resized
    return new_image_array.astype(int) if new_dim[0] == 3 else new_image_array

def split_data(images,labels,num_clients,shuffle,name_dataset,dim,data_size=None):
    extra = False
    images = image_reshape(images,dim)
    image_shape = images.shape
    
    if num_clients>len(images):
        print("Impossible Split!!")
        exit()
        
    if shuffle:
        images, labels = shuffle_image_array(images, labels)
    
    client_list = []
    for i in range(num_clients):
        client_list.append("client_"+str(i))
    
    images_normalized,labels_tensor = transform_image_array(images,labels,name_dataset)
    
    
    # Nonedefined Datasize
    if data_size==None:
        if(len(images)%num_clients != 0):
            extra_images = len(images)%num_clients
            extra = True
        len_data_per_clients = len(images)//num_clients
    
    # Predefined Datasize
    else:
        len_data_per_clients = data_size
        
    Data_Split_Dict = {} # Client_name: (image,label)
    for index,name in enumerate(client_list): 
        array_split = images_normalized[index*len_data_per_clients:(index*len_data_per_clients)+len_data_per_clients]
        label_split = labels_tensor[index*len_data_per_clients:(index*len_data_per_clients)+len_data_per_clients]
        Data_Split_Dict[name] = [array_split,label_split]

    
    client_names = [k for k,v in Data_Split_Dict.items()]
    if extra:
        for i, (image,label) in enumerate(zip(images_normalized[-1*extra_images:],labels_tensor[-1*extra_images:])):   
            new_data = torch.reshape(image,(-1,image.size()[0],image.size()[1],image.size()[2]))
            Data_Split_Dict[client_names[i%num_clients]][0] = torch.cat((Data_Split_Dict[client_names[i%num_clients]][0],new_data),dim=0)

            label_list = torch.reshape(Data_Split_Dict[client_names[i%num_clients]][1],(-1,1))
            new_label = torch.reshape(label,(-1,1))
            Data_Split_Dict[client_names[i%num_clients]][1] = torch.flatten(torch.cat((label_list,new_label),dim=0))
    
    return client_names,Data_Split_Dict         

def make_specific_label(label,specific_int):
    new_label = torch.zeros(label.size())
    for i in range(new_label.size()[0]):
        if label[i] == specific_int:
            new_label[i] = 1
        else: 
            new_label[i] = 0
    return new_label

def collect_batch(data,label,batch_num,batch_size):
    extra = len(data)-(len(data)//batch_size)*batch_size
    batch_count = len(data)//batch_size if extra == 0 else len(data)//batch_size+1
    
    if batch_num == batch_count and extra != 0:
        batch = (data[batch_num*batch_size:],label[batch_num*batch_size:])
    else:
        batch = (data[batch_num*batch_size:batch_num*batch_size+batch_size],label[batch_num*batch_size:batch_num*batch_size+batch_size])
    if batch_num >= batch_count:
        batch = (-1,-1)
        
    return batch


##############################################################################
##############################################################################
##############################################################################

def specific_split(own_label,other_labels_and_ratios,data,size,own_label_ratio,start_index):
    own_label_size = int(size*own_label_ratio)
    #other_label_size_each = (size-own_label_size)//len(other_labels)
    # other_labels_and_ratios= {0:0.32,1:0.236,...}
    valid_other_label = {l:r for l,r in other_labels_and_ratios.items() if r != 0}
    
    
    cnt = 0
    own_image_data = torch.zeros([own_label_size]+[i for i in data[0].shape[1:]])
    own_label_data = torch.zeros(own_label_size)
    for index,(i,l) in enumerate(zip(data[0][start_index:],data[1][start_index:])):
        if l == own_label:
            own_image_data[cnt] = i
            own_label_data[cnt] = l
            cnt = cnt+1
        if cnt>=own_label_size:
            break

    other_image_data = torch.zeros([size-own_label_size]+[i for i in data[0].shape[1:]])
    other_label_data = torch.zeros(size-own_label_size)

    end_index,itr = index,0
    for other_label,ratio in valid_other_label.items():
        cnt = 0
        other_size = int(len(other_label_data)*ratio)
        #print("{}  {}  {}".format(other_label,ratio,other_size))
        for index,(i,l) in enumerate(zip(data[0][start_index:],data[1][start_index:])):
            if l == other_label:
                other_image_data[itr] = i
                other_label_data[itr] = l
                cnt,itr = cnt+1,itr+1
            if cnt>=other_size:
                break
        if index>=end_index:
            end_index = index 
            
    client_data = torch.cat([own_image_data,other_image_data],dim=0)
    client_label = torch.cat([own_label_data,other_label_data],dim=0)
    
    client_data,client_label = shuffle_image_array(client_data,client_label)
    return end_index,client_data,client_label

def specific_split_data(images,labels,num_clients,shuffle,name_dataset,dim,sel_label,other_labels,data_size=None,specific_ratio=0.25):
    extra = False
    images = image_reshape(images,dim)
    image_shape = images.shape
    
    if num_clients>len(images):
        print("Impossible Split!!")
        exit()
        
    if shuffle:
        images, labels = shuffle_image_array(images, labels)
    
    client_list = []
    for i in range(num_clients):
        client_list.append("client_"+str(i))
    
    images_normalized,labels_tensor = transform_image_array(images,labels,name_dataset)
    
    
    # Nonedefined Datasize
    if data_size==None:
        if(len(images)%num_clients != 0):
            extra_images = len(images)%num_clients
            extra = True
        len_data_per_clients = len(images)//num_clients
    
    # Predefined Datasize
    else:
        len_data_per_clients = data_size
        
    Data_Split_Dict = {} # Client_name: (image,label)
    start_index_specific = 0
    for index,name in enumerate(client_list):
        if name[-1] != str(sel_label):
            array_split = images_normalized[index*len_data_per_clients:(index*len_data_per_clients)+len_data_per_clients]
            label_split = labels_tensor[index*len_data_per_clients:(index*len_data_per_clients)+len_data_per_clients]
            Data_Split_Dict[name] = [array_split,label_split]
        else: #Specific Split Part
            print("Specific Split for {}".format(name))
            end_,s_data,s_label = specific_split(own_label=sel_label,other_labels_and_ratios=other_labels,data=[images_normalized,labels_tensor],
                                                 size=len_data_per_clients,own_label_ratio=specific_ratio,start_index=start_index_specific)
            Data_Split_Dict[name] = [s_data,s_label]
            start_index_specific = end_
    
    client_names = [k for k,v in Data_Split_Dict.items()]
    if extra:
        for i, (image,label) in enumerate(zip(images_normalized[-1*extra_images:],labels_tensor[-1*extra_images:])):
            if name[-1] != str(sel_label):
                new_data = torch.reshape(image,(-1,image.size()[0],image.size()[1],image.size()[2]))
                Data_Split_Dict[client_names[i%num_clients]][0] = torch.cat((Data_Split_Dict[client_names[i%num_clients]][0],new_data),dim=0)

                label_list = torch.reshape(Data_Split_Dict[client_names[i%num_clients]][1],(-1,1))
                new_label = torch.reshape(label,(-1,1))
                Data_Split_Dict[client_names[i%num_clients]][1] = torch.flatten(torch.cat((label_list,new_label),dim=0))
    
    return client_names,Data_Split_Dict       













