import torch 
import torchvision
from torch.utils.data import Dataset , DataLoader
import numpy as np
import math

# gradient computation etc. not efficient for whole data set
# -> divide dataset into small batches- "Using Dataloader"

#use datasets to make our own dataset that is customizable

# epoch = one forward and backward pass of ALL training samples
# batch_size = number of training samples used in one forward/backward pass
# number of iterations = number of passes, each pass (forward+backward) using [batch_size] number of sampes
# e.g : 100 samples, batch_size=20 -> 100/20=5 iterations for 1 epoch



# Dataset iis used to make your own customizable dataset
class WineDataset(Dataset):
    
    # to load data
    def __init__(self):
        xy=np.loadtxt('wine.csv',delimiter=',',skiprows=1,dtype=np.float32)
        self.n_samples=xy.shape[0]

        # creating two tensors 
        self.x=torch.from_numpy(xy[:,1:].astype(np.float32)) # it contains the entire dataset except the first row [n_samples, n_features]
        self.y=torch.from_numpy(xy[:,[0]].astype(np.float32)) # it contains all the row  and a single column [n_samples, 1]
    
    # to get the length of the datasets
    def __len__(self):
        return self.n_samples
    
    # for indexing throught dataset
    def __getitem__(self,index):
        return self.x[index], self.y[index]
    

data=WineDataset()
features,labels=data[0]
print(features,labels)

# Load whole dataset with DataLoader
# shuffle: shuffle data, good for training
# num_workers: faster loading with multiple subprocesses
# !!! IF YOU GET AN ERROR DURING LOADING, SET num_workers TO 0 !!!

dataload=DataLoader(dataset=data,shuffle=True,batch_size=4,num_workers=0)

# convert to an iterator and look at one random sample
data_iter=iter(dataload) # Creates an iterator from DataLoader
features,labels=next(data_iter) # fetch one batch from dataset
print(features,labels)

#-enumerate() helps track the batch index while iterating over DataLoader.

num_epoch=2
total_samples=len(dataload)
for epoch in range(num_epoch):
    for i ,(feat,labe) in enumerate(dataload):
        if (i+1) % 5 == 0:
         print(f'epoch{epoch+1} ,features{feat},label{labe}')




