import torch 
import numpy as np
import torch.nn as nn

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


#using numpy array
a1=np.array([2.0,1.0,0.1])
d1=softmax(a1)
print(f'logits probability:{d1}')

x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim=0) # along values along first axis
print('softmax torch:', outputs)