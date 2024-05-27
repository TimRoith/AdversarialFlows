import torch.nn as nn

def get_two_moons_model(act_fun='ReLU'):
    '''
    Defines a simple neural network for the two moons dataset. 
    The structure is copied from ["A 'Hello World' for PyTorch"](https://seanhoward.me/blog/2022/hello_world_pytorch/) 
    tutorial by Sean T. Howard.
    '''
    if act_fun == 'ReLU':
        sigma = nn.ReLU()
    elif act_fun == 'GeLU':
        sigma = nn.GELU()
    else:
        raise ValueError('Unknown activation: ' + str(act_fun) +' function specified, please choose from: "ReLU", "GeLU"')
    
    return nn.Sequential(
    nn.Linear(2, 20),
    sigma,
    nn.BatchNorm1d(20),
    nn.Linear(20, 20),
    sigma,
    nn.BatchNorm1d(20),
    nn.Linear(20, 20),
    sigma,
    nn.BatchNorm1d(20),
    nn.Linear(20, 1),
    nn.Sigmoid()
)