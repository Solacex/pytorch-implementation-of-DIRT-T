import torch
import torch.nn
from torch.nn import init

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight, a=0, mode='fan_out')
        init.constant(m.bias, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight, 1.0, 0.02)
        init.constant(m.bias, 0.0)
     print('initializing the weight of {} using Kaiming Initialization Method'.format(classname))
   
def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal(m.weight.data, std=0.001)
        init.constant(m.bias.data, 0.0)
    print('initializing the weight of {} using Normal Initialization Method'.format(classname))