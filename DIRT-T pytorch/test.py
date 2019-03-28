import time
from options.test_options import TestOptions
from data import create_dataset
import models
from util.util import *
import os.path as osp
import sys,os
from models.testor import Testor
import torch.nn as nn

if __name__ == '__main__':
    ##################################
    #Preparing the Logger and Backup
    ##################################
    opt,logger = TestOptions().parse()  

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    ##################################
    #Preparing the Dataset
    ##################################
    test_loader = create_dataset(opt) 

    ##################################
    #Initilizing the Model
    ##################################
    model = models.create_model(opt)

    criterion = nn.CrossEntropyLoss().cuda()
    runner = Testor(opt, model, criterion)
    runner.load_network(opt)
 
    runner.test(test_loader)

