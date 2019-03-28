import time
from options.test_options import TestOptions
from data import create_dataset
from models import *
from util.util import *
import os.path as osp
import sys,os
from models.base_visual import BaseVisual
import torch.nn as nn


if __name__ == '__main__':

    ##################################
    #Preparing the Logger and Backup
    ##################################
    opt,logger = TestOptions().parse()  
    opt.batch_size=1
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    ##################################
    #Preparing the Dataset
    ##################################
    test_loader = create_dataset(opt) 

    ##################################
    #Initilizing the Model
    ##################################
    model = create_model(opt)
   # the_model = torch.load(osp.join(opt.backup_path, 'Best.pth'))
  #  model.load_state_dict(the_model)

    print(model)
#    grad_cam = GradCam(model=model, target_layer_names=['layer4'],use_cuda=True)

    criterion = nn.CrossEntropyLoss().cuda()
    runner = BaseVisual(opt, model)
     
    runner.test(test_loader)

