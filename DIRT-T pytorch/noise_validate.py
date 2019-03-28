import time
from options.train_options import TrainOptions
from data import create_dataset
import models
from util.util import *
import os.path as osp
import sys,os
from models.dirtt_runner import DirtTRunner
import torch.nn as nn
from tensorboardX import SummaryWriter
import time
from data.base_loader import *
from models.dirt_t import DirtT  
from models.runner import Runner

if __name__ == '__main__':
    start_time = time.time()

    opt, logger = TrainOptions().parse()   
    writer = SummaryWriter(osp.join(opt.backup,'visual'))

    src_train_loader, src_val_loader, tgt_train_loader, tgt_val_loader = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    torch.backends.cudnn.benchmark = True    

    ##################################
    #Initilizing the Model
    ##################################
 #   model = DirtT(opt).cuda()
    model,discriminator = models.create_model(opt)

    criterion =  nn.CrossEntropyLoss().cuda()
    runner = Runner(opt, model,discriminator, criterion, writer)
    
#    model.load_state_dict(torch.load('/home/liguangrui/domain_adaptation_base_code/checkpoints/baselineV4+VADA/Best.pth'))
 #   teacher.load_state_dict(torch.load('/home/liguangrui/domain_adaptation_base_code/checkpoints/baselineV4+VADA/Best.pth'))

    model.load_state_dict(torch.load('/home/liguangrui/domain_adaptation_base_code/checkpoints/baselineV9+vada+noNoise/latest.pth'))
        
    

#    criterion =  nn.CrossEntropyLoss().cuda() 
  #  runner = Runner(opt, model, criterion, writer)
    
    ##################################
    #Start Training
    ##################################
    #train_loader = JointLoader(src_train_loader, tgt_train_loader)

    for epoch in range(1, 10):    
        runner.noise_validate(epoch, tgt_val_loader, logger)



    time_elapsed = time.time()-start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) 
