import time
from options.train_options import TrainOptions
from data import create_dataset
import models
from util.util import *
import os.path as osp
import sys,os
from models.runner import Runner
import torch.nn as nn
from tensorboardX import SummaryWriter
import time
from tsnecuda import TSNE
from data.base_loader import *
if __name__ == '__main__':
    start_time = time.time()

    opt, logger = TrainOptions().parse()   
    writer = SummaryWriter(osp.join(opt.backup,'visual'))

    src_train_loader, src_val_loader, tgt_train_loader, tgt_val_loader = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    torch.backends.cudnn.benchmark = True    

    ##################################
    #Initilizing the Model
    ##################################
    model,discriminator = models.create_model(opt) 
    
    criterion =  nn.CrossEntropyLoss().cuda() 
    runner = Runner(opt, model,discriminator, criterion, writer)
    
    ##################################
    #Start Training
    ##################################
    train_loader = JointLoader(src_train_loader, tgt_train_loader)

    for epoch in range(1, opt.epoch_count + 1):    
        runner.train(epoch, train_loader, logger)
        if epoch % opt.validate_freq == 0 and opt.val:            
       #     runner.validate(epoch, src_val_loader, logger)
            runner.validate2(epoch, tgt_val_loader, logger)
       #     runner.visualize(src_val_loader, tgt_val_loader, epoch)

        if epoch % opt.save_epoch_freq == 0: 
            print('saving the model at the end of epoch %d' % (epoch))
            runner.save_network('latest')
            runner.save_network(epoch)


    time_elapsed = time.time()-start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) 
