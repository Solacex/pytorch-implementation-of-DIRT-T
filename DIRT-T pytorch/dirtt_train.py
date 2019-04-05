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

if __name__ == '__main__':
    start_time = time.time()

    opt, logger = TrainOptions().parse()   
    writer = SummaryWriter(osp.join(opt.backup,'visual'))

    src_train_loader, src_val_loader, tgt_train_loader, tgt_val_loader = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    torch.backends.cudnn.benchmark = True    

    ##################################
    #Initilizing the Model
    ##################################
    model = DirtT(opt).cuda()
    teacher = DirtT(opt).cuda()
    
#    model.load_state_dict(torch.load('/home/liguangrui/domain_adaptation_base_code/checkpoints/baselineV4+VADA/Best.pth'))
 #   teacher.load_state_dict(torch.load('/home/liguangrui/domain_adaptation_base_code/checkpoints/baselineV4+VADA/Best.pth'))

    model.load_state_dict(torch.load('/home/liguangrui/domain_adaptation_base_code/checkpoints/baselineV10+vada/latest.pth'))
    teacher.load_state_dict(torch.load('/home/liguangrui/domain_adaptation_base_code/checkpoints/baselineV10+vada/latest.pth'))


    criterion =  nn.CrossEntropyLoss().cuda() 
    runner = DirtTRunner(opt, model, teacher, criterion, writer)
    
    ##################################
    #Start Training
    ##################################
    #train_loader = JointLoader(src_train_loader, tgt_train_loader)

    for epoch in range(1, opt.epoch_count + 1):    
        if epoch ==1:
            runner.validate(epoch, tgt_val_loader, logger)
        runner.train(epoch, tgt_train_loader, logger)
        if epoch % opt.validate_freq == 0 and opt.val:            
            runner.validate(epoch, tgt_val_loader, logger)
       #     runner.visualize(src_val_loader, tgt_val_loader, epoch)

        if epoch % opt.save_epoch_freq == 0: 
            print('saving the model at the end of epoch %d' % (epoch))
            runner.save_network('latest')
            runner.save_network(epoch)


    time_elapsed = time.time()-start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) 
