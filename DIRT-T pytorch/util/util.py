"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
from .logger import Logger
import os.path as osp
import json
import time

def mkdirs(paths):
    """create empty directories if they don't exist
    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def save_opt(opt, dir_name):
    with open('%s/opts.json'%dir_name,'w') as fp:
        json.dump(vars(opt), fp, indent=1)


def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def initialize_parser(opt):
    for op, value in opt.__dict__.items():
        print('{}   {}'.format(op,value))
    backup_dir = osp.join(opt.checkpoints_dir, opt.name)
    mkdir(backup_dir)
    save_opt(opt, backup_dir)
    logger = Logger(osp.join(backup_dir, 'log'+ time.strftime(".%m_%d_%H:%M:%S") + '.txt'))

    return backup_dir, logger 
def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
    """print current losses on console; also save the losses to the disk
    Parameters:
        epoch (int) -- current epoch
        iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
        losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
        t_comp (float) -- computational time per data point (normalized by batch_size)
        t_data (float) -- data loading time per data point (normalized by batch_size)
    """
    message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
    for k, v in losses.items():
        message += '%s: %.3f ' % (k, v)

    print(message)  # print the message
    with open(self.log_name, "a") as log_file:
        log_file.write('%s\n' % message)  # save the message
