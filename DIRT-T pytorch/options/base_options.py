import argparse
import os
from util import util
import torch
import models
import data
import os.path as osp
from util.logger import Logger 
import time

class BaseOptions():

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options."""

        # Print and Backup
        parser.add_argument('--print_freq', '-p', default=100, type=int, metavar='N', help='print frequency (default: 10)')
        parser.add_argument('--checkpoints_dir','--chk', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')

        # Dataset and Data-loading
        parser.add_argument('--dataroot', type=str, default="/home/liguangrui/data/domain_adaptation_dataset")
        parser.add_argument('--num_threads', '--workers', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=64, help='input batch size')

        # Model modification
        parser.add_argument('--arch', type=str, default="resnet50")
        parser.add_argument('--model', type=str, default='dirt_t', help='chooses which model to use. [cycle_gan | pix2pix | test | colorization]')

        # Training and validate Hyper-parameter 
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=100, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--val', type=bool, default=True, help='validate or not')
        parser.add_argument('--validate_freq','--val_freq', type = int, default = 5, help = 'validate_freq')

        # Training Scheduler
        parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
        parser.add_argument('--lr_policy', type=str, default='step', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_step', default=40, type=float, metavar='LRSteps', help='epochs to decay learning rate by 10')
        parser.add_argument('--milestone', default=[30,60,90], nargs="+",  help='milestone for MultiStep scheduler')
        #Training Optimizer
        parser.add_argument('--optim', type=str, default="adam")
        parser.add_argument('--momentum', default=0.998, type=float, metavar='M', help='momentum')
        parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float, metavar='W', help='weight decay (default: 5e-4)') 
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')

        # Loss setting
        parser.add_argument('--loss_type', type=str, default="nll", choices=['nll'])

        # Model modifications
        parser.add_argument('--lr_steps', default=40, type=float, nargs="+", metavar='LRSteps', help='epochs to decay learning rate by 10')
        parser.add_argument('--dropout', type=float, default=0.5, help='dropout for the model')
        parser.add_argument('--clip-gradient', '--gd', default=20, type=float,metavar='W', help='gradient norm clipping (default: disabled)')
        parser.add_argument('--img_feature_dim', default=256, type=int, help="the feature dimension for each frame")
        parser.add_argument('--partialbn', '--pb', default=False, action="store_true")
  
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
        parser.add_argument('--modality', default='', type=str, help='modality')

        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
   #     model_name = opt.arch
       # model_option_setter = models.get_option_setter(model_name)
    #    parser = model_option_setter(parser, self.isTrain)
    #    opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
    #    dataset_name = opt.dataset_mode
     #   dataset_option_setter = data.get_option_setter(dataset_name)
  #      parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = opt#parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options
        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k in sorted(vars(opt)):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix
     #   self.print_options(opt)
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
        for op, value in opt.__dict__.items():
            print('{:>25}: {:<30}'.format(op,str(value)))
        backup_dir = osp.join(opt.checkpoints_dir, opt.name)
        util.mkdir(backup_dir)
        logger = Logger(osp.join(backup_dir, 'log'+ time.strftime(".%m_%d_%H:%M:%S") + '.txt'))
        opt.backup = backup_dir
        util.save_opt(opt, backup_dir)
        self.opt = opt
        return self.opt,logger
