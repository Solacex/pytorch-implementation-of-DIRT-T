from torch.utils.data import DataLoader 
from . import opencv_transforms as transforms
from torchvision.datasets import MNIST

from torchvision.transforms import Compose, ToTensor, Normalize
from  torchvision import  transforms,datasets
from .LMDB_dataset import LMDBDataset
from . import opencv_transforms as o_transforms
import numpy as np
import os.path as osp

def create_dataset(opt):
    
    image_datasets = {}

    mean = np.array([0.44, 0.44, 0.44])
    std = np.array([0.19, 0.19, 0.19])
    src_train_db = osp.join(opt.dataroot,'_'.join([opt.dataset, opt.source, 'train'])+'.lmdb')
    src_val_db =  osp.join(opt.dataroot,'_'.join([opt.dataset, opt.source, 'val'])+'.lmdb')
    tgt_train_db =  osp.join(opt.dataroot,'_'.join([opt.dataset, opt.target, 'train'])+'.lmdb')
    tgt_val_db = osp.join(opt.dataroot, '_'.join([opt.dataset, opt.target, 'val'])+'.lmdb')

    if opt.isTrain:
        train_trans = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])

        src_train_loader = DataLoader(LMDBDataset(src_train_db,train_trans),
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=int(opt.num_threads),
            pin_memory=True,
            drop_last = opt.isTrain)

        tgt_train_loader = DataLoader(LMDBDataset(tgt_train_db,train_trans),
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=int(opt.num_threads),
            pin_memory=True,
            drop_last = opt.isTrain)

        if opt.val:
            val_trans = transforms.Compose([
                transforms.Resize((32,32)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
                ])

            src_val_loader = DataLoader(LMDBDataset(src_val_db,val_trans),
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=int(opt.num_threads),
                pin_memory=True,
                drop_last = opt.isTrain)
            tgt_val_loader = DataLoader(LMDBDataset(tgt_val_db,val_trans),
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=int(opt.num_threads),
                pin_memory=True,
                drop_last = opt.isTrain)

            return src_train_loader, src_val_loader, tgt_train_loader, tgt_val_loader
        return train_loader, None

    else:
        val_trans = transforms.Compose([
            transforms.Resize(size=(256,128),interpolation=3), #Image.BICUBIC
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        test_loader = DataLoader(LMDBDataset(opt.val_db, val_trans),
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=int(opt.num_threads),
            pin_memory=True,
            drop_last = opt.isTrain)
        return test_loader
