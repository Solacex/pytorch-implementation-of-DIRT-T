import torch
from .backbone import *
import torch.nn
from .FCBlcok import *

class reidBaseline(nn.Module):
    def __init__(self, opt):
    	super(reidBaseline， self).__init__()
        self.isTrain = opt.isTrain
    	self.backbone = backbone.get_pretrained_model(opt.arch)
    	self.fcblcok = fcblcok(opt.feat_dim, opt.bottle_dim, cls_num=opt.cls，isTrain=self.isTrain)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fcblcok(x)
        return x