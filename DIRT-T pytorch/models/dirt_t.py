import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F

class Noise_layer(nn.Module):
    def __init__(self,mid,w):
        super(Noise_layer, self).__init__()
        self.mid = mid
        self.w = w
        self.noise = Parameter(torch.zeros(mid, w, w).normal_(0, 1))

    def forward(self, input):
        return input + self.noise

class Discriminator(nn.Module):
    def __init__(self,opt):
        super(Discriminator, self).__init__()
        if opt.small:
            self.hidden = 64
        else:
            self.hidden = 192
        self.model = nn.Sequential(
            nn.Linear(81*self.hidden, 100),
            nn.ReLU(),
            nn.Linear(100,1)
            )
    def forward(self, input):
        
        return self.model(input)

class DirtT(nn.Module):
    def __init__(self,opt, n_domains=2):
        super(DirtT, self).__init__()
        
        self.ndf = 64
        self.opt = opt
        self.n_domains = n_domains
        self.conditional_layers = []
        if self.opt.ins_norm:
            self.ins_norm = nn.InstanceNorm2d(3, affine=False, momentum=0,track_running_stats=False)
        if self.opt.small:
            self.mid1 = 64
            self.mid2 = 64
        else:
            self.mid1 = 96
            self.mid2=192
        if self.opt.noise:
            print('Noise Enabled and Noise tensor initialized')
            self.gau1 = Noise_layer(self.mid1 ,16)#torch.zeros(self.mid1, 16, 16).normal_(0, 1).cuda()
            self.gau2 = Noise_layer(self.mid2 ,8)#torch.zeros(self.mid2, 8, 8).normal_(0, 1).cuda()
        
        self.conv1_1    = nn.Conv2d(3, self.mid1, (3, 3))
        self.conv1_1_bn = nn.BatchNorm2d(self.mid1, momentum = opt.momentum)#self._batch_norm(self.mid1)
        self.conv1_2    = nn.Conv2d(self.mid1, self.mid1, (3, 3))
        self.conv1_2_bn = nn.BatchNorm2d(self.mid1, momentum = opt.momentum)#self._batch_norm(self.mid1)
        self.conv1_3    = nn.Conv2d(self.mid1, self.mid1, (3, 3))
        self.conv1_3_bn = nn.BatchNorm2d(self.mid1, momentum = opt.momentum)#self._batch_norm(self.mid1)
        self.pool1      = nn.MaxPool2d(2, stride=2)

        self.conv2_1    = nn.Conv2d(self.mid2, self.mid2, (3, 3))
        self.conv2_1_bn = nn.BatchNorm2d(self.mid2, momentum = opt.momentum)#self._batch_norm(self.mid2)
        self.conv2_2    = nn.Conv2d(self.mid2, self.mid2, (3, 3))
        self.conv2_2_bn = nn.BatchNorm2d(self.mid2, momentum = opt.momentum)#self._batch_norm(self.mid2)
        self.conv2_3    = nn.Conv2d(self.mid2, self.mid2, (3, 3))
        self.conv2_3_bn = nn.BatchNorm2d(self.mid2, momentum = opt.momentum)#self._batch_norm(self.mid2)
        self.pool2      = nn.MaxPool2d(2, stride=2)

        self.conv3_1    = nn.Conv2d(self.mid2, self.mid2, (3, 3))
        self.conv3_1_bn = nn.BatchNorm2d(self.mid2, momentum = opt.momentum)#self._batch_norm(self.mid2)
        self.nin3_2     = nn.Conv2d(self.mid2, self.mid2, (3, 3))
        self.nin3_2_bn  = nn.BatchNorm2d(self.mid2, momentum = opt.momentum)#self._batch_norm(self.mid2)
        self.nin3_3     = nn.Conv2d(self.mid2, self.mid2, (3, 3))
        self.nin3_3_bn  = nn.BatchNorm2d(self.mid2, momentum = opt.momentum)#self._batch_norm(self.mid2)

        self.pool       = nn.AdaptiveMaxPool2d((1,1))
        self.classfier  = nn.Linear(self.mid2,10)
        self.cls_bn     = nn.BatchNorm1d(10, momentum = opt.momentum)

    def forward(self, x, noise=True, training=True):
        if self.opt.ins_norm:
            x = self.ins_norm(x)
        x = F.leaky_relu(self.conv1_1_bn(self.conv1_1(F.pad(x,(1,1,1,1), "replicate" ))), negative_slope=0.1)#, d))
        x = F.leaky_relu(self.conv1_2_bn(self.conv1_2(F.pad(x,(1,1,1,1), "replicate" ))), negative_slope=0.1)#, d))
        x = F.leaky_relu(self.conv1_3_bn(self.conv1_3(F.pad(x,(1,1,1,1), "replicate" ))), negative_slope=0.1)#, d)))
        x = self.pool1(F.pad(x,(1,1,1,1), "replicate" ))
        x = F.dropout(x, p=0.5,training=training)

        if noise:
            x += torch.zeros(self.mid1, 17, 17).normal_(0, 1).cuda()  
        x = F.leaky_relu(self.conv2_1_bn(self.conv2_1(F.pad(x,(1,1,1,1), "replicate" ))), negative_slope=0.1)#, d))
        x = F.leaky_relu(self.conv2_2_bn(self.conv2_2(F.pad(x,(1,1,1,1), "replicate" ))), negative_slope=0.1)#, d))
        x = F.leaky_relu(self.conv2_3_bn(self.conv2_3(F.pad(x,(1,1,1,1), "replicate" ))), negative_slope=0.1)#, d)))
        x = self.pool2(F.pad(x,(1,1,1,1), "replicate" ))
        x = F.dropout(x, p=0.5,training=training)
        if noise :
            x += torch.zeros(self.mid1, 9, 9).normal_(0, 1).cuda()
        feat = x 
        x = F.leaky_relu(self.conv3_1_bn(self.conv3_1(F.pad(x,(1,1,1,1), "replicate" ))), negative_slope=0.1)#, d))
        x = F.leaky_relu(self.nin3_2_bn(self.nin3_2(F.pad(x,(1,1,1,1), "replicate" ))), negative_slope=0.1)#, d))
        x = F.leaky_relu(self.nin3_3_bn(self.nin3_3(F.pad(x,(1,1,1,1), "replicate" ))), negative_slope=0.1)#, d))

        x = self.pool(x)
        x = x.squeeze()
#        x = self.cls_bn(self.classfier(x))
        x = self.classfier(x)
        feat = feat.view(x.shape[0],-1)
        return feat, x 


