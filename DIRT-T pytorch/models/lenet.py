import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self,opt):
        super(LeNet, self).__init__()
        
        self.ndf = 64
        self.opt = opt
        if self.opt.ins_norm:
            self.ins_norm = nn.InstanceNorm2d(3)
        self.feature = nn.Sequential(
            nn.Conv2d(3, self.ndf, 5, 1, 0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(self.ndf, self.ndf, 5, 1, 0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
                    
            nn.Conv2d(self.ndf, self.ndf*2, 5, 1,0),
            nn.ReLU(inplace=True)
        )
        self.classfier = nn.Linear(128,10)

    def extract_feature(self, input):
        if self.opt.ins_norm:
            input = self.ins_norm(input)
        output = self.feature(input)
        return output.view(-1, 2*self.ndf)

    def forward(self, input):
        if self.opt.ins_norm:
            input = self.ins_norm(input)
        output = self.feature(input)
        output = output.view(-1, 2*self.ndf)
        output = self.classfier(output)
        return output

