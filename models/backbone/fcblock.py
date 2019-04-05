import torch, torch.nn
from .weight_init import weights_init_kaiming, weights_init_classifier

class FCBlock(nn.Module):
    def __init__(self, input_dim, class_num, isTrain=True, dropout=0.5, relu=True, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if dropout:
            add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
        self.isTrain = isTrain
    def forward(self, x):
        x = self.add_block(x)
        if not self.isTrain:
        	return x
        x = self.classifier(x)
        return x
