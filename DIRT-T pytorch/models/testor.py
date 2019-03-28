from __future__ import print_function, absolute_import
import time
import torch
from .networks import AverageMeter
from .base_testor import BaseTestor
from . import networks
class Testor(BaseTestor):

    def _parse_data(self, inputs):
        input, targets = inputs
        input = input.cuda()
        targets = targets.cuda()
        return input, targets

    def _forward(self, inputs, targets):
        outputs = self.model.forward(inputs)
        loss = self.criterion(outputs, targets)
        prec1, prec5 = self.accuracy(outputs, targets)
        return loss, prec1, prec5

    def accuracy(self, outputs, targets):
        return networks.accuracy(outputs, targets)
