from __future__ import print_function, absolute_import
import time
from collections import OrderedDict

import torch

from .utils.meters import AverageMeter

class Testor(BaseTestor):

   def _parse_data(self, inputs):
        inputs, targets = inputs
        inputs = inputs.cuda()
        targets = targets.cuda()
        return inputs, targets

    def _forward(self, inputs, targets):
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        prec1, prec5 = self.accuracy(outputs.data, targets.data)
        return loss, prec1, prec5
