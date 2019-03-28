from __future__ import print_function, absolute_import
import time
import torch
from .networks import AverageMeter
from .base_runner import BaseRunner
from . import networks
from .vat_loss import *

class Runner(BaseRunner):

    def _parse_data(self, inputs):
        (s_input, s_label), (t_input, t_label) = inputs
        s_input = s_input.cuda()
        s_label = s_label.cuda()
        t_input = t_input.cuda()
        t_label = t_label.cuda()
        return s_input, s_label, t_input, t_label

    def _parse_val_data(self, inputs):
        input, label = inputs
        input = input.cuda()
        label = label.cuda()
        return input, label

    def G_forward(self, inputs):
        s_input, s_label, t_input, t_label = self._parse_data(inputs)
        s_feat, s_output = self.model.forward(s_input, noise=False)
        t_feat, t_output = self.model.forward(t_input, noise=False)
        crossE_loss     = self.crossE(s_output, s_label)#cross entropy loss in source domain
          
        s_score = self.D(s_feat)
        t_score = self.D(t_feat)       
        domain_loss     = 0.5*self.disc(s_score,torch.zeros_like(s_score)) + 0.5*self.disc(t_score, torch.ones_like(t_score))

        conditionE_loss = self.conditionE(t_output) # condition entropy
        vat_src_loss    = 0#self.vat()
        vat_tgt_loss    = self.tgt_vat(t_input,t_output)

        loss = crossE_loss +0.01*domain_loss + 0.01*conditionE_loss+ 0.01*vat_tgt_loss

        prec1, prec5 = self.accuracy(t_output, t_label)
        return s_feat, t_feat, loss, prec1, prec5
    def D_forward(self, s_feat, t_feat):
        s_score = self.D(s_feat)
        t_score = self.D(t_feat)

        disc_loss       = 0.5*self.disc(s_score,torch.ones_like(s_score)) + 0.5*self.disc(t_score, torch.zeros_like(t_score))

        return disc_loss

    def _val_forward(self, inputs,targets):
        _, output = self.model.forward(inputs, noise=False,training=False)
        loss  = self.crossE(output, targets)#cross entropy loss in source domain
        __, output_noise = self.model.forward(inputs, noise=True, training=False)

        prec1, prec5 = self.accuracy(output, targets)
        noise_prec1, ___ = self.accuracy(output_noise, targets)
        return loss, prec1, prec5, noise_prec1

    def _noise_val_forward(self, inputs, targets, noise1, noise2):
        _, output_noise = self.model.noise_forward(inputs, noise1, noise2, training=False)
        loss  = self.crossE(output_noise, targets)#cross entropy loss in source domain
        __, output = self.model.forward(inputs, noise=False, training=False)

        prec1, prec5 = self.accuracy(output, targets)
        noise_prec1, ___ = self.accuracy(output_noise, targets)
        return loss, prec1, prec5, noise_prec1

    def accuracy(self, outputs, targets):
        return networks.accuracy(outputs, targets)
