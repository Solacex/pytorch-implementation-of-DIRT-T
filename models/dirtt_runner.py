from __future__ import print_function, absolute_import
import time
import torch
import os.path as osp

from .networks import AverageMeter
from .networks import *
from tsnecuda import TSNE
import matplotlib.pyplot as plt
import numpy as np 
import torch.nn as nn
from .vat_loss import *
from .optimizer import *
from .optimizer import EMA

class DirtTRunner(object):
    def __init__(self, opt, model, teacher, criterion=None, writer=None):
        super(DirtTRunner, self).__init__()
        self.model = model
        self.teacher = teacher
        self.criterion = criterion
        self.opt = opt
        # Setting on screen printing, logger and Backup
        self.print_freq = opt.print_freq
        self.best_prec1 = 0.0
        self.best_prec1_noise = 0.0
        self.best_prec1_tgt = 0.0 
        self.backup_path = opt.backup
        self.writer = writer
        
        #Modification on parameters
        self.partialBN = opt.partialbn
        
        #Optimizer and Scheduler
        student_params   = list(self.model.parameters())
        teacher_params   = list(self.teacher.parameters())

        for param in teacher_params:
            param.requires_grad = False

        self.optimizer   = optim.Adam(self.model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer2  = WeightEMA(teacher_params, student_params)#DelayedWeight(teacher_params, student_params)
   #     self.optimizer2  = WeightEMA(student_params, teacher_params)
    #    self.optimizer2  = DelayedWeight(teacher_params, student_params)


        self.crossE      = nn.CrossEntropyLoss().cuda()
        self.conditionE  = ConditionalEntropy().cuda()
        self.tgt_vat     = VATLoss(self.model, radius=self.opt.radius).cuda()
        self.dirt        = KLDivWithLogits()           

    def train(self, epoch, data_loader, logger):
        self.model.train(True)
  #      self.teacher.eval()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        dirt_losses       = AverageMeter()
        conditionE_losses = AverageMeter()
        vat_tgt_losses    = AverageMeter()

        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()

        for i, inputs in enumerate(data_loader):#zip(src_loader, tgt_loader)):
            data_time.update(time.time() - end)

            
            dirt_loss, conditionE_loss, vat_tgt_loss, prec1, prec5 = self.G_forward(inputs)
              
            dirt_losses.update(dirt_loss.item(), self.opt.batch_size)
            conditionE_losses.update(conditionE_loss.item(), self.opt.batch_size)
            vat_tgt_losses.update(vat_tgt_loss.item(), self.opt.batch_size)

            loss = (conditionE_loss+ vat_tgt_loss + dirt_loss)

            losses.update(loss.item(), self.opt.batch_size)#targets.size(0))
            top1.update(prec1.item(), self.opt.batch_size)#targets.size(0))
            top5.update(prec5.item(), self.opt.batch_size)#targets.size(0))
            
            #self.optimizer2.step()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.optimizer2.step()
               
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % self.print_freq == 0:
                output = ('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'DIRT Loss {:.3f} ({:.3f})\t'
                      'conE Loss {:.3f} ({:.3f})\t'
                      'VAT Loss {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec1 {:.2f} ({:.2f})\t'
                      'Prec5 {:.2f} ({:.2f})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              dirt_losses.val, dirt_losses.avg, 
                              conditionE_losses.val, conditionE_losses.avg, 
                              vat_tgt_losses.val,vat_tgt_losses.avg, 
                              losses.val, losses.avg,
                              top1.val, top1.avg,
                              top5.val, top5.avg))
               # print(output)
                logger.write(output+'\n')
                logger.flush()
                self.writer.add_scalar('train/loss', losses.avg, (epoch-1)*(len(data_loader))+i)

    def validate(self, epoch, data_loader, logger):

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        noise_top1 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            with torch.no_grad():
                inputs, targets = self._parse_val_data(inputs)
                loss, prec1, prec5, noise_prec1  = self._val_forward(inputs, targets)


            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            noise_top1.update(noise_prec1.item(), inputs.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % self.print_freq == 0:
                output = ('Val Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec1 {:.2f} ({:.2f})\t'
                      'Prec5 {:.2f} ({:.2f})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              top1.val, top1.avg,
                              top5.val, top5.avg))
                logger.write(output+'\n')
                logger.flush()
        self.writer.add_scalar('val/target-prec1', top1.avg, epoch)
        self.writer.add_scalar('val/noise-prec1', noise_top1.avg, epoch)

        output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec1@(Noise) {noise_top1.avg:.3f} Loss {loss.avg:.5f}'
              .format(top1=top1, noise_top1=noise_top1, loss=losses))
        current_prec1 = top1.avg
        current_prec1_noise = noise_top1.avg

        if current_prec1 > self.best_prec1_tgt:
            self.best_prec1_tgt = current_prec1
            self.save_network('Best')

        if current_prec1_noise > self.best_prec1_noise:
            self.best_prec1_noise = current_prec1_noise

        output_best = '\nTarget domain Best Prec@1: %.3f'%(self.best_prec1_tgt)
        self.writer.add_scalar('val/target-best-prec1', self.best_prec1_tgt, epoch)
        print('Target domain Best Prec@1(Noise): %.3f'%(self.best_prec1_noise))
        self.writer.add_scalar('val/Noise-best-prec1', self.best_prec1_noise, epoch)

        logger.write(output + ' ' + output_best + '\n')
        logger.flush()
        return prec1

    def _parse_data(self, inputs):
        (s_input, s_label), (t_input, _) = inputs
        s_input = s_input.cuda()
        s_label = s_label.cuda()
        t_input = t_input.cuda()
        return s_input, s_label, t_input

    def _parse_val_data(self, inputs):
        input, label = inputs
        input = input.cuda()
        label = label.cuda()
        return input, label

    def G_forward(self, inputs):
#        s_input, s_label, t_input = self._parse_data(inputs)
        t_input, t_label = self._parse_val_data(inputs)

        t_feat, t_output = self.model.forward(t_input,noise=False)#, training=False)  
    #    t_feat_noise, t_output_noise = self.model.forward(t_input,noise=True, training=True)
        tea_feat, tea_output = self.teacher.forward(t_input,noise=False)#, training=False)


        conditionE_loss = self.conditionE(t_output) # condition entropy
        dirt_loss       = self.dirt(t_output, tea_output)
        vat_tgt_loss    = self.tgt_vat(t_input,t_output)

        prec1, prec5 = self.accuracy(t_output, t_label)
        return dirt_loss,conditionE_loss,vat_tgt_loss, prec1, prec5

    def _val_forward(self, inputs,targets):
        _, output = self.model.forward(inputs, noise=False, training=False)
        loss  = self.crossE(output, targets)#cross entropy loss in source domain
        __, output_noise = self.model.forward(inputs,noise=True,training=False)

        prec1, prec5 = self.accuracy(output, targets)
        noise_prec1, ___ = self.accuracy(output_noise, targets)
        return loss, prec1, prec5, noise_prec1

    def accuracy(self, outputs, targets):
        return accuracy(outputs, targets)

    def save_network(self, name):
        torch.save(self.model.state_dict(), osp.join(self.backup_path, str(name)+'.pth'))

    def load_network(self, name):
        the_model = torch.load(osp.join(self.backup_path, +'Best.pth'))
        if isinstance(the_model, torch.nn.DataParallel):
            the_model = the_model.module
        self.model.load_state_dict(the_model)


