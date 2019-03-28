from __future__ import print_function, absolute_import
import time
import torch
import os.path as osp

from .networks import AverageMeter
from .networks import *
from tsnecuda import TSNE
import matplotlib.pyplot as plt
import numpy as np 

class DARunner(object):
    def __init__(self, opt, model, criterion=None, writer=None):
        super(DARunner, self).__init__()
        self.model = model
        self.criterion = criterion
      
        # Setting on screen printing, logger and Backup
        self.print_freq = opt.print_freq
        self.best_prec1 = 0.0
        self.best_prec1_tgt = 0.0 
        self.backup_path = opt.backup
        self.writer = writer

        #Modification on parameters
        self.partialBN = opt.partialbn
        
        #Optimizer and Scheduler
    #    policies = self.get_optim_policies(self.model)
        self.optimizer = optimizer = optim.Adam(self.model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))#get_optimizer(opt, policies)
        self.scheduler = get_scheduler(self.optimizer, opt)

    def train(self, epoch, data_loader, logger):
        self.model.train(True)
   #     self.scheduler.step()
        print('Current learning rate:{}'.format(self.scheduler.get_lr()))
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()

        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)
            inputs, targets = self._parse_data(inputs)
            loss, prec1, prec5 = self._forward(inputs, targets)
            losses.update(loss.item(), targets.size(0))
            top1.update(prec1.item(), targets.size(0))
            top5.update(prec5.item(), targets.size(0))

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % self.print_freq == 0:
                output = ('Epoch: [{}][{}/{}]\t'
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

        # switch to evaluate mode
        self.model.eval()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            with torch.no_grad():
                inputs, targets = self._parse_data(inputs)
                loss, prec1, prec5 = self._forward(inputs, targets)


            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

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
        self.writer.add_scalar('val/source-prec1', top1.avg, epoch)
        output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
              .format(top1=top1, top5=top5, loss=losses))
        current_prec1 = top1.avg
        if current_prec1 > self.best_prec1:
            self.best_prec1 = current_prec1
            self.save_network('Best')
        self.writer.add_scalar('val/source-best-prec1', self.best_prec1, epoch)
        output_best = '\nBest Prec@1: %.3f'%(self.best_prec1)
        logger.write(output + ' ' + output_best + '\n')
        logger.flush()
        return current_prec1
    def validate2(self, epoch, data_loader, logger):

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            with torch.no_grad():
                inputs, targets = self._parse_data(inputs)
                loss, prec1, prec5 = self._forward(inputs, targets)


            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

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
        output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
              .format(top1=top1, top5=top5, loss=losses))
        current_prec1 = top1.avg
        if current_prec1 > self.best_prec1_tgt:
            self.best_prec1_tgt = current_prec1
            self.save_network('Best')
        output_best = '\nTarget domain Best Prec@1: %.3f'%(self.best_prec1_tgt)
        self.writer.add_scalar('val/target-best-prec1', self.best_prec1_tgt, epoch)
        logger.write(output + ' ' + output_best + '\n')
        logger.flush()
        return prec1

    
    def visualize(self, src_data_loader, tgt_data_loader, epoch):
        fig = plt.figure()
        features = []
        labels = []
        for data_loader in [src_data_loader, tgt_data_loader]:
            tmp_features = []
            tmp_labels = []
            for i, inputs in enumerate(data_loader):
                with torch.no_grad():
                    inputs, targets = self._parse_data(inputs)
                    feature = self.model.extract_feature(inputs)
                tmp_features.append(feature)
                tmp_labels.append(targets)
            tmp_features = torch.cat(tmp_features, dim=0)
            tmp_labels = torch.cat(tmp_labels, dim=0)
            tmp_features = tmp_features.detach().cpu().numpy()
            tmp_labels = tmp_labels.detach().cpu().numpy()
            features.append(tmp_features)
            labels.append(tmp_labels)

        mid = features[0].shape[0]
        features = np.concatenate((features[0], features[1]), axis=0)
        feat_tsne = TSNE().fit_transform(features)

        plt.scatter(feat_tsne[:mid,0], feat_tsne[:mid,1], c=labels[0], alpha=0.4)
        #plt.scatter(feat_tsne[mid:,0], feat_tsne[mid:,1], c=labels[1]/2, alpha=0.5)

        save_path = osp.join(self.backup_path, str(epoch)+'.png')
        plt.savefig(save_path)
        self.writer.add_figure('visualization', fig,global_step=epoch)

        plt.clf()
        
   
    def _parse_data(self, inputs):
        pass

    def _forward(self, inputs, targets):
        pass

    def _accuracy(self, outputs, targets):
        return networks.accuracy(outputs, targets)

    def save_network(self, name):
        torch.save(self.model.state_dict(), osp.join(self.backup_path, str(name)+'.pth'))
    def load_network(self, name):
        the_model = torch.load(osp.join(self.backup_path, +'Best.pth'))
        if isinstance(the_model, torch.nn.DataParallel):
            the_model = the_model.module
        self.model.load_state_dict(the_model)

    def get_optim_policies(self, model):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []

        conv_cnt = 0
        bn_cnt = 0
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                ps = list(m.parameters())
                conv_cnt += 1
                first_conv_weight.append(ps[0])
                if len(ps) == 2:
                    first_conv_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self.partialBN or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        policy =  [
            {'params': first_conv_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 10, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 10, 'decay_mult': 1,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},]
        for group in policy:
            print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
                group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))
        return policy


