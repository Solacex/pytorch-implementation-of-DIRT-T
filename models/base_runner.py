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
from .optimizer import EMA
class BaseRunner(object):
    def __init__(self, opt, model, D, criterion=None, writer=None):
        super(BaseRunner, self).__init__()
        self.model = model
        self.D = D
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
        self.optimizer = optim.Adam(self.model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer2 = optim.Adam(self.D.parameters(), lr = opt.lr, betas=(opt.beta1, 0.999))   

        self.crossE      = nn.CrossEntropyLoss().cuda() 
        self.conditionE  = ConditionalEntropy().cuda()
        self.src_vat     = VATLoss(self.model, radius=self.opt.radius).cuda()
        self.tgt_vat     = VATLoss(self.model, radius=self.opt.radius).cuda()
        self.disc        = nn.BCEWithLogitsLoss().cuda()
           
        self.ema = EMA(0.998)

    def train(self, epoch, data_loader, logger):
        self.model.train(True)
        self.D.train(True)
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()

    #    self.ema.register_model(self.model)         

        for i, inputs in enumerate(data_loader):#zip(src_loader, tgt_loader)):
            data_time.update(time.time() - end)
            s_feat, t_feat, loss, prec1, prec5 = self.G_forward(inputs)
            losses.update(loss.item(), self.opt.batch_size)#targets.size(0))
            top1.update(prec1.item(), self.opt.batch_size)#targets.size(0))
            top5.update(prec5.item(), self.opt.batch_size)#targets.size(0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            disc_loss = self.D_forward(s_feat.detach(), t_feat.detach())
            self.optimizer2.zero_grad()
            disc_loss.backward()
            self.optimizer2.step()
            
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

    def validate2(self, epoch, data_loader, logger):

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
                loss, prec1, prec5, prec1_noise = self._val_forward(inputs, targets)


            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            noise_top1.update(prec1_noise.item(),inputs.size(0))

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

        output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
              .format(top1=top1, top5=top5, loss=losses))
        current_prec1 = top1.avg
        current_prec1_noise = noise_top1.avg

        if current_prec1 > self.best_prec1_tgt:
            self.best_prec1_tgt = current_prec1
            self.save_network('Best')
        if current_prec1_noise > self.best_prec1_noise:
            self.best_prec1_noise = current_prec1_noise

        output_best = '\nTarget domain Best Prec@1: %.3f'%(self.best_prec1_tgt)
        self.writer.add_scalar('val/target-best-prec1', self.best_prec1_tgt, epoch)
        self.writer.add_scalar('val/Noise-best-prec1', self.best_prec1_noise, epoch)
        logger.write(output + ' ' + output_best + '\n')
        logger.flush()
        return prec1

    def noise_validate(self, epoch, data_loader, logger):

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        noise_top1 = AverageMeter()
        # switch to evaluate mode
        self.model.eval()
        noise1 = torch.zeros(64, 16, 16).normal_(0, 1).cuda()
        noise2 = torch.zeros(64, 8,  8 ).normal_(0, 1).cuda()
        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            with torch.no_grad():
                inputs, targets = self._parse_val_data(inputs)
                loss, prec1, prec5, prec1_noise = self._noise_val_forward(inputs, targets, noise1, noise2)


            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            noise_top1.update(prec1_noise.item(),inputs.size(0))

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

        output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@1(noise) {noise_top1.avg:.3f} Loss {loss.avg:.5f}'
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
        self.writer.add_scalar('val/Noise-best-prec1', self.best_prec1_noise, epoch)
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



    def cuda(self, obj):
        """ Move nested iterables between CUDA or CPU
        """
        if isinstance(obj, tuple) or isinstance(obj, list):
            obj = [self.cuda(el) for el in obj]
        else:
            obj = self.to_cuda(obj)
        return obj

    def register_loss(self, func, weight = 1.,
                      name=None, display = True, 
                      override = False, **kwargs):
        """ Register a new loss function

        Parameters
        ----------

        func : ZZ
            pass
        weight : float

        """

        assert name is None or isinstance(name, str)

        # TODO refactor, add checks

        if isinstance(weight, int):
            weight = float(weight)

        assert weight is None or isinstance(weight, float)
        
        # TODO assert for torch Function
        # assert isinstance(func, nn.Module)
        if name is None:
            name = 'unnamed_{}'.format(len(self.loss_funcs)+1)

        if isinstance(func, nn.Module):
            self.cuda(func)

        if name in self.loss_funcs:
            if override:
                if name in self.display_loss:
                    self.display_loss.remove(name)

            else:
                raise ValueError('Name {} for loss func {} already taken.'.format(
                    name, self.loss_funcs[name].__class__.__name__
                )
                    + ' Call register_loss with the override=True option if this was intended.'
                )

        self.loss_funcs[name]   = func
        if weight is not None:
            self.loss_weights[name] = weight
        self.loss_kwargs[name]  = kwargs

        if display:
            self.display_loss.append(name)

        print('Registered {} as "{}" with weight {}'.format(
            func.__class__.__name__, name, weight))
