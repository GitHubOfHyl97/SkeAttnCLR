#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
import yaml
import math
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor
from .pretrain import PT_Processor
from .knn_monitor import knn_predict
from tqdm import tqdm

SMALL_NUM = np.log(1e-45)


class SkeletonCLR_Processor(PT_Processor):
    """
        Processor for SkeletonCLR Pretraining.
    """

    @torch.no_grad()
    def calc_mask_accuracy(self, pred_mask, target_mask):
        '''
        :param pred: N, K
        :param targets: N, K
        :return: acc
        '''
        right_num = (pred_mask * target_mask).sum()
        total_num = pred_mask.sum()
        return right_num, max(total_num, 1)

    def multi_nce_loss(self, logits, mask):
        mask_sum = mask.sum(1)
        loss = - torch.log((F.softmax(logits, dim=1) * mask).sum(1) / mask_sum)
        return loss.mean()

    def dcl_loss(self, logits, mask):
        postive_loss = - (logits * mask).sum(1)
        negative_loss = torch.logsumexp(logits + (mask > 1e-5) * SMALL_NUM, dim=1, keepdim=False)
        # loss = - (logits * mask).sum(1) / mask.sum(1) + torch.log( (torch.exp(logits) * (mask == 0)).sum(1) )
        loss = (postive_loss + negative_loss).mean()
        return loss

    def train(self, epoch):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []
        # loss_ins_value = []
        # loss_reg_value = []
        # right_num = 0
        # total_num = 0
        batch_iter = len(loader)
        max_iter = float(batch_iter*self.arg.num_epoch)
        i_iter = 0
        if epoch <= self.arg.mining_epoch:
            self.io.print_log('Training Mode: Normal')
        else:
            self.io.print_log('Training Mode: Mining -- {}'.format(self.arg.topk))
        self.io.print_log('Loss: {}'.format(self.arg.loss))

        for [data1, data2, index], label in loader:
            curr_iter = float(epoch*batch_iter + i_iter)
            i_iter += 1
            self.global_step += 1
            # get data
            data1 = data1.float().to(self.dev, non_blocking=True)
            data2 = data2.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)

            # data1 = self.view_gen(data1)
            # data2 = self.view_gen(data2)

            # forward
            loss_z, loss_p, loss_c = self.model(data1, data2, self.arg.view)
            if hasattr(self.model, 'module'):
                self.model.module.update_ptr(data1.size(0))
            else:
                self.model.update_ptr(data1.size(0))
            loss = loss_z + self.arg.mu*loss_p + (1-self.arg.mu)*loss_c

            # acc = self.calc_mask_accuracy(mask, target_mask)
            # right_num += acc[0]
            # total_num += acc[1]

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if hasattr(self.model, 'module'):
                self.model.module.momentum_update(curr_iter, max_iter)
            else:
                self.model.momentum_update(curr_iter, max_iter)
                
            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['loss_z'] = loss_z.data.item()
            self.iter_info['loss_p'] = loss_p.data.item()
            self.iter_info['loss_c'] = loss_c.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1
            self.train_log_writer(epoch)

        self.epoch_info['train_mean_loss']= np.mean(loss_value)
        self.train_writer.add_scalar('loss', self.epoch_info['train_mean_loss'], epoch)
        self.show_epoch_info()

    def view_gen(self, data):
        if self.arg.view == 'joint':
            pass
        elif self.arg.view == 'motion':
            motion = torch.zeros_like(data)

            motion[:, :, :-1, :, :] = data[:, :, 1:, :, :] - data[:, :, :-1, :, :]

            data = motion
        elif self.arg.view == 'bone':
            Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                    (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                    (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]

            bone = torch.zeros_like(data)

            for v1, v2 in Bone:
                bone[:, :, :, v1 - 1, :] = data[:, :, :, v1 - 1, :] - data[:, :, :, v2 - 1, :]
            data = bone
        else:
            raise ValueError

        return data

    @torch.no_grad()
    def knn_monitor(self, epoch):
        if len(self.gpus) > 1:
            self.model.module.encoder_q.eval()
        else:
            self.model.encoder_q.eval()
        feature_bank, label_bank = [], []
        with torch.no_grad():
            # generate feature bank
            for data, label in tqdm(self.data_loader['mem_train'], desc='Feature extracting'):
                data = data.float().to(self.dev, non_blocking=True)
                label = label.long().to(self.dev, non_blocking=True)

                data = self.view_gen(data)
                if len(self.gpus) > 1:
                    out = self.model.module.encoder_q(data)
                    q = out
                    q = self.model.module.predictor(q)
                else:
                    out = self.model.encoder_q(data)
                    q = out
                    q = self.model.predictor(q)
                feature = F.normalize(q.squeeze(-1), dim=1)
                feature_bank.append(feature)
                label_bank.append(label)
            # [D, N]
            feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
            # [N]
            feature_labels = torch.cat(label_bank).to(feature_bank.device)
            # loop test data to predict the label by weighted knn search
            for i in self.arg.knn_k:
                total_top1, total_top5, total_num = 0, 0, 0
                test_bar = tqdm(self.data_loader['mem_test'], desc='kNN-{}'.format(i))
                for data, label in test_bar:
                    data = data.float().to(self.dev, non_blocking=True)
                    label = label.float().to(self.dev, non_blocking=True)

                    data = self.view_gen(data)

                    if len(self.gpus) > 1:
                        out = self.model.module.encoder_q(data)
                        q = out
                        q = self.model.module.predictor(q)
                    else:
                        out = self.model.encoder_q(data)
                        q = out
                        q = self.model.predictor(q)
                    feature = F.normalize(q.squeeze(-1), dim=1)
                    pred_labels = knn_predict(feature, feature_bank, feature_labels, self.arg.knn_classes, i,
                                              self.arg.knn_t)

                    total_num += data.size(0)
                    total_top1 += (pred_labels[:, 0] == label).float().sum().item()
                    test_bar.set_postfix({'k': i, 'Accuracy': total_top1 / total_num * 100})
                acc = total_top1 / total_num * 100

                self.knn_results[i][epoch] = acc

                self.train_writer.add_scalar('KNN-{}'.format(i), acc, epoch)
            

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+',
                            help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        parser.add_argument('--view', type=str, default='joint', help='the view of input')
        parser.add_argument('--topk', type=int, default=1, help='topk to mine')
        parser.add_argument('--mining_epoch', type=int, default=1e6, help='the starting epoch of mining training')
        parser.add_argument('--loss', type=str, default='ce', help='loss')
        parser.add_argument('--weight', type=str2bool, default=False, help='use dcw or not')
        parser.add_argument('--cos', type=int, default=0, help='use cosine lr schedule')
        parser.add_argument('--alpha', type=float, default=1.)
        parser.add_argument('--mu', type=float, default=0.5)
#        parser.add_argument('--knn_k', type=int, default=[], nargs='+', help='KNN-K')
#        parser.add_argument('--knn_classes', type=int, default=60, help='use cosine lr schedule')
#        parser.add_argument('--knn_t', type=int, default=0.1, help='use cosine lr schedule')
#        parser.add_argument('--KNN_show', type=int, default=[], nargs='+',
#                            help='the epoch to show the best KNN result')
        # endregion yapf: enable

        return parser
