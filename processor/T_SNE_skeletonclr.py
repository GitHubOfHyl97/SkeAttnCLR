import sys
import argparse
import yaml
import math
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class
from .knn_monitor import knn_predict
from tqdm import tqdm
from .processor import Processor
from .tsne_torch import TSNE

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1 or classname.find('Conv2d') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class FT_Processor(Processor):
    """
        Processor for Finetune Evaluation.
    """

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        self.model.apply(weights_init)
        # for name, param in self.model.encoder_q.named_parameters():
        #     if name.split('.')[0] != 'backbone':
        #         param.requires_grad = False
        self.loss = nn.CrossEntropyLoss()

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def adjust_lr(self):
        lr = self.arg.base_lr
        if self.arg.optimizer == 'SGD' and self.arg.step:
            if self.meta_info['epoch'] < self.arg.warm_up_epoch:
                lr *= self.meta_info['epoch'] / self.arg.warm_up_epoch
            else:
                lr *= (0.1**np.sum(self.meta_info['epoch'] > np.array(self.arg.step)))
        else:
            lr = self.arg.base_lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.lr = lr

    def show_topk(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))

    def show_best(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = 100 * sum(hit_top_k) * 1.0 / len(hit_top_k)
        accuracy = round(accuracy, 5)
        self.current_result = accuracy
        if self.best_result <= accuracy:
            self.best_result = accuracy
        self.io.print_log('\tBest Top{}: {:.2f}%'.format(k, self.best_result))

    def train(self, epoch):
        for _ in range(100):
            self.iter_info['loss'] = 0
            self.show_iter_info()
            self.meta_info['iter'] += 1
        self.epoch_info['train_mean_loss'] = 0
        self.show_epoch_info()

    def test(self, epoch):
        for _ in range(100):
            self.iter_info['loss'] = 1
            self.show_iter_info()
        self.eval_info['test_mean_loss'] = 1
        self.show_eval_info()

    def view_gen(self, data):
        if self.arg.view == 'joint' or self.arg.view == 'all':
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
                    q,_  = self.model.module.encoder_q(data)
                else:
                    q,_ = self.model.encoder_q(data)
                feature = F.normalize(q, dim=1)
                feature_bank.append(feature)
                label_bank.append(label)
            # [D, N]
            feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
            # [N]
            feature_labels = torch.cat(label_bank).to(feature_bank.device)
            features = feature_bank.permute(1, 0).contiguous()
            labels = feature_labels.clone().detach().cpu().numpy()
            print(features.shape, labels.shape)
            TSNE(features, labels, self.arg.work_dir)
            # loop test data to predict the label by weighted knn search
            for i in self.arg.knn_k:
                total_top1, total_top5, total_num = 0, 0, 0
                test_bar = tqdm(self.data_loader['mem_test'], desc='kNN-{}'.format(i))
                for data, label in test_bar:
                    data = data.float().to(self.dev, non_blocking=True)
                    label = label.float().to(self.dev, non_blocking=True)

                    data = self.view_gen(data)

                    if len(self.gpus) > 1:
                        q,_ = self.model.module.encoder_q(data)
                    else:
                        q,_ = self.model.encoder_q(data)
                    feature = F.normalize(q, dim=1)
                    pred_labels = knn_predict(feature, feature_bank, feature_labels, self.arg.knn_classes, i,
                                              self.arg.knn_t)

                    total_num += data.size(0)
                    total_top1 += (pred_labels[:, 0] == label).float().sum().item()
                    test_bar.set_postfix({'k': i, 'Accuracy': total_top1 / total_num * 100})
                acc = total_top1 / total_num * 100

                self.knn_results[i][epoch] = acc

                self.train_writer.add_scalar('KNN-{}'.format(i), acc, epoch)
            # print(self.knn_results)
    @staticmethod
    def get_parser(add_help=False):
        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # evaluation
        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+',
                            help='which Top K accuracy will be shown')
        # optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+',
                            help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        parser.add_argument('--view', type=str, default='joint', help='the view of input')
        parser.add_argument('--warm_up_epoch', type=int, default=0, help='warm up epochs')

        parser.add_argument('--clip_norm', type=str2bool, default=True, help='use clip_norm or not')
        parser.add_argument('--max_grad', type=float, default=35., help='max number of grad')

        return parser
