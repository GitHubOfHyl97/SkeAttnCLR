import torch
from math import cos, pi, ceil
import torch.nn as nn
import torch.nn.functional as F
from torchlight import import_class
from net.utils.LEWLE import ObjectNeck_V3, ObjectNeck_V4, MLP1D
from net.st_gcn import Model
import random
trunk_ori_index = [4, 3, 21, 2, 1]
left_hand_ori_index = [9, 10, 11, 12, 24, 25]
right_hand_ori_index = [5, 6, 7, 8, 22, 23]
left_leg_ori_index = [17, 18, 19, 20]
right_leg_ori_index = [13, 14, 15, 16]

trunk = [i - 1 for i in trunk_ori_index]
left_hand = [i - 1 for i in left_hand_ori_index]
right_hand = [i - 1 for i in right_hand_ori_index]
left_leg = [i - 1 for i in left_leg_ori_index]
right_leg = [i - 1 for i in right_leg_ori_index]

body_parts = [trunk, left_hand, right_hand, left_leg, right_leg]

def multi_nce_loss(logits, mask):
    mask_sum = mask.sum(1)
    loss = - torch.log((F.softmax(logits, dim=1) * mask).sum(1) / mask_sum)
    return loss.mean()

class KLDiv(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(KLDiv, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T**2)
        return loss

class EncoderObj(nn.Module):
    def __init__(self, base_encoder, hid_dim, out_dim, norm_layer=None, num_mlp=2,
                 scale=1., l2_norm=True, num_heads=8, in_channels=3, hidden_channels=64,
                 hidden_dim=256, num_class=60, dropout=0.5, attn_drop=0., proj_drop=0.,
                 drop_path=0.1,
                 graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                 edge_importance_weighting=True, **kwargs):
        super(EncoderObj, self).__init__()
        self.backbone = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=num_class,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
        in_dim = hidden_dim
        self.neck = ObjectNeck_V4(in_channels=in_dim, hid_channels=hid_dim, out_channels=out_dim,
                               norm_layer=norm_layer, num_layers=num_mlp, attn_drop=attn_drop, proj_drop=proj_drop,
                               drop_path=drop_path, scale=scale, l2_norm=l2_norm, num_heads=num_heads)
        self.neck.init_weights(init_linear='kaiming')

    def forward(self, im, drop=False, mask=None):
        _, f = self.backbone(im)
        out = self.neck(f, drop, mask)
        return out

class Ske_LWLCLR_M(nn.Module):
    """ Referring to the code of MOCO, https://arxiv.org/abs/1911.05722 """

    def __init__(self, base_encoder=None, pretrain=True, feature_dim=128, queue_size=32768,
                 momentum=0.999, Temperature=0.07, mlp=True, in_channels=3, hidden_channels=64,
                 hidden_dim=256, num_class=60, dropout=0.5, attn_drop=0., proj_drop=0.,
                 drop_path=0.1, loss_weight=0.5, heads=8, norm_layer=None, num_mlp=2, scale=1.,
                 graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                 edge_importance_weighting=True, **kwargs):
        """
        K: queue size; number of negative keys (default: 32768)
        m: momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """

        super().__init__()
        base_encoder = import_class(base_encoder)
        self.pretrain = pretrain

        if not self.pretrain:
            self.encoder_q = EncoderObj(base_encoder=base_encoder,hid_dim=hidden_dim, out_dim=feature_dim, norm_layer=norm_layer, num_mlp=num_mlp,
                                          scale=scale, l2_norm=True, num_heads=heads,
                                          in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=num_class,
                                          dropout=dropout, attn_drop=attn_drop, proj_drop=proj_drop,
                                          drop_path=drop_path, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
        else:
            self.K = queue_size
            self.m = momentum
            self.curr_m = momentum
            self.T = Temperature

            self.loss_weight = loss_weight

            self.encoder_q = EncoderObj(base_encoder=base_encoder,hid_dim=hidden_dim, out_dim=feature_dim, norm_layer=norm_layer, num_mlp=num_mlp,
                                          scale=scale, l2_norm=True, num_heads=heads,
                                          in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=num_class,
                                          dropout=dropout, attn_drop=attn_drop, proj_drop=proj_drop,
                                          drop_path=drop_path, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
            self.encoder_k = EncoderObj(base_encoder=base_encoder,hid_dim=hidden_dim, out_dim=feature_dim, norm_layer=norm_layer, num_mlp=num_mlp,
                                          scale=scale, l2_norm=True, num_heads=heads,
                                          in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=num_class,
                                          dropout=dropout, attn_drop=attn_drop, proj_drop=proj_drop,
                                          drop_path=drop_path, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)

            self.predictor = MLP1D(feature_dim, hidden_dim, feature_dim)
            self.predictor.init_weights()
            self.predictor_obj = MLP1D(feature_dim, hidden_dim, feature_dim)
            self.predictor_obj.init_weights()
            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)    # initialize
                param_k.requires_grad = False       # not update by gradient

            # create the queue
            self.register_buffer("queue_z", torch.randn(feature_dim, queue_size))
            self.queue_z = F.normalize(self.queue_z, dim=0)
            self.register_buffer("queue_ptr_z", torch.zeros(1, dtype=torch.long))

            # create the queue
            self.register_buffer("queue_obj", torch.randn(feature_dim, queue_size))
            self.queue_obj = F.normalize(self.queue_obj, dim=0)
            self.register_buffer("queue_ptr_obj", torch.zeros(1, dtype=torch.long))
    @torch.no_grad()
    def momentum_update(self, cur_iter, max_iter):
        """
        Momentum update of the target network.
        """
        # momentum anneling
        momentum = 1. - (1. - self.m) * (cos(pi * cur_iter / float(max_iter)) + 1) / 2.0
        self.curr_m = momentum
        # parameter update for target network
        state_dict_ol = self.encoder_q.state_dict()
        state_dict_tgt = self.encoder_k.state_dict()
        for (k_ol, v_ol), (k_tgt, v_tgt) in zip(state_dict_ol.items(), state_dict_tgt.items()):
            assert k_tgt == k_ol, "state_dict names are different!"
            assert v_ol.shape == v_tgt.shape, "state_dict shapes are different!"
            if 'num_batches_tracked' in k_tgt:
                v_tgt.copy_(v_ol)
            else:
                v_tgt.copy_(v_tgt * momentum + (1. - momentum) * v_ol)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, key0, key1):
        batch_size0 = key0.shape[0]
        batch_size1 = key1.shape[0]
        ptr0 = int(self.queue_ptr_z)
        ptr1 = int(self.queue_ptr_obj)
        gpu_index0 = key0.device.index
        gpu_index1 = key1.device.index
        self.queue_z[:, (ptr0 + batch_size0 * gpu_index0):(ptr0 + batch_size0 * (gpu_index0 + 1))] = key0.T
        self.queue_obj[:, (ptr1 + batch_size1 * gpu_index1):(ptr1 + batch_size1 * (gpu_index1 + 1))] = key1.T
        
    @torch.no_grad()
    def update_ptr(self, batch_size):
        assert self.K % batch_size == 0 #  for simplicity
        self.queue_ptr_z[0] = (self.queue_ptr_z[0] + batch_size) % self.K
        self.queue_ptr_obj[0] = (self.queue_ptr_obj[0] + batch_size) % self.K

    def forward(self, xq, xk=None):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        """

        if not self.pretrain:
            q, _ = self.encoder_q.backbone(xq)
            return q

        # compute query features
        q = self.encoder_q(xq)  # queries: NxC
        q_z, q_obj = q
        q_z = F.normalize(self.predictor(q_z).squeeze(-1), dim=1)
        b, c, n = q_obj.shape
        q_obj = F.normalize(self.predictor_obj(q_obj).transpose(2, 1).reshape(b * n, c), dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # self._momentum_update_key_encoder()  # update the key encoder
            k = [x.clone().detach() for x in self.encoder_k(xk)]
            k_z, k_obj = k
            k_z = F.normalize(k_z.squeeze(-1), dim=1)
            b, c, n = k_obj.shape
            k_obj = F.normalize(k_obj.transpose(2, 1).reshape(b*n, c), dim=1)

        l_pos_z = torch.einsum('nc,nc->n', [q_z, k_z]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_z = torch.einsum('nc,ck->nk', [q_z, self.queue_z.clone().detach()])

        # logits: Nx(1+K)
        logits_z = torch.cat([l_pos_z, l_neg_z], dim=1)

        topk_onehot_z = torch.zeros_like(l_neg_z)
        pos_mask_z = torch.cat([torch.ones(topk_onehot_z.size(0), 1).cuda(), topk_onehot_z], dim=1)

        # apply temperature
        logits_z /= self.T

        l_pos_obj = torch.einsum('nc,nc->n', [q_obj, k_obj]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_obj = torch.einsum('nc,ck->nk', [q_obj, self.queue_obj.clone().detach()])

        # logits: Nx(1+K)
        logits_obj = torch.cat([l_pos_obj, l_neg_obj], dim=1)

        topk_onehot_obj = torch.zeros_like(l_neg_obj)
        pos_mask_obj = torch.cat([torch.ones(topk_onehot_obj.size(0), 1).cuda(), topk_onehot_obj], dim=1)

        # apply temperature
        logits_obj /= self.T

        # dequeue and enqueue
        self._dequeue_and_enqueue(k_z, k_obj)
        return logits_z, pos_mask_z, logits_obj, pos_mask_obj

class SkeMix_LWLCLR(nn.Module):
    """ Referring to the code of MOCO, https://arxiv.org/abs/1911.05722 """

    def __init__(self, base_encoder=None, pretrain=True, feature_dim=128, queue_size=32768,
                 momentum=0.999, Temperature=0.07, mlp=True, in_channels=3, hidden_channels=64,
                 hidden_dim=256, num_class=60, dropout=0.5, loss_weight=0.5, heads=8,
                 graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},sigma=0.5, spa_l=1, spa_u=4, tem_l=1, tem_u=12,
                 swap_mode='swap', spatial_mode='semantic',
                 edge_importance_weighting=True, **kwargs):
        """
        K: queue size; number of negative keys (default: 32768)
        m: momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """

        super().__init__()
        base_encoder = import_class(base_encoder)
        self.pretrain = pretrain

        if not self.pretrain:
            self.encoder_q = EncoderObj(base_encoder=base_encoder,hid_dim=hidden_dim, out_dim=feature_dim, norm_layer=None, num_mlp=2,
                                          scale=1., l2_norm=True, num_heads=heads,
                                          in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=num_class,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
        else:
            self.K = queue_size
            self.m = momentum
            self.curr_m = momentum
            self.T = Temperature

            self.sigma = sigma
            self.spa_l = spa_l
            self.spa_u = spa_u
            self.tem_l = tem_l
            self.tem_u = tem_u
            self.swap_mode = swap_mode
            self.spatial_mode = spatial_mode
            self.loss_weight = loss_weight

            self.encoder_q = EncoderObj(base_encoder=base_encoder, hid_dim=hidden_dim, out_dim=feature_dim, norm_layer=None, num_mlp=2,
                                          scale=1., l2_norm=True, num_heads=heads,
                                          in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=num_class,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
            self.encoder_k = EncoderObj(base_encoder=base_encoder, hid_dim=hidden_dim, out_dim=feature_dim, norm_layer=None, num_mlp=2,
                                          scale=1., l2_norm=True, num_heads=heads,
                                          in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=num_class,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)

            self.predictor = MLP1D(feature_dim, hidden_dim, feature_dim)
            self.predictor.init_weights()
            self.predictor_obj = MLP1D(feature_dim, hidden_dim, feature_dim)
            self.predictor_obj.init_weights()
            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)    # initialize
                param_k.requires_grad = False       # not update by gradient

            # create the queue
            self.register_buffer("queue_z", torch.randn(feature_dim, queue_size))
            self.queue_z = F.normalize(self.queue_z, dim=0)
            self.register_buffer("queue_ptr_z", torch.zeros(1, dtype=torch.long))

            # create the queue
            self.register_buffer("queue_obj", torch.randn(feature_dim, queue_size))
            self.queue_obj = F.normalize(self.queue_obj, dim=0)
            self.register_buffer("queue_ptr_obj", torch.zeros(1, dtype=torch.long))
    @torch.no_grad()
    def momentum_update(self, cur_iter, max_iter):
        """
        Momentum update of the target network.
        """
        # momentum anneling
        momentum = 1. - (1. - self.m) * (cos(pi * cur_iter / float(max_iter)) + 1) / 2.0
        self.curr_m = momentum
        # parameter update for target network
        state_dict_ol = self.encoder_q.state_dict()
        state_dict_tgt = self.encoder_k.state_dict()
        for (k_ol, v_ol), (k_tgt, v_tgt) in zip(state_dict_ol.items(), state_dict_tgt.items()):
            assert k_tgt == k_ol, "state_dict names are different!"
            assert v_ol.shape == v_tgt.shape, "state_dict shapes are different!"
            if 'num_batches_tracked' in k_tgt:
                v_tgt.copy_(v_ol)
            else:
                v_tgt.copy_(v_tgt * momentum + (1. - momentum) * v_ol)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, key0, key1):
        bs0 = key0.shape[0]
        bs1 = key1.shape[0]
        ptr_z = int(self.queue_ptr_z)
        ptr_obj = int(self.queue_ptr_obj)
        gpu_index0 = key0.device.index
        gpu_index1 = key1.device.index
        self.queue_z[:, (ptr_z + bs0 * gpu_index0):(ptr_z + bs0 * (gpu_index0 + 1))] = key0.T
        self.queue_obj[:, (ptr_obj + bs1 * gpu_index1):(ptr_obj + bs1 * (gpu_index1 + 1))] = key1.T

    @torch.no_grad()
    def update_ptr(self, batch_size):
        assert self.K % batch_size == 0 #  for simplicity
        self.queue_ptr_z[0] = (self.queue_ptr_z[0] + batch_size) % self.K
        self.queue_ptr_obj[0] = (self.queue_ptr_obj[0] + batch_size) % self.K

    @torch.no_grad()
    def ske_swap(self, x):
        '''
        swap a batch skeleton
        T   64 --> 32 --> 16    # 8n
        S   25 --> 25 --> 25 (5 parts)
        '''
        N, C, T, V, M = x.size()
        tem_downsample_ratio = 4

        # generate swap swap idx
        idx = torch.arange(N)
        n = torch.randint(1, N - 1, (1,))
        randidx = (idx + n) % N

        # ------ Spatial ------ #
        if self.spatial_mode == 'semantic':
            Cs = random.randint(self.spa_l, self.spa_u)
            # sample the parts index
            parts_idx = random.sample(body_parts, Cs)
            # generate spa_idx
            spa_idx = []
            for part_idx in parts_idx:
                spa_idx += part_idx
            spa_idx.sort()
        elif self.spatial_mode == 'random':
            spa_num = random.randint(10, 15)
            spa_idx = random.sample(list(range(V)), spa_num)
            spa_idx.sort()
        else:
            raise ValueError('Not supported operation {}'.format(self.spatial_mode))
        # spa_idx = torch.tensor(spa_idx, dtype=torch.long).cuda()

        # ------ Temporal ------ #
        Ct = random.randint(self.tem_l, self.tem_u)
        # print(int(ceil(T / tem_downsample_ratio)), Ct)
        tem_idx = random.randint(0, int(ceil(T / tem_downsample_ratio)) - Ct)
        rt = Ct * tem_downsample_ratio

        xst = x.clone()
        # begin swap
        if self.swap_mode == 'swap':
            xst[:, :, tem_idx * tem_downsample_ratio: tem_idx * tem_downsample_ratio + rt, spa_idx, :] = \
                xst[randidx][:, :, tem_idx * tem_downsample_ratio: tem_idx * tem_downsample_ratio + rt, spa_idx, :]
        elif self.swap_mode == 'zeros':
            xst[:, :, tem_idx * tem_downsample_ratio: tem_idx * tem_downsample_ratio + rt, spa_idx, :] = 0
        elif self.swap_mode == 'Gaussian':
            xst[:, :, tem_idx * tem_downsample_ratio: tem_idx * tem_downsample_ratio + rt, spa_idx, :] = \
                torch.randn(N, C, rt, len(spa_idx), M).cuda()
        else:
            raise ValueError('Not supported operation {}'.format(self.swap_mode))
        # generate mask
        mask = torch.zeros(int(ceil(T / tem_downsample_ratio)), V).cuda()
        mask[tem_idx:tem_idx + Ct, spa_idx] = 1

        return randidx, xst, mask.bool()

    def forward(self, xq, xk=None):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        """

        if not self.pretrain:
            q, _ = self.encoder_q.backbone(xq)
            return q
        randidx, x_pc, mask = self.ske_swap(xq)

        # compute query features
        q = self.encoder_q(xq)  # queries: NxC
        q_z, q_obj = q
        q_z = F.normalize(self.predictor(q_z).squeeze(-1), dim=1)
        b, c, n = q_obj.shape
        q_obj = F.normalize(self.predictor_obj(q_obj).transpose(2, 1).reshape(b * n, c), dim=1)

        p, c, obj_swap = self.encoder_q(x_pc, mask=mask)  # queries: NxC
        q_p, q_c = F.normalize(self.predictor(p).squeeze(-1), dim=1), F.normalize(self.predictor(c).squeeze(-1), dim=1)
        b, c, n = obj_swap.shape
        q_obj_swap = F.normalize(self.predictor_obj(obj_swap).transpose(2, 1).reshape(b * n, c), dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # self._momentum_update_key_encoder()  # update the key encoder
            k = [x.clone().detach() for x in self.encoder_k(xk)]
            k_z, k_obj = k
            k_z = F.normalize(k_z.squeeze(-1), dim=1)
            b, c, n = k_obj.shape
            k_obj = F.normalize(k_obj.transpose(2, 1).reshape(b*n, c), dim=1)

        l_pos_z = torch.einsum('nc,nc->n', [q_z, k_z]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_z = torch.einsum('nc,ck->nk', [q_z, self.queue_z.clone().detach()])
        # logits: Nx(1+K)
        logits_z = torch.cat([l_pos_z, l_neg_z], dim=1)
        topk_onehot_z = torch.zeros_like(l_neg_z)
        pos_mask_z = torch.cat([torch.ones(topk_onehot_z.size(0), 1).cuda(), topk_onehot_z], dim=1)
        # apply temperature
        logits_z /= self.T

        l_pos_obj = torch.einsum('nc,nc->n', [q_obj, k_obj]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_obj = torch.einsum('nc,ck->nk', [q_obj, self.queue_obj.clone().detach()])
        # logits: Nx(1+K)
        logits_obj = torch.cat([l_pos_obj, l_neg_obj], dim=1)
        topk_onehot_obj = torch.zeros_like(l_neg_obj)
        pos_mask_obj = torch.cat([torch.ones(topk_onehot_obj.size(0), 1).cuda(), topk_onehot_obj], dim=1)
        # apply temperature
        logits_obj /= self.T

        l_pos_obj_swap = torch.einsum('nc,nc->n', [q_obj_swap, k_obj]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_obj_swap = torch.einsum('nc,ck->nk', [q_obj_swap, self.queue_obj.clone().detach()])
        # logits: Nx(1+K)
        logits_obj_swap = torch.cat([l_pos_obj_swap, l_neg_obj_swap], dim=1)
        topk_onehot_obj_swap = torch.zeros_like(l_neg_obj_swap)
        pos_mask_obj_swap = torch.cat([torch.ones(topk_onehot_obj_swap.size(0), 1).cuda(), topk_onehot_obj_swap], dim=1)
        # apply temperature
        logits_obj_swap /= self.T

        # Loss region p
        l_pos = torch.einsum('nc,nc->n', [q_p, k_z[randidx]]).unsqueeze(-1)
        l_neg_1 = torch.einsum('nc,ck->nk', [q_p, self.queue_z.clone().detach()])
        l_neg_2 = torch.einsum('nc,nc->n', [q_p, q_c.clone().detach()]).unsqueeze(-1)
        logits_reg_1 = torch.cat([l_pos, l_neg_1, l_neg_2], dim=1)
        logits_reg_1 /= self.T

        # Loss region c
        l_pos = torch.einsum('nc,nc->n', [q_c, k_z]).unsqueeze(-1)
        l_neg_1 = torch.einsum('nc,ck->nk', [q_c, self.queue_z.clone().detach()])
        l_neg_2 = torch.einsum('nc,nc->n', [q_c, q_p.clone().detach()]).unsqueeze(-1)
        logits_reg_2 = torch.cat([l_pos, l_neg_1, l_neg_2], dim=1)
        logits_reg_2 /= self.T

        topk_onehot_neg_p = torch.zeros_like(l_neg_1)
        topk_onehot_neg_c = torch.zeros_like(l_neg_2)

        pos_mask_reg1 = torch.cat([torch.ones(topk_onehot_z.size(0), 1).cuda(), topk_onehot_neg_p, topk_onehot_neg_c],
                                  dim=1)
        pos_mask_reg2 = torch.cat([torch.ones(topk_onehot_z.size(0), 1).cuda(), topk_onehot_neg_p, topk_onehot_neg_c],
                                  dim=1)

        loss_z = multi_nce_loss(logits_z, pos_mask_z)
        loss_obj = multi_nce_loss(logits_obj, pos_mask_obj)
        loss_obj_swap = multi_nce_loss(logits_obj_swap, pos_mask_obj_swap)
        loss_zp = multi_nce_loss(logits_reg_1, pos_mask_reg1)
        loss_zc = multi_nce_loss(logits_reg_2, pos_mask_reg2)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k_z, k_obj)
        return loss_z, loss_obj, loss_obj_swap, loss_zp, loss_zc


class MutualLWL(nn.Module):
    """ Referring to the code of MOCO, https://arxiv.org/abs/1911.05722 """

    def __init__(self, base_encoder=None, pretrain=True, feature_dim=128, queue_size=32768,
                 momentum=0.996, Temperature=0.07, mlp=True, in_channels=3, hidden_channels=64,
                 hidden_dim=256, num_class=60, dropout=0.5, attn_drop=0., proj_drop=0.,
                 drop_path=0.1, loss_weight=0.5, heads=8, norm_layer=None, num_mlp=2, scale=1.,
                 graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                 edge_importance_weighting=True, **kwargs):
        """
        K: queue size; number of negative keys (default: 32768)
        m: momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """

        super().__init__()
        base_encoder = import_class(base_encoder)
        self.pretrain = pretrain
        self.kl = KLDiv(T=3)
        if not self.pretrain:
            self.encoder_q = EncoderObj(base_encoder=base_encoder, hid_dim=hidden_dim, out_dim=feature_dim,
                                        norm_layer=norm_layer, num_mlp=num_mlp,
                                        scale=scale, l2_norm=True, num_heads=heads,
                                        in_channels=in_channels, hidden_channels=hidden_channels,
                                        hidden_dim=hidden_dim, num_class=num_class,
                                        dropout=dropout, attn_drop=attn_drop, proj_drop=proj_drop,
                                        drop_path=drop_path, graph_args=graph_args,
                                        edge_importance_weighting=edge_importance_weighting,
                                        **kwargs)
        else:
            self.K = queue_size
            self.m = momentum
            self.curr_m = momentum
            self.T = Temperature

            self.loss_weight = loss_weight

            self.encoder_q = EncoderObj(base_encoder=base_encoder, hid_dim=hidden_dim, out_dim=feature_dim,
                                        norm_layer=norm_layer, num_mlp=num_mlp,
                                        scale=scale, l2_norm=True, num_heads=heads,
                                        in_channels=in_channels, hidden_channels=hidden_channels,
                                        hidden_dim=hidden_dim, num_class=num_class,
                                        dropout=dropout, attn_drop=attn_drop, proj_drop=proj_drop,
                                        drop_path=drop_path, graph_args=graph_args,
                                        edge_importance_weighting=edge_importance_weighting,
                                        **kwargs)
            self.encoder_k = EncoderObj(base_encoder=base_encoder, hid_dim=hidden_dim, out_dim=feature_dim,
                                        norm_layer=norm_layer, num_mlp=num_mlp,
                                        scale=scale, l2_norm=True, num_heads=heads,
                                        in_channels=in_channels, hidden_channels=hidden_channels,
                                        hidden_dim=hidden_dim, num_class=num_class,
                                        dropout=dropout, attn_drop=attn_drop, proj_drop=proj_drop,
                                        drop_path=drop_path, graph_args=graph_args,
                                        edge_importance_weighting=edge_importance_weighting,
                                        **kwargs)

            self.predictor = MLP1D(feature_dim, hidden_dim, feature_dim)
            self.predictor.init_weights()
            self.predictor_obj = MLP1D(feature_dim, hidden_dim, feature_dim)
            self.predictor_obj.init_weights()
            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient

            # create the queue
            self.register_buffer("queue_z0", torch.randn(feature_dim, queue_size))
            self.queue_z0 = F.normalize(self.queue_z0, dim=0)
            self.register_buffer("queue_ptr_z0", torch.zeros(1, dtype=torch.long))

            self.register_buffer("queue_z1", torch.randn(feature_dim, queue_size))
            self.queue_z1 = F.normalize(self.queue_z1, dim=0)
            self.register_buffer("queue_ptr_z1", torch.zeros(1, dtype=torch.long))

            # create the queue
            self.register_buffer("queue_obj0", torch.randn(feature_dim, queue_size))
            self.queue_obj0 = F.normalize(self.queue_obj0, dim=0)
            self.register_buffer("queue_ptr_obj0", torch.zeros(1, dtype=torch.long))

            self.register_buffer("queue_obj1", torch.randn(feature_dim, queue_size))
            self.queue_obj1 = F.normalize(self.queue_obj1, dim=0)
            self.register_buffer("queue_ptr_obj1", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def momentum_update(self, cur_iter, max_iter):
        """
        Momentum update of the target network.
        """
        # momentum anneling
        momentum = 1. - (1. - self.m) * (cos(pi * cur_iter / float(max_iter)) + 1) / 2.0
        self.curr_m = momentum
        # parameter update for target network
        state_dict_ol = self.encoder_q.state_dict()
        state_dict_tgt = self.encoder_k.state_dict()
        for (k_ol, v_ol), (k_tgt, v_tgt) in zip(state_dict_ol.items(), state_dict_tgt.items()):
            assert k_tgt == k_ol, "state_dict names are different!"
            assert v_ol.shape == v_tgt.shape, "state_dict shapes are different!"
            if 'num_batches_tracked' in k_tgt:
                v_tgt.copy_(v_ol)
            else:
                v_tgt.copy_(v_tgt * momentum + (1. - momentum) * v_ol)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, kz0, kz1, kobj0, kobj1):
        bs_z0 = kz0.shape[0]
        bs_z1 = kz1.shape[0]
        bs_obj0 = kobj0.shape[0]
        bs_obj1 = kobj1.shape[0]
        ptr_z0 = int(self.queue_ptr_z0)
        ptr_z1 = int(self.queue_ptr_z1)
        ptr_obj0 = int(self.queue_ptr_obj0)
        ptr_obj1 = int(self.queue_ptr_obj1)
        gi_z0 = kz0.device.index
        gi_z1 = kz1.device.index
        gi_obj0 = kobj0.device.index
        gi_obj1 = kobj1.device.index
        self.queue_z0[:, (ptr_z0 + bs_z0 * gi_z0):(ptr_z0 + bs_z0 * (gi_z0 + 1))] = kz0.T
        self.queue_z1[:, (ptr_z1 + bs_z1 * gi_z1):(ptr_z1 + bs_z1 * (gi_z1 + 1))] = kz1.T
        self.queue_obj0[:, (ptr_obj0 + bs_obj0 * gi_z0):(ptr_obj0 + bs_obj0 * (gi_obj0 + 1))] = kobj0.T
        self.queue_obj1[:, (ptr_obj1 + bs_obj1 * gi_z1):(ptr_obj1 + bs_obj1 * (gi_obj1 + 1))] = kobj1.T

    @torch.no_grad()
    def update_ptr(self, batch_size):
        assert self.K % batch_size == 0  # for simplicity
        self.queue_ptr_z0[0] = (self.queue_ptr_z0[0] + batch_size) % self.K
        self.queue_ptr_z1[0] = (self.queue_ptr_z1[0] + batch_size) % self.K
        self.queue_ptr_obj0[0] = (self.queue_ptr_obj0[0] + batch_size) % self.K
        self.queue_ptr_obj1[0] = (self.queue_ptr_obj1[0] + batch_size) % self.K

    def forward(self, x0, x1=None):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        """

        if not self.pretrain:
            q, _ = self.encoder_q.backbone(x0)
            return q

        # compute query features
        q0 = self.encoder_q(x0)  # queries: NxC
        q_z0, q_obj0 = q0
        q_z0 = F.normalize(self.predictor(q_z0).squeeze(-1), dim=1)
        b, c, n = q_obj0.shape
        q_obj0 = F.normalize(self.predictor_obj(q_obj0).transpose(2, 1).reshape(b * n, c), dim=1)

        q1 = self.encoder_q(x1)  # queries: NxC
        q_z1, q_obj1 = q1
        q_z1 = F.normalize(self.predictor(q_z1).squeeze(-1), dim=1)
        b, c, n = q_obj1.shape
        q_obj1 = F.normalize(self.predictor_obj(q_obj1).transpose(2, 1).reshape(b * n, c), dim=1)

        qz_list = [q_z0, q_z1]
        qobj_list = [q_obj0, q_obj1]

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # self._momentum_update_key_encoder()  # update the key encoder
            k0 = [x.clone().detach() for x in self.encoder_k(x0)]
            k_z0, k_obj0 = k0
            k_z0 = F.normalize(k_z0.squeeze(-1), dim=1)
            b, c, n = k_obj0.shape
            k_obj0 = F.normalize(k_obj0.transpose(2, 1).reshape(b * n, c), dim=1)

            k1 = [x.clone().detach() for x in self.encoder_k(x1)]
            k_z1, k_obj1 = k1
            k_z1 = F.normalize(k_z1.squeeze(-1), dim=1)
            b, c, n = k_obj1.shape
            k_obj1 = F.normalize(k_obj1.transpose(2, 1).reshape(b * n, c), dim=1)

        kz_list = [k_z0, k_z1]
        kobj_list = [k_obj0, k_obj1]
        queue_z_list = [self.queue_z0, self.queue_z1]
        queue_obj_list = [self.queue_obj0, self.queue_obj1]
        loss_icl_z = 0.
        # loss_soft_icl_z = 0.
        for i in range(len(qz_list)):
            for j in range(i+1, len(qz_list)):
                l_pos_ij = torch.einsum('nc,nc->n', [qz_list[i], kz_list[j]]).unsqueeze(-1)
                # negative logits: NxK
                l_neg_ij = torch.einsum('nc,ck->nk', [qz_list[i], queue_z_list[j].clone().detach()])
                # logits: Nx(1+K)
                logits_ij = torch.cat([l_pos_ij, l_neg_ij], dim=1)
                # apply temperature
                logits_ij /= self.T

                l_pos_ji = torch.einsum('nc,nc->n', [qz_list[j], kz_list[i]]).unsqueeze(-1)
                # negative logits: NxK
                l_neg_ji = torch.einsum('nc,ck->nk', [qz_list[j], queue_z_list[i].clone().detach()])
                # logits: Nx(1+K)
                logits_ji = torch.cat([l_pos_ji, l_neg_ji], dim=1)

                topk_onehot = torch.zeros_like(l_neg_ij)
                pos_mask = torch.cat([torch.ones(topk_onehot.size(0), 1).cuda(), topk_onehot], dim=1)
                # apply temperature
                logits_ji /= self.T
                loss_icl_z += multi_nce_loss(logits_ij, pos_mask)
                loss_icl_z += multi_nce_loss(logits_ji, pos_mask)
                # loss_soft_icl_z += self.kl(logits_ij, logits_ji.detach())
                # loss_soft_icl_z += self.kl(logits_ji, logits_ij.detach())


        loss_icl_obj = 0.
        # loss_soft_icl_obj = 0.
        for i in range(len(qobj_list)):
            for j in range(i+1, len(qobj_list)):
                l_pos_ij = torch.einsum('nc,nc->n', [qobj_list[i], kobj_list[j]]).unsqueeze(-1)
                # negative logits: NxK
                l_neg_ij = torch.einsum('nc,ck->nk', [qobj_list[i], queue_obj_list[j].clone().detach()])
                # logits: Nx(1+K)
                logits_ij = torch.cat([l_pos_ij, l_neg_ij], dim=1)
                # apply temperature
                logits_ij /= self.T

                l_pos_ji = torch.einsum('nc,nc->n', [qobj_list[j], kobj_list[i]]).unsqueeze(-1)
                # negative logits: NxK
                l_neg_ji = torch.einsum('nc,ck->nk', [qobj_list[j], queue_obj_list[i].clone().detach()])
                # logits: Nx(1+K)
                logits_ji = torch.cat([l_pos_ji, l_neg_ji], dim=1)

                topk_onehot = torch.zeros_like(l_neg_ij)
                pos_mask = torch.cat([torch.ones(topk_onehot.size(0), 1).cuda(), topk_onehot], dim=1)
                # apply temperature
                logits_ji /= self.T
                loss_icl_obj += multi_nce_loss(logits_ij, pos_mask)
                loss_icl_obj += multi_nce_loss(logits_ji, pos_mask)
                # loss_soft_icl_obj += self.kl(logits_ij, logits_ji.detach())
                # loss_soft_icl_obj += self.kl(logits_ji, logits_ij.detach())

        # dequeue and enqueue
        self._dequeue_and_enqueue(k_z0, k_z1, k_obj0, k_obj1)
        return loss_icl_z, loss_icl_obj

if __name__ == '__main__':
    # x = torch.randn((16, 256, 13, 25, 2))
    x0 = torch.randn((16, 128, 1))
    x1 = torch.randn((16, 128, 16))
    x2 = torch.randn((16, 128, 1))
    x3 = torch.randn((16, 128, 16))
    # test = ObjectNeck(in_channels=256, out_channels=128)
    test = Ske_LWLCLR_M(base_encoder=Model)

    out = test.loss_func([x0, x1], [x2, x3])
    print(out.shape)
    # print(out1.shape)