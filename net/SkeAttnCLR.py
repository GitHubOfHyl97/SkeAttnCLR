import torch
from math import cos, pi, ceil
import torch.nn as nn
import torch.nn.functional as F
from torchlight import import_class
from net.utils.AttnMask import ObjectNeck_AM, ObjectNeck_GRU, ObjectNeck_TR, MLP1D
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
                 drop_path=0.1, Lambda=2,
                 graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                 edge_importance_weighting=True, **kwargs):
        super(EncoderObj, self).__init__()
        self.backbone = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=num_class,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
        in_dim = hidden_dim
        self.neck = ObjectNeck_AM(in_channels=in_dim, hid_channels=hid_dim, out_channels=out_dim, Lambda=Lambda,
                               norm_layer=norm_layer, num_layers=num_mlp, attn_drop=attn_drop, proj_drop=proj_drop,
                               drop_path=drop_path, scale=scale, l2_norm=l2_norm, num_heads=num_heads)
        self.neck.init_weights(init_linear='kaiming')

    def forward(self, im, mask=None, attn_mask=False):
        _, f = self.backbone(im)
        out = self.neck(f, mask=mask, attn_mask=attn_mask)
        return out

class SkeAttnMask(nn.Module):
    """ Referring to the code of MOCO, https://arxiv.org/abs/1911.05722 """

    def __init__(self, base_encoder=None, pretrain=True, feature_dim=128, queue_size=32768,
                 momentum=0.999, Temperature=0.07, mlp=True, in_channels=3, hidden_channels=64,
                 hidden_dim=256, num_class=60, dropout=0.5, loss_weight=0.5, heads=8,
                 graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},sigma=0.5, spa_l=2, spa_u=3, tem_l=7, tem_u=11,
                 swap_mode='swap', spatial_mode='semantic', Lambda=2,
                 edge_importance_weighting=True, **kwargs):
        """
        K: queue size; number of negative keys (default: 32768)
        m: momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """

        super().__init__()
        base_encoder = import_class(base_encoder)
        self.pretrain = pretrain
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
        
        if not self.pretrain:
            self.encoder_q = EncoderObj(base_encoder=base_encoder,hid_dim=hidden_dim, out_dim=feature_dim, norm_layer=None, num_mlp=2,
                                          scale=1., l2_norm=True, num_heads=heads, Lambda=Lambda,
                                          in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=num_class,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
        else:
            self.encoder_q = EncoderObj(base_encoder=base_encoder, hid_dim=hidden_dim, out_dim=feature_dim, norm_layer=None, num_mlp=2,
                                          scale=1., l2_norm=True, num_heads=heads, Lambda=Lambda,
                                          in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=num_class,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
            self.encoder_k = EncoderObj(base_encoder=base_encoder, hid_dim=hidden_dim, out_dim=feature_dim, norm_layer=None, num_mlp=2,
                                          scale=1., l2_norm=True, num_heads=heads, Lambda=Lambda,
                                          in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=num_class,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)

            self.predictor = MLP1D(feature_dim, hidden_dim, feature_dim)
            self.predictor.init_weights()
            self.predictor_p = MLP1D(feature_dim, hidden_dim, feature_dim)
            self.predictor_p.init_weights()
            self.predictor_c = MLP1D(feature_dim, hidden_dim, feature_dim)
            self.predictor_c.init_weights()
            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)    # initialize
                param_k.requires_grad = False       # not update by gradient

            # create the queue
            self.register_buffer("queue_z", torch.randn(feature_dim, queue_size))
            self.queue_z = F.normalize(self.queue_z, dim=0)
            self.register_buffer("queue_ptr_z", torch.zeros(1, dtype=torch.long))

            # create the queue
            self.register_buffer("queue_p", torch.randn(feature_dim, queue_size))
            self.queue_p = F.normalize(self.queue_p, dim=0)
            self.register_buffer("queue_ptr_p", torch.zeros(1, dtype=torch.long))

            # create the queue
            self.register_buffer("queue_c", torch.randn(feature_dim, queue_size))
            self.queue_c = F.normalize(self.queue_c, dim=0)
            self.register_buffer("queue_ptr_c", torch.zeros(1, dtype=torch.long))
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
    def _dequeue_and_enqueue(self, key0, key1, key2):
        bs0 = key0.shape[0]
        bs1 = key1.shape[0]
        bs2 = key2.shape[0]
        ptr_z = int(self.queue_ptr_z)
        ptr_p = int(self.queue_ptr_p)
        ptr_c = int(self.queue_ptr_c)
        gpu_index0 = key0.device.index
        gpu_index1 = key1.device.index
        gpu_index2 = key2.device.index
        self.queue_z[:, (ptr_z + bs0 * gpu_index0):(ptr_z + bs0 * (gpu_index0 + 1))] = key0.T
        self.queue_p[:, (ptr_p + bs1 * gpu_index1):(ptr_p + bs1 * (gpu_index1 + 1))] = key1.T
        self.queue_c[:, (ptr_c + bs2 * gpu_index2):(ptr_c + bs2 * (gpu_index2 + 1))] = key2.T

    @torch.no_grad()
    def update_ptr(self, batch_size):
        assert self.K % batch_size == 0 #  for simplicity
        self.queue_ptr_z[0] = (self.queue_ptr_z[0] + batch_size) % self.K
        self.queue_ptr_p[0] = (self.queue_ptr_p[0] + batch_size) % self.K
        self.queue_ptr_c[0] = (self.queue_ptr_c[0] + batch_size) % self.K

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
    @torch.no_grad()
    def view_gen(self, data, view):
        if view == 'joint':
            pass
        elif view == 'motion':
            motion = torch.zeros_like(data)

            motion[:, :, :-1, :, :] = data[:, :, 1:, :, :] - data[:, :, :-1, :, :]

            data = motion
        elif view == 'bone':
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
    
    def forward(self, xq, xk=None, view='joint'):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        """
        _, x_swap, _ = self.ske_swap(xq)
        xq = self.view_gen(xq, view)
        if not self.pretrain:
            q, _ = self.encoder_q.backbone(xq)
            return q
        xk = self.view_gen(xk, view)
        x_swap = self.view_gen(x_swap, view)
        # compute query features
        q_z = self.encoder_q(xq)  # queries: NxC
        q_z = F.normalize(self.predictor(q_z).squeeze(-1), dim=1)

        q_z_swap, q_s, q_ns, mask = self.encoder_q(x_swap, attn_mask=True)  # queries: NxC
        q_z_swap = F.normalize(self.predictor(q_z_swap).squeeze(-1), dim=1)
        q_s = F.normalize(self.predictor_p(q_s).squeeze(-1), dim=1)
        q_ns = F.normalize(self.predictor_c(q_ns).squeeze(-1), dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # self._momentum_update_key_encoder()  # update the key encoder
            k = [x.clone().detach() for x in self.encoder_k(xk, mask=mask)]
            k_z, k_s, k_ns = k
            k_z = F.normalize(k_z.squeeze(-1), dim=1)
            k_s = F.normalize(k_s.squeeze(-1), dim=1)
            k_ns = F.normalize(k_ns.squeeze(-1), dim=1)


        l_pos_z = torch.einsum('nc,nc->n', [q_z, k_z]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_z = torch.einsum('nc,ck->nk', [q_z, self.queue_z.clone().detach()])
        # logits: Nx(1+K)
        logits_z = torch.cat([l_pos_z, l_neg_z], dim=1)
        topk_onehot_z = torch.zeros_like(l_neg_z)
        pos_mask_z = torch.cat([torch.ones(topk_onehot_z.size(0), 1).cuda(), topk_onehot_z], dim=1)
        # apply temperature
        logits_z /= self.T

        # Loss region salient
        l_pos = torch.einsum('nc,nc->n', [q_s, k_s]).unsqueeze(-1)
        l_neg_1 = torch.einsum('nc,ck->nk', [q_s, self.queue_z.clone().detach()])
        l_neg_2 = torch.einsum('nc,nc->n', [q_s, q_ns.clone().detach()]).unsqueeze(-1)
        logits_reg_1 = torch.cat([l_pos, l_neg_1, l_neg_2], dim=1)
        # logits_reg_1 = torch.cat([l_pos, l_neg_1], dim=1)
        logits_reg_1 /= self.T

        # Loss region non-salient
        l_pos = torch.einsum('nc,nc->n', [q_ns, k_ns]).unsqueeze(-1)
        l_neg_1 = torch.einsum('nc,ck->nk', [q_ns, self.queue_z.clone().detach()])
        l_neg_2 = torch.einsum('nc,nc->n', [q_ns, q_s.clone().detach()]).unsqueeze(-1)
        logits_reg_2 = torch.cat([l_pos, l_neg_1, l_neg_2], dim=1)
        # logits_reg_2 = torch.cat([l_pos, l_neg_1], dim=1)
        logits_reg_2 /= self.T

        topk_onehot_neg_p = torch.zeros_like(l_neg_1)
        topk_onehot_neg_c = torch.zeros_like(l_neg_2)

        pos_mask_reg1 = torch.cat([torch.ones(topk_onehot_z.size(0), 1).cuda(), topk_onehot_neg_p, topk_onehot_neg_c],
                                  dim=1)
        pos_mask_reg2 = torch.cat([torch.ones(topk_onehot_z.size(0), 1).cuda(), topk_onehot_neg_p, topk_onehot_neg_c],
                                  dim=1)
        
        # pos_mask_reg1 = torch.cat([torch.ones(topk_onehot_z.size(0), 1).cuda(), topk_onehot_neg_p],
        #                           dim=1)
        # pos_mask_reg2 = torch.cat([torch.ones(topk_onehot_z.size(0), 1).cuda(), topk_onehot_neg_p],
        #                           dim=1)

        loss_z = multi_nce_loss(logits_z, pos_mask_z)
        loss_zp = multi_nce_loss(logits_reg_1, pos_mask_reg1)
        loss_zc = multi_nce_loss(logits_reg_2, pos_mask_reg2)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k_z, k_s, k_ns)
        # self._dequeue_and_enqueue(k_z)
        return loss_z, loss_zp, loss_zc

# initilize weight
def weights_init_gru(model):
    with torch.no_grad():
        for child in list(model.children()):
            print(child)
            for param in list(child.parameters()):
                if param.dim() == 2:
                    nn.init.xavier_uniform_(param)
    print('GRU weights initialization finished!')

class EncoderObj_GRU(nn.Module):
    def __init__(self, base_encoder, hid_dim, out_dim, norm_layer=None, num_mlp=2,
                 scale=1., l2_norm=True, num_heads=8, en_input_size=150, en_hidden_size=1024, en_num_layers=3,
                 num_class=60, attn_drop=0., proj_drop=0., Lambda=2,
                 drop_path=0.1):
        super(EncoderObj_GRU, self).__init__()
        self.backbone = base_encoder(en_input_size=en_input_size, en_hidden_size=en_hidden_size, en_num_layers=en_num_layers, num_class=num_class)
        weights_init_gru(self.backbone)
        in_dim = en_hidden_size*2
        self.neck = ObjectNeck_GRU(in_channels=in_dim, hid_channels=hid_dim, out_channels=out_dim,
                                   norm_layer=norm_layer, num_layers=num_mlp, attn_drop=attn_drop, proj_drop=proj_drop,
                                   drop_path=drop_path, scale=scale, l2_norm=l2_norm, num_heads=num_heads)

        self.neck.init_weights(init_linear='kaiming')

    def forward(self, im, mask=None, attn_mask=False):
        _, f = self.backbone(im)
        out = self.neck(f, mask=mask, attn_mask=attn_mask)
        return out

class SkeAttnMask_GRU(nn.Module):
    """ Referring to the code of MOCO, https://arxiv.org/abs/1911.05722 """

    def __init__(self, base_encoder=None, pretrain=True, feature_dim=128, queue_size=32768,
                 momentum=0.999, Temperature=0.07, mlp=True, en_input_size=150, en_hidden_size=1024, en_num_layers=3,
                 num_class=60, loss_weight=0.5, heads=8, sigma=0.5, spa_l=1, spa_u=4, tem_l=1, tem_u=12, Lambda=2,
                 swap_mode='swap', spatial_mode='semantic'):
        """
        K: queue size; number of negative keys (default: 32768)
        m: momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """

        super().__init__()
        base_encoder = import_class(base_encoder)
        self.pretrain = pretrain
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
        if not self.pretrain:
            self.encoder_q = EncoderObj_GRU(base_encoder=base_encoder, hid_dim=en_hidden_size, out_dim=feature_dim,
                                            num_heads=heads, en_input_size=en_input_size, en_hidden_size=en_hidden_size,
                                            en_num_layers=en_num_layers, Lambda=Lambda,
                                            num_class=num_class)
        else:
            self.encoder_q = EncoderObj_GRU(base_encoder=base_encoder, hid_dim=en_hidden_size, out_dim=feature_dim,
                                            num_heads=heads, en_input_size=en_input_size, en_hidden_size=en_hidden_size,
                                            en_num_layers=en_num_layers, Lambda=Lambda,
                                            num_class=num_class)
            self.encoder_k = EncoderObj_GRU(base_encoder=base_encoder, hid_dim=en_hidden_size, out_dim=feature_dim,
                                            num_heads=heads, en_input_size=en_input_size, en_hidden_size=en_hidden_size,
                                            en_num_layers=en_num_layers, Lambda=Lambda,
                                            num_class=num_class)

            self.predictor = MLP1D(feature_dim, feature_dim*2, feature_dim)
            self.predictor.init_weights()
            self.predictor_p = MLP1D(feature_dim, feature_dim*2, feature_dim)
            self.predictor_p.init_weights()
            self.predictor_c = MLP1D(feature_dim, feature_dim*2, feature_dim)
            self.predictor_c.init_weights()
            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)    # initialize
                param_k.requires_grad = False       # not update by gradient

            # create the queue
            self.register_buffer("queue_z", torch.randn(feature_dim, queue_size))
            self.queue_z = F.normalize(self.queue_z, dim=0)
            self.register_buffer("queue_ptr_z", torch.zeros(1, dtype=torch.long))

            # create the queue
            self.register_buffer("queue_p", torch.randn(feature_dim, queue_size))
            self.queue_p = F.normalize(self.queue_p, dim=0)
            self.register_buffer("queue_ptr_p", torch.zeros(1, dtype=torch.long))

            # create the queue
            self.register_buffer("queue_c", torch.randn(feature_dim, queue_size))
            self.queue_c = F.normalize(self.queue_c, dim=0)
            self.register_buffer("queue_ptr_c", torch.zeros(1, dtype=torch.long))
    @torch.no_grad()
    def momentum_update(self, cur_iter, max_iter):
        """
        Momentum update of the target network.
        """
        # momentum anneling
        momentum = 1. - (1. - self.m) * (cos(pi * cur_iter / float(max_iter)) + 1) / 2.0
        # if momentum > 0.999:
        #     momentum = 0.999
        if cur_iter > max_iter:
            momentum = 0.999
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
    def _dequeue_and_enqueue(self, key0, key1, key2):
        bs0 = key0.shape[0]
        bs1 = key1.shape[0]
        bs2 = key2.shape[0]
        ptr_z = int(self.queue_ptr_z)
        ptr_p = int(self.queue_ptr_p)
        ptr_c = int(self.queue_ptr_c)
        gpu_index0 = key0.device.index
        gpu_index1 = key1.device.index
        gpu_index2 = key2.device.index
        self.queue_z[:, (ptr_z + bs0 * gpu_index0):(ptr_z + bs0 * (gpu_index0 + 1))] = key0.T
        self.queue_p[:, (ptr_p + bs1 * gpu_index1):(ptr_p + bs1 * (gpu_index1 + 1))] = key1.T
        self.queue_c[:, (ptr_c + bs2 * gpu_index2):(ptr_c + bs2 * (gpu_index2 + 1))] = key2.T

    @torch.no_grad()
    def update_ptr(self, batch_size):
        assert self.K % batch_size == 0 #  for simplicity
        self.queue_ptr_z[0] = (self.queue_ptr_z[0] + batch_size) % self.K
        self.queue_ptr_p[0] = (self.queue_ptr_p[0] + batch_size) % self.K
        self.queue_ptr_c[0] = (self.queue_ptr_c[0] + batch_size) % self.K

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
    @torch.no_grad()
    def view_gen(self, data, view):
        if view == 'joint':
            pass
        elif view == 'motion':
            motion = torch.zeros_like(data)

            motion[:, :, :-1, :, :] = data[:, :, 1:, :, :] - data[:, :, :-1, :, :]

            data = motion
        elif view == 'bone':
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
    
    def forward(self, xq, xk=None, view='joint'):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        """
        _, x_swap, _ = self.ske_swap(xq)
        xq = self.view_gen(xq, view)
        # Permute and Reshape
        N, C, T, V, M = xq.size()
        im_q = xq.permute(0, 2, 3, 1, 4).reshape(N, T, -1)

        if not self.pretrain:
            q, _ = self.encoder_q.backbone(im_q)
            return q
        xk = self.view_gen(xk, view)
        x_swap = self.view_gen(x_swap, view)
        im_k = xk.permute(0, 2, 3, 1, 4).reshape(N, T, -1)
        im_swap = x_swap.permute(0, 2, 3, 1, 4).reshape(N, T, -1)

        # compute query features
        q_z = self.encoder_q(im_q)  # queries: NxC
        q_z = F.normalize(self.predictor(q_z).squeeze(-1), dim=1)

        q_z_swap, q_s, q_ns, mask = self.encoder_q(im_swap, mask=None, attn_mask=True)  # queries: NxC
        q_z_swap = F.normalize(self.predictor(q_z_swap).squeeze(-1), dim=1)
        q_s = F.normalize(self.predictor_p(q_s).squeeze(-1), dim=1)
        q_ns = F.normalize(self.predictor_c(q_ns).squeeze(-1), dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # self._momentum_update_key_encoder()  # update the key encoder
            k = [x.clone().detach() for x in self.encoder_k(im_k, mask=mask)]
            k_z, k_s, k_ns = k
            k_z = F.normalize(k_z.squeeze(-1), dim=1)
            k_s = F.normalize(k_s.squeeze(-1), dim=1)
            k_ns = F.normalize(k_ns.squeeze(-1), dim=1)


        l_pos_z = torch.einsum('nc,nc->n', [q_z, k_z]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_z = torch.einsum('nc,ck->nk', [q_z, self.queue_z.clone().detach()])
        # logits: Nx(1+K)
        logits_z = torch.cat([l_pos_z, l_neg_z], dim=1)
        topk_onehot_z = torch.zeros_like(l_neg_z)
        pos_mask_z = torch.cat([torch.ones(topk_onehot_z.size(0), 1).cuda(), topk_onehot_z], dim=1)
        # apply temperature
        logits_z /= self.T

        # Loss region salient
        l_pos = torch.einsum('nc,nc->n', [q_s, k_s]).unsqueeze(-1)
        l_neg_1 = torch.einsum('nc,ck->nk', [q_s, self.queue_z.clone().detach()])
        l_neg_2 = torch.einsum('nc,nc->n', [q_s, q_ns.clone().detach()]).unsqueeze(-1)
        logits_reg_1 = torch.cat([l_pos, l_neg_1, l_neg_2], dim=1)
        logits_reg_1 /= self.T

        # Loss region non-salient
        l_pos = torch.einsum('nc,nc->n', [q_ns, k_ns]).unsqueeze(-1)
        l_neg_1 = torch.einsum('nc,ck->nk', [q_ns, self.queue_z.clone().detach()])
        l_neg_2 = torch.einsum('nc,nc->n', [q_ns, q_s.clone().detach()]).unsqueeze(-1)
        logits_reg_2 = torch.cat([l_pos, l_neg_1, l_neg_2], dim=1)
        logits_reg_2 /= self.T

        topk_onehot_neg_p = torch.zeros_like(l_neg_1)
        topk_onehot_neg_c = torch.zeros_like(l_neg_2)

        pos_mask_reg1 = torch.cat([torch.ones(topk_onehot_z.size(0), 1).cuda(), topk_onehot_neg_p, topk_onehot_neg_c],
                                  dim=1)
        pos_mask_reg2 = torch.cat([torch.ones(topk_onehot_z.size(0), 1).cuda(), topk_onehot_neg_p, topk_onehot_neg_c],
                                  dim=1)

        loss_z = multi_nce_loss(logits_z, pos_mask_z)
        loss_zp = multi_nce_loss(logits_reg_1, pos_mask_reg1)
        loss_zc = multi_nce_loss(logits_reg_2, pos_mask_reg2)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k_z, k_s, k_ns)
        return loss_z, loss_zp, loss_zc
    
class EncoderObj_TR(nn.Module):
    def __init__(self, base_encoder, hid_dim, out_dim, num_class=60, num_point=25, num_frame=64, num_subset=3, dropout=0., config=None, num_person=2,
                 num_channel=3,  attn_drop=0., norm_layer=None, num_mlp=2, scale=1., l2_norm=True, num_heads=8,
                 proj_drop=0., drop_path=0.1, Lambda=2, **kwargs):
        super(EncoderObj_TR, self).__init__()
        self.backbone = base_encoder(num_class=num_class, num_point=num_point, num_frame=num_frame, num_subset=num_subset,
                                     dropout=dropout, config=config, num_person=num_person, num_channel=num_channel, **kwargs)
        in_dim = hid_dim
        self.neck = ObjectNeck_AM(in_channels=in_dim, hid_channels=hid_dim, out_channels=out_dim, Lambda=Lambda,
                               norm_layer=norm_layer, num_layers=num_mlp, attn_drop=attn_drop, proj_drop=proj_drop,
                               drop_path=drop_path, scale=scale, l2_norm=l2_norm, num_heads=num_heads)
        self.neck.init_weights(init_linear='kaiming')

    def forward(self, im, mask=None, attn_mask=False):
        _, f = self.backbone(im)
        out = self.neck(f, mask=mask, attn_mask=attn_mask)
        return out

class SkeAttnMask_TR(nn.Module):
    """ Referring to the code of MOCO, https://arxiv.org/abs/1911.05722 """

    def __init__(self, base_encoder=None, pretrain=True, feature_dim=128, queue_size=32768, momentum=0.996, Temperature=0.2,
                 num_frame=64, num_joint=25, input_channel=3, hidden_dim=256, num_class=60, config=None, num_heads=8,
                 dropout=0.5,  Lambda=2, loss_weight=0.5, sigma=0.5, spa_l=2, spa_u=3, tem_l=7, tem_u=11,
                 swap_mode='swap', spatial_mode='semantic', **kwargs):
        """
        K: queue size; number of negative keys (default: 32768)
        m: momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """

        super().__init__()
        base_encoder = import_class(base_encoder)
        self.pretrain = pretrain
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
        
        if not self.pretrain:
            self.encoder_q = EncoderObj_TR(base_encoder=base_encoder, hid_dim=hidden_dim, out_dim=feature_dim,
                                           num_channel=input_channel, num_class=num_class, num_point=num_joint,
                                           num_frame=num_frame, dropout=dropout, config=config, num_heads=num_heads,
                                           Lambda=Lambda,
                                           **kwargs)
        else:
            self.encoder_q = EncoderObj_TR(base_encoder=base_encoder, hid_dim=hidden_dim, out_dim=feature_dim,
                                           num_channel=input_channel, num_class=num_class, num_point=num_joint,
                                           num_frame=num_frame, dropout=dropout, config=config, num_heads=num_heads,
                                           Lambda=Lambda,
                                           **kwargs)
            self.encoder_k = EncoderObj_TR(base_encoder=base_encoder, hid_dim=hidden_dim, out_dim=feature_dim,
                                           num_channel=input_channel, num_class=num_class, num_point=num_joint,
                                           num_frame=num_frame, dropout=dropout, config=config, num_heads=num_heads,
                                           Lambda=Lambda,
                                           **kwargs)

            self.predictor = MLP1D(feature_dim, hidden_dim, feature_dim)
            self.predictor.init_weights()
            self.predictor_p = MLP1D(feature_dim, hidden_dim, feature_dim)
            self.predictor_p.init_weights()
            self.predictor_c = MLP1D(feature_dim, hidden_dim, feature_dim)
            self.predictor_c.init_weights()
            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)    # initialize
                param_k.requires_grad = False       # not update by gradient

            # create the queue
            self.register_buffer("queue_z", torch.randn(feature_dim, queue_size))
            self.queue_z = F.normalize(self.queue_z, dim=0)
            self.register_buffer("queue_ptr_z", torch.zeros(1, dtype=torch.long))

            # create the queue
            self.register_buffer("queue_p", torch.randn(feature_dim, queue_size))
            self.queue_p = F.normalize(self.queue_p, dim=0)
            self.register_buffer("queue_ptr_p", torch.zeros(1, dtype=torch.long))

            # create the queue
            self.register_buffer("queue_c", torch.randn(feature_dim, queue_size))
            self.queue_c = F.normalize(self.queue_c, dim=0)
            self.register_buffer("queue_ptr_c", torch.zeros(1, dtype=torch.long))
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
    def _dequeue_and_enqueue(self, key0, key1, key2):
        bs0 = key0.shape[0]
        bs1 = key1.shape[0]
        bs2 = key2.shape[0]
        ptr_z = int(self.queue_ptr_z)
        ptr_p = int(self.queue_ptr_p)
        ptr_c = int(self.queue_ptr_c)
        gpu_index0 = key0.device.index
        gpu_index1 = key1.device.index
        gpu_index2 = key2.device.index
        self.queue_z[:, (ptr_z + bs0 * gpu_index0):(ptr_z + bs0 * (gpu_index0 + 1))] = key0.T
        self.queue_p[:, (ptr_p + bs1 * gpu_index1):(ptr_p + bs1 * (gpu_index1 + 1))] = key1.T
        self.queue_c[:, (ptr_c + bs2 * gpu_index2):(ptr_c + bs2 * (gpu_index2 + 1))] = key2.T

    @torch.no_grad()
    def update_ptr(self, batch_size):
        assert self.K % batch_size == 0 #  for simplicity
        self.queue_ptr_z[0] = (self.queue_ptr_z[0] + batch_size) % self.K
        self.queue_ptr_p[0] = (self.queue_ptr_p[0] + batch_size) % self.K
        self.queue_ptr_c[0] = (self.queue_ptr_c[0] + batch_size) % self.K

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
    @torch.no_grad()
    def view_gen(self, data, view):
        if view == 'joint':
            pass
        elif view == 'motion':
            motion = torch.zeros_like(data)

            motion[:, :, :-1, :, :] = data[:, :, 1:, :, :] - data[:, :, :-1, :, :]

            data = motion
        elif view == 'bone':
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
    
    def forward(self, xq, xk=None, view='joint'):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        """
        _, x_swap, _ = self.ske_swap(xq)
        xq = self.view_gen(xq, view)
        if not self.pretrain:
            q, _ = self.encoder_q.backbone(xq)
            return q
        xk = self.view_gen(xk, view)
        x_swap = self.view_gen(x_swap, view)
        # compute query features
        q_z = self.encoder_q(xq)  # queries: NxC
        q_z = F.normalize(self.predictor(q_z).squeeze(-1), dim=1)

        q_z_swap, q_s, q_ns, mask = self.encoder_q(x_swap, attn_mask=True)  # queries: NxC
        q_z_swap = F.normalize(self.predictor(q_z_swap).squeeze(-1), dim=1)
        q_s = F.normalize(self.predictor_p(q_s).squeeze(-1), dim=1)
        q_ns = F.normalize(self.predictor_c(q_ns).squeeze(-1), dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # self._momentum_update_key_encoder()  # update the key encoder
            k = [x.clone().detach() for x in self.encoder_k(xk, mask=mask)]
            k_z, k_s, k_ns = k
            k_z = F.normalize(k_z.squeeze(-1), dim=1)
            k_s = F.normalize(k_s.squeeze(-1), dim=1)
            k_ns = F.normalize(k_ns.squeeze(-1), dim=1)


        l_pos_z = torch.einsum('nc,nc->n', [q_z, k_z]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_z = torch.einsum('nc,ck->nk', [q_z, self.queue_z.clone().detach()])
        # logits: Nx(1+K)
        logits_z = torch.cat([l_pos_z, l_neg_z], dim=1)
        topk_onehot_z = torch.zeros_like(l_neg_z)
        pos_mask_z = torch.cat([torch.ones(topk_onehot_z.size(0), 1).cuda(), topk_onehot_z], dim=1)
        # apply temperature
        logits_z /= self.T

        # Loss region salient
        l_pos = torch.einsum('nc,nc->n', [q_s, k_s]).unsqueeze(-1)
        l_neg_1 = torch.einsum('nc,ck->nk', [q_s, self.queue_z.clone().detach()])
        l_neg_2 = torch.einsum('nc,nc->n', [q_s, q_ns.clone().detach()]).unsqueeze(-1)
        logits_reg_1 = torch.cat([l_pos, l_neg_1, l_neg_2], dim=1)
        logits_reg_1 /= self.T

        # Loss region non-salient
        l_pos = torch.einsum('nc,nc->n', [q_ns, k_ns]).unsqueeze(-1)
        l_neg_1 = torch.einsum('nc,ck->nk', [q_ns, self.queue_z.clone().detach()])
        l_neg_2 = torch.einsum('nc,nc->n', [q_ns, q_s.clone().detach()]).unsqueeze(-1)
        logits_reg_2 = torch.cat([l_pos, l_neg_1, l_neg_2], dim=1)
        logits_reg_2 /= self.T

        topk_onehot_neg_p = torch.zeros_like(l_neg_1)
        topk_onehot_neg_c = torch.zeros_like(l_neg_2)

        pos_mask_reg1 = torch.cat([torch.ones(topk_onehot_z.size(0), 1).cuda(), topk_onehot_neg_p, topk_onehot_neg_c],
                                  dim=1)
        pos_mask_reg2 = torch.cat([torch.ones(topk_onehot_z.size(0), 1).cuda(), topk_onehot_neg_p, topk_onehot_neg_c],
                                  dim=1)

        loss_z = multi_nce_loss(logits_z, pos_mask_z)
        loss_zp = multi_nce_loss(logits_reg_1, pos_mask_reg1)
        loss_zc = multi_nce_loss(logits_reg_2, pos_mask_reg2)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k_z, k_s, k_ns)
        return loss_z, loss_zp, loss_zc