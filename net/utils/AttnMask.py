from math import cos, pi
import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.models.vision_transformer import Attention
from timm.models.layers import DropPath
from torch.nn.modules import loss
from net.utils import init
from net.utils.att_drop import Simam_Drop
from einops import rearrange
import numbers

class MLP1D(nn.Module):
    """
    The non-linear neck in byol: fc-bn-relu-fc
    """

    def __init__(self, in_channels, hid_channels, out_channels,
                 norm_layer=None, bias=False, num_mlp=2):
        super(MLP1D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        mlps = []
        for _ in range(num_mlp - 1):
            mlps.append(nn.Conv1d(in_channels, hid_channels, 1, bias=bias))
            mlps.append(norm_layer(hid_channels))
            mlps.append(nn.ReLU(inplace=True))
            in_channels = hid_channels
        mlps.append(nn.Conv1d(hid_channels, out_channels, 1, bias=bias))
        self.mlp = nn.Sequential(*mlps)

    def init_weights(self, init_linear='normal'):
        init.init_weights(self, init_linear)

    def forward(self, x):
        x = self.mlp(x)
        return x

class MLP(nn.Module):
    """
    The non-linear neck in byol: fc-bn-relu-fc
    """
    def __init__(self, dim_mlp, fc):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(dim_mlp, 2*dim_mlp),
                            nn.BatchNorm1d(2*dim_mlp),
                            nn.ReLU(),
                            fc)
    def forward(self, x):
        x = self.mlp(x)
        return x


class Attention_MaskV2(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = \
                torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index", relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None, return_attention=False, return_qkv=False):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple) (B, H, N, C)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.relative_position_bias_table is not None:
            relative_position_bias = \
                self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if return_attention:
            return attn

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        mask = F.sigmoid(2 * x)
        if return_qkv:
            return mask, qkv

        return mask

class Flow_Attention_Mask(nn.Module):
    def __init__(self, dim, num_heads=12):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def kernel(self, x):
        x = torch.sigmoid(x)
        return x

    def my_sum(self, a, b):
        # "nhld,nhd->nhl"
        return torch.sum(a * b[:, :, None, :], dim=-1)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # kernel
        q, k = self.kernel(q), self.kernel(k)
        # normalizer
        sink_incoming = 1.0 / (self.my_sum(q + 1e-6, k.sum(dim=2) + 1e-6) + 1e-6)
        source_outgoing = 1.0 / (self.my_sum(k + 1e-6, q.sum(dim=2) + 1e-6) + 1e-6)
        conserved_sink = self.my_sum(q + 1e-6, (k * source_outgoing[:, :, :, None]).sum(dim=2) + 1e-6) + 1e-6
        conserved_source = self.my_sum(k + 1e-6, (q * sink_incoming[:, :, :, None]).sum(dim=2) + 1e-6) + 1e-6
        conserved_source = torch.clamp(conserved_source, min=-1.0, max=1.0)  # for stability
        # allocation
        sink_allocation = torch.sigmoid(conserved_sink * (float(q.shape[2]) / float(k.shape[2])))
        # competition
        source_competition = torch.softmax(conserved_source, dim=-1) * float(k.shape[2])
        # multiply
        kv = k.transpose(-2, -1) @ (v * source_competition[:, :, :, None])
        x_update = ((q @ kv) * sink_incoming[:, :, :, None]) * sink_allocation[:, :, :, None]
        x = (x_update).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        mask = F.sigmoid(2*x)
        # mask = torch.where(mask > 0.5, torch.ones_like(mask), torch.zeros_like(mask))
        return mask

#for SkeMix
class Attention_MASK(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., Lambda=2):
        super().__init__()
        self.num_heads = num_heads
        self.Lambda = Lambda
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        mask = F.sigmoid(self.Lambda*x)
        return x, mask

#for SkeAttnMask
class ObjectNeck_AM(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hid_channels=None,
                 num_layers=1,
                 scale=1.,
                 l2_norm=True,
                 num_heads=8,
                 norm_layer=None,
                 qkv_bias=True,
                 attn_drop=0.,
                 proj_drop=0.,
                 drop_path=0.1,
                 Lambda=2,
                 **kwargs):
        super(ObjectNeck_AM, self).__init__()

        self.scale = scale
        self.l2_norm = l2_norm
        assert l2_norm
        self.num_heads = num_heads

        hid_channels = hid_channels or in_channels
        self.norm0 = nn.LayerNorm(in_channels)
        self.Attn = Attention_MASK(dim=in_channels, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                   proj_drop=proj_drop, Lambda=Lambda)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.proj = MLP1D(in_channels, hid_channels, out_channels, norm_layer, num_mlp=num_layers)
        self.proj_obj = MLP1D(in_channels, hid_channels, out_channels, norm_layer, num_mlp=num_layers)
        self.dropout = Simam_Drop(num_point=25, keep_prob=0.7)

    def init_weights(self, init_linear='kaiming'):
        self.proj.init_weights(init_linear)
        self.proj_obj.init_weights(init_linear)

    def _mask_pooling(self, x, mask):
        '''
        :param x:       B N C
        :param mask:    B N C (bool) # paste True
        :return:        p [B C 1]   c [B C 1]
        '''
        mask = mask.permute(0, 2, 1)
        p = (x * mask)
        p = F.adaptive_avg_pool1d(p, 1)

        ones = torch.ones_like(mask).to(mask.device)
        _mask = ones - mask

        c = (x * _mask)
        c = F.adaptive_avg_pool1d(c, 1)
        return p, c

    def forward(self, x, mask=None, attn_mask=False):
        b, c, h, w, m = x.shape
        x = x.view(b*m, c, h, w)
        x_pool = F.adaptive_avg_pool2d(x, 1).flatten(2)
        x_pool = x_pool.view(b, m, c, -1).mean(dim=1)
        z_x = self.proj(x_pool)
        if attn_mask:
            x = x.flatten(2)  # (b*m, c, h*w)
            _, attn_mask = self.Attn(self.norm0(x.permute(0, 2, 1)))
            z_p, z_c = self._mask_pooling(x, attn_mask)
            # print(p.shape), print(c.shape)
            z_p, z_c = z_p.view(b, m, c).mean(dim=1).unsqueeze(-1), z_c.view(b, m, c).mean(dim=1).unsqueeze(-1)
            z_p, z_c = self.proj(z_p), self.proj(z_c)
            return z_x, z_p, z_c, attn_mask
        elif mask is not None:
            x = x.flatten(2)  # (b*m, c, h*w)
            z_p, z_c = self._mask_pooling(x, mask)
            z_p, z_c = z_p.view(b, m, c).mean(dim=1).unsqueeze(-1), z_c.view(b, m, c).mean(dim=1).unsqueeze(-1)
            z_p, z_c = self.proj(z_p), self.proj(z_c)
            return z_x, z_p, z_c
        else:
            return z_x

    def extra_repr(self) -> str:
        parts = []
        for name in ["scale", "l2_norm", "num_heads"]:
            parts.append(f"{name}={getattr(self, name)}")
        return ", ".join(parts)

#for SkeAttnMask
class ObjectNeck_AMV2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hid_channels=None,
                 num_layers=1,
                 scale=1.,
                 l2_norm=True,
                 num_heads=8,
                 norm_layer=None,
                 qkv_bias=True,
                 attn_drop=0.,
                 proj_drop=0.,
                 drop_path=0.1,
                 **kwargs):
        super(ObjectNeck_AMV2, self).__init__()

        self.scale = scale
        self.l2_norm = l2_norm
        assert l2_norm
        self.num_heads = num_heads

        hid_channels = hid_channels or in_channels
        self.norm0 = nn.LayerNorm(in_channels)
        self.Attn = Attention_MASK(dim=in_channels, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                   proj_drop=proj_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.proj = MLP1D(in_channels, hid_channels, out_channels, norm_layer, num_mlp=num_layers)
        self.proj_obj = MLP1D(in_channels, hid_channels, out_channels, norm_layer, num_mlp=num_layers)
        self.dropout = Simam_Drop(num_point=25, keep_prob=0.7)

    def init_weights(self, init_linear='kaiming'):
        self.proj.init_weights(init_linear)
        self.proj_obj.init_weights(init_linear)

    def _mask_pooling(self, x, mask):
        '''
        :param x:       B N C
        :param mask:    B N C (bool) # paste True
        :return:        p [B C 1]   c [B C 1]
        '''
        mask = mask.permute(0, 2, 1)
        p = (x * mask)
        p = F.adaptive_avg_pool1d(p, 1)

        ones = torch.ones_like(mask).to(mask.device)
        _mask = ones - mask

        c = (x * _mask)
        c = F.adaptive_avg_pool1d(c, 1)
        return p, c

    def forward(self, x, mask=None, mask_flag=False):
        b, c, h, w, m = x.shape
        x = x.view(b*m, c, h, w)
        x_pool = F.adaptive_avg_pool2d(x, 1).flatten(2)
        x_pool = x_pool.view(b, m, c, -1).mean(dim=1)
        z_x = self.proj(x_pool)

        x = x.flatten(2)  # (b*m, c, h*w)
        x_attn, attn_mask = self.Attn(self.norm0(x.permute(0, 2, 1)))
        obj_val = x.permute(0, 2, 1) + self.drop_path(x_attn)
        obj_val = F.adaptive_avg_pool1d(obj_val.permute(0, 2, 1), 1)
        obj_val = obj_val.view(b, m, -1).mean(dim=1)
        # print(obj_val.shape)
        obj_val = self.proj_obj(obj_val.unsqueeze(-1))
        if mask_flag:
            z_p, z_c = self._mask_pooling(x, attn_mask)
            # print(p.shape), print(c.shape)
            z_p, z_c = z_p.view(b, m, c).mean(dim=1).unsqueeze(-1), z_c.view(b, m, c).mean(dim=1).unsqueeze(-1)
            z_p, z_c = self.proj(z_p), self.proj(z_c)
            return z_x, z_p, z_c, attn_mask
        elif mask is not None:
            z_p, z_c = self._mask_pooling(x, mask)
            z_p, z_c = z_p.view(b, m, c).mean(dim=1).unsqueeze(-1), z_c.view(b, m, c).mean(dim=1).unsqueeze(-1)
            z_p, z_c = self.proj(z_p), self.proj(z_c)
            return z_x, obj_val, z_p, z_c
        else:
            return z_x, obj_val

    def extra_repr(self) -> str:
        parts = []
        for name in ["scale", "l2_norm", "num_heads"]:
            parts.append(f"{name}={getattr(self, name)}")
        return ", ".join(parts)

#for BIGRU
class ObjectNeck_GRU(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hid_channels=None,
                 num_layers=1,
                 scale=1.,
                 l2_norm=True,
                 num_heads=8,
                 norm_layer=None,
                 qkv_bias=True,
                 attn_drop=0.,
                 proj_drop=0.,
                 drop_path=0.1,
                 Lambda=2,
                 **kwargs):
        super(ObjectNeck_GRU, self).__init__()

        self.scale = scale
        self.l2_norm = l2_norm
        assert l2_norm
        self.num_heads = num_heads

        hid_channels = hid_channels or in_channels
        self.norm0 = nn.LayerNorm(in_channels)
        self.Attn = Attention_MASK(dim=in_channels, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                   proj_drop=proj_drop, Lambda=Lambda)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.proj = MLP1D(in_channels, hid_channels, out_channels, norm_layer, num_mlp=num_layers)
        self.proj_obj = MLP1D(in_channels, hid_channels, out_channels, norm_layer, num_mlp=num_layers)
        self.dropout = Simam_Drop(num_point=25, keep_prob=0.7)

    def init_weights(self, init_linear='kaiming'):
        self.proj.init_weights(init_linear)
        self.proj_obj.init_weights(init_linear)

    def _mask_pooling(self, x, mask):
        '''
        :param x:       B N C
        :param mask:    B N C  # paste True
        :return:        p [B N C]   c [B N C]
        '''
        # print(mask)
        p = x * mask
        # p = F.adaptive_avg_pool1d(p, 1)

        ones = torch.ones_like(mask).to(mask.device)
        _mask = ones - mask

        c = x * _mask
        # c = F.adaptive_avg_pool1d(c, 1)
        return p, c
    def _GRU_pooling(self, x):
        b, T, c = x.shape
        seq_len = torch.zeros(b, dtype=int) + T
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hidden = torch.empty((1, len(seq_len), x.shape[-1])).to(device)
        count = 0
        for ith_len in seq_len:
            hidden[0, count, :] = x[count, ith_len - 1, :]
            count += 1
        return hidden
    def forward(self, x, mask=None, attn_mask=False):
        hidden = self._GRU_pooling(x)
        z_x = self.proj(hidden[0].unsqueeze(-1))
        if attn_mask:
            _, am = self.Attn(self.norm0(x))
            z_p, z_c = self._mask_pooling(x, am)
            hidden_p, hidden_c = self._GRU_pooling(z_p), self._GRU_pooling(z_c)
            z_p, z_c = self.proj(hidden_p[0].unsqueeze(-1)), self.proj(hidden_c[0].unsqueeze(-1))
            return z_x, z_p, z_c, am
        elif mask is not None:
            z_p, z_c = self._mask_pooling(x, mask)
            hidden_p, hidden_c = self._GRU_pooling(z_p), self._GRU_pooling(z_c)
            z_p, z_c = self.proj(hidden_p[0].unsqueeze(-1)), self.proj(hidden_c[0].unsqueeze(-1))
            return z_x, z_p, z_c
        else:
            return z_x

    def extra_repr(self) -> str:
        parts = []
        for name in ["scale", "l2_norm", "num_heads"]:
            parts.append(f"{name}={getattr(self, name)}")
        return ", ".join(parts)
    
#for transformer
class ObjectNeck_TR(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hid_channels=None,
                 num_layers=1,
                 scale=1.,
                 l2_norm=True,
                 num_heads=8,
                 norm_layer=None,
                 qkv_bias=True,
                 attn_drop=0.,
                 proj_drop=0.,
                 drop_path=0.1,
                 **kwargs):
        super(ObjectNeck_TR, self).__init__()

        self.scale = scale
        self.l2_norm = l2_norm
        assert l2_norm
        self.num_heads = num_heads

        hid_channels = hid_channels or in_channels
        self.norm0 = nn.LayerNorm(in_channels)
        self.Attn = Attention_MASK(dim=in_channels, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                   proj_drop=proj_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.proj = MLP1D(in_channels, hid_channels, out_channels, norm_layer, num_mlp=num_layers)
        self.proj_obj = MLP1D(in_channels, hid_channels, out_channels, norm_layer, num_mlp=num_layers)
        self.dropout = Simam_Drop(num_point=25, keep_prob=0.7)

    def init_weights(self, init_linear='kaiming'):
        self.proj.init_weights(init_linear)
        self.proj_obj.init_weights(init_linear)

    def _mask_pooling(self, x, mask):
        '''
        :param x:       B N C
        :param mask:    B N C (bool) # paste True
        :return:        p [B C 1]   c [B C 1]
        '''
        mask = mask.permute(0, 2, 1)
        p = (x * mask)
        p = F.adaptive_avg_pool1d(p, 1)

        ones = torch.ones_like(mask).to(mask.device)
        _mask = ones - mask

        c = (x * _mask)
        c = F.adaptive_avg_pool1d(c, 1)
        return p, c

    def forward(self, x, mask=None, attn_mask=False):
        b, h, w, c = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()
        x_pool = F.adaptive_avg_pool2d(x, 1).flatten(2)
        z_x = self.proj(x_pool)
        if attn_mask:
            x = x.flatten(2)  # (b, c, h*w)
            _, attn_mask = self.Attn(self.norm0(x.permute(0, 2, 1)))
            z_p, z_c = self._mask_pooling(x, attn_mask)
            # print(p.shape), print(c.shape)
            # print(z_p.shape)
            z_p, z_c = self.proj(z_p), self.proj(z_c)
            return z_x, z_p, z_c, attn_mask
        elif mask is not None:
            x = x.flatten(2)  # (b*m, c, h*w)
            z_p, z_c = self._mask_pooling(x, mask)
            # z_p, z_c = z_p.unsqueeze(-1), z_c.unsqueeze(-1)
            z_p, z_c = self.proj(z_p), self.proj(z_c)
            return z_x, z_p, z_c
        else:
            return z_x

    def extra_repr(self) -> str:
        parts = []
        for name in ["scale", "l2_norm", "num_heads"]:
            parts.append(f"{name}={getattr(self, name)}")
        return ", ".join(parts)
    
if __name__ == '__main__':
    # mask = torch.zeros(32, 400, 128)
    # mask = torch.zeros(16, 25)
    # mask[5:13, [2, 5, 9, 12]] = 1
    # mask = mask.bool()
    x = torch.randn((16, 256, 16, 25, 2))
    # x = torch.randn((16, 64, 2048))
    test = ObjectNeck_TR(in_channels=256)
    z, p, c, mask = test(x, attn_mask=True)
    # z, p, c = test(x, mask=mask)
    print(z.shape, p.shape, c.shape)
    # test = ObjectNeck_GRU(in_channels=2048, out_channels=128)
    # out0, out1, out2 = test(x, mask=mask)
    # print(out0.shape)
    # print(out1.shape)
    # print(out2.shape)
