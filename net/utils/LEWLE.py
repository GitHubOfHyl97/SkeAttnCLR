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
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]

        return to_4d(self.body(to_3d(x)), h, w)

#与卷积结合
class Attention_V2(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention_V2, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        #Spacial-Temporal Dconv
        kernel_size = (3, 5)
        pads = int((kernel_size[1] - 1) / 2)
        padt = int((kernel_size[0] - 1) / 2)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=(1, 1), bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=kernel_size, stride=(1, 1), padding=(padt, pads), groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=(1, 1), bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        # print(qkv.shape)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # print(q.shape)
        # print(k.shape)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        # print(out.shape)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

#for SkeMix
class Attention_MASK(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
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
        mask = F.sigmoid(2*x)
        # mask = torch.where(mask > 0.5, torch.ones_like(mask), torch.zeros_like(mask))
        return mask

class ObjectNeck(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hid_channels=None,
                 num_layers=1,
                 scale=1.,
                 l2_norm=True,
                 num_heads=8,
                 norm_layer=None,
                 **kwargs):
        super(ObjectNeck, self).__init__()

        self.scale = scale
        self.l2_norm = l2_norm
        assert l2_norm
        self.num_heads = num_heads

        hid_channels = hid_channels or in_channels
        self.proj = MLP1D(in_channels, hid_channels, out_channels, norm_layer, num_mlp=num_layers)
        self.proj_obj = MLP1D(in_channels, hid_channels, out_channels, norm_layer, num_mlp=num_layers)

    def init_weights(self, init_linear='kaiming'):
        self.proj.init_weights(init_linear)
        self.proj_obj.init_weights(init_linear)

    def forward(self, x):
        b, c, h, w, m = x.shape
        x = x.view(b*m, c, h, w)
        # flatten and projection
        x_pool = F.adaptive_avg_pool2d(x, 1).flatten(2)
        x = x.flatten(2)  # (bs, c, h*w*m)
        x = x.view(b, m, c, -1).mean(dim=1)
        x_pool = x_pool.view(b, m, c, -1).mean(dim=1)
        z = self.proj(torch.cat([x_pool, x], dim=2))
        # print(z.shape)

        z_g, obj_attn = torch.split(z, [1, x.shape[2]], dim=2)  # (bs, nH*k, 1), (bs, nH*k, h*w)# (bs, k, 1+h*w)
        # do attention according to obj attention map
        obj_attn = F.normalize(obj_attn, dim=1) if self.l2_norm else obj_attn
        obj_attn /= self.scale
        obj_attn = F.softmax(obj_attn, dim=2)
        obj_attn = obj_attn.view(b, self.num_heads, -1, h * w)
        x = x.view(b, self.num_heads, -1, h * w)
        obj_val = torch.matmul(x, obj_attn.transpose(3, 2))  # (bs, nH, c//Nh, k)
        obj_val = obj_val.view(b, c, obj_attn.shape[-2])  # (bs, c, k)
        # projection
        obj_val = self.proj_obj(obj_val)  # (bs, c, k)
        return z_g, obj_val  # (bs, c, 1), (bs, c, k), where the second dim is channel

    def extra_repr(self) -> str:
        parts = []
        for name in ["scale", "l2_norm", "num_heads"]:
            parts.append(f"{name}={getattr(self, name)}")
        return ", ".join(parts)

#for SkeAttnMask
class ObjectNeck_V2(nn.Module):
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
        super(ObjectNeck_V2, self).__init__()

        self.scale = scale
        self.l2_norm = l2_norm
        assert l2_norm
        self.num_heads = num_heads

        hid_channels = hid_channels or in_channels
        self.norm0 = nn.LayerNorm(in_channels)
        self.Attn = Attention_MASK(dim=in_channels, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
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
        :return:        p [B N C]   c [B N C]
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
            attn_mask = self.Attn(self.norm0(x.permute(0, 2, 1)))
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

#用单独的Multi heads self-Attention机制来改进LWL
class ObjectNeck_V3(nn.Module):
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
        super(ObjectNeck_V3, self).__init__()

        self.scale = scale
        self.l2_norm = l2_norm
        assert l2_norm
        self.num_heads = num_heads

        hid_channels = hid_channels or in_channels
        self.norm0 = LayerNorm(in_channels, 'withbias')
        self.Attn = Attention_V2(dim=in_channels, num_heads=num_heads, bias=qkv_bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.proj = MLP1D(in_channels, hid_channels, out_channels, norm_layer, num_mlp=num_layers)
        self.proj_obj = MLP1D(in_channels, hid_channels, out_channels, norm_layer, num_mlp=num_layers)
        self.dropout = Simam_Drop(num_point=25, keep_prob=0.7)

    def init_weights(self, init_linear='kaiming'):
        self.proj.init_weights(init_linear)
        self.proj_obj.init_weights(init_linear)

    def _mask_pool(self, x, mask):
        '''
        :param x:       N C T V
        :param mask:    T V (bool) # paste True
        :return:        p [N C]   c [N C]
        '''
        # print((x * mask).shape)
        # print((x * (~mask)).shape)
        p = (x * mask).sum(-1).sum(-1) / mask.sum()
        c = (x * (~mask)).sum(-1).sum(-1) / (~mask).sum()
        return p, c

    def forward(self, x, drop=False, mask=None):
        b, c, h, w, m = x.shape
        x = x.view(b*m, c, h, w)
        if drop:
            y = self.dropout(x)
            y_pool = F.adaptive_avg_pool2d(y, 1).flatten(2)
            y_pool = y_pool.view(b, m, c, -1).mean(dim=1)
            z_y = self.proj(y_pool)

            x_pool = F.adaptive_avg_pool2d(x, 1).flatten(2)
            x_pool = x_pool.view(b, m, c, -1).mean(dim=1)
            z_x = self.proj(x_pool)

            attn = self.Attn(self.norm0(x))
            obj_val = x + self.drop_path(attn)
            obj_val = F.adaptive_avg_pool2d(obj_val, 1).flatten(2)
            obj_val = obj_val.view(b, m, c, -1).mean(dim=1)
            obj_val = self.proj_obj(obj_val)
            return z_x, z_y, obj_val
        elif mask is not None:
            p, c_ = self._mask_pool(x, mask)
            # print(p.shape), print(c.shape)
            p, c_ = p.view(b, m, -1).mean(dim=1).unsqueeze(-1), c_.view(b, m, -1).mean(dim=1).unsqueeze(-1)
            z_p, z_c = self.proj(p), self.proj(c_)

            attn = self.Attn(self.norm0(x))
            obj_val = x + self.drop_path(attn)
            obj_val = F.adaptive_avg_pool2d(obj_val, 1).flatten(2)
            obj_val = obj_val.view(b, m, c, -1).mean(dim=1)
            obj_val = self.proj_obj(obj_val)
            return z_p, z_c, obj_val
        else:
            x_pool = F.adaptive_avg_pool2d(x, 1).flatten(2)
            x_pool = x_pool.view(b, m, c, -1).mean(dim=1)
            z_x = self.proj(x_pool)

            attn = self.Attn(self.norm0(x))
            obj_val = x + self.drop_path(attn)
            obj_val = F.adaptive_avg_pool2d(obj_val, 1).flatten(2)
            obj_val = obj_val.view(b, m, c, -1).mean(dim=1)
            obj_val = self.proj_obj(obj_val)
            return z_x, obj_val

#用单独的Multi heads self-Attention机制来改进LWL
class ObjectNeck_V4(nn.Module):
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
        super(ObjectNeck_V4, self).__init__()

        self.scale = scale
        self.l2_norm = l2_norm
        assert l2_norm
        self.num_heads = num_heads

        hid_channels = hid_channels or in_channels
        self.norm0 = nn.LayerNorm(in_channels)
        self.Attn = Attention(dim=in_channels, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.proj = MLP1D(in_channels, hid_channels, out_channels, norm_layer, num_mlp=num_layers)
        self.proj_obj = MLP1D(in_channels, hid_channels, out_channels, norm_layer, num_mlp=num_layers)
        self.dropout = Simam_Drop(num_point=25, keep_prob=0.7)

    def init_weights(self, init_linear='kaiming'):
        self.proj.init_weights(init_linear)
        self.proj_obj.init_weights(init_linear)

    def _mask_pool(self, x, mask):
        '''
        :param x:       N C T V
        :param mask:    T V (bool) # paste True
        :return:        p [N C]   c [N C]
        '''
        p = (x * mask).sum(-1).sum(-1) / mask.sum()
        c = (x * (~mask)).sum(-1).sum(-1) / (~mask).sum()
        return p, c

    def forward(self, x, drop=False, mask=None):
        b, c, h, w, m = x.shape
        x = x.view(b*m, c, h, w)
        if drop:
            y = self.dropout(x)
            y_pool = F.adaptive_avg_pool2d(y, 1).flatten(2)
            y_pool = y_pool.view(b, m, c, -1).mean(dim=1)
            z_y = self.proj(y_pool)
            # flatten and projection
            x_pool = F.adaptive_avg_pool2d(x, 1).flatten(2)
            x = x.flatten(2)  # (bs, c, h*w*m)
            x = x.view(b, m, c, -1).mean(dim=1)
            x_pool = x_pool.view(b, m, c, -1).mean(dim=1)
            z_x = self.proj(x_pool)
            attn = self.Attn(self.norm0(x.permute(0, 2, 1)))
            obj_val = x.permute(0, 2, 1) + self.drop_path(attn)
            obj_val = F.adaptive_avg_pool1d(obj_val.permute(0, 2, 1), 1)
            # print(obj_val.shape)
            obj_val = self.proj_obj(obj_val)
            return z_x, z_y, obj_val
        elif mask is not None:
            p, c = self._mask_pool(x, mask)
            # print(p.shape), print(c.shape)
            p, c = p.view(b, m, -1).mean(dim=1).unsqueeze(-1), c.view(b, m, -1).mean(dim=1).unsqueeze(-1)
            z_p, z_c = self.proj(p), self.proj(c)
            x = x.flatten(2)  # (bs, c, h*w*m)
            # print(x.shape)
            x = x.view(b, m, x.shape[1], -1).mean(dim=1)
            attn = self.Attn(self.norm0(x.permute(0, 2, 1)))
            obj_val = x.permute(0, 2, 1) + self.drop_path(attn)
            obj_val = F.adaptive_avg_pool1d(obj_val.permute(0, 2, 1), 1)
            obj_val = self.proj_obj(obj_val)
            return z_p, z_c, obj_val
        else:
            x_pool = F.adaptive_avg_pool2d(x, 1).flatten(2)
            x = x.flatten(2)  # (bs, c, h*w*m)
            x = x.view(b, m, c, -1).mean(dim=1)
            x_pool = x_pool.view(b, m, c, -1).mean(dim=1)
            z_x = self.proj(x_pool)
            attn = self.Attn(self.norm0(x.permute(0, 2, 1)))
            obj_val = x.permute(0, 2, 1) + self.drop_path(attn)
            obj_val = F.adaptive_avg_pool1d(obj_val.permute(0, 2, 1), 1)
            obj_val = self.proj_obj(obj_val)
            return z_x, obj_val

    def extra_repr(self) -> str:
        parts = []
        for name in ["scale", "l2_norm", "num_heads"]:
            parts.append(f"{name}={getattr(self, name)}")
        return ", ".join(parts)

class ObjectNeck_Sim(nn.Module):
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
        super(ObjectNeck_Sim, self).__init__()

        self.scale = scale
        self.l2_norm = l2_norm
        assert l2_norm
        self.num_heads = num_heads

        hid_channels = hid_channels or in_channels
        self.norm0 = nn.LayerNorm(in_channels)
        self.Attn = Attention(dim=in_channels, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.proj = MLP1D(in_channels, hid_channels, out_channels, norm_layer, num_mlp=num_layers)
        self.proj_obj = MLP1D(in_channels, hid_channels, out_channels, norm_layer, num_mlp=num_layers)
        self.dropout = Simam_Drop(num_point=25, keep_prob=0.7)

    def init_weights(self, init_linear='kaiming'):
        # self.proj.init_weights(init_linear)
        self.proj_obj.init_weights(init_linear)

    def forward(self, x):
        b, c, h, w, m = x.shape
        x = x.view(b*m, c, h, w)
        x = x.flatten(2)  # (bs, c, h*w*m)
        x = x.view(b, m, c, -1).mean(dim=1)
        attn = self.Attn(self.norm0(x.permute(0, 2, 1)))
        obj_val = x.permute(0, 2, 1) + self.drop_path(attn)
        obj_val = F.adaptive_avg_pool1d(obj_val.permute(0, 2, 1), 1)
        obj_val = self.proj_obj(obj_val)
        return obj_val

    def extra_repr(self) -> str:
        parts = []
        for name in ["scale", "l2_norm", "num_heads"]:
            parts.append(f"{name}={getattr(self, name)}")
        return ", ".join(parts)

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
                 **kwargs):
        super(ObjectNeck_GRU, self).__init__()

        self.scale = scale
        self.l2_norm = l2_norm
        assert l2_norm
        self.num_heads = num_heads

        hid_channels = hid_channels or in_channels
        self.proj = MLP1D(in_channels, hid_channels, out_channels, norm_layer, num_mlp=num_layers)
        self.proj_obj = MLP1D(in_channels, hid_channels, out_channels, norm_layer, num_mlp=num_layers)

    def init_weights(self, init_linear='kaiming'):
        self.proj.init_weights(init_linear)
        self.proj_obj.init_weights(init_linear)

    def forward(self, x):
        b, T, c = x.shape
        x = x.transpose(1, 2)
        # flatten and projection
        x_pool = F.adaptive_avg_pool2d(x.unsqueeze(-1), 1).squeeze(-1)

        z = self.proj(torch.cat([x_pool, x], dim=2))
        z_g, obj_attn = torch.split(z, [1, x.shape[2]], dim=2)  # (bs, nH*k, 1), (bs, nH*k, h*w)# (bs, k, 1+h*w)

        # do attention according to obj attention map
        obj_attn = F.normalize(obj_attn, dim=1) if self.l2_norm else obj_attn
        obj_attn /= self.scale
        obj_attn = F.softmax(obj_attn, dim=2)
        obj_attn = obj_attn.view(b, self.num_heads, -1, T)
        x = x.view(b, self.num_heads, -1, T)
        obj_val = torch.matmul(x, obj_attn.transpose(3, 2))  # (bs, nH, c//Nh, k)
        obj_val = obj_val.view(b, c, obj_attn.shape[-2])  # (bs, c, k)
        # projection
        obj_val = self.proj_obj(obj_val)  # (bs, k, k)
        return z_g, obj_val  # (bs, k, 1), (bs, k, k), where the second dim is channel

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
    # x = torch.randn((16, 50, 2048))
    test = ObjectNeck_V2(in_channels=256, out_channels=128)
    z, p, c = test(x)
    print(z.shape, p.shape, c.shape)
    # test = ObjectNeck_GRU(in_channels=2048, out_channels=128)
    # out0, out1, out2 = test(x, mask=mask)
    # print(out0.shape)
    # print(out1.shape)
    # print(out2.shape)
