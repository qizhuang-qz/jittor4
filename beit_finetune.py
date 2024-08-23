import math
from functools import partial

import jittor as jt
from jittor import nn
import numpy as np
from beit_pretraining import Block, PatchEmbed, RelativePositionBias

from jimm.models.layers.drop import drop_path
from jimm.models.layers.helpers import to_2tuple
from jimm.models.layers.weight_init import trunc_normal_
from jimm.models.registry import register_model
import ipdb

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def execute(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # 注释掉这一行以与原始 BERT 实现保持一致
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
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
            self.q_bias = jt.array(np.zeros(all_head_dim))
            self.v_bias = jt.array(np.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(
                jt.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = jt.arange(window_size[0])
            coords_w = jt.arange(window_size[1])
            coords = jt.stack(jt.meshgrid(coords_h, coords_w))  # 2, Wh, Ww
            coords_flatten = coords.flatten(1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1

            relative_position_index = \
                jt.zeros((window_size[0] * window_size[1] + 1, ) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.relative_position_index = relative_position_index
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def execute(self, x, rel_pos_bias=None, return_attention=False, return_qkv=False):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = jt.concat((self.q_bias, jt.zeros_like(self.v_bias), self.v_bias))
        qkv = jt.nn.linear(x, self.qkv.weight, qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, num_heads, N, head_dim

        q = q * self.scale
        attn = jt.matmul(q, k.transpose(-2, -1))

        if self.relative_position_bias_table is not None:
            relative_position_bias = \
                self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias
        
        attn = nn.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        if return_attention:
            return attn

        x = jt.matmul(attn, v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_qkv:
            return x, qkv

        return x



class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * jt.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * jt.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def execute(self, x, rel_pos_bias=None, return_attention=False, return_qkv=False):
        if return_attention:
            return self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, return_attention=True)
        if return_qkv:
            y, qkv = self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, return_qkv=return_qkv)
            x = x + self.drop_path(self.gamma_1 * y)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
            return x, qkv

        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x



class RelativePositionBias(nn.Module):
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(
            jt.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = jt.arange(window_size[0])
        coords_w = jt.arange(window_size[1])
        coords = jt.stack(jt.meshgrid(coords_h, coords_w))  # 2, Wh, Ww
        coords_flatten = jt.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0)  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = \
            jt.zeros((window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer("relative_position_index", relative_position_index)

        # trunc_normal_(self.relative_position_bias_table, std=.02)

    def execute(self):
        relative_position_bias = \
            self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
        return relative_position_bias.permute(2, 0, 1)  # nH, Wh*Ww, Wh*Ww


class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()

        self.localization = nn.Sequential(
            nn.Conv(3, 5, kernel_size=5),
            nn.Pool(2, stride=2, op='maximum'),
            nn.ReLU(),
            nn.Conv(5, 8, kernel_size=7),
            nn.Pool(2, stride=2, op='maximum'),
            nn.ReLU(),
            nn.Conv(8, 10, kernel_size=9),
            nn.Pool(2, stride=2, op='maximum'),
            nn.ReLU(),
            nn.Conv(10, 16, kernel_size=11),
            nn.Pool(2, stride=2, op='maximum'),
            nn.ReLU(),
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(16 * 6 * 6, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )
        # 初始化输出为恒等变换
        self.fc_loc[-1].weight.data = jt.zeros((6, 64))
        self.fc_loc[-1].bias.data = jt.array([1, 0, 0, 0, 1, 0], dtype=jt.float32)

    def stn(self, x):
        y = self.localization(x)
        y = y.view(-1, 16 * 6 * 6)
        # 回归theta参数
        theta = self.fc_loc(y)
        theta = theta.view(-1, 2, 3)
        # 利用theta参数计算变换后图片的位置
        grid = nn.affine_grid(theta, x.size(), align_corners=True)
        # 根据输入图片计算变换后图片位置填充的像素值
        x = nn.grid_sample(x, grid, align_corners=True)

        return x

    def execute(self, x):
        return self.stn(x)



class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=0.1,
                 use_abs_pos_emb=False, use_rel_pos_bias=True, use_shared_rel_pos_bias=False,
                 use_mean_pooling=True, init_scale=0.001):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = jt.Var(jt.zeros((1, 1, embed_dim)))
        # self.mask_token = jt.Var(jt.zeros((1, 1, embed_dim)))
        if use_abs_pos_emb:
            self.pos_embed = jt.Var(jt.zeros((1, num_patches + 1, embed_dim)))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        dpr = [x.item() for x in jt.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.use_rel_pos_bias = use_rel_pos_bias
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None)
            for i in range(depth)])
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        # self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.stn = STN()
        if self.pos_embed is not None:
            self.pos_embed.gauss_(0, .02)
        self.cls_token.gauss_(0, .02)
        # self.mask_token.gauss_(0, .02)
        if isinstance(self.head, nn.Linear):
            self.head.weight.gauss_(0, .02)
        self.apply(self._init_weights)
        self.fix_init_weight()

        dpr_ = [x.item() for x in jt.linspace(0, drop_path_rate, 2)]
        self.block25 = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr_[0], norm_layer=norm_layer,
                init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None)

        self.block26 = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr_[1], norm_layer=norm_layer,
                init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None)
        
        
        if isinstance(self.head, nn.Linear):
            self.head.weight *= init_scale
            self.head.bias *= init_scale

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param /= math.sqrt(2.0 * layer_id)

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight, layer_id + 1)
            rescale(layer.mlp.fc2.weight, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight.gauss_(0, .02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.zero_()
        elif isinstance(m, nn.LayerNorm):
            m.bias.zero_()
            m.weight.fill_(1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @jt.no_grad()
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[0]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return jt.concat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward_features(self, x, return_patch_tokens=False, return_all_tokens=False, **kwargs):
        
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        
        batch_size, seq_len, _ = x.size()  
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        x = jt.concat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            if x.shape[1] != self.pos_embed.shape[1]:
                x = x + self.interpolate_pos_encoding(x, w, h)
            else:
                x = x + self.pos_embed
                
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)
        x25 = self.block25(x)
        x26 = self.block26(x25)
        
        x = self.norm(x)
        x25 = self.norm(x25)
        x26 = self.norm(x26)

        return x[:, 0], x25[:, 0], x26[:, 0]

    def stn_trans(self, x):
        x1 = self.stn(x)
        return x1

    def execute(self, x, return_patch_tokens=False, return_all_tokens=False, **kwargs):
        x_ = self.stn_trans(x)
        
        x, x25, x26 = self.forward_features(x, return_patch_tokens=return_patch_tokens, return_all_tokens=return_all_tokens, **kwargs)
        x_, x_25, x_26 = self.forward_features(x_, return_patch_tokens=return_patch_tokens, return_all_tokens=return_all_tokens, **kwargs)

        x_all26 = jt.cat([x26,x_26])
        
        out_all = self.head(x_all26)   
        # out1 = self.head(x1)
        return out_all

    def forward_intermediate(self, x, layer_id=12, norm_output=False):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = jt.concat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        if isinstance(layer_id, list):
            output_list = []
            for l, blk in enumerate(self.blocks):
                x = blk(x, rel_pos_bias=rel_pos_bias)
                # use last norm for all intermediate layers
                if l in layer_id:
                    if norm_output:
                        x_norm = self.fc_norm(self.norm(x[:, 1:]))
                        output_list.append(x_norm)
                    else:
                        output_list.append(x[:, 1:])
            return output_list
        elif isinstance(layer_id, int):
            for l, blk in enumerate(self.blocks):
                if l < layer_id:
                    x = blk(x, rel_pos_bias=rel_pos_bias)
                elif l == layer_id:
                    x = blk.norm1(x)
                else:
                    break
            return x[:, 1:]
        else:
            raise NotImplementedError(f"Not support for layer id is {layer_id} now!")
    
    def get_intermediate_layers(self, x, use_last_norm=False):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()
    
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = jt.concat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)
    
        features = []
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias)
            if use_last_norm:
                features.append(self.norm(x))
            else:
                features.append(x)
    
        return features


class ood_block(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.oodblock1 = nn.Linear(embed_dim, embed_dim * 2)
        self.oodblock2 = nn.Linear(embed_dim * 2, embed_dim)

        self.ood25block1 = nn.Linear(embed_dim, embed_dim * 2)
        self.ood25block2 = nn.Linear(embed_dim * 2, embed_dim)

        self.ood26block1 = nn.Linear(embed_dim, embed_dim * 2)
        self.ood26block2 = nn.Linear(embed_dim * 2, embed_dim)


    def execute(self, x, x25, x26):
        x = self.oodblock1(x)
        x = self.oodblock2(x)
        
        x25 = self.ood25block1(x25)
        x25 = self.ood25block1(x25)

        x26 = self.ood26block1(x26)
        x26 = self.ood26block1(x26)
        
        return x, x25, x26