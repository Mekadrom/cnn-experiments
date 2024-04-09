from timm.layers import DropPath, to_2tuple, trunc_normal_
from tutel import moe as tutel_moe

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

class MultiLayerPerceptron(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, activation=nn.GELU, dropout=0.0):
        super(MultiLayerPerceptron, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.ffn = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, out_features),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ffn(x)
    
class MoEMlp(nn.Module):
    def __init__(self, in_features, hidden_features, num_local_experts, top_value, capacity_factor=1.25,
                 cosine_router=False, normalize_gate=False, use_bpr=True, is_gshard_loss=True,
                 gate_noise=1.0, cosine_router_dim=256, cosine_router_init_t=0.5, moe_dropout=0.0, init_std=0.02,
                 mlp_fc2_bias=True):
        super().__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.num_local_experts = num_local_experts
        self.top_value = top_value
        self.capacity_factor = capacity_factor
        self.cosine_router = cosine_router
        self.normalize_gate = normalize_gate
        self.use_bpr = use_bpr
        self.init_std = init_std
        self.mlp_fc2_bias = mlp_fc2_bias

        self.dist_rank = dist.get_rank()

        self._dropout = nn.Dropout(p=moe_dropout)

        _gate_type = {'type': 'cosine_top' if cosine_router else 'top',
                      'k': top_value, 'capacity_factor': capacity_factor,
                      'gate_noise': gate_noise, 'fp32_gate': True}
        if cosine_router:
            _gate_type['proj_dim'] = cosine_router_dim
            _gate_type['init_t'] = cosine_router_init_t
        self._moe_layer = tutel_moe.moe_layer(
            gate_type=_gate_type,
            model_dim=in_features,
            experts={'type': 'ffn', 'count_per_node': num_local_experts, 'hidden_size_per_expert': hidden_features,
                     'activation_fn': lambda x: self._dropout(F.gelu(x))},
            scan_expert_func=lambda name, param: setattr(param, 'skip_allreduce', True),
            seeds=(1, self.dist_rank + 1, self.dist_rank + 1),
            batch_prioritized_routing=use_bpr,
            normalize_gate=normalize_gate,
            is_gshard_loss=is_gshard_loss,

        )
        if not self.mlp_fc2_bias:
            self._moe_layer.experts.batched_fc2_bias.requires_grad = False

    def forward(self, x):
        x = self._moe_layer(x)
        return x, x.l_aux

    def _init_weights(self):
        if hasattr(self._moe_layer, "experts"):
            trunc_normal_(self._moe_layer.experts.batched_fc1_w, std=self.init_std)
            trunc_normal_(self._moe_layer.experts.batched_fc2_w, std=self.init_std)
            nn.init.constant_(self._moe_layer.experts.batched_fc1_bias, 0)
            nn.init.constant_(self._moe_layer.experts.batched_fc2_bias, 0)
    
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attention_dropout=0.0, output_dropout=0.0, pretrained_window_size=[0, 0]):
        super(WindowAttention, self).__init__()

        self.dim = dim
        self.window_size = window_size # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads
        
        self.logit_scale = nn.Parameter(torch.log(10 * torch.onez((num_heads, 1, 1))), requires_grad=True)

        self.cast_continuous_positional_bias = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False)
        )

        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0) # 1, 2*Wh, 2*Ww, 2

        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)

        relative_coords_table *= 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / np.log2(8.0)

        self.register_buffer("relative_coords_table", relative_coords_table)

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.windows_size[0] - 1
        relative_coords[:, :, 1] += self.windows_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.windows_size[1] - 1
        relative_position_index = relative_coords.sum(dim=-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.cast_qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = self.v_bias = None

        self.attention_dropout = nn.Dropout(attention_dropout)
        self.cast_output = nn.Linear(dim, dim)
        self.output_dropout = nn.Dropout(output_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape

        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat([self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias])

        qkv = F.linear(input=x, weight=self.cast_qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        queries, keys, values = qkv[0], qkv[1], qkv[2]

        attention_weights = (F.normalize(queries, dim=-1) @ F.normalize(keys, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1.0 / 0.01))).exp()
        attention_weights = attention_weights * logit_scale

        # positional encoding
        relative_position_bias_table = self.cast_continuous_positional_bias(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attention_weights = attention_weights + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attention_weights = attention_weights.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attention_weights = attention_weights.view(-1, self.num_heads, N, N)
            attention_weights = self.softmax(attention_weights)
        else:
            attention_weights = self.softmax(attention_weights)

        attention_weights = self.attention_dropout(attention_weights)

        x = (attention_weights @ values).transpose(1, 2).reshape(B_ N, C)
        x = self.cast_output(x)
        x = self.output_dropout(x)

        return x
    
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, dropout=0.0, attention_dropout=0.0, drop_path=0.0,
                 activation=nn.GELU, norm_layer=nn.LayerNorm, mlp_fc2_bias=True, init_std=0.02, pretrained_window_size=0,
                 is_moe=False, num_local_experts=1, top_value=1, capacity_factor=1.25, cosine_router=False,
                 normalize_gate=False, use_bpr=True, is_gshard_loss=True, gate_noise=1.0,
                 cosine_router_dim=256, cosine_router_init_t=0.5, moe_dropout=0.0):
        super(SwinTransformerBlock, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.is_moe = is_moe
        
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

            assert 0 <= self.shift_size < self.window_size, "shift_size must be in [0, window_size)"

        self.norm1 = norm_layer(dim)
        self.window_attn = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, qkv_bias=qkv_bias, attention_dropout=attention_dropout, pretrained_window_size=to_2tuple(pretrained_window_size))

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        if self.use_moe:
            self.mlp = MoEMlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                num_local_experts=num_local_experts,
                top_value=top_value,
                capacity_factor=capacity_factor,
                cosine_router=cosine_router,
                normalize_gate=normalize_gate,
                use_bpr=use_bpr,
                is_gshard_loss=is_gshard_loss,
                gate_noise=gate_noise,
                cosine_router_dim=cosine_router_dim,
                cosine_router_init_t=cosine_router_init_t,
                moe_dropout=moe_dropout,
                mlp_fc2_bias=mlp_fc2_bias,
                init_std=init_std
            )
        else:
            self.mlp = MultiLayerPerceptron(in_features=dim, hidden_features=mlp_hidden_dim, activation=activation, dropout=dropout)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size_, slice(-self.shift_size, None)))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size_, slice(-self.shift_size, None)))

            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attention_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attention_mask = attention_mask.masked_fill(attention_mask != 0, float(-100.0)).masked_fill(attention_mask == 0, float(0.0))
        else:
            attention_mask = None

        self.register_buffer("attention_mask", attention_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attention_windows = self.window_attn(x_windows, mask=self.attention_mask)

        attention_windows = attention_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attention_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))

        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x
    
class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super(PatchMerging, self).__init__()

        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)

        x = self.reduction(x)
        x = self.norm(x)

        return x

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, dropout=0.0, attention_dropout=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None,
                 mlp_fc2_bias=True, init_std=0.02, use_checkpoint=False, pretrained_window_size=0,
                 moe_block=[-1], num_local_experts=1, top_value=1, capacity_factor=1.25, cosine_router=False,
                 normalize_gate=False, use_bpr=True, is_gshard_loss=True,
                 cosine_router_dim=256, cosine_router_init_t=0.5, gate_noise=1.0, moe_dropout=0.0):
        super(BasicLayer, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, 
                input_resolution=input_resolution, 
                num_heads=num_heads, 
                window_size=window_size, 
                shift_size=0, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, 
                dropout=dropout, 
                attention_dropout=attention_dropout, 
                drop_path=drop_path[i], 
                norm_layer=norm_layer, 
                pretrained_window_size=pretrained_window_size[i],

                is_moe=True if i in moe_block else False,
                num_local_experts=num_local_experts,
                top_value=top_value,
                capacity_factor=capacity_factor,
                cosine_router=cosine_router,
                normalize_gate=normalize_gate,
                use_bpr=use_bpr,
                is_gshard_loss=is_gshard_loss,
                gate_noise=gate_noise,
                cosine_router_dim=cosine_router_dim,
                cosine_router_init_t=cosine_router_init_t,
                moe_dropout=moe_dropout
            ) 
            for i in range(depth)
        ])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x
    
    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super(PatchEmbed, self).__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape

        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})"

        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)

        return x
    
class SwinTransformerV2(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
                 dropout=0.0, attention_dropout=0.0, drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 mlp_fc2_bias=True, init_std=0.02, use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0],
                 moe_blocks=[[-1], [-1], [-1], [-1]], num_local_experts=1, top_value=1, capacity_factor=1.25,
                 cosine_router=False, normalize_gate=False, use_bpr=True, is_gshard_loss=True, gate_noise=1.0,
                 cosine_router_dim=256, cosine_router_init_t=0.5, moe_dropout=0.0, aux_loss_weight=0.01, **kwargs):
        super(SwinTransformerV2, self).__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_dropout = nn.Dropout(p=dropout)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()

        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer), patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                dropout=dropout, attention_dropout=attention_dropout,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                pretrained_window_size=pretrained_window_sizes[i_layer],

                moe_block=moe_blocks[i_layer],
                num_local_experts=num_local_experts,
                top_value=top_value,
                capacity_factor=capacity_factor,
                cosine_router=cosine_router,
                normalize_gate=normalize_gate,
                use_bpr=use_bpr,
                is_gshard_loss=is_gshard_loss,
                gate_noise=gate_noise,
                cosine_router_dim=cosine_router_dim,
                cosine_router_init_t=cosine_router_init_t,
                moe_dropout=moe_dropout
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weights, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed

        x = self.pos_dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
    