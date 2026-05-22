# ------------------------------------------------------------------------
# Copyright (c) 2024-present, BAAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Vision Transformer."""

from typing import Tuple
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as apply_ckpt

from diffnext.models.embeddings import PatchEmbed, RotaryEmbed3D
from diffnext.models.feature_decompose import Fourier_filter_1d_func_compose
from diffnext.models.normalization import adaptive_instance_normalization_1d_both


class MLP(nn.Module):
    """Two layers MLP."""

    def __init__(self, dim, mlp_ratio=4):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(dim, int(dim * mlp_ratio))
        self.fc2 = nn.Linear(int(dim * mlp_ratio), dim)
        self.activation = nn.GELU()

    def forward(self, x) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(x)))


class Attention(nn.Module):
    """Multihead attention."""

    def __init__(self, dim, num_heads, qkv_bias=True):
        super(Attention, self).__init__()
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.scale = self.head_dim ** -0.5
        self.proj = nn.Linear(dim, dim)
        self.attn_mask, self.cache_kv, self.pe_func = None, None, None

    def forward(self, x, atten_entro=None) -> torch.Tensor:
        
        qkv_shape = [-1, x.size(1), 3, self.num_heads, self.head_dim] #[B, L, 3, H, D/H]
        #[B, L, 3, H, D/H] -> [3, B, H, L, D/H]
        q, k, v = self.qkv(x).view(qkv_shape).permute(2, 0, 3, 1, 4).unbind(dim=0) # q, k, v: [B, H, L, D/H]
        q, k = (self.pe_func(q), self.pe_func(k)) if self.pe_func else (q, k)
        
        if self.cache_kv is not None:
            if isinstance(self.cache_kv, list):
                k = self.cache_kv[0] = torch.cat([self.cache_kv[0], k], dim=2)
                v = self.cache_kv[1] = torch.cat([self.cache_kv[1], v], dim=2)
            else:
                self.cache_kv = [k, v]
        
        c_length = 1024
        
        if atten_entro is None:  
            o = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=self.attn_mask)
        else:
            coder_name = atten_entro[0]
            layer_name = atten_entro[1]
            entropy_value = atten_entro[2]
            
            if coder_name == 'encoder': 
                if layer_name in [0, 1]:
                    scale_factor = 1 / math.sqrt(q.size(-1))
                    attn_weight = q @ k.transpose(-2, -1) * scale_factor
                    attn_weight = attn_weight * (entropy_value) ** 0.25
                    attn_weight = torch.softmax(attn_weight, dim=-1)
                    o = attn_weight @ v
                else:
                    o = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=self.attn_mask)
            elif coder_name == 'decoder': #q, k, v [B, H, L, D/H]
                
                if layer_name in [0,1,2,3,4,5,6,7,8,9,10,11,12]:
                    scale_factor = 1 / math.sqrt(q.size(-1))
                    attn_weight = q @ k.transpose(-2, -1) * scale_factor
                    
                    B, H, L = q.size(0), q.size(1), q.size(2)
                    inpaint_token_ids = atten_entro[3][0, :, 0] # [B, l1, 1]
                    uninpaint_token_ids = atten_entro[4][0, :, 0] # [B, l2, 1]
                    all_inpaint_token_ids = torch.cat([inpaint_token_ids, uninpaint_token_ids], dim=0) # [B, l3, 1]
                    text_token_ids = torch.arange(c_length)
                    
                    B_idx = torch.arange(B)[:, None, None, None].to(device=q.device)
                    H_idx = torch.arange(H)[None, :, None, None].to(device=q.device)
                    
                    all_inpaint_token_ids_q = all_inpaint_token_ids[None, None, :, None].to(device=q.device)
                    text_token_ids_k = text_token_ids[None, None, None, :].to(device=q.device)
                    uninpaint_token_ids_q = uninpaint_token_ids[None, None, :, None].to(device=q.device)
                    inpaint_token_ids_k = inpaint_token_ids[None, None, None, :].to(device=q.device)
                    
                    uninpaint_token_num = len(uninpaint_token_ids)
                    all_inpaint_num = len(all_inpaint_token_ids)
                    inpaint_token_num = len(inpaint_token_ids)
                    
                    attn_weight[B_idx, H_idx, all_inpaint_token_ids_q, text_token_ids_k] *= (entropy_value) ** 0.3
                    attn_weight[B_idx, H_idx, uninpaint_token_ids_q, inpaint_token_ids_k] *= (entropy_value) ** 0.1 * math.log(uninpaint_token_num, inpaint_token_num + 1) ** 0.03
                    
                    attn_weight = torch.softmax(attn_weight, dim=-1)
                    o = attn_weight @ v
                else:
                    o = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=self.attn_mask)
            else:
                RuntimeError('No such atten_entro type!')
                
        return self.proj(o.transpose(1, 2).flatten(2))


class Block(nn.Module):
    """Transformer block."""

    def __init__(self, dim, num_heads, mlp_ratio=4, qkv_bias=True):
        super(Block, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio)
        self.attn_checkpointing, self.mlp_checkpointing = False, False

    def forward_attn(self, x) -> torch.Tensor:
        x = self.attn(x, atten_entro=self.atten_entro)
        return self.norm1(x)

    def forward_mlp(self, x) -> torch.Tensor:
        return self.norm2(self.mlp(x))

    def forward_ckpt(self, x, name) -> torch.Tensor:
        if getattr(self, f"{name}_checkpointing", False) and x.requires_grad:
            return apply_ckpt(getattr(self, f"forward_{name}"), x, use_reentrant=False)
        return getattr(self, f"forward_{name}")(x)

    def forward(self, x, pe_func: callable = None, atten_entro = None) -> torch.Tensor:
        self.attn.pe_func = pe_func
        self.atten_entro = atten_entro
        x = self.forward_ckpt(x, "attn").add_(x)
        return self.forward_ckpt(x, "mlp").add_(x)


class VisionTransformer(nn.Module):
    """Vision transformer."""

    def __init__(
        self,
        depth,
        embed_dim,
        num_heads,
        mlp_ratio=4,
        patch_size=2,
        image_size=32,
        image_dim=4,
        encoder_depth=None,
    ):
        super(VisionTransformer, self).__init__()
        self.embed_dim, self.image_size, self.image_dim = embed_dim, image_size, image_dim
        self.patch_embed = PatchEmbed(image_dim, embed_dim, patch_size)
        self.pos_embed, self.rope = nn.Identity(), RotaryEmbed3D(embed_dim // num_heads)
        self.blocks = nn.ModuleList(Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth))
        self.norm, self.mixer = nn.LayerNorm(embed_dim), nn.Identity()
        self.encoder_depth = len(self.blocks) // 2 if encoder_depth is None else encoder_depth

    def prepare_pe(self, c=None, ids=None, pos=None) -> Tuple[callable, callable]:
        pad = 0 if c is None else c.size(1)
        pe1 = pe2 = self.rope.get_func(pos, pad)
        pe1 = self.rope.get_func(pos, pad, ids.expand(-1, -1, 3)) if ids is not None else pe1
        return pe1, pe2

    def forward(self, x, c=None, prev_ids=None, pred_ids=None, pos=None) -> torch.Tensor:
        x, prev_ids = x if isinstance(x, (tuple, list)) else (x, prev_ids)
        prev_ids = prev_ids if self.encoder_depth else None
        
        x = x_masked = self.pos_embed(self.patch_embed(x)) #[2, 4096, 1024]
        pe1, pe2 = self.prepare_pe(c, prev_ids, pos) if pos is not None else [None] * 2
        
        if prev_ids is not None:  # Split mask from x.
            prev_ids = prev_ids.expand(-1, -1, x.size(-1))
            x = x.gather(1, prev_ids)
                
        x = x if c is None else torch.cat([c, x], dim=1)
        
        for blk in self.blocks[: self.encoder_depth]:
            x = blk(x, pe1)
            
        if prev_ids is not None and c is not None:  # Split c from x.
            c, x = x.split((c.size(1), x.size(1) - c.size(1)), dim=1)
        
        if prev_ids is not None: 
            x = x_masked.to(dtype=x.dtype).scatter(1, prev_ids, x)
            x = x if c is None else torch.cat([c, x], dim=1)
        
        for blk in self.blocks[self.encoder_depth :]:
            x = blk(x, pe2)
        return self.norm(x if c is None else x[:, c.size(1) :])
        
    def inpaint_forward(self, x, c=None, prev_ids=None, pred_ids=None, cond_para=None, pos=None, inpaint_token_ids=None, uninpaint_token_ids=None, atten_entro_value=None) -> torch.Tensor:
        x, prev_ids = x if isinstance(x, (tuple, list)) else (x, prev_ids)
        prev_ids = prev_ids if self.encoder_depth else None
        
        x = x_masked = self.pos_embed(self.patch_embed(x)) #[2, 4096, 1024]
        pe1, pe2 = self.prepare_pe(c, prev_ids, pos) if pos is not None else [None] * 2
        
        if prev_ids is not None:
            prev_ids = prev_ids.expand(-1, -1, x.size(-1))
            x = x.gather(1, prev_ids)
        
        if cond_para is not None:
            c_origin = c.clone()
            for ind, blk in enumerate(self.blocks[: self.encoder_depth]):
                atten_entro = ['encoder', ind, atten_entro_value] if atten_entro_value else None
                c_origin = blk(c_origin, atten_entro=atten_entro)
                
        x = x if c is None else torch.cat([c, x], dim=1)
        
        for ind, blk in enumerate(self.blocks[: self.encoder_depth]):
            atten_entro = ['encoder', ind, atten_entro_value] if atten_entro_value else None
            x = blk(x, pe1, atten_entro=atten_entro)

        if prev_ids is not None and c is not None:
            c, x = x.split((c.size(1), x.size(1) - c.size(1)), dim=1)
        
        if cond_para is not None:
            
            if cond_para['func'] != 'linear_inter':
                c, c_origin = adaptive_instance_normalization_1d_both(c, c_origin) if cond_para['ada_in'] else (c, c_origin)
                c = Fourier_filter_1d_func_compose(c_origin, c, cond_para)
            else:
                c = c_origin * cond_para['condweight'] + c * (1-cond_para['condweight']) # [N, L, C]
        
        if prev_ids is not None: 
            x = x_masked.to(dtype=x.dtype).scatter(1, prev_ids, x)
            x = x if c is None else torch.cat([c, x], dim=1)
        
        for ind, blk in enumerate(self.blocks[self.encoder_depth :]):
            atten_entro = ['decoder', ind, atten_entro_value, inpaint_token_ids, uninpaint_token_ids] if atten_entro_value else None
            x = blk(x, pe2, atten_entro=atten_entro)
        return self.norm(x if c is None else x[:, c.size(1) :])
