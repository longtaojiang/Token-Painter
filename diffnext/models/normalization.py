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
"""Normalization Layers."""

from typing import Tuple

import torch
from torch import nn


class AdaLayerNormZero(nn.Module):
    """Adaptive LayerNorm with residual stats."""

    def __init__(self, dim, rank=None, num_stats=2, eps=1e-6):
        super(AdaLayerNormZero, self).__init__()
        self.lora = nn.Linear(dim, rank, bias=False) if rank else nn.Identity()
        self.proj = nn.Linear(rank if rank else dim, num_stats * dim)
        self.norm = nn.LayerNorm(dim, eps, elementwise_affine=False) if eps else nn.Identity()
        self.activation, self.num_stats = nn.SiLU(), num_stats

    def forward(self, x, z) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        stats = self.proj(self.lora(self.activation(z))).chunk(self.num_stats, dim=-1)
        return self.norm(x).mul(1 + stats[0]).add_(stats[1]), stats[2:]


class AdaLayerNorm(AdaLayerNormZero):
    """Adaptive LayerNorm."""

    def __init__(self, dim, rank=None, eps=1e-6):
        super(AdaLayerNorm, self).__init__(dim, rank, num_stats=2, eps=eps)

    def forward(self, x, z) -> torch.Tensor:
        return super().forward(x, z)[0]
    
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 3)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat):
    # content_feat, style_feat, shape is [N, C, L]
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def adaptive_instance_normalization_1d(content_feat, style_feat):
    # content_feat, style_feat, shape is [N, C, L]
    content_feat = content_feat.permute(0,2,1)
    style_feat = style_feat.permute(0,2,1)
    
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return (normalized_feat * style_std.expand(size) + style_mean.expand(size)).permute(0,2,1)

def adaptive_instance_normalization_1d_both(content_feat, style_feat):
    # content_feat, style_feat, shape is [N, C, L]
    content_feat = content_feat.permute(0,2,1)
    style_feat = style_feat.permute(0,2,1)
    
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    
    need_mean, need_std = content_mean*0.3 + style_mean*0.7, content_std*0.3 + style_std*0.7

    normalized_content_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    
    normalized_style_feat = (style_feat - style_mean.expand(
        size)) / style_std.expand(size)
    
    return (normalized_content_feat * need_std.expand(size) + need_mean.expand(size)).permute(0,2,1), (normalized_style_feat * need_std.expand(size) + need_mean.expand(size)).permute(0,2,1)

def tensor_normlization(feat):
    size = feat.size()
    assert (len(size) == 3)
    style_mean, style_std = calc_mean_std(feat)

    normalized_feat = (feat - style_mean.expand(
        size)) / style_std.expand(size)
    return normalized_feat

def tensor_decentralization(feat):
    size = feat.size()
    assert (len(size) == 3)
    style_mean, style_std = calc_mean_std(feat)

    normalized_feat = (feat - style_mean.expand(
        size))
    return normalized_feat
