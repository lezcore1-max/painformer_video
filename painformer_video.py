"""
painformer_video.py
===================
Spatiotemporal extension of PainFormer (SpectFormer architecture).

Original image encoder:  SpectFormer  [B, 3, 224, 224]  → [B, 160]
This video encoder:      SpectFormer  [B, T, 3, 224, 224] → [B, 160]

Strategy B – true spatiotemporal:  the architecture itself is modified,
frames are NOT processed independently.

Key changes vs. the image version
-----------------------------------
1.  Stem       : Conv2d → Conv3d  (temporal kernel=1, so no temporal stride)
2.  PosEmbed   : factorised spatial + temporal learnable parameters
3.  SGN        : factorised spatial (rfft2) + temporal (rfft along T)
4.  Attention  : divided space-time (TimeSformer style)
5.  DownSamples: Conv2d kept; reshape logic updated to handle T
6.  DWConv     : reshape [B,T*N,C] → [B*T,C,H,W] → [B,T*N,C]
7.  forward_cls: mean-pool over T*N tokens to build CLS token
8.  forward_features: threads T through every stage

Hyperparameters (unchanged from original painformer)
----------------------------------------------------
embed_dims  = [64, 128, 320, 160]
depths      = [3, 4, 12, 3]
num_heads   = [2, 4, 10, 16]
mlp_ratios  = [8, 8, 4, 4]
Output dim  = 160
"""

import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_weights(m):
    """Shared weight initialiser used by all modules."""
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
        # fan-out init
        ksize = m.kernel_size[0] * m.kernel_size[1]
        if isinstance(m, nn.Conv3d):
            ksize *= m.kernel_size[2]
        fan_out = ksize * m.out_channels // m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()


def add_pos_embed(x, T, H, W, C, spatial_pe, temporal_pe):
    """
    Add factorised learnable positional embeddings.

    Parameters
    ----------
    x           : [B, T*H*W, C]
    T, H, W, C  : ints
    spatial_pe  : nn.Parameter  [1, H*W, C]
    temporal_pe : nn.Parameter  [1, T,   C]

    Returns
    -------
    x           : [B, T*H*W, C]
    """
    B = x.shape[0]

    # Reshape so spatial and temporal axes are explicit
    x = x.view(B, T, H * W, C)                          # [B, T, H*W, C]

    # Spatial PE:  broadcast over T dimension  →  add [1, 1, H*W, C]
    x = x + spatial_pe.unsqueeze(1)                      # [B, T, H*W, C]

    # Temporal PE: broadcast over H*W dimension → add [1, T, 1, C]
    x = x + temporal_pe.unsqueeze(2)                     # [B, T, H*W, C]

    # Flatten back
    x = x.reshape(B, T * H * W, C)                      # [B, T*H*W, C]
    return x


# ---------------------------------------------------------------------------
# SpectralGatingNetwork  (Stages 0 & 1) – factorised spatial + temporal
# ---------------------------------------------------------------------------

class SpectralGatingNetwork(nn.Module):
    """
    Factorised spectral gating:
      1. rfft2 over (H, W)  – spatial frequency filtering
      2. rfft  over T       – temporal frequency filtering

    The spatial complex_weight is the same learnable tensor as in the
    original PainFormer (per-spatial-frequency, per-channel).
    A new temporal_weight [T_freq, C, 2] is added for temporal filtering.

    forward signature:  forward(x, H, W, T)
                        x : [B, T*H*W, C]
    """

    def __init__(self, dim, T=8):
        super().__init__()
        self.T = T

        # Spatial weight – hardcoded H/W sizes match the original paper
        # (after the Stem: 224 → 56 spatial for dim=64; 56→28 for dim=128)
        if dim == 64:
            self.h, self.w = 56, 29           # H, (W/2)+1
        elif dim == 128:
            self.h, self.w = 28, 15           # H, (W/2)+1
        else:
            raise ValueError(f"SpectralGatingNetwork: unsupported dim={dim}. "
                             "Expected one of {64, 128}.")

        # Spatial learnable weight: [H, W_rfft, C, 2(re/im)]
        self.complex_weight = nn.Parameter(
            torch.randn(self.h, self.w, dim, 2, dtype=torch.float32) * 0.02
        )

        # Temporal learnable weight: [T_freq, C, 2(re/im)]
        # rfft of a real signal of length T produces floor(T/2)+1 coefficients.
        T_freq = T // 2 + 1
        self.temporal_weight = nn.Parameter(
            torch.randn(T_freq, dim, 2, dtype=torch.float32) * 0.02
        )

    def forward(self, x, H, W, T):
        """
        x : [B, T*H*W, C]
        returns: [B, T*H*W, C]

        Factorised spectral gating — two sequential passes, each fully
        completed before the next starts so inputs are always real:
          Pass 1 (spatial) : rfft2 → multiply → irfft2  → real output
          Pass 2 (temporal): rfft  → multiply → irfft   → real output
        """
        B, _N, C = x.shape                               # [B, T*H*W, C]

        x = x.view(B, T, H, W, C)                        # [B, T, H, W, C]
        x = x.to(torch.float32)                          # ensure real float32

        # ── Pass 1: Spatial filter ─────────────────────────────────────────
        # Flatten (B,T) so rfft2 sees (B*T) independent 2-D frames.
        x = x.reshape(B * T, H, W, C)                    # [B*T, H, W, C]       real
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho') # [B*T, H, W//2+1, C]  complex

        w_spatial = torch.view_as_complex(self.complex_weight)  # [H, W//2+1, C]
        x = x * w_spatial                                 # [B*T, H, W//2+1, C]  complex

        # irfft2 brings x back to real — temporal rfft requires real input
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')  # [B*T, H, W, C]  real

        # ── Pass 2: Temporal filter ────────────────────────────────────────
        # x is real here so rfft is valid.
        x = x.view(B, T, H, W, C)                        # [B, T, H, W, C]      real
        x = torch.fft.rfft(x, dim=1, norm='ortho')       # [B, T_freq, H, W, C] complex

        w_temporal = torch.view_as_complex(self.temporal_weight)  # [T_freq, C]
        # broadcast over (B, H, W)
        x = x * w_temporal.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # [B, T_freq, H, W, C]

        x = torch.fft.irfft(x, n=T, dim=1, norm='ortho')  # [B, T, H, W, C]  real

        # ── Reshape back ───────────────────────────────────────────────────
        x = x.reshape(B, T * H * W, C)                    # [B, T*H*W, C]
        return x


# ---------------------------------------------------------------------------
# ClassAttention  (unchanged from original)
# ---------------------------------------------------------------------------

class ClassAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.kv = nn.Linear(dim, dim * 2)
        self.q = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.apply(_init_weights)

    def forward(self, x):
        B, N, C = x.shape                                # [B, 1+T*H*W, C]
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q = self.q(x[:, :1, :]).reshape(B, self.num_heads, 1, self.head_dim)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        cls_embed = (attn @ v).transpose(1, 2).reshape(B, 1, self.head_dim * self.num_heads)
        cls_embed = self.proj(cls_embed)
        return cls_embed


# ---------------------------------------------------------------------------
# FFN  (unchanged from original)
# ---------------------------------------------------------------------------

class FFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.apply(_init_weights)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


# ---------------------------------------------------------------------------
# ClassBlock  (unchanged from original)
# ---------------------------------------------------------------------------

class ClassBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = ClassAttention(dim, num_heads)
        self.mlp = FFN(dim, int(dim * mlp_ratio))
        self.apply(_init_weights)

    def forward(self, x):
        cls_embed = x[:, :1]
        cls_embed = cls_embed + self.attn(self.norm1(x))
        cls_embed = cls_embed + self.mlp(self.norm2(cls_embed))
        return torch.cat([cls_embed, x[:, 1:]], dim=1)


# ---------------------------------------------------------------------------
# DWConv  – video-aware depthwise conv
# ---------------------------------------------------------------------------

class DWConv(nn.Module):
    """
    Depthwise 2-D convolution applied frame-by-frame.

    Input  : [B, T*N, C]   where N = H*W
    Output : [B, T*N, C]
    """

    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,
                                bias=True, groups=dim)

    def forward(self, x, H, W, T):
        B, _TN, C = x.shape                              # [B, T*H*W, C]
        N = H * W

        # Merge batch and temporal dims so Conv2d sees (B*T) images
        x = x.view(B * T, N, C)                          # [B*T, H*W, C]
        x = x.transpose(1, 2).view(B * T, C, H, W)      # [B*T, C, H, W]
        x = self.dwconv(x)                               # [B*T, C, H, W]
        x = x.flatten(2).transpose(1, 2)                 # [B*T, H*W, C]

        # Restore leading batch dimension
        x = x.reshape(B, T * N, C)                      # [B, T*H*W, C]
        return x


# ---------------------------------------------------------------------------
# PVT2FFN  – video-aware FFN with depthwise conv
# ---------------------------------------------------------------------------

class PVT2FFN(nn.Module):
    """
    Input / output: [B, T*N, C]  (N = H*W)
    """

    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.apply(_init_weights)

    def forward(self, x, H, W, T):
        x = self.fc1(x)                                  # [B, T*H*W, hidden]
        x = self.dwconv(x, H, W, T)                     # [B, T*H*W, hidden]
        x = self.act(x)
        x = self.fc2(x)                                  # [B, T*H*W, C]
        return x


# ---------------------------------------------------------------------------
# Attention  – divided space-time (TimeSformer style)
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """
    Divided space-time attention.

    Spatial pass  : process each frame separately → [B*T, N, C]
    Temporal pass : process each spatial token across T → [B*N, T, C]

    Both passes use the same QKV projection + output proj structure,
    but the temporal pass has its own independent projection.

    forward signature : forward(x, H, W, T)
                        x : [B, T*N, C]   N = H*W
    returns           : (x, attn_weights)   attn_weights from spatial pass
    """

    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0, (
            f"dim {dim} must be divisible by num_heads {num_heads}.")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # ── Spatial attention projections ──────────────────────────────────
        self.q_s = nn.Linear(dim, dim)
        self.kv_s = nn.Linear(dim, dim * 2)
        self.proj_s = nn.Linear(dim, dim)

        # ── Temporal attention projections ─────────────────────────────────
        self.q_t = nn.Linear(dim, dim)
        self.kv_t = nn.Linear(dim, dim * 2)
        self.proj_t = nn.Linear(dim, dim)

        self.apply(_init_weights)

    def _mhsa(self, x, q_lin, kv_lin, proj_lin):
        """
        Standard multi-head self-attention.
        x         : [M, L, C]
        returns   : (out [M, L, C], attn [M, num_heads, L, L])
        """
        M, L, C = x.shape
        q = q_lin(x).reshape(M, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                                                                          # [M, nh, L, hd]
        kv = kv_lin(x).reshape(M, L, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]                                              # [M, nh, L, hd]
        attn = (q @ k.transpose(-2, -1)) * self.scale                   # [M, nh, L, L]
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(M, L, C)               # [M, L, C]
        out = proj_lin(out)                                              # [M, L, C]
        return out, attn

    def forward(self, x, H, W, T):
        """
        x  : [B, T*H*W, C]
        returns : (x_out [B, T*H*W, C], attn_spatial [B*T, nh, N, N])
        """
        B, _TN, C = x.shape                              # [B, T*H*W, C]
        N = H * W

        # ── Spatial attention  ─────────────────────────────────────────────
        # Reshape: treat (B*T) as the batch, N = H*W as the sequence
        x_s = x.view(B * T, N, C)                        # [B*T, N, C]
        x_s_out, attn_spatial = self._mhsa(              # [B*T, N, C], [B*T, nh, N, N]
            x_s, self.q_s, self.kv_s, self.proj_s)       # x_s_out is Delta_s

        # Feed spatially attended representation into temporal pass (x + Delta_s)
        x_mid = x_s + x_s_out                            # [B*T, N, C]

        # ── Temporal attention ─────────────────────────────────────────────
        # Reshape: treat (B*N) as the batch, T as the sequence
        x_t = x_mid.view(B, T, N, C)                     # [B, T, H*W, C]
        x_t = x_t.permute(0, 2, 1, 3)                    # [B, N, T, C]
        x_t = x_t.reshape(B * N, T, C)                   # [B*N, T, C]

        x_t_out, _attn_temporal = self._mhsa(            # [B*N, T, C]
            x_t, self.q_t, self.kv_t, self.proj_t)       # x_t_out is Delta_t

        # Delta_s reshape
        delta_s = x_s_out.view(B, T * N, C)              # [B, T*N, C]

        # Delta_t reshape
        delta_t = x_t_out.view(B, N, T, C)               # [B, N, T, C]
        delta_t = delta_t.permute(0, 2, 1, 3)            # [B, T, N, C]
        delta_t = delta_t.reshape(B, T * N, C)           # [B, T*N, C]

        # Return only the spatial delta + temporal delta to Block
        attn_out = delta_s + delta_t                     # [B, T*N, C]
        return attn_out, attn_spatial                    # [B, T*N, C], attn


# ---------------------------------------------------------------------------
# Block  – wraps SpectralGatingNetwork or divided space-time Attention
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """
    Universal block for both 'wave' (stages 0-1) and 'std_att' (stages 2-3).

    forward(x, H, W, T)
      x : [B, T*H*W, C]

    Returns:
      wave:    x                         [B, T*H*W, C]      (no attn map)
      std_att: (x, attn_spatial_weights)
    """

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 sr_ratio=1,
                 block_type='wave',
                 T=8):
        super().__init__()
        self.block_type = block_type
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        if block_type == 'std_att':
            self.attn = Attention(dim, num_heads)
        else:  # 'wave'
            self.attn = SpectralGatingNetwork(dim, T=T)

        self.mlp = PVT2FFN(in_features=dim,
                           hidden_features=int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(_init_weights)

    def forward(self, x, H, W, T):
        """
        x      : [B, T*H*W, C]
        returns: x  OR  (x, attn_weights)
        """
        if self.block_type == 'std_att':
            # Divided space-time attention – returns (out, attn_map)
            attn_out, attn_weights = self.attn(self.norm1(x), H, W, T)  # [B, T*H*W, C]
            x = x + self.drop_path(attn_out)                            # [B, T*H*W, C]
            x = x + self.drop_path(self.mlp(self.norm2(x), H, W, T))   # [B, T*H*W, C]
            return x, attn_weights

        else:  # 'wave'
            # Factorised spectral gating – no attention map
            wave_out = self.attn(self.norm1(x), H, W, T)                # [B, T*H*W, C]
            x = x + self.drop_path(wave_out)                            # [B, T*H*W, C]
            x = x + self.drop_path(self.mlp(self.norm2(x), H, W, T))   # [B, T*H*W, C]
            return x                                                     # [B, T*H*W, C]


# ---------------------------------------------------------------------------
# DownSamples  – spatial only, but video-aware reshape
# ---------------------------------------------------------------------------

class DownSamples(nn.Module):
    """
    Spatial 2x2 downsampling (Conv2d, stride=2) applied frame-by-frame.

    Input  : [B, T*H*W, C_in]   (flat tokens from previous stage)
    Output : (x [B, T*H_new*W_new, C_out],  H_new, W_new)

    T is unchanged; only (H, W) are halved.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm(out_channels)
        self.apply(_init_weights)

    def forward(self, x, T):
        """
        x  : [B, T*H*W, C_in]
        returns (x_out [B, T*H_new*W_new, C_out], H_new, W_new)
        """
        B, TN, C = x.shape                               # [B, T*H*W, C_in]
        N = TN // T
        H = W = int(math.isqrt(N))                       # assumes square spatial grid

        # Merge (B, T) into a single leading dim so Conv2d sees (B*T) images
        x = x.view(B * T, N, C)                          # [B*T, H*W, C_in]
        x = x.transpose(1, 2).view(B * T, C, H, W)      # [B*T, C_in, H, W]
        x = self.proj(x)                                  # [B*T, C_out, H_new, W_new]
        _, C_out, H_new, W_new = x.shape

        x = x.flatten(2).transpose(1, 2)                 # [B*T, H_new*W_new, C_out]
        x = self.norm(x)                                  # [B*T, H_new*W_new, C_out]

        # Restore batch dimension
        x = x.reshape(B, T * H_new * W_new, C_out)      # [B, T*H_new*W_new, C_out]
        return x, H_new, W_new


# ---------------------------------------------------------------------------
# Stem  – 3-D conv version (temporal kernel = 1, so no temporal stride)
# ---------------------------------------------------------------------------

class Stem(nn.Module):
    """
    Input  : [B, T, 3, H, W]
    Output : ([B, T*H_out*W_out, C],  H_out, W_out, T)

    Conv3d kernel/stride/padding:
      conv1 : k=(1,7,7)  s=(1,2,2)  p=(0,3,3)   224 → 112
      conv2 : k=(1,3,3)  s=(1,1,1)  p=(0,1,1)   112 → 112
      conv3 : k=(1,3,3)  s=(1,1,1)  p=(0,1,1)   112 → 112
      proj  : k=(1,3,3)  s=(1,2,2)  p=(0,1,1)   112 → 56
    """

    def __init__(self, in_channels, stem_hidden_dim, out_channels):
        super().__init__()
        hd = stem_hidden_dim

        # Three 3-D convolutions (no temporal downsampling)
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, hd,
                      kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3),
                      bias=False),
            nn.BatchNorm3d(hd),
            nn.ReLU(inplace=True),

            # conv2: feature enrichment 112 → 112
            nn.Conv3d(hd, hd,
                      kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1),
                      bias=False),
            nn.BatchNorm3d(hd),
            nn.ReLU(inplace=True),

            # conv3: feature enrichment 112 → 112
            nn.Conv3d(hd, hd,
                      kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1),
                      bias=False),
            nn.BatchNorm3d(hd),
            nn.ReLU(inplace=True),
        )

        # proj: spatial downsampling 112 → 56 (no temporal stride)
        self.proj = nn.Conv3d(hd, out_channels,
                              kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.norm = nn.LayerNorm(out_channels)
        self.apply(_init_weights)

    def forward(self, x):
        """
        x   : [B, T, 3, H, W]
        returns ([B, T*H_out*W_out, C],  H_out, W_out, T)
        """
        B, T, C_in, H_in, W_in = x.shape                # [B, T, 3, 224, 224]

        # Conv3d expects [B, C, T, H, W]
        x = x.permute(0, 2, 1, 3, 4)                    # [B, 3, T, 224, 224]

        x = self.conv(x)                                  # [B, hd, T, 112, 112]
        x = self.proj(x)                                  # [B, C_out, T, 56, 56]

        B, C_out, T_out, H_out, W_out = x.shape
        # T_out == T  (no temporal stride anywhere in Stem)

        # Flatten spatial only; keep T absorbed into the token sequence
        x = x.permute(0, 2, 3, 4, 1)                    # [B, T, H_out, W_out, C_out]
        x = x.reshape(B, T_out * H_out * W_out, C_out)  # [B, T*H_out*W_out, C_out]
        x = self.norm(x)                                  # [B, T*H_out*W_out, C_out]

        return x, H_out, W_out, T_out                    # T_out == T


# ---------------------------------------------------------------------------
# SpectFormer  – full video model
# ---------------------------------------------------------------------------

class SpectFormer(nn.Module):
    """
    Spatiotemporal SpectFormer.

    Parameters
    ----------
    T               : number of input frames  (default 8)
    in_chans        : input channels (3 for RGB)
    num_classes     : number of output classes
    stem_hidden_dim : hidden dim in the Stem conv block
    embed_dims      : list of 4 channel dims for each stage
    num_heads       : list of 4 head counts
    mlp_ratios      : list of 4 MLP expansion ratios
    drop_path_rate  : stochastic depth rate
    depths          : list of 4 block depths
    sr_ratios       : (unused spatial reduction ratios, kept for API compat)
    num_stages      : 4 for PainFormer
    """

    def __init__(self,
                 T=8,
                 in_chans=3,
                 num_classes=1000,
                 stem_hidden_dim=32,
                 embed_dims=None,
                 num_heads=None,
                 mlp_ratios=None,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 depths=None,
                 sr_ratios=None,
                 num_stages=4,
                 token_label=False,
                 **kwargs):
        super().__init__()

        # Default PainFormer hyperparameters
        embed_dims = embed_dims or [64, 128, 320, 160]
        num_heads  = num_heads  or [2, 4, 10, 16]
        mlp_ratios = mlp_ratios or [8, 8, 4, 4]
        depths     = depths     or [3, 4, 12, 3]
        sr_ratios  = sr_ratios  or [4, 2, 1, 1]

        self.T = T
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        # ── Factorised positional embeddings (for stage 0 output) ──────────
        # After Stem: spatial grid is 56×56 = 3136 for the first stage
        stem_H = stem_W = 56    # 224 → ÷2 (conv1) → ÷2 (proj) = 56
        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, stem_H * stem_W, embed_dims[0])  # [1, 3136, 64]
        )
        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, T, embed_dims[0])                 # [1, T, 64]
        )
        trunc_normal_(self.spatial_pos_embed, std=0.02)
        trunc_normal_(self.temporal_pos_embed, std=0.02)

        # ── Build stages ──────────────────────────────────────────────────
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = Stem(in_chans, stem_hidden_dim, embed_dims[i])
            else:
                patch_embed = DownSamples(embed_dims[i - 1], embed_dims[i])

            block = nn.ModuleList([
                Block(
                    dim=embed_dims[i],
                    num_heads=num_heads[i],
                    mlp_ratio=mlp_ratios[i],
                    drop_path=dpr[cur + j],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[i],
                    block_type='wave' if i < 2 else 'std_att',
                    T=T,
                )
                for j in range(depths[i])
            ])

            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # ── Class token network ───────────────────────────────────────────
        self.post_network = nn.ModuleList([
            ClassBlock(
                dim=embed_dims[-1],
                num_heads=num_heads[-1],
                mlp_ratio=mlp_ratios[-1],
                norm_layer=norm_layer,
            )
        ])

        # ── Classification head ───────────────────────────────────────────
        self.head = (nn.Linear(embed_dims[-1], num_classes)
                     if num_classes > 0 else nn.Identity())

        # ── Token labelling (optional, kept for API compat) ───────────────
        self.return_dense  = token_label
        self.mix_token     = token_label
        self.beta          = 1.0
        self.pooling_scale = 8
        if self.return_dense:
            self.aux_head = (nn.Linear(embed_dims[-1], num_classes)
                             if num_classes > 0 else nn.Identity())

        self.apply(_init_weights)

    # ------------------------------------------------------------------
    # forward_cls
    # ------------------------------------------------------------------
    def forward_cls(self, x):
        """
        Create CLS token by mean-pooling over ALL T*N spatial-temporal tokens,
        prepend it, then run ClassBlock(s).

        x   : [B, T*N, C]   (N = H*W for the last stage)
        returns [B, 1+T*N, C]
        """
        B, _TN, C = x.shape                              # [B, T*N, C]

        # Mean over all tokens (spatial AND temporal) → CLS
        cls_tokens = x.mean(dim=1, keepdim=True)         # [B, 1, C]
        x = torch.cat((cls_tokens, x), dim=1)            # [B, 1+T*N, C]

        for block in self.post_network:
            x = block(x)                                 # [B, 1+T*N, C]

        return x                                         # [B, 1+T*N, C]

    # ------------------------------------------------------------------
    # forward_features
    # ------------------------------------------------------------------
    def forward_features(self, x):
        """
        x   : [B, T, 3, H, W]

        Returns
        -------
        x             : [B, 160]          (CLS embedding after norm)
        tokens        : [B, T*H_last*W_last, 160]
        attention_maps: list of spatial attention tensors (from std_att stages)
        """
        B = x.shape[0]
        T = self.T
        attention_maps = []
        tokens = None

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block       = getattr(self, f"block{i + 1}")

            # ── Embedding / downsampling ───────────────────────────────
            if i == 0:
                # Stem: takes [B, T, 3, H, W] → ([B, T*H*W, C], H, W, T)
                x, H, W, T = patch_embed(x)              # [B, T*H*W, C]

                # Add factorised positional embeddings right after Stem
                x = add_pos_embed(
                    x, T, H, W, x.shape[-1],
                    self.spatial_pos_embed,
                    self.temporal_pos_embed,
                )                                         # [B, T*H*W, C]
            else:
                # DownSamples: takes ([B, T*H*W, C_in], T) → ([B, T*H_new*W_new, C_out], H_new, W_new)
                x, H, W = patch_embed(x, T)              # [B, T*H_new*W_new, C_out]

            # ── Transformer / Wave blocks ──────────────────────────────
            for blk in block:
                outputs = blk(x, H, W, T)                # [B, T*H*W, C] or tuple
                if isinstance(outputs, tuple):
                    x, attn_weights = outputs
                    attention_maps.append(attn_weights)   # store spatial attn map
                else:
                    x = outputs                           # [B, T*H*W, C]

            tokens = x                                   # [B, T*H*W, C]

            # ── Normalise before feeding into next stage ───────────────
            if i != self.num_stages - 1:
                norm = getattr(self, f"norm{i + 1}")
                x = norm(x)                              # [B, T*H*W, C]
                # x stays flat [B, T*H*W, C]; DownSamples handles reshape internally

        # ── CLS token → final embedding ───────────────────────────────
        x = self.forward_cls(x)[:, 0]                   # [B, C_last=160]
        norm = getattr(self, f"norm{self.num_stages}")
        x = norm(x)                                      # [B, 160]

        return x, tokens, attention_maps

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, x):
        """
        x : [B, T, 3, H, W]

        Returns
        -------
        x             : [B, num_classes]
        tokens        : [B, T*H_last*W_last, 160]
        attention_maps: list of spatial attention maps
        """
        attention_maps = []

        if not self.return_dense:
            x, tokens, new_attn = self.forward_features(x)  # [B, 160]
            attention_maps.extend(new_attn)
            x = self.head(x)                                 # [B, num_classes]
            return x, tokens, attention_maps

        else:
            raise NotImplementedError(
                "Dense token-label mode is not yet adapted for video input. "
                "Set token_label=False (default).")


# ---------------------------------------------------------------------------
# painformer()  factory
# ---------------------------------------------------------------------------

@register_model
def painformer_video(pretrained=False, T=8, **kwargs):
    """
    PainFormer video encoder (spatiotemporal).

    Parameters
    ----------
    pretrained : bool   – no pre-trained weights for video version
    T          : int    – number of input frames (default 8)
    **kwargs   : passed to SpectFormer (e.g. num_classes=4)

    Input  : [B, T, 3, 224, 224]
    Output : ([B, num_classes], [B, T*49, 160], attention_maps)
             tokens shape: T=8 → [B, 392, 160]
    """
    model = SpectFormer(
        T=T,
        stem_hidden_dim=64,
        embed_dims=[64, 128, 320, 160],
        num_heads=[2, 4, 10, 16],
        mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 4, 12, 3],
        sr_ratios=[4, 2, 1, 1],
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("PainFormer Video – Sanity Check")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Build model
    model = painformer_video(T=8, num_classes=4).to(device)
    model.eval()

    # Dummy video input: [B=2, T=8, C=3, H=224, W=224]
    x = torch.randn(2, 8, 3, 224, 224, device=device)

    with torch.no_grad():
        out, tokens, attn_maps = model(x)

    print(f"\nout.shape    : {out.shape}")       # [2, 4]
    print(f"tokens.shape : {tokens.shape}")     # [2, 392, 160]
    print(f"#attn_maps   : {len(attn_maps)}")   # one per std_att block (stages 2+3)
    if attn_maps:
        print(f"attn_maps[0] : {attn_maps[0].shape}")  # [B*T, nh, N, N]

    # Parameter count
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable parameters: {n_params / 1e6:.2f} M")

    assert out.shape    == (2, 4),         f"Wrong out shape: {out.shape}"
    assert tokens.shape == (2, 392, 160),  f"Wrong tokens shape: {tokens.shape}"
    print("\n[OK]  All assertions passed.")
