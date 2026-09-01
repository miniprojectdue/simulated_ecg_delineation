#!/usr/bin/env python3
"""
model.py  -  the 1-D U-Net that performs the delineation, its lead conditioning and its
attention.

Per-sample four-class labelling over a 1280-sample crop: the one-dimensional analogue of the
standard encoder-decoder with skip connections, plus two attention mechanisms acting at two
resolutions -- bottleneck self-attention over 80 tokens at 32 ms, and attention gates on the
skips at 2 to 16 ms per token.

Every attention setting reads from the config and defaults to off, so omitting the
model.attention block builds the convolution-only network. That is what the removal arms of
the ablation table do, and it is why those comparisons differ in one setting rather than in a
codebase.

FiLM conditioning, the attention output projection and the gate projection are all
zero-initialised, so at step zero this network computes exactly what the published one
computes and any difference is something training chose.
"""
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import N_CLASSES, N_LEADS  # noqa: E402


def make_norm(kind, channels):
    if kind == 'batch':
        return nn.BatchNorm1d(channels)
    if kind == 'group':
        groups = 8
        while groups > 1 and channels % groups:
            groups //= 2
        return nn.GroupNorm(groups, channels)
    if kind in (None, 'none'):
        return nn.Identity()
    raise ValueError('unknown norm %r' % kind)


def make_activation(kind):
    if kind == 'relu':
        return nn.ReLU(inplace=True)
    if kind == 'gelu':
        return nn.GELU()
    if kind == 'silu':
        return nn.SiLU(inplace=True)
    raise ValueError('unknown activation %r' % kind)


class ConvBlock(nn.Module):
    """Two padded convolutions, each followed by normalisation and an activation."""

    def __init__(self, in_ch, out_ch, kernel_size=9, norm='batch', activation='relu', dropout=0.0):
        super(ConvBlock, self).__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, bias=(norm in (None, 'none')))
        self.norm1 = make_norm(norm, out_ch)
        self.act1 = make_activation(activation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, bias=(norm in (None, 'none')))
        self.norm2 = make_norm(norm, out_ch)
        self.act2 = make_activation(activation)
        self.drop = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, x):
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.drop(x)
        x = self.act2(self.norm2(self.conv2(x)))
        return x


class FiLM(nn.Module):
    """Per-channel scale and shift produced from the lead embedding."""

    def __init__(self, embed_dim, channels):
        super(FiLM, self).__init__()
        self.to_gamma = nn.Linear(embed_dim, channels)
        self.to_beta = nn.Linear(embed_dim, channels)
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.zeros_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, x, embedding):
        gamma = 1.0 + self.to_gamma(embedding).unsqueeze(-1)
        beta = self.to_beta(embedding).unsqueeze(-1)
        return x * gamma + beta


def sinusoidal_positions(length, channels, device, dtype):
    """Fixed sinusoidal positional encoding, so the block works at any crop length."""
    pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    idx = torch.arange(0, channels, 2, device=device, dtype=torch.float32)
    div = torch.exp(-math.log(10000.0) * idx / channels)
    enc = torch.zeros(length, channels, device=device, dtype=torch.float32)
    enc[:, 0::2] = torch.sin(pos * div)
    enc[:, 1::2] = torch.cos(pos * div)[:, :enc[:, 1::2].shape[1]]
    return enc.to(dtype).unsqueeze(0)


class BottleneckAttention(nn.Module):
    """Pre-norm multi-head self-attention with a zero-initialised output projection.

    Attention is permutation invariant, so positional handling is explicit. ``qk`` adds the
    fixed encoding to query and key but keeps values content-only; ``none`` relies on the local
    band and convolutional features; ``legacy_qkv`` reproduces older checkpoints that added the
    encoding to all three streams. It is never added directly to the residual stream, which keeps
    the identity property at initialisation exact.

    neighborhood = 0 gives global attention. A positive value restricts each token to that many
    neighbours on either side through an additive mask, which is the local attention used by the
    neighborhood transformer literature and costs the same to run at this sequence length.
    """

    def __init__(self, channels, heads=8, neighborhood=0, dropout=0.0,
                 position_encoding='legacy_qkv', mask_invalid=False):
        super(BottleneckAttention, self).__init__()
        while heads > 1 and channels % heads:
            heads //= 2
        self.heads = heads
        self.neighborhood = int(neighborhood)
        self.position_encoding = str(position_encoding or 'legacy_qkv').lower()
        self.mask_invalid = bool(mask_invalid)
        if self.position_encoding not in ('legacy_qkv', 'qk', 'none'):
            raise ValueError('attention.position_encoding %r is not one of legacy_qkv, qk or none'
                             % self.position_encoding)
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        nn.init.zeros_(self.attn.out_proj.weight)
        if self.attn.out_proj.bias is not None:
            nn.init.zeros_(self.attn.out_proj.bias)
        self.last_map = None

    def band_mask(self, length, device):
        i = torch.arange(length, device=device)
        return (i.unsqueeze(0) - i.unsqueeze(1)).abs() > self.neighborhood

    def token_validity(self, valid, length):
        """Downsample an input validity mask to the bottleneck token grid."""
        if valid is None:
            return None
        mask = valid.to(dtype=torch.float32)
        if mask.shape[-1] != length:
            mask = F.adaptive_max_pool1d(mask.unsqueeze(1), length).squeeze(1)
        return mask > 0

    def forward(self, x, valid=None, keep_map=False):
        b, c, t = x.shape
        content = self.norm(x.transpose(1, 2))                  # B, T, C
        if self.position_encoding == 'none':
            query = key = value = content
        else:
            positioned = content + sinusoidal_positions(t, c, x.device, content.dtype)
            query = key = positioned
            # legacy_qkv exactly reproduces checkpoints trained before this option existed.
            # qk implements the documented design: absolute position chooses what is attended
            # to, but is not itself written into the residual value stream.
            value = positioned if self.position_encoding == 'legacy_qkv' else content

        mask = self.band_mask(t, x.device) if self.neighborhood > 0 else None
        token_valid = self.token_validity(valid, t) if self.mask_invalid else None
        key_padding = None if token_valid is None else ~token_valid
        if key_padding is not None:
            # MultiheadAttention cannot handle a row whose every key is masked. Such a crop is
            # invalid upstream, but leaving one harmless key open keeps a diagnostic from
            # turning the complete batch into NaNs before the loader can report the bad unit.
            empty = key_padding.all(dim=1)
            if bool(empty.any()):
                key_padding = key_padding.clone()
                key_padding[empty, 0] = False
        out, weights = self.attn(query, key, value, attn_mask=mask,
                                 key_padding_mask=key_padding, need_weights=keep_map,
                                 average_attn_weights=True)
        if token_valid is not None:
            out = out * token_valid.unsqueeze(-1).to(out.dtype)
        if keep_map:
            self.last_map = weights.detach()
        return x + out.transpose(1, 2)


class SkipAttentionGate(nn.Module):
    """Gate an encoder skip on the decoder features that are about to consume it.

    The gate equals one everywhere at initialisation, since the projection that produces it is
    zero-initialised and the gate is twice a sigmoid. It can therefore suppress a skip towards
    zero or amplify it up to twofold, and it begins by doing neither.
    """

    def __init__(self, skip_ch, gate_ch, reduction=2):
        super(SkipAttentionGate, self).__init__()
        inter = max(1, skip_ch // reduction)
        self.theta = nn.Conv1d(skip_ch, inter, kernel_size=1, bias=False)
        self.phi = nn.Conv1d(gate_ch, inter, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.psi = nn.Conv1d(inter, 1, kernel_size=1, bias=True)
        nn.init.zeros_(self.psi.weight)
        nn.init.zeros_(self.psi.bias)
        self.last_gate = None

    def forward(self, skip, gate, keep_map=False):
        g = self.phi(gate)
        if g.shape[-1] != skip.shape[-1]:
            g = F.interpolate(g, size=skip.shape[-1], mode='nearest')
        a = 2.0 * torch.sigmoid(self.psi(self.act(self.theta(skip) + g)))
        if keep_map:
            self.last_gate = a.detach()
        return skip * a



class PObservabilityHead(nn.Module):
    """Does the observed part of this crop contain a P wave?

    The pooling is validity aware. A crop truncated at the QRS onset is sixty one per cent
    padding, and a mean over the whole token grid would be sixty one per cent a constant, so the
    pooled vector would describe how much of the record is missing rather than what is in it.
    Masking the pool to the observed tokens removes that shortcut from the pooling itself. What
    stops the head reading the answer off the validity mask more generally is the augmentation
    draw rather than anything here, since truncation and P removal are drawn separately.
    """

    def __init__(self, channels, hidden=128, dropout=0.0):
        super(PObservabilityHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(channels, hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, 1))

    def forward(self, h, valid=None):
        # h is [B, C, T_b] at the bottleneck resolution
        if valid is not None:
            m = valid.to(h.dtype)
            if m.shape[-1] != h.shape[-1]:
                m = F.adaptive_max_pool1d(m.unsqueeze(1), h.shape[-1]).squeeze(1)
            m = m.unsqueeze(1)
            denom = m.sum(dim=-1).clamp(min=1.0)
            pooled = (h * m).sum(dim=-1) / denom
        else:
            pooled = h.mean(dim=-1)
        return self.net(pooled).squeeze(-1)


class UNet1D(nn.Module):
    """Encoder, bottleneck and decoder over a one-dimensional signal."""

    def __init__(self, in_channels=N_LEADS, n_classes=N_CLASSES, base_width=32, depth=5,
                 kernel_size=9, dropout=0.1, norm='batch', activation='relu',
                 lead_embedding_dim=32, lead_conditioning='film', n_leads=N_LEADS,
                 attention=None, p_head=None):
        super(UNet1D, self).__init__()
        if depth < 2:
            raise ValueError('depth must be at least 2')
        self.depth = depth
        self.lead_conditioning = lead_conditioning
        self.divisor = 2 ** (depth - 1)

        widths = [base_width * (2 ** i) for i in range(depth)]
        self.widths = widths

        if lead_conditioning in ('film', 'concat'):
            self.lead_embed = nn.Embedding(n_leads, lead_embedding_dim)
        else:
            self.lead_embed = None

        stem_in = in_channels + (lead_embedding_dim if lead_conditioning == 'concat' else 0)

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev = stem_in
        for i in range(depth - 1):
            self.encoders.append(ConvBlock(prev, widths[i], kernel_size, norm, activation, dropout))
            self.pools.append(nn.MaxPool1d(2))
            prev = widths[i]
        self.bottleneck = ConvBlock(prev, widths[-1], kernel_size, norm, activation, dropout)

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev = widths[-1]
        for i in range(depth - 2, -1, -1):
            self.ups.append(nn.ConvTranspose1d(prev, widths[i], kernel_size=2, stride=2))
            self.decoders.append(ConvBlock(widths[i] * 2, widths[i], kernel_size, norm, activation, dropout))
            prev = widths[i]

        if lead_conditioning == 'film':
            self.films = nn.ModuleList([FiLM(lead_embedding_dim, w) for w in widths[:-1]])
            self.film_bottleneck = FiLM(lead_embedding_dim, widths[-1])
            self.film_decoders = nn.ModuleList(
                [FiLM(lead_embedding_dim, widths[i]) for i in range(depth - 2, -1, -1)])
        else:
            self.films = None
            self.film_bottleneck = None
            self.film_decoders = None

        a = dict(attention or {})
        self.attn_enabled = bool(a.get('bottleneck', False))
        self.gates_enabled = bool(a.get('skip_gates', False))
        self.keep_maps = bool(a.get('keep_maps', False))

        if self.attn_enabled:
            self.bottleneck_attn = BottleneckAttention(
                widths[-1], heads=int(a.get('heads', 8)),
                neighborhood=int(a.get('neighborhood', 0)),
                dropout=float(a.get('attn_dropout', 0.0)),
                position_encoding=a.get('position_encoding', 'legacy_qkv'),
                mask_invalid=bool(a.get('mask_invalid', False)))
        else:
            self.bottleneck_attn = None

        if self.gates_enabled:
            red = int(a.get('gate_reduction', 2))
            self.skip_gates = nn.ModuleList(
                [SkipAttentionGate(widths[i], widths[i], reduction=red)
                 for i in range(depth - 2, -1, -1)])
        else:
            self.skip_gates = None

        self.head = nn.Conv1d(widths[0], n_classes, kernel_size=1)

        ph = dict(p_head or {})
        self.p_head = (PObservabilityHead(widths[-1], int(ph.get('hidden', 128)),
                                          float(ph.get('dropout', 0.0)))
                       if ph.get('enabled', False) else None)

    def encoder_parameters(self):
        """Everything up to and including the bottleneck. Used by the fine-tuning strategies."""
        modules = [self.encoders, self.bottleneck]
        if self.bottleneck_attn is not None:
            modules.append(self.bottleneck_attn)
        if self.lead_embed is not None:
            modules.append(self.lead_embed)
        # FiLM modulates the encoder and bottleneck activations. Omitting it here silently gave
        # these tensors the decoder learning rate and left them trainable under freeze=encoder.
        if self.films is not None:
            modules.extend([self.films, self.film_bottleneck])
        for module in modules:
            for param in module.parameters():
                yield param

    def decoder_parameters(self):
        modules = [self.ups, self.decoders, self.head]
        if getattr(self, 'p_head', None) is not None:
            modules.append(self.p_head)
        if self.film_decoders is not None:
            modules.append(self.film_decoders)
        if self.skip_gates is not None:
            modules.append(self.skip_gates)
        for module in modules:
            for param in module.parameters():
                yield param

    def forward(self, x, lead_idx=None, valid=None):
        length = x.shape[-1]
        if length % self.divisor:
            raise ValueError('input length %d is not a multiple of %d, which the %d level U-Net '
                             'needs for the skip connections to line up'
                             % (length, self.divisor, self.depth))

        embedding = None
        if self.lead_embed is not None:
            if lead_idx is None:
                raise ValueError('lead conditioning is on, so lead_idx must be supplied')
            embedding = self.lead_embed(lead_idx)
            if self.lead_conditioning == 'concat':
                x = torch.cat([x, embedding.unsqueeze(-1).expand(-1, -1, length)], dim=1)

        skips = []
        for i, (block, pool) in enumerate(zip(self.encoders, self.pools)):
            x = block(x)
            if self.films is not None:
                x = self.films[i](x, embedding)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)
        if self.bottleneck_attn is not None:
            x = self.bottleneck_attn(x, valid=valid, keep_map=self.keep_maps)
        if self.film_bottleneck is not None:
            x = self.film_bottleneck(x, embedding)

        # The auxiliary branch reads the bottleneck after attention and conditioning and does
        # not feed back into the decoder, so the segmentation path is unchanged by its presence.
        p_logit = self.p_head(x, valid) if self.p_head is not None else None

        for i, (up, block) in enumerate(zip(self.ups, self.decoders)):
            x = up(x)
            skip = skips[-(i + 1)]
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-1], mode='nearest')
            if self.skip_gates is not None:
                skip = self.skip_gates[i](skip, x, keep_map=self.keep_maps)
            x = block(torch.cat([x, skip], dim=1))
            if self.film_decoders is not None:
                x = self.film_decoders[i](x, embedding)

        logits = self.head(x)
        return (logits, p_logit) if p_logit is not None else logits


def build_model(cfg):
    """Instantiate the model named by a merged config."""
    m = cfg['model']
    name = str(m.get('name', 'unet1d')).lower()
    if name != 'unet1d':
        raise SystemExit('model.name %r is not implemented, only unet1d is' % name)
    return UNet1D(
        in_channels=int(m.get('in_channels', N_LEADS)),
        n_classes=int(m.get('n_classes', N_CLASSES)),
        base_width=int(m.get('base_width', 32)),
        depth=int(m.get('depth', 5)),
        kernel_size=int(m.get('kernel_size', 9)),
        dropout=float(m.get('dropout', 0.0)),
        norm=m.get('norm', 'batch'),
        activation=m.get('activation', 'relu'),
        lead_embedding_dim=int(m.get('lead_embedding_dim', 32)),
        lead_conditioning=m.get('lead_conditioning', 'film'),
        attention=m.get('attention'),
        p_head=m.get('p_head'),
    )


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable}
