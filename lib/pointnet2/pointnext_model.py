"""
PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies.

Qian et al., NeurIPS 2022.

Standalone implementation that plugs into the existing precastconcrete-qc pipeline.
Forward interface: forward(xyz, geo_features=None, scale=None) -> (B, N, C)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

# ── reuse existing utilities from model.py ──
from .model import (
    farthest_point_sample,
    ball_query,
    _index_points,
    _square_distance,
)


# ═══════════════════════════════════════════════════════════════
#  Channel mapping for feature_type
# ═══════════════════════════════════════════════════════════════

def _map_channels(feature_type: str, in_ch: int) -> int:
    """Compute effective input channels after feature concatenation."""
    if feature_type == 'dp_fj':          # delta_xyz + neighbor_features
        return 3 + in_ch
    elif feature_type == 'fj':           # just neighbor features
        return in_ch
    elif feature_type == 'dp':           # just delta_xyz
        return 3
    elif feature_type == 'df':           # feature differences
        return in_ch
    elif feature_type == 'dp_fj_df':     # delta_xyz + neighbor_feat + feat_diff
        return 3 + in_ch * 2
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")


def _get_aggregation_features(
    p: torch.Tensor, dp: torch.Tensor, f: torch.Tensor, fj: torch.Tensor,
    feature_type: str = 'dp_fj',
) -> torch.Tensor:
    """Concatenate features according to feature_type."""
    if feature_type == 'dp_fj':
        return torch.cat([dp, fj], dim=1)
    elif feature_type == 'fj':
        return fj
    elif feature_type == 'dp':
        return dp
    elif feature_type == 'dp_fj_df':
        df = fj - f.unsqueeze(-1)
        return torch.cat([dp, fj, df], dim=1)
    elif feature_type == 'df':
        df = fj - f.unsqueeze(-1)
        return df
    else:
        return torch.cat([dp, fj], dim=1)


# ═══════════════════════════════════════════════════════════════
#  Conv block helpers (match PointNeXt design)
# ═══════════════════════════════════════════════════════════════

class Conv1dBN(nn.Module):
    """1x1 Conv1d + BatchNorm + ReLU (optional)."""
    def __init__(self, in_ch: int, out_ch: int, act: bool = True):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Conv2dBN(nn.Module):
    """1x1 Conv2d + BatchNorm + ReLU (optional)."""
    def __init__(self, in_ch: int, out_ch: int, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


# ═══════════════════════════════════════════════════════════════
#  LocalAggregation — depthwise conv on grouped neighbors
# ═══════════════════════════════════════════════════════════════

class LocalAggregation(nn.Module):
    """
    Aggregates neighbor features via ball query → shared conv → max pool.

    This is the core building block used inside InvResMLP.
    """
    def __init__(
        self,
        channels: List[int],
        radius: float = 0.1,
        nsample: int = 16,
        feature_type: str = 'dp_fj',
        reduction: str = 'max',
    ):
        super().__init__()
        self.feature_type = feature_type
        self.radius = radius
        self.nsample = nsample

        # adjust first channel for feature_type
        ch0 = _map_channels(feature_type, channels[0])
        convs = []
        for i in range(len(channels) - 1):
            last = (i == len(channels) - 2)
            convs.append(Conv2dBN(ch0 if i == 0 else channels[i],
                                  channels[i + 1],
                                  act=not last))
        self.convs = nn.Sequential(*convs)

        if reduction == 'max':
            self.pool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]
        elif reduction == 'mean':
            self.pool = lambda x: torch.mean(x, dim=-1, keepdim=False)
        else:
            raise ValueError(f"Unknown reduction: {reduction}")

    def forward(
        self, p: torch.Tensor, f: torch.Tensor
    ) -> torch.Tensor:
        """p: (B, N, 3), f: (B, C, N) -> (B, C_out, N)"""
        B, N = p.shape[:2]

        # ball query: find neighbors
        idx = ball_query(self.radius, self.nsample, p, p)  # (B, N, nsample)

        # grouped relative positions
        grouped_xyz = _index_points(p, idx)  # (B, N, nsample, 3)
        dp = grouped_xyz - p.unsqueeze(2)     # (B, N, nsample, 3)
        dp = dp.permute(0, 3, 1, 2)           # (B, 3, N, nsample)

        # grouped features
        if f is not None:
            fj = _index_points(f.transpose(-1, -2), idx)  # (B, N, nsample, C)
            fj = fj.permute(0, 3, 1, 2)        # (B, C, N, nsample)
        else:
            fj = None

        # aggregate features
        agg = _get_aggregation_features(p, dp, f, fj, self.feature_type)
        out = self.convs(agg)   # (B, C_out, N, nsample)
        out = self.pool(out)    # (B, C_out, N)
        return out


# ═══════════════════════════════════════════════════════════════
#  InvResMLP — Inverted Residual MLP block
# ═══════════════════════════════════════════════════════════════

class InvResMLP(nn.Module):
    """
    Inverted Residual MLP: LocalAggregation → pointwise expand → compress.

    Design:  channels → (expand) → channels
    """
    def __init__(
        self,
        channels: int,
        expansion: float = 4.0,
        radius: float = 0.1,
        nsample: int = 16,
        feature_type: str = 'dp_fj',
    ):
        super().__init__()
        mid_channels = int(channels * expansion)

        # Depthwise (local aggregation)
        self.local = LocalAggregation(
            [channels, channels],
            radius=radius, nsample=nsample,
            feature_type=feature_type,
        )
        # Pointwise conv: expand then compress back
        self.pw = nn.Sequential(
            Conv1dBN(channels, mid_channels, act=True),
            Conv1dBN(mid_channels, channels, act=False),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, p: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        identity = f
        f = self.local(p, f)
        f = self.pw(f)
        if f.shape == identity.shape:
            f = self.act(f + identity)
        else:
            f = self.act(f)
        return p, f


# ═══════════════════════════════════════════════════════════════
#  PointNeXt SetAbstraction (downsampling)
# ═══════════════════════════════════════════════════════════════

class PointNextSetAbstraction(nn.Module):
    """
    Downsampling SetAbstraction: FPS → ball query → conv → max pool.

    Supports residual skip connection and feature_type.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        radius: float = 0.1,
        nsample: int = 16,
        feature_type: str = 'dp_fj',
        use_res: bool = False,
        num_layers: int = 1,
    ):
        super().__init__()
        self.stride = stride
        self.radius = radius
        self.nsample = nsample
        self.feature_type = feature_type
        self.use_res = use_res

        # build conv blocks
        mid_ch = out_channels // 2 if stride > 1 else out_channels
        ch = [in_channels] + [mid_ch] * (num_layers - 1) + [out_channels]
        ch[0] = _map_channels(feature_type, ch[0])

        convs = []
        for i in range(len(ch) - 1):
            convs.append(Conv2dBN(ch[i], ch[i + 1],
                                  act=not (use_res and i == len(ch) - 2)))
        self.convs = nn.Sequential(*convs)

        # residual skip
        if use_res:
            self.skip = (Conv1dBN(in_channels, out_channels, act=False)
                         if in_channels != out_channels else nn.Identity())
            self.act = nn.ReLU(inplace=True)

        # pooling
        self.pool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]

    def forward(
        self, p: torch.Tensor, f: torch.Tensor
    ) -> tuple:
        """Returns (new_p, new_f)"""
        B, N, _ = p.shape

        # FPS downsampling
        if self.stride > 1:
            npoint = N // self.stride
            fps_idx = farthest_point_sample(p, npoint)
            new_p = _index_points(p, fps_idx.unsqueeze(-1)).squeeze(2)
            idx = fps_idx
        else:
            new_p = p
            idx = None

        # ball query
        idx_bq = ball_query(self.radius, self.nsample, p, new_p)  # (B, M, nsample)
        grouped_xyz = _index_points(p, idx_bq)  # (B, M, nsample, 3)
        dp = grouped_xyz - new_p.unsqueeze(2)
        dp = dp.permute(0, 3, 1, 2)  # (B, 3, M, nsample)

        if f is not None:
            fj = _index_points(f.transpose(-1, -2), idx_bq)
            fj = fj.permute(0, 3, 1, 2)  # (B, C, M, nsample)
        else:
            fj = None

        agg = _get_aggregation_features(new_p, dp, f, fj, self.feature_type)
        out = self.convs(agg)   # (B, C_out, M, nsample)
        out = self.pool(out)    # (B, C_out, M)

        if self.use_res and idx is not None and f is not None:
            # gather features for residual
            gathered = _index_points(f.transpose(-1, -2), idx.unsqueeze(-1)).squeeze(2)
            gathered = gathered.transpose(-1, -2)  # (B, C, M)
            out = self.act(out + self.skip(gathered))

        return new_p, out


# ═══════════════════════════════════════════════════════════════
#  PointNeXt FeaturePropagation (upsampling)
# ═══════════════════════════════════════════════════════════════

class PointNextFeaturePropagation(nn.Module):
    """
    Upsampling via 3-nearest-neighbor interpolation + skip + MLP.
    """
    def __init__(self, mlp: List[int]):
        super().__init__()
        convs = []
        for i in range(len(mlp) - 1):
            convs.append(Conv1dBN(mlp[i], mlp[i + 1], act=True))
        self.convs = nn.Sequential(*convs)

    def forward(
        self,
        p1: torch.Tensor, f1: Optional[torch.Tensor],  # fine level (target)
        p2: torch.Tensor, f2: torch.Tensor,             # coarse level (source)
    ) -> torch.Tensor:
        """Returns upsampled features at p1 resolution."""
        B, N1, _ = p1.shape
        N2 = p2.shape[1]

        if N2 == 1:
            interpolated = f2.expand(-1, -1, N1)
        else:
            # 3-NN weighted interpolation
            sq = _square_distance(p1, p2)  # (B, N1, N2)
            dist, idx = sq.topk(3, dim=-1, largest=False)  # (B, N1, 3)
            w = 1.0 / (dist + 1e-8)
            w = w / w.sum(dim=-1, keepdim=True)
            feat_src = _index_points(f2.transpose(-1, -2), idx)  # (B, N1, 3, C)
            interpolated = (feat_src * w.unsqueeze(-1)).sum(dim=2)  # (B, N1, C)
            interpolated = interpolated.transpose(-1, -2)  # (B, C, N1)

        if f1 is not None:
            cat = torch.cat([interpolated, f1], dim=1)
        else:
            cat = interpolated
        return self.convs(cat)


# ═══════════════════════════════════════════════════════════════
#  PointNeXt Segmentation Model (full encoder-decoder-head)
# ═══════════════════════════════════════════════════════════════

class PointNeXtSeg(nn.Module):
    """
    PointNeXt for semantic segmentation.

    Designed as a drop-in replacement for PointNetPlusPlus in the
    precastconcrete-qc pipeline.

    Forward: (B, N, 3) xyz, (B, N, 3) geo_features, (B,) scale
      → (B, N, num_classes) logits

    Args:
        num_classes: number of output classes (default 3)
        in_channels: input feature channels (default 6 = 3 XYZ + 3 geo)
        width: base channel width (default 32)
        blocks: number of InvResMLP blocks per stage (default [1, 2, 3, 2, 3])
        strides: downsampling stride per stage (default [4, 4, 4, 4])
        expansion: InvResMLP expansion ratio (default 4)
        nsample: neighbors for ball query (default 24)
        radius: initial ball query radius in normalized space (default 0.1)
        use_res_sa: use residual in SA blocks (default False for small model)
    """
    def __init__(
        self,
        num_classes: int = 3,
        in_channels: int = 6,
        width: int = 32,
        blocks: List[int] = None,
        strides: List[int] = None,
        expansion: float = 4.0,
        nsample: int = 24,
        radius: float = 0.1,
        use_res_sa: bool = False,
    ):
        super().__init__()
        if blocks is None:
            blocks = [1, 2, 3, 2, 3]   # PointNeXt-S style
        if strides is None:
            strides = [4, 4, 4, 4]     # each stage downsamples 4x

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.blocks = blocks
        self.strides = strides
        self.radius_base = radius

        # ── Encoder: build stages ──
        self.enc_channels = []
        encoder = []
        cur_ch = in_channels
        w = width
        for stage_idx, (n_blocks, stride) in enumerate(zip(blocks, strides)):
            if stride != 1 and stage_idx > 0:
                w *= 2
            stage_channels = w
            self.enc_channels.append(stage_channels)

            stage_modules = []

            # First block: SA for downsampling
            stage_modules.append(PointNextSetAbstraction(
                in_channels=cur_ch,
                out_channels=stage_channels,
                stride=stride,
                radius=self.radius_base,
                nsample=nsample,
                use_res=use_res_sa,
                num_layers=1,
            ))
            cur_ch = stage_channels

            # Remaining blocks: InvResMLP for refinement
            for _ in range(n_blocks - 1):
                stage_modules.append(InvResMLP(
                    channels=cur_ch,
                    expansion=expansion,
                    radius=self.radius_base,
                    nsample=nsample,
                ))

            encoder.append(nn.Sequential(*stage_modules))

        self.encoder = nn.ModuleList(encoder)

        # ── Decoder: Feature Propagation stages ──
        # upsample from deepest back to original resolution
        decoder = []
        dec_in = self.enc_channels[-1]
        skip_list = list(reversed(self.enc_channels[:-1])) + [in_channels]
        for skip_ch in skip_list:
            mlp = [skip_ch + dec_in, skip_ch, skip_ch]
            decoder.append(PointNextFeaturePropagation(mlp))
            dec_in = skip_ch
        self.decoder = nn.ModuleList(decoder)

        # ── Head ──
        head_in = dec_in  # channels after final decoder stage
        self.head = nn.Sequential(
            Conv1dBN(head_in, 64, act=True),
            nn.Dropout(0.3),
            nn.Conv1d(64, num_classes, 1),
        )

    def forward(
        self,
        xyz: torch.Tensor,
        geo_features: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            xyz: (B, N, 3) point coordinates
            geo_features: (B, N, 3) geometric features (linearity, planarity, sphericity)
            scale: (B,) normalization scale in mm for dynamic radius scaling
        Returns:
            logits: (B, N, num_classes)
        """
        B, N = xyz.shape[:2]

        # Build input features
        if geo_features is not None:
            f = torch.cat([xyz, geo_features], dim=-1)  # (B, N, C_in)
        else:
            f = xyz  # (B, N, 3)
        f = f.transpose(-1, -2)  # (B, C_in, N)

        # Store original point positions for the decoder's last skip
        p_orig = xyz
        f_orig = f

        # Scale-aware radius
        if scale is not None:
            scale = scale.to(xyz.device, non_blocking=True)
            # Map 80mm physical radius to normalized space
            r_eff = 80.0 / scale.view(-1, 1).float()  # (B, 1)
        else:
            r_eff = None

        # ── Encoder forward with intermediate storage for skip ──
        # First entry is the original resolution
        p_skips = [p_orig]
        f_skips = [f_orig]

        p = xyz
        for stage_idx, stage in enumerate(self.encoder):
            # Update radii for SA blocks with scale
            for module in stage:
                if isinstance(module, PointNextSetAbstraction):
                    if r_eff is not None:
                        module.radius = r_eff.mean().item()
                elif isinstance(module, InvResMLP):
                    if r_eff is not None:
                        module.local.radius = (r_eff.mean().item() * 0.5)
                p, f = module(p, f)
            p_skips.append(p)
            f_skips.append(f)

        # ── Decoder: upsample with skip connections ──
        f = f_skips[-1]
        for i, dec_stage in enumerate(self.decoder):
            p_target = p_skips[-(i + 2)]
            f_skip = f_skips[-(i + 2)]
            f = dec_stage(p_target, f_skip, p_skips[-(i + 1)], f)

        # ── Head ──
        logits = self.head(f)  # (B, num_classes, N)
        logits = logits.transpose(-1, -2)  # (B, N, num_classes)
        return logits


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
