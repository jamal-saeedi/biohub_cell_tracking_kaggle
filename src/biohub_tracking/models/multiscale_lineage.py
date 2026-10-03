"""`MultiScaleLineageNet`: `IsotropicLineageNet`'s contract with a richer trunk and linker.

Differences from `IsotropicLineageNet`:

1. Gated temporal fusion at the coarse (2,8,8) and isotropic (1,4,4) scales,
   several learned displaced samples per neighbour frame (`GatedTemporalFusion`).
2. One residual adapter each for detection and association on top of the
   shared decoder; `DetectionOutput.features` is the association adapter's map.
3. A learned node descriptor (`LearnedDescriptorSampler`, width 3C): the fixed
   centre + local average plus attention-pooled samples at learned offsets.
4. Motion refined inside the association graph (`RefiningAssociationHead`).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from biohub_tracking.models.isotropic_lineage import (
    AssociationOutput,
    DetectionOutput,
    IsotropicLineageNet,
    ResidualBlock,
    SparseAssociationBlock,
    conv_block,
    edge_geometry,
    normalized_grid,
    resize_at_stride,
    sample_node_features,
    segment_softmax,
    voxel_grid,
)

__all__ = [
    "GatedTemporalFusion",
    "LearnedDescriptorSampler",
    "MultiScaleLineageNet",
    "RefiningAssociationHead",
    "build_lineage_model",
]

#: Nominal voxel spacing (Z, Y, X) in microns, which sets the sampling bounds in cells.
NOMINAL_SPACING_ZYX: tuple[float, float, float] = (1.625, 0.40625, 0.40625)


def _cells(radius_um: float, stride_zyx: tuple[int, int, int]) -> tuple[float, float, float]:
    """A physical radius in cells of the grid at `stride_zyx`."""
    return tuple(radius_um / (s * k) for s, k in zip(NOMINAL_SPACING_ZYX, stride_zyx))


class GatedTemporalFusion(nn.Module):
    """`samples` learned displaced samples per neighbour frame, attention over all
    of them, and a per-voxel gate on the result:

        out_t = x_t + g_t * W( sum_{s != t, k} a_{t,s,k} x_s(p + d_{t,s,k}) )

    Displacements are bounded per axis by `radius_cells` per elapsed interval.
    The offset and attention logits come from `conv(cat(x_t, x_s))`, computed as
    `ref(x_t) + nbr(x_s)` so each frame is convolved once per role (`project`).
    """

    def __init__(self, channels: int, samples: int, radius_cells: tuple[float, float, float]) -> None:
        super().__init__()
        self.samples = samples
        self.radius_cells = tuple(float(r) for r in radius_cells)
        self.ref = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.nbr = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.elapsed = nn.Parameter(torch.zeros(channels))
        self.alignment = nn.Sequential(
            nn.GroupNorm(math.gcd(8, channels), channels), nn.GELU(),
            nn.Conv3d(channels, 4 * samples, 1),
        )
        self.mix = nn.Conv3d(channels, channels, 1)
        self.gate = nn.Conv3d(2 * channels, 1, 1)

    def project(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """`(ref(x_t), nbr(x_t))` per frame, B,T,C,Z,Y,X each (cacheable per frame)."""
        batch, frames = x.shape[:2]
        flat = x.flatten(0, 1)
        return (self.ref(flat).unflatten(0, (batch, frames)),
                self.nbr(flat).unflatten(0, (batch, frames)))

    def forward(
        self,
        x: Tensor,
        times: Tensor,
        positions: list[int] | None = None,
        projected: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        """Fused features for the reference frames `positions` (`None` = all);
        `projected` = `self.project(x)`."""
        batch, frames, channels = x.shape[:3]
        if frames == 1:
            return x if positions is None else x[:, positions]
        shape = tuple(x.shape[-3:])
        k = self.samples
        # float32 grids: half precision cannot index a 64-cell axis to sub-cell accuracy.
        base = voxel_grid(shape, x.new_zeros((), dtype=torch.float32))  # Z,Y,X,3
        base = base.movedim(-1, 0)[None, None]  # 1,1,3,Z,Y,X
        radius = torch.tensor(self.radius_cells, dtype=torch.float32, device=x.device)
        radius = radius.view(1, 1, 3, 1, 1, 1)
        ref, nbr = self.project(x) if projected is None else projected
        outputs = []
        for i in range(frames) if positions is None else positions:
            values, logits = [], []
            for j in range(frames):
                if j == i:
                    continue
                dt = (times[:, j] - times[:, i]).float()
                elapsed = (dt[:, None] * self.elapsed.float()).view(batch, channels, 1, 1, 1)
                params = self.alignment(ref[:, i] + nbr[:, j] + elapsed.to(ref.dtype))
                params = params.float().view(batch, k, 4, *shape)
                reach = radius * dt.abs().clamp_min(1.0).view(batch, 1, 1, 1, 1, 1)
                coords = base + reach * params[:, :, :3].tanh()  # B,K,3,Z,Y,X
                grid = normalized_grid(coords.movedim(2, -1), shape)  # B,K,Z,Y,X,3
                grid = grid.reshape(batch, k * shape[0], *shape[1:], 3)
                sampled = F.grid_sample(
                    x[:, j], grid.to(x.dtype), align_corners=False, padding_mode="border"
                )
                values.append(sampled.view(batch, channels, k, *shape))
                logits.append(params[:, :, 3])
            weights = torch.cat(logits, 1).softmax(1).to(x.dtype)  # B,(T-1)K,Z,Y,X
            context = torch.einsum("bcnzyx,bnzyx->bczyx", torch.cat(values, 2), weights)
            gate = torch.sigmoid(self.gate(torch.cat((x[:, i], context), 1)))
            outputs.append(x[:, i] + gate * self.mix(context))
        return torch.stack(outputs, 1)


class LearnedDescriptorSampler(nn.Module):
    """Node descriptor = [centre, local average, attention-pooled learned samples], width 3C.

    The first two parts are `sample_node_features`; from them a small MLP
    predicts `samples` offsets (bounded per axis by `radius_cells` on the
    (1,2,2) grid) and one attention logit each.
    """

    def __init__(self, channels: int, samples: int, radius_cells: tuple[float, float, float]) -> None:
        super().__init__()
        self.channels = channels
        self.samples = samples
        self.radius_cells = tuple(float(r) for r in radius_cells)
        self.out_channels = 3 * channels
        self.query = nn.Sequential(
            nn.Linear(2 * channels, channels), nn.GELU(), nn.Linear(channels, 4 * samples)
        )

    def forward(
        self, features: Tensor, coords_native: Tensor, stride_zyx: tuple[int, int, int] = (1, 2, 2)
    ) -> Tensor:
        base = sample_node_features(features, coords_native, stride_zyx)
        batch, nodes = coords_native.shape[:2]
        if nodes == 0:
            return features.new_empty(batch, 0, self.out_channels)
        k = self.samples
        params = self.query(base).float().view(batch, nodes, k, 4)
        radius = torch.tensor(self.radius_cells, dtype=torch.float32, device=features.device)
        stride = torch.tensor(stride_zyx, dtype=torch.float32, device=features.device)
        centre = (coords_native.float() / stride)[:, :, None]  # B,N,1,3
        points = centre + radius * params[..., :3].tanh()  # B,N,K,3
        grid = normalized_grid(points, features.shape[-3:]).reshape(batch, nodes * k, 1, 1, 3)
        sampled = F.grid_sample(
            features, grid.to(features.dtype), align_corners=False, padding_mode="border"
        )  # B,C,N*K,1,1
        sampled = sampled[:, :, :, 0, 0].view(batch, -1, nodes, k).permute(0, 2, 3, 1)
        weights = params[..., 3].softmax(-1).to(sampled.dtype)  # B,N,K
        pooled = (sampled * weights[..., None]).sum(2)  # B,N,C
        return torch.cat((base, pooled.to(base.dtype)), -1)


class RefiningAssociationHead(nn.Module):
    """Association with motion refined inside the graph.

    One message-passing block, then a residual velocity / log-variance update
    from each source's soft-assigned candidate displacements, recomputed edge
    geometry, and `blocks - 1` further blocks before scoring. Same call and
    output contract as `LineageAssociationHead`.
    """

    def __init__(
        self,
        descriptor_channels: int,
        hidden: int = 96,
        blocks: int = 2,
        sampler: LearnedDescriptorSampler | None = None,
    ) -> None:
        super().__init__()
        self.sampler = sampler
        self.project = nn.Sequential(
            nn.Linear(descriptor_channels, hidden), nn.LayerNorm(hidden), nn.GELU()
        )
        self.motion = nn.Linear(hidden, 6)
        self.first = SparseAssociationBlock(hidden)
        self.provisional = nn.Sequential(
            nn.Linear(2 * hidden + 8, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.refine = nn.Sequential(nn.Linear(hidden + 7, hidden), nn.GELU(), nn.Linear(hidden, 6))
        self.blocks = nn.ModuleList([SparseAssociationBlock(hidden) for _ in range(blocks - 1)])
        self.edge = nn.Sequential(
            nn.Linear(2 * hidden + 8, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.no_parent = nn.Linear(hidden, 1)
        self.division = nn.Sequential(
            nn.Linear(hidden + 1, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1)
        )
        # Daughter-pair head: trained, not used at inference.
        self.daughters = nn.Sequential(
            nn.Linear(3 * hidden + 8, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )

    def forward(
        self,
        src_features: Tensor,
        tgt_features: Tensor,
        src_um: Tensor,
        tgt_um: Tensor,
        edge_index: Tensor,
        dt: float = 1.0,
    ) -> AssociationOutput:
        src, tgt = self.project(src_features), self.project(tgt_features)
        motion = self.motion(src)
        velocity, raw_logvar = motion[:, :3], motion[:, 3:]
        i, j = edge_index
        delta = (tgt_um[j] - src_um[i]).to(src)
        geometry = edge_geometry(delta, velocity[i], raw_logvar.clamp(-6, 6)[i], dt)
        src, tgt = self.first(src, tgt, edge_index, geometry)

        # Where the candidates sit, weighted by how likely each continues this source.
        provisional = self.provisional(torch.cat((src[i], tgt[j], geometry), -1)).squeeze(-1)
        weight = segment_softmax(provisional, i, len(src))  # float32
        expected = torch.zeros(len(src), 3, device=src.device, dtype=torch.float32)
        expected = expected.index_add(0, i, weight[:, None] * delta.float())
        confidence = torch.zeros(len(src), device=src.device, dtype=torch.float32)
        confidence = confidence.scatter_reduce(0, i, weight, reduce="amax", include_self=True)
        evidence = torch.cat(
            (expected / 10, (expected - velocity.float() * dt) / 10, confidence[:, None]), -1
        ).to(src.dtype)
        update = self.refine(torch.cat((src, evidence), -1))
        velocity = velocity + update[:, :3]
        logvar = (raw_logvar + update[:, 3:]).clamp(-6, 6)

        geometry = edge_geometry(delta, velocity[i], logvar[i], dt)
        for block in self.blocks:
            src, tgt = block(src, tgt, edge_index, geometry)
        logits = self.edge(torch.cat((src[i], tgt[j], geometry), -1)).squeeze(-1)
        division_input = torch.cat((src, src.new_full((len(src), 1), dt)), -1)
        return AssociationOutput(
            logits,
            self.no_parent(tgt).squeeze(-1),
            self.division(division_input).squeeze(-1),
            velocity,
        )


class _Residual(nn.Module):
    def __init__(self, body: nn.Module) -> None:
        super().__init__()
        self.body = body

    def forward(self, x: Tensor) -> Tensor:
        return x + self.body(x)


class MultiScaleLineageNet(nn.Module):
    """Detector and linker with two-scale gated temporal fusion, task adapters,
    learned descriptors and in-graph motion refinement.

    Channel plan for `feature_channels = c`: fine (1,2,2) c/2, isotropic (1,4,4)
    c with `iso_blocks` residual blocks, coarse (2,8,8) 2c, decoder c.
    `temporal_samples` / `temporal_radius_um` are (coarse, isotropic).
    """

    def __init__(
        self,
        stem_channels: int = 8,
        feature_channels: int = 32,
        association_hidden: int = 96,
        association_blocks: int = 2,
        temporal_samples: tuple[int, int] = (4, 2),
        temporal_radius_um: tuple[float, float] = (9.75, 6.5),
        descriptor_samples: int = 4,
        descriptor_radius_um: float = 3.0,
        iso_blocks: int = 2,
    ) -> None:
        super().__init__()
        temporal_samples = tuple(int(v) for v in temporal_samples)
        temporal_radius_um = tuple(float(v) for v in temporal_radius_um)
        self.feature_channels = feature_channels
        c = feature_channels
        self.stem = conv_block(1, stem_channels, (1, 3, 3))
        self.fine = conv_block(stem_channels, c // 2, (1, 3, 3), (1, 2, 2))
        self.isotropic = nn.Sequential(
            conv_block(c // 2, c, (3, 3, 3), (1, 2, 2)),
            *[ResidualBlock(c) for _ in range(iso_blocks)],
        )
        self.coarse = nn.Sequential(
            conv_block(c, 2 * c, stride=(2, 2, 2)), ResidualBlock(2 * c)
        )
        self.temporal_coarse = GatedTemporalFusion(
            2 * c, temporal_samples[0], _cells(temporal_radius_um[0], (2, 8, 8))
        )
        self.temporal_iso = GatedTemporalFusion(
            c, temporal_samples[1], _cells(temporal_radius_um[1], (1, 4, 4))
        )
        self.decode_iso = conv_block(3 * c, c)
        self.decode_fine = conv_block(c + c // 2, c)
        self.detect_adapter = _Residual(conv_block(c, c))
        self.associate_adapter = _Residual(conv_block(c, c))
        self.center = nn.Conv3d(c, 1, 1)
        self.offset = nn.Conv3d(c, 3, 1)
        sampler = LearnedDescriptorSampler(
            c, descriptor_samples, _cells(descriptor_radius_um, (1, 2, 2))
        )
        self.association = RefiningAssociationHead(
            sampler.out_channels, association_hidden, association_blocks, sampler=sampler
        )

    def encode(self, images: Tensor) -> dict[str, Tensor]:
        """Everything that sees one frame alone, each value B,T,C,Z,Y,X: the trunk
        to the coarse scale plus both fusions' per-frame projections."""
        if images.ndim != 6 or images.shape[2] != 1:
            raise ValueError("images must have shape B,T,1,Z,Y,X")
        batch, frames = images.shape[:2]
        x = images.flatten(0, 1)
        fine = self.fine(self.stem(x))
        iso = self.isotropic(fine)
        coarse = self.coarse(iso)
        out = {name: value.unflatten(0, (batch, frames))
               for name, value in (("fine", fine), ("iso", iso), ("coarse", coarse))}
        if frames > 1:
            out["coarse_ref"], out["coarse_nbr"] = self.temporal_coarse.project(out["coarse"])
            out["iso_ref"], out["iso_nbr"] = self.temporal_iso.project(out["iso"])
        return out

    def decode_positions(
        self,
        encoded: dict[str, Tensor],
        times: Tensor,
        positions: list[int] | None = None,
    ) -> DetectionOutput:
        """Fusion, decoder, adapters and heads for the window's `positions` only
        (`None` = every frame); the output's frame axis is `positions`."""
        coarse, iso, fine = encoded["coarse"], encoded["iso"], encoded["fine"]
        batch, frames = coarse.shape[:2]
        times = times.float()
        kept = frames if positions is None else len(positions)
        projected = {}
        if "coarse_ref" in encoded:
            projected = {"coarse": (encoded["coarse_ref"], encoded["coarse_nbr"]),
                         "iso": (encoded["iso_ref"], encoded["iso_nbr"])}
        coarse = self.temporal_coarse(coarse, times, positions, projected.get("coarse"))
        iso = self.temporal_iso(iso, times, positions, projected.get("iso"))
        if positions is not None:
            fine = fine[:, positions]
        coarse, iso, fine = coarse.flatten(0, 1), iso.flatten(0, 1), fine.flatten(0, 1)
        up = resize_at_stride(coarse, iso.shape[-3:], (0.5, 0.5, 0.5))
        decoded = self.decode_iso(torch.cat((iso, up), 1))
        up = resize_at_stride(decoded, fine.shape[-3:], (1.0, 0.5, 0.5))
        shared = self.decode_fine(torch.cat((fine, up), 1))
        detection = self.detect_adapter(shared)
        features = self.associate_adapter(shared)
        with torch.autocast(detection.device.type, enabled=False):  # float32 heads
            head_input = detection.float()
            center = self.center(head_input)
            offset = 0.5 * self.offset(head_input).tanh()
        return DetectionOutput(
            center.unflatten(0, (batch, kept)),
            offset.unflatten(0, (batch, kept)),
            features.unflatten(0, (batch, kept)),
            descriptor=self.association.sampler,
        )


def build_lineage_model(architecture: str = "isotropic_lineage", **kwargs) -> nn.Module:
    """The model a checkpoint's `config.model` describes."""
    if architecture == "isotropic_lineage":
        return IsotropicLineageNet(**kwargs)
    if architecture == "multiscale_lineage":
        return MultiScaleLineageNet(**kwargs)
    raise ValueError(f"unknown architecture {architecture!r}")
