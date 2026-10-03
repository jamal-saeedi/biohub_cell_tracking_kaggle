"""`IsotropicLineageNet`: native-resolution cell detector with a sparse lineage association head.

Coordinates and vector components are Z, Y, X. The detector reads native
anisotropic volumes (spacing 4:1:1), learns its features on a near-isotropic
grid, and predicts centre logits and sub-voxel offsets on the (1, 2, 2) grid.
The association head scores externally supplied candidate edges between two
frames' node descriptors, together with an explicit no-parent class.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def conv_block(
    cin: int,
    cout: int,
    kernel: tuple[int, int, int] = (3, 3, 3),
    stride: tuple[int, int, int] = (1, 1, 1),
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(
            cin,
            cout,
            kernel,
            stride=stride,
            padding=tuple(k // 2 for k in kernel),
            bias=False,
        ),
        nn.GroupNorm(gcd(8, cout), cout),
        nn.GELU(),
    )


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            conv_block(channels, channels), conv_block(channels, channels)
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.layers(x)


def voxel_grid(shape: tuple[int, ...], reference: Tensor) -> Tensor:
    axes = [
        torch.arange(s, device=reference.device, dtype=reference.dtype) for s in shape
    ]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)


def normalized_grid(coords_zyx: Tensor, shape: tuple[int, ...]) -> Tensor:
    size = coords_zyx.new_tensor(shape)
    # align_corners=False: index 0 is the centre of the first voxel.
    return (2 * (coords_zyx + 0.5) / size - 1).flip(-1)


def resize_at_stride(
    x: Tensor, shape: tuple[int, ...], ratio: tuple[float, float, float]
) -> Tensor:
    """Resample at known voxel centres, without interpolate half-pixel shifts."""
    coords = voxel_grid(shape, x) * x.new_tensor(ratio)
    grid = normalized_grid(coords, x.shape[-3:])
    return F.grid_sample(
        x,
        grid[None].expand(x.shape[0], *grid.shape),
        align_corners=False,
        padding_mode="border",
    )


class MotionTemporalFusion(nn.Module):
    """One learned displaced sample per neighbour frame, with attention over time,
    on the coarse features."""

    def __init__(self, channels: int, radius: float = 2.0) -> None:
        super().__init__()
        self.radius = radius
        self.alignment = nn.Sequential(
            conv_block(2 * channels + 1, channels), nn.Conv3d(channels, 4, 1)
        )
        # Zero displacement and uniform attention at initialisation.
        nn.init.zeros_(self.alignment[-1].weight)
        nn.init.zeros_(self.alignment[-1].bias)
        self.mix = nn.Conv3d(channels, channels, 1)

    def forward(self, x: Tensor, times: Tensor, positions: list[int] | None = None) -> Tensor:
        """Fused features for the reference frames `positions` (`None` = all)."""
        batch, frames = x.shape[:2]
        shape = x.shape[-3:]
        base = voxel_grid(shape, x)[None]
        outputs = []
        for i in range(frames) if positions is None else positions:
            values, scores = [x[:, i]], [x.new_zeros(batch, 1, *shape)]
            for j in range(frames):
                if i == j:
                    continue
                dt = (times[:, j] - times[:, i]).to(x.dtype)
                dt = dt[:, None, None, None, None].expand(batch, 1, *shape)
                params = self.alignment(torch.cat((x[:, i], x[:, j], dt), 1))
                offset = self.radius * params[:, :3].tanh()
                grid = normalized_grid(base + offset.movedim(1, -1), shape)
                values.append(
                    F.grid_sample(
                        x[:, j], grid, align_corners=False, padding_mode="border"
                    )
                )
                scores.append(params[:, 3:4])
            weights = torch.stack(scores, 1).softmax(1)
            context = (weights * torch.stack(values, 1)).sum(1)
            outputs.append(x[:, i] + self.mix(context))
        return torch.stack(outputs, 1)


@dataclass
class DetectionOutput:
    """Dense detector outputs on the (1, 2, 2) grid.

    Native centre = (integer peak + offsets_zyx[peak]) * stride_zyx. Offsets
    are in detection-grid voxels; logits are before the sigmoid.
    """

    center_logits: Tensor  # B,T,1,Z,ceil(Y/2),ceil(X/2)
    offsets_zyx: Tensor  # B,T,3,Z,ceil(Y/2),ceil(X/2)
    features: Tensor  # B,T,C,Z,ceil(Y/2),ceil(X/2)
    stride_zyx: tuple[int, int, int] = (1, 2, 2)
    #: The model's node-descriptor sampler, or `None` for `sample_node_features`.
    descriptor: nn.Module | None = None

    @property
    def descriptor_channels(self) -> int:
        """Width of one node descriptor."""
        if self.descriptor is None:
            return 2 * self.features.shape[2]
        return int(self.descriptor.out_channels)


def sample_node_features(
    features: Tensor,
    coords_native: Tensor,
    stride_zyx: tuple[int, int, int] = (1, 2, 2),
) -> Tensor:
    """Centre + local-average descriptors at continuous native coordinates.

    features: B,C,Z,Y,X; coords_native: B,N,3. Returns B,N,2C.
    """
    if coords_native.ndim != 3 or coords_native.shape[-1] != 3:
        raise ValueError("coords_native must have shape B,N,3")
    if features.shape[0] != coords_native.shape[0]:
        raise ValueError("feature and coordinate batches must match")
    if coords_native.shape[1] == 0:
        return features.new_empty(features.shape[0], 0, features.shape[1] * 2)
    # float32 coordinates: in half precision a native index of 128 cannot hold
    # a sub-voxel offset.
    coords = coords_native.float() / torch.tensor(
        stride_zyx, dtype=torch.float32, device=coords_native.device
    )
    grid = normalized_grid(coords, features.shape[-3:])[:, :, None, None]
    local = F.avg_pool3d(
        features, (3, 5, 5), stride=1, padding=(1, 2, 2), count_include_pad=False
    )
    both = torch.cat((features, local), 1)
    sampled = F.grid_sample(
        both, grid.to(both.dtype), align_corners=False, padding_mode="border"
    )
    return sampled[:, :, :, 0, 0].transpose(1, 2)


def nodes_from_peaks(
    output: DetectionOutput, batch: int, frame: int, peaks_zyx: Tensor
) -> tuple[Tensor, Tensor]:
    """Integer peaks (N,3 on the detection grid) -> native coordinates and descriptors."""
    z, y, x = peaks_zyx.unbind(-1)
    offsets = output.offsets_zyx[batch, frame, :, z, y, x].transpose(0, 1).float()
    # float32: in half precision `index + offset` rounds the offset away.
    coords = (peaks_zyx.float() + offsets) * torch.tensor(
        output.stride_zyx, dtype=torch.float32, device=offsets.device
    )
    return coords, node_descriptors(output, batch, frame, coords)


def node_descriptors(
    output: DetectionOutput, batch: int, frame: int, coords_native: Tensor
) -> Tensor:
    """Descriptors (N, `output.descriptor_channels`) at continuous native coordinates."""
    sampler = sample_node_features if output.descriptor is None else output.descriptor
    return sampler(
        output.features[batch : batch + 1, frame], coords_native[None], output.stride_zyx
    )[0]


def segment_softmax(scores: Tensor, index: Tensor, size: int) -> Tensor:
    """Stable scalar softmax for a sparse set of edges grouped by endpoint."""
    scores = scores.float()
    maximum = scores.new_full((size,), -torch.inf)
    maximum.scatter_reduce_(0, index, scores, reduce="amax", include_self=True)
    weights = (scores - maximum[index]).exp()
    denominator = scores.new_zeros(size).index_add(0, index, weights)
    return weights / denominator[index].clamp_min(1e-12)


class SparseAssociationBlock(nn.Module):
    """Bidirectional graph attention over the candidate edges."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(2 * hidden + 8, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.message = nn.Linear(hidden, hidden)
        self.update = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(
        self, src: Tensor, tgt: Tensor, edge_index: Tensor, geometry: Tensor
    ) -> tuple[Tensor, Tensor]:
        i, j = edge_index
        scores = self.score(torch.cat((src[i], tgt[j], geometry), -1)).squeeze(-1)
        a = segment_softmax(scores, j, len(tgt)).to(src.dtype)
        b = segment_softmax(scores, i, len(src)).to(src.dtype)
        to_tgt = torch.zeros_like(tgt).index_add(
            0, j, a[:, None] * self.message(src[i])
        )
        to_src = torch.zeros_like(src).index_add(
            0, i, b[:, None] * self.message(tgt[j])
        )
        return (
            self.norm(src + self.update(torch.cat((src, to_src), -1))),
            self.norm(tgt + self.update(torch.cat((tgt, to_tgt), -1))),
        )


@dataclass
class AssociationOutput:
    edge_logits: Tensor  # E, in edge_index order
    no_parent_logits: Tensor  # N_target
    division_logits: Tensor  # N_source
    velocity_um: Tensor  # N_source,3, microns per frame interval
    log_variance: Tensor  # N_source,3, velocity log-variance
    source_embeddings: Tensor
    target_embeddings: Tensor
    dt: float = 1.0


def parent_log_probabilities(
    edge_logits: Tensor, no_parent_logits: Tensor, edge_index: Tensor
) -> tuple[Tensor, Tensor]:
    """Normalise each target's candidate parents together with its no-parent class
    (float32). A target without candidates gets null log-probability zero."""
    edge = edge_logits.float()
    null = no_parent_logits.float()
    j = edge_index[1]
    maximum = null.scatter_reduce(0, j, edge, reduce="amax", include_self=True)
    total = (null - maximum).exp().index_add(0, j, (edge - maximum[j]).exp())
    normalizer = maximum + total.log()
    return edge - normalizer[j], null - normalizer


def edge_geometry(delta: Tensor, velocity: Tensor, logvar: Tensor, dt: float) -> Tensor:
    """The 8 edge features: displacement, distance, motion residual standardised
    by the predicted variance, and elapsed time."""
    residual = delta - velocity * dt
    return torch.cat(
        (
            delta / 10,
            delta.norm(dim=-1, keepdim=True) / 10,
            residual / (logvar.mul(0.5).exp() * dt + 1e-3),
            delta.new_full((len(delta), 1), dt),
        ),
        -1,
    )


class LineageAssociationHead(nn.Module):
    """Association for one frame pair: unique candidate `edge_index` (2,E), coordinates in microns."""

    def __init__(
        self, descriptor_channels: int, hidden: int = 96, blocks: int = 2
    ) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(descriptor_channels, hidden), nn.LayerNorm(hidden), nn.GELU()
        )
        self.motion = nn.Linear(hidden, 6)
        self.blocks = nn.ModuleList(
            [SparseAssociationBlock(hidden) for _ in range(blocks)]
        )
        self.edge = nn.Sequential(
            nn.Linear(2 * hidden + 8, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.no_parent = nn.Linear(hidden, 1)
        self.division = nn.Sequential(
            nn.Linear(hidden + 1, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1)
        )
        # Daughter-pair head: a training signal, not used at inference.
        self.daughters = nn.Sequential(
            nn.Linear(3 * hidden + 3, hidden), nn.GELU(), nn.Linear(hidden, 1)
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
        velocity, logvar = motion[:, :3], motion[:, 3:].clamp(-6, 6)
        i, j = edge_index
        delta = (tgt_um[j] - src_um[i]).to(src)
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
            logvar,
            src,
            tgt,
        )

    def score_daughter_pairs(
        self, output: AssociationOutput, triplets: Tensor, src_um: Tensor, tgt_um: Tensor
    ) -> Tensor:
        """Logits of [parent, daughter_a, daughter_b] rows (P,3), symmetric in the daughters."""
        p, a, b = triplets.unbind(-1)
        src, tgt = output.source_embeddings, output.target_embeddings
        da = (tgt_um[a] - src_um[p]).norm(dim=-1)
        db = (tgt_um[b] - src_um[p]).norm(dim=-1)
        separation = (tgt_um[a] - tgt_um[b]).norm(dim=-1)
        geometry = torch.stack((da + db, (da - db).abs(), separation), -1).to(src) / 10
        features = torch.cat(
            (src[p], tgt[a] + tgt[b], (tgt[a] - tgt[b]).abs(), geometry), -1
        )
        return self.daughters(features).squeeze(-1)


def check_window(images: Tensor, times: Tensor | None) -> Tensor:
    """Validate a B,T,1,Z,Y,X window; return its B,T frame times (default 0..T-1)."""
    if images.ndim != 6 or images.shape[2] != 1:
        raise ValueError("images must have shape B,T,1,Z,Y,X")
    batch, frames = images.shape[:2]
    if times is None:
        times = torch.arange(frames, device=images.device, dtype=images.dtype)
        times = times[None].expand(batch, -1)
    if times.shape != (batch, frames):
        raise ValueError("times must have shape B,T")
    return times.to(images)


class IsotropicLineageNet(nn.Module):
    """Detector and linker. Channel plan for `feature_channels = c`: fine (1,2,2)
    c/2, isotropic (1,4,4) c, coarse (2,8,8) 2c with temporal fusion, decoder c.

    `encode` runs per frame; `decode_positions` fuses a window's frames and
    decodes the requested positions. The association head is called separately
    on node descriptors.
    """

    def __init__(
        self,
        stem_channels: int = 8,
        feature_channels: int = 32,
        association_hidden: int = 96,
        association_blocks: int = 2,
        feature_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.feature_channels = feature_channels
        c = feature_channels
        self.stem = conv_block(1, stem_channels, (1, 3, 3))
        self.fine = conv_block(stem_channels, c // 2, (1, 3, 3), (1, 2, 2))
        self.isotropic = nn.Sequential(
            conv_block(c // 2, c, (3, 3, 3), (1, 2, 2)), ResidualBlock(c)
        )
        self.coarse = nn.Sequential(
            conv_block(c, 2 * c, stride=(2, 2, 2)), ResidualBlock(2 * c)
        )
        self.temporal = MotionTemporalFusion(2 * c)
        self.decode_iso = conv_block(3 * c, c)
        self.decode_fine = conv_block(c + c // 2, c)
        self.center = nn.Conv3d(c, 1, 1)
        self.offset = nn.Conv3d(c, 3, 1)
        nn.init.constant_(self.center.bias, -4.0)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        self.association = LineageAssociationHead(
            2 * c, association_hidden, association_blocks
        )
        # Noisy-student model noise: channel dropout on the bottleneck and the
        # first decoder stage; parameter-free and inactive in eval().
        self.feature_dropout = nn.Dropout3d(feature_dropout) if feature_dropout > 0 else nn.Identity()

    def forward(self, images: Tensor, times: Tensor | None = None) -> DetectionOutput:
        """Every frame of B,T,1,Z,Y,X windows (training)."""
        times = check_window(images, times)
        return self.decode_positions(self.encode(images), times)

    def encode(self, images: Tensor) -> dict[str, Tensor]:
        """The per-frame trunk (stem to coarse) of B,T,1,Z,Y,X volumes, each value
        B,T,C,Z,Y,X. A frame's encoding does not depend on the other frames."""
        if images.ndim != 6 or images.shape[2] != 1:
            raise ValueError("images must have shape B,T,1,Z,Y,X")
        batch, frames = images.shape[:2]
        x = images.flatten(0, 1)
        fine = self.fine(self.stem(x))
        iso = self.isotropic(fine)
        coarse = self.feature_dropout(self.coarse(iso))
        return {name: value.unflatten(0, (batch, frames))
                for name, value in (("fine", fine), ("iso", iso), ("coarse", coarse))}

    def decode_positions(
        self, encoded: dict[str, Tensor], times: Tensor, positions: list[int] | None = None
    ) -> DetectionOutput:
        """Temporal fusion, decoder and heads for the window's `positions` only
        (`None` = every frame); the output's frame axis is `positions`."""
        coarse = encoded["coarse"]
        batch, frames = coarse.shape[:2]
        temporal = self.temporal(coarse, times, positions).flatten(0, 1)
        kept = frames if positions is None else len(positions)
        iso, fine = encoded["iso"], encoded["fine"]
        if positions is not None:
            iso, fine = iso[:, positions], fine[:, positions]
        iso, fine = iso.flatten(0, 1), fine.flatten(0, 1)
        up = resize_at_stride(temporal, iso.shape[-3:], (0.5, 0.5, 0.5))
        decoded = self.feature_dropout(self.decode_iso(torch.cat((iso, up), 1)))
        up = resize_at_stride(decoded, fine.shape[-3:], (1.0, 0.5, 0.5))
        features = self.decode_fine(torch.cat((fine, up), 1))
        # The heads run in float32 even under autocast: in bf16 the centre
        # probability is too coarse near 1 and neighbouring peaks tie.
        with torch.autocast(features.device.type, enabled=False):
            head_input = features.float()
            center = self.center(head_input)
            offset = 0.5 * self.offset(head_input).tanh()
        return DetectionOutput(
            center.unflatten(0, (batch, kept)),
            offset.unflatten(0, (batch, kept)),
            features.unflatten(0, (batch, kept)),
        )
