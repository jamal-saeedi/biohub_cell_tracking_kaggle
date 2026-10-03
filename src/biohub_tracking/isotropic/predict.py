"""Per-movie inference: decode -> association -> edge re-scoring -> event ILP -> `.geff`.

Ensemble (association mode): every model runs its own test-time augmentation;
the members' centre logits and offsets are averaged into one shared node set;
each model scores the shared candidate edges with its own association head and
its own re-scorer; the re-scored parent distributions are averaged in
probability space and handed to the solver.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import torch

from biohub_tracking.isotropic.checkpoint import LoadedModel, load_isotropic_model
from biohub_tracking.isotropic.config import IsotropicConfig
from biohub_tracking.isotropic.decode import FrameNodes, decode_frames, window_plan
from biohub_tracking.isotropic.graph import PairAssociation, associate_pair
from biohub_tracking.isotropic.solver import (
    build_event_graph,
    drift_corrected_distance,
    solve_event_ilp,
)
from biohub_tracking.isotropic.volumes import (
    NativeMovie,
    normalize_native,
    open_native_movie,
)
from biohub_tracking.tracking_io import save_graph

__all__ = [
    "MoviePrediction",
    "MovieTerms",
    "degraded_config",
    "load_for_inference",
    "predict",
    "predict_movie",
    "primary_only_config",
    "release_memory",
]


@dataclass
class MoviePrediction:
    """One solved movie and its per-stage counts and timings."""

    stem: str
    graph: object  # td.graph.BaseGraph
    stats: dict = field(default_factory=dict)


def inference_autocast_dtype(amp, device: torch.device):
    """The autocast dtype for `amp` = None (fp32) or an `IsotropicConfig.amp_dtype`.

    ``"auto"`` is bf16 only where the GPU supports it natively (compute
    capability >= 8) and fp16 elsewhere: on a T4 bf16 is emulated and much
    slower.
    """
    if amp is None or device.type != "cuda":
        return None
    if amp == "auto":
        native = torch.cuda.is_bf16_supported(including_emulation=False)
        return torch.bfloat16 if native else torch.float16
    if amp in ("bf16", "fp16"):
        return torch.bfloat16 if amp == "bf16" else torch.float16
    raise ValueError(f"unknown amp_dtype {amp!r}")


# Lateral test-time augmentation: views 0-3 are the rotations by k * 90 degrees,
# views 4-7 add the transpose (the dihedral group D4 on Y, X). Offsets need the
# inverse transform applied to their (dy, dx) components as well as to the grid.
TTA_MAX_VIEWS = 8


def _view_forward(volume, k: int, transpose: bool):
    out = torch.rot90(volume, k, dims=(-2, -1)) if k else volume
    return out.transpose(-2, -1) if transpose else out


def _view_inverse(tensor, k: int, transpose: bool):
    out = tensor.transpose(-2, -1) if transpose else tensor
    return torch.rot90(out, -k, dims=(-2, -1)) if k else out


def _unrotate_offsets(offsets, k: int, transpose: bool):
    """Undo the view's action on the (dy, dx) components of `offsets_zyx`."""
    out = offsets.clone()
    if transpose:  # its own inverse
        out[:, :, 1], out[:, :, 2] = out[:, :, 2].clone(), out[:, :, 1].clone()
    for _ in range(k % 4):  # (dy, dx) -> (dx, -dy)
        dy, dx = out[:, :, 1].clone(), out[:, :, 2].clone()
        out[:, :, 1], out[:, :, 2] = dx, -dy
    return out


def _lateral_axis_sources(k: int, transpose: bool) -> list[tuple[int, int]]:
    """For each view lateral axis, the original axis it reads (0 = y, 1 = x) and its sign."""
    axes = [(0, 1), (1, 1)]
    for _ in range(k % 4):  # rot90: (p, q) -> (W-1-q, p)
        (a0, s0), (a1, s1) = axes
        axes = [(a1, -s1), (a0, s0)]
    if transpose:
        axes = [axes[1], axes[0]]
    return axes


def _anchor_correction(k: int, transpose: bool, shape_yx, stride_zyx):
    """Per-axis offset correction for a reflected axis.

    A cell is decoded as `(index + offset) * stride`, anchored at its lower
    native corner. Reflecting the native volume about `S - 1` and the detection
    grid about `(G - 1) * stride` differ by

        delta = (S - 1 - (G - 1) * stride) / stride,  G = ceil(S / stride)

    which is half a cell for an even size and zero for an odd one.
    """
    correction = [0.0, 0.0, 0.0]
    for source_axis, sign in _lateral_axis_sources(k, transpose):
        if sign > 0:
            continue
        size = int(shape_yx[source_axis])
        stride = int(stride_zyx[source_axis + 1])
        cells = -(-size // stride)  # ceil
        correction[source_axis + 1] = (size - 1 - (cells - 1) * stride) / stride
    return correction


def _tta_forward(model, volume, *, views: int, run):
    """Average `views` D4-augmented passes of `model`, mapped back to the original frame.

    `run(model, k, transpose)` returns one view's dense output (`EncoderCache.runner`).
    `views=1` is the plain forward pass.
    """
    from biohub_tracking.models.isotropic_lineage import DetectionOutput

    if views <= 1:
        return run(model, 0, False)

    plan = [(k, t) for t in (False, True) for k in range(4)][:max(1, min(views, TTA_MAX_VIEWS))]
    if any(t for _, t in plan) and volume.shape[-2] != volume.shape[-1]:
        # A transpose only round-trips on a square lateral grid.
        plan = plan[:4]

    center, offsets, features, stride, descriptor = None, None, None, None, None
    for k, transpose in plan:
        out = run(model, k, transpose)
        # Accumulate in float32: the half-cell anchor shift and an 8-view sum
        # would both lose precision in half precision.
        c = _view_inverse(out.center_logits, k, transpose).float()
        f = _view_inverse(out.features, k, transpose).float()
        o = _unrotate_offsets(_view_inverse(out.offsets_zyx, k, transpose), k, transpose)
        o = o.float() + torch.tensor(
            _anchor_correction(k, transpose, volume.shape[-2:], out.stride_zyx),
            dtype=torch.float32, device=o.device,
        ).view(1, 1, 3, 1, 1, 1)
        center = c if center is None else center + c
        offsets = o if offsets is None else offsets + o
        features = f if features is None else features + f
        stride, descriptor = out.stride_zyx, out.descriptor
        del out

    scale = 1.0 / float(len(plan))
    return DetectionOutput(
        center_logits=center * scale,  # averaged in logit space, as the solver reads them
        offsets_zyx=offsets * scale,
        features=features * scale,
        stride_zyx=stride,
        descriptor=descriptor,
    )


def member_view_counts(config: IsotropicConfig, members: int) -> list[int]:
    """Each model's TTA view count: `detection.member_tta_views`, else `tta_views` for all."""
    views = config.detection.member_tta_views
    if not views:
        return [int(config.detection.tta_views)] * members
    if members == 0:
        raise ValueError("detection.member_tta_views needs an ensemble")
    if len(views) != members:
        raise ValueError(f"member_tta_views has {len(views)} view counts for {members} models")
    return [int(v) for v in views]


def shared_detection(outs):
    """The shared node set's centre logits and offsets: the mean of the members' TTA means."""
    return (sum(o.center_logits for o in outs) / len(outs),
            sum(o.offsets_zyx for o in outs) / len(outs))


class EncoderCache:
    """Per-frame encodings (`model.encode`) of one movie, reused across windows.

    Every frame sits in up to three 3-frame windows. A frame is encoded once per
    (model, view) -- alone, in the view's orientation -- and each window decodes
    only its kept positions (`model.decode_positions`). Windows arrive in
    increasing start order, so a frame is dropped once no later window reads it.
    Under fp16 autocast the encodings are stored in fp16 to fit a 16 GB GPU.
    """

    def __init__(self, frames: np.ndarray, *, size: int, amp, device) -> None:
        self.frames = frames  # (T, Z, Y, X) float32, normalised
        self.size, self.amp, self.device = size, amp, device
        self.store: dict[tuple, dict[int, dict]] = {}

    def runner(self, start: int, positions: list[int]):
        """A `_tta_forward` `run` callable for the window starting at `start`."""
        def run(model, k, transpose):
            store = self.store.setdefault((id(model), k, transpose), {})
            for frame in [f for f in store if f < start]:
                del store[frame]
            dtype = inference_autocast_dtype(self.amp, self.device)
            parts = []
            for frame in range(start, start + self.size):
                if frame not in store:
                    volume = torch.as_tensor(self.frames[frame], device=self.device)[None, None, None]
                    volume = _view_forward(volume, k, transpose).contiguous()
                    with torch.autocast(self.device.type, dtype=dtype or torch.float32,
                                        enabled=dtype is not None):
                        encoded = model.encode(volume)
                    if dtype == torch.float16:
                        encoded = {name: value.to(dtype) for name, value in encoded.items()}
                    store[frame] = encoded
                parts.append(store[frame])
            encoded = {name: torch.cat([p[name] for p in parts], 1) for name in parts[0]}
            del parts
            times = torch.arange(self.size, dtype=torch.float32, device=self.device)[None]
            with torch.autocast(self.device.type, dtype=dtype or torch.float32,
                                enabled=dtype is not None):
                out = model.decode_positions(encoded, times, positions)
            store.pop(start, None)  # every later window starts past it
            return out
        return run


def load_for_inference(config: IsotropicConfig, device) -> LoadedModel:
    """The primary checkpoint and `config.ensemble_checkpoints`, as one `LoadedModel`."""
    loaded = load_isotropic_model(config.checkpoint, device)
    if not config.ensemble_checkpoints:
        return loaded
    members = [load_isotropic_model(path, device).model for path in config.ensemble_checkpoints]
    print(f"[isotropic] ensemble of {1 + len(members)} models", flush=True)
    return replace(loaded, ensemble=tuple(members))


@torch.no_grad()
def decode_movie(
    loaded: LoadedModel, movie: NativeMovie, config: IsotropicConfig
) -> list[FrameNodes]:
    """Decode every frame of a movie from its own canonical window."""
    device = loaded.device
    normalized = normalize_native(movie.image)
    plan = window_plan(movie.frames, config.window)
    per_member = bool(loaded.ensemble)
    member_views = member_view_counts(config, 1 + len(loaded.ensemble) if per_member else 0)
    size = min(config.window.frames, movie.frames)
    cache = EncoderCache(normalized, size=size, amp=config.amp_dtype, device=device)
    nodes: dict[int, FrameNodes] = {}
    for start, positions in plan:
        # Only its shape is read: the frames come from the cache.
        volume = torch.as_tensor(normalized[start : start + size][None], device=device).unsqueeze(2)
        run = cache.runner(start, positions)
        others: list = []
        if per_member:
            outs = [
                _tta_forward(m, volume, views=views, run=run)
                for m, views in zip((loaded.model, *loaded.ensemble), member_views)
            ]
            center, offsets = shared_detection(outs)
            output = replace(outs[0], center_logits=center, offsets_zyx=offsets)
            others = [replace(o, center_logits=center, offsets_zyx=offsets) for o in outs[1:]]
            del outs
        else:
            output = _tta_forward(loaded.model, volume,
                                  views=int(config.detection.tta_views), run=run)
        decode_args = dict(
            positions=list(range(len(positions))),  # the cached forward returns the kept positions only
            frame_ids=[start + p for p in positions],
            config=config.detection,
            shape_zyx=movie.shape_zyx,
        )
        decoded = decode_frames(output, **decode_args)
        extra = [decode_frames(o, **decode_args) for o in others]
        for k, frame_nodes in enumerate(decoded):
            if extra:
                for member in extra:  # same centres and offsets, so the same nodes
                    if not np.array_equal(member[k].coords_native, frame_nodes.coords_native):
                        raise RuntimeError(
                            f"frame {frame_nodes.frame}: ensemble members decoded different nodes"
                        )
                frame_nodes = replace(
                    frame_nodes,
                    member_descriptors=tuple(member[k].descriptors for member in extra),
                )
            nodes[frame_nodes.frame] = frame_nodes
        del output, volume, others

    missing = sorted(set(range(movie.frames)) - set(nodes))
    if missing:  # pragma: no cover - window_plan covers every frame
        raise RuntimeError(f"{movie.stem}: frames not decoded: {missing[:5]}")
    return [nodes[f] for f in range(movie.frames)]


def combine_pair_terms(
    pairs: list[PairAssociation], offset: np.ndarray, total: int
) -> tuple[np.ndarray, np.ndarray, list, list, list, list]:
    """Fold every scored frame pair into the flat per-node arrays the solver takes.

    `offset[f]` is where frame `f`'s nodes start in the flat node order. Returns
    `(null_logp, division_logit, rows, cols, logp, distance)`; the four lists
    are the per-pair pieces of the edge arrays.
    """
    # Frame-0 nodes have no candidate parents, so they pay no appearance cost.
    null_logp = np.zeros(total, dtype=np.float64)
    division_logit = np.zeros(total, dtype=np.float64)
    edge_rows, edge_cols, edge_logp, edge_distance = [], [], [], []
    for pair in pairs:
        src_base, tgt_base = offset[pair.source_frame], offset[pair.target_frame]
        if pair.target_frame > 0:
            null_logp[tgt_base: tgt_base + len(pair.null_logp)] += pair.null_logp
        division_logit[src_base: src_base + len(pair.division_logit)] = pair.division_logit
        if pair.n_edges:
            edge_rows.append(pair.edge_index[0] + src_base)
            edge_cols.append(pair.edge_index[1] + tgt_base)
            edge_logp.append(pair.edge_logp)
            edge_distance.append(pair.distance_um)
    return null_logp, division_logit, edge_rows, edge_cols, edge_logp, edge_distance


def combine_member_pairs(pairs: list[PairAssociation]) -> PairAssociation:
    """Average the ensemble members' scores of one shared candidate set.

    Each target's parent distribution (candidates plus the null class) is
    averaged in probability space; division logits and velocities are averaged
    as they are.
    """
    first = pairs[0]
    for other in pairs[1:]:
        if not np.array_equal(other.edge_index, first.edge_index):
            raise RuntimeError("ensemble members kept different candidate edges")
    log_k = np.log(float(len(pairs)))
    return PairAssociation(
        source_frame=first.source_frame,
        target_frame=first.target_frame,
        edge_index=first.edge_index,
        edge_logp=(np.logaddexp.reduce(np.stack([p.edge_logp for p in pairs]), axis=0)
                   - log_k).astype(np.float32),
        null_logp=(np.logaddexp.reduce(np.stack([p.null_logp for p in pairs]), axis=0)
                   - log_k).astype(np.float32),
        division_logit=np.mean([p.division_logit for p in pairs], axis=0).astype(np.float32),
        distance_um=first.distance_um,
        velocity_um=np.mean([p.velocity_um for p in pairs], axis=0).astype(np.float32),
    )


def associate_members(
    loaded: LoadedModel,
    source: FrameNodes,
    target: FrameNodes,
    *,
    spacing: tuple[float, float, float],
    config,
) -> list[PairAssociation]:
    """Every model's own scores of one frame pair's shared candidates, primary first."""
    if not source.member_descriptors and not target.member_descriptors:
        return [associate_pair(loaded.model, source, target, spacing=spacing, config=config)]
    models = [loaded.model, *loaded.ensemble]
    if not (len(source.member_descriptors) == len(target.member_descriptors) == len(models) - 1):
        raise RuntimeError("member descriptors do not match the loaded ensemble")
    scored = []
    for k, model in enumerate(models):
        src = source if k == 0 else replace(source, descriptors=source.member_descriptors[k - 1])
        tgt = target if k == 0 else replace(target, descriptors=target.member_descriptors[k - 1])
        scored.append(associate_pair(model, src, tgt, spacing=spacing, config=config))
    return scored


def _combined(scored: list[PairAssociation]) -> PairAssociation:
    return scored[0] if len(scored) == 1 else combine_member_pairs(scored)


def gap1_velocity(pairs, offset: np.ndarray, total: int) -> np.ndarray:
    """Each node's predicted velocity (um/frame) from its pair, zeros elsewhere."""
    velocity = np.zeros((total, 3))
    for p in pairs:
        if p.target_frame - p.source_frame == 1:
            base = offset[p.source_frame]
            velocity[base: base + len(p.velocity_um)] = p.velocity_um
    return velocity


_RESCORERS: dict = {}


def _load_rescorer(path):
    """`TreeEnsemble.load`, once per path per process."""
    from biohub_tracking.isotropic.edge_rescorer import TreeEnsemble

    ensemble = _RESCORERS.get(path)
    if ensemble is None:
        ensemble = _RESCORERS[path] = TreeEnsemble.load(path)
    return ensemble


def member_edge_terms(member_pairs: list[list[PairAssociation]], k: int,
                      offset: np.ndarray, total: int):
    """Model k's own flat `(edge_index, logp, null_logp, division_logit, velocity)`
    on the shared nodes: the inputs of its re-scorer."""
    own = [scored[k] for scored in member_pairs]
    null, division, rows, cols, logps, _ = combine_pair_terms(own, offset, total)
    edge_index = (np.stack((np.concatenate(rows), np.concatenate(cols))) if rows
                  else np.empty((2, 0), dtype=np.int64))
    logp = np.concatenate(logps) if logps else np.zeros(0)
    return edge_index, logp, null, division, gap1_velocity(own, offset, total)


def fuse_member_rescored(
    rescorers, pairs: list[PairAssociation], member_pairs: list[list[PairAssociation]],
    offset: np.ndarray, total: int, coords: np.ndarray, node_logit: np.ndarray, spacing,
) -> list[PairAssociation]:
    """`pairs` (the members' average) with each target's parent distribution replaced
    by the probability mean of the members' own re-scored ones.

    `member_pairs[i]` holds every model's own scores of `pairs[i]`, primary
    first, and `rescorers` one tree ensemble per model in the same order. Member
    k's own log-probabilities, null, division logits and velocities are
    re-scored on the shared nodes. The float32 round trips match how the trees'
    training data was stored.
    """
    from biohub_tracking.isotropic.edge_rescorer import rescore_edges

    models = len(rescorers)
    if any(len(scored) != models for scored in member_pairs) or len(member_pairs) != len(pairs):
        raise RuntimeError(f"{models} re-scorers, but the pairs were scored by a different "
                           "number of models")
    rescored = []
    for k, trees in enumerate(rescorers):
        edge_index, logp, null, division, velocity = member_edge_terms(
            member_pairs, k, offset, total)
        rescored.append(rescore_edges(
            trees, coords, edge_index, logp, null, node_logit, division, velocity, spacing,
        ).astype(np.float32))
    log_k = np.log(float(models))
    fused, cursor = [], 0
    for pair, scored in zip(pairs, member_pairs):
        m = pair.n_edges
        for member in scored:
            if not np.array_equal(member.edge_index, pair.edge_index):
                raise RuntimeError("ensemble members kept different candidate edges")
        edges = np.stack([r[cursor: cursor + m] for r in rescored]).astype(np.float64)
        nulls = np.stack([member.null_logp for member in scored]).astype(np.float64)
        cursor += m
        fused.append(replace(
            pair,
            edge_logp=(np.logaddexp.reduce(edges, axis=0) - log_k).astype(np.float32),
            null_logp=(np.logaddexp.reduce(nulls, axis=0) - log_k).astype(np.float32),
        ))
    if any(cursor != len(r) for r in rescored):  # pragma: no cover - same edges, same order
        raise RuntimeError(f"fused {cursor} edges of {[len(r) for r in rescored]}")
    return fused


def degraded_config(config: IsotropicConfig) -> IsotropicConfig:
    """The deadline fallback: one TTA view per model, everything else unchanged."""
    return config.with_detection(tta_views=1, member_tta_views=())


def primary_only_config(config: IsotropicConfig) -> IsotropicConfig:
    """The last model fallback: the primary checkpoint alone at one view, with its own
    re-scorer (`ensemble_rescorers[0]`) as `edge_rescorer`."""
    rescorer = config.ensemble_rescorers[0] if config.ensemble_rescorers else config.edge_rescorer
    return replace(degraded_config(config), ensemble_checkpoints=(), ensemble_rescorers=(),
                   edge_rescorer=rescorer)


@dataclass
class MovieTerms:
    """A decoded movie's shared nodes and every model's scores of its candidate edges."""

    movie: NativeMovie
    pairs: list[PairAssociation]  # the models' average, per frame pair
    member_pairs: list[list[PairAssociation]]  # every model's own scores, per pair
    offset: np.ndarray  # where each frame's nodes start in the flat order
    coords: np.ndarray  # (N,4) t,z,y,x native
    node_logit: np.ndarray  # (N,)
    decode_seconds: float
    associate_seconds: float

    @property
    def total(self) -> int:
        return int(self.offset[-1])


def movie_terms(
    loaded: LoadedModel, ds_path: Path, config: IsotropicConfig, *,
    max_frames: int | None = None, keep_members: bool = False,
) -> MovieTerms:
    """Decode a movie and score every consecutive frame pair with every model."""
    movie = open_native_movie(ds_path, max_frames=max_frames)

    t0 = time.perf_counter()
    frames = decode_movie(loaded, movie, config)
    decode_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    pairs: list[PairAssociation] = []
    member_pairs: list[list[PairAssociation]] = []
    for index in range(len(frames) - 1):
        if len(frames[index]) == 0 or len(frames[index + 1]) == 0:
            continue
        scored = associate_members(
            loaded, frames[index], frames[index + 1],
            spacing=movie.spacing, config=config.candidates,
        )
        pairs.append(_combined(scored))
        if keep_members:
            member_pairs.append(scored)
    associate_seconds = time.perf_counter() - t0

    offset = np.cumsum([0] + [len(f) for f in frames])
    if int(offset[-1]) == 0:
        raise RuntimeError(f"{movie.stem}: the detector produced no cells at all")
    coords = np.concatenate(
        [
            np.column_stack(
                (np.full(len(f), f.frame, dtype=np.float64), f.coords_native)
            )
            for f in frames
        ]
    )
    node_logit = np.concatenate([f.logits for f in frames]).astype(np.float64)
    return MovieTerms(movie, pairs, member_pairs, offset, coords, node_logit,
                      decode_seconds, associate_seconds)


def predict_movie(
    loaded: LoadedModel,
    ds_path: Path,
    config: IsotropicConfig,
    *,
    max_frames: int | None = None,
    ilp_timeout: float | None = None,
) -> MoviePrediction:
    """Decode, associate, re-score and solve one movie."""
    member_rescore = bool(config.ensemble_rescorers)
    if member_rescore and len(config.ensemble_rescorers) != 1 + len(loaded.ensemble):
        raise ValueError(f"{len(config.ensemble_rescorers)} re-scorers for "
                         f"{1 + len(loaded.ensemble)} loaded models")
    terms = movie_terms(loaded, ds_path, config, max_frames=max_frames,
                        keep_members=member_rescore)
    if member_rescore and any(len(scored) != len(config.ensemble_rescorers)
                              for scored in terms.member_pairs):
        raise RuntimeError("member re-scorers need per-member descriptors")
    movie, pairs, member_pairs = terms.movie, terms.pairs, terms.member_pairs
    offset, total, coords, node_logit = terms.offset, terms.total, terms.coords, terms.node_logit
    decode_seconds, associate_seconds = terms.decode_seconds, terms.associate_seconds
    rescore_seconds = 0.0
    if member_rescore:
        t0 = time.perf_counter()
        pairs = fuse_member_rescored(
            [_load_rescorer(p) for p in config.ensemble_rescorers], pairs, member_pairs,
            offset, total, coords, node_logit, movie.spacing,
        )
        rescore_seconds = time.perf_counter() - t0
    del member_pairs, terms
    null_logp, division_logit, edge_rows, edge_cols, edge_logp, _ = (
        combine_pair_terms(pairs, offset, total)
    )
    edge_index = (
        np.stack((np.concatenate(edge_rows), np.concatenate(edge_cols)))
        if edge_rows
        else np.empty((2, 0), dtype=np.int64)
    )
    logp = np.concatenate(edge_logp) if edge_logp else np.zeros(0)

    if config.edge_rescorer is not None:
        from biohub_tracking.isotropic.edge_rescorer import rescore_edges

        t0 = time.perf_counter()
        logp = rescore_edges(_load_rescorer(config.edge_rescorer), coords, edge_index, logp,
                             null_logp, node_logit, division_logit,
                             gap1_velocity(pairs, offset, total), movie.spacing)
        rescore_seconds += time.perf_counter() - t0

    distance = drift_corrected_distance(coords, edge_index, logp, movie.spacing)
    track = build_event_graph(
        coords, node_logit, null_logp, division_logit,
        edge_index, logp, distance, config.solver,
    )
    t0 = time.perf_counter()
    solved = solve_event_ilp(track, config.solver, timeout=ilp_timeout)
    ilp_seconds = time.perf_counter() - t0

    out_degree: dict[int, int] = {}
    if solved.num_edges():
        for source in solved.edge_attrs().select("source_id").to_numpy().ravel():
            out_degree[int(source)] = out_degree.get(int(source), 0) + 1

    stats = {
        "frames": movie.frames,
        "decoded_nodes": total,
        "candidate_edges": int(edge_index.shape[1]),
        "solved_nodes": int(solved.num_nodes()),
        "solved_edges": int(solved.num_edges()),
        "divisions": sum(1 for v in out_degree.values() if v == 2),
        "node_retention": solved.num_nodes() / total,
        "decode_seconds": decode_seconds,
        "associate_seconds": associate_seconds,
        "ilp_seconds": ilp_seconds,
        "rescore_seconds": rescore_seconds,
    }
    return MoviePrediction(stem=movie.stem, graph=solved, stats=stats)


def predict(
    data_dir: Path,
    output_dir: Path,
    config: IsotropicConfig,
    dataset_stems: Sequence[str] | None = None,
    device: torch.device | None = None,
    max_frames: int | None = None,
    ilp_timeout: float | None = None,
    session_deadline_seconds: float | None = None,
    ilp_reserve_fraction: float = 0.3,
    ilp_min_timeout_seconds: float = 30.0,
    skip_failures: bool = False,
) -> list[Path]:
    """Run every movie in `dataset_stems` (default: every `*.zarr`) and save one `.geff` each.

    With `session_deadline_seconds`, each movie's ILP may use at most
    `ilp_reserve_fraction * remaining session / remaining movies` (at least
    `ilp_min_timeout_seconds`), measured against elapsed wall-clock time.
    `skip_failures` logs a movie that raises and goes on; the caller re-runs it.
    """
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaded = load_for_inference(config, device)
    if dataset_stems is None:
        dataset_stems = sorted(p.stem for p in data_dir.glob("*.zarr"))
    output_dir.mkdir(parents=True, exist_ok=True)

    run_start = time.perf_counter()
    saved: list[Path] = []
    n_movies = len(dataset_stems)
    for index, stem in enumerate(dataset_stems):
        caps = [ilp_timeout] if ilp_timeout is not None else []
        if session_deadline_seconds is not None:
            remaining = session_deadline_seconds - (time.perf_counter() - run_start)
            caps.append(ilp_reserve_fraction * remaining / max(n_movies - index, 1))
        movie_timeout = max(ilp_min_timeout_seconds, min(caps)) if caps else None
        try:
            prediction = predict_movie(
                loaded, data_dir / stem, config,
                max_frames=max_frames, ilp_timeout=movie_timeout,
            )
        except Exception as error:  # noqa: BLE001 -- one movie must not cost the run
            if not skip_failures:
                raise
            error.__traceback__ = None  # its frames hold the failed attempt's GPU tensors
            print(f"  [{stem}] {index + 1}/{len(dataset_stems)}: FAILED ({error!r}); "
                  "left to the re-run", flush=True)
            release_memory()
            continue
        out_path = output_dir / f"{stem}.geff"
        save_graph(prediction.graph, out_path)
        saved.append(out_path)
        stats = prediction.stats
        print(
            f"  [{stem}] {index + 1}/{len(dataset_stems)}: "
            f"{stats['solved_nodes']} nodes ({stats['node_retention']:.2f} kept), "
            f"{stats['solved_edges']} edges, {stats['divisions']} divisions, "
            f"decode {stats['decode_seconds']:.1f}s assoc "
            f"{stats['associate_seconds']:.1f}s ilp {stats['ilp_seconds']:.1f}s, "
            f"elapsed {(time.perf_counter() - run_start) / 60:.1f} min",
            flush=True,
        )
        del prediction
        release_memory()
    return saved


def release_memory() -> None:
    """Between movies: collect garbage and return cached CUDA blocks, so host and
    GPU memory stay flat over a long run."""
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
