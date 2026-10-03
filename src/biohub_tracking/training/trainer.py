"""Training loop for the lineage models: AdamW, warm-up + cosine, EMA, mixed precision.

Each epoch writes `history.json` and `last.pt`; `best.pt` is the epoch with the
best `selection_metric` on the validation movies. Both checkpoints hold the EMA
weights and the model's constructor arguments, the format inference loads.

The association losses ramp in after a detection-only warm-up, so the linker
does not train on features that cannot localise anything yet.
"""

from __future__ import annotations

import json
import platform
import shutil
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from biohub_tracking.models.multiscale_lineage import build_lineage_model
from biohub_tracking.training.dataset import (
    DataConfig,
    EpochIndexSampler,
    LineageCropDataset,
    collate,
)
from biohub_tracking.training.ema import WeightAverage
from biohub_tracking.training.evaluate import batch_losses, validate
from biohub_tracking.training.losses import LossWeights
from biohub_tracking.training.optim import (
    OptimConfig,
    autocast_dtype,
    build_optimizer,
    learning_rate_at,
    set_learning_rate,
)
from biohub_tracking.training.splits import (
    DEFAULT_RATIOS,
    SplitManifest,
    build_split_from_dir,
    read_manifest,
)

#: Constructor arguments that exist only for `multiscale_lineage`.
MULTISCALE_ONLY_FIELDS = (
    "temporal_samples", "temporal_radius_um", "descriptor_samples",
    "descriptor_radius_um", "iso_blocks",
)


@dataclass(frozen=True)
class ModelConfig:
    architecture: str = "isotropic_lineage"  # or "multiscale_lineage"
    stem_channels: int = 8
    feature_channels: int = 32
    association_hidden: int = 96
    association_blocks: int = 2
    #: Noisy-student channel dropout (training only).
    feature_dropout: float = 0.0
    #: `multiscale_lineage` only: (coarse, isotropic) temporal samples per
    #: neighbour frame and their bound in um, learned descriptor samples and
    #: their bound, residual blocks at the isotropic stage.
    temporal_samples: tuple[int, int] = (4, 2)
    temporal_radius_um: tuple[float, float] = (9.75, 6.5)
    descriptor_samples: int = 4
    descriptor_radius_um: float = 3.0
    iso_blocks: int = 2

    def __post_init__(self) -> None:
        for name in ("temporal_samples", "temporal_radius_um"):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    def constructor_args(self) -> dict:
        """What a checkpoint records: everything that shapes the weights."""
        args = asdict(self)
        args.pop("feature_dropout")
        if self.architecture == "isotropic_lineage":
            for name in MULTISCALE_ONLY_FIELDS:
                args.pop(name)
        else:
            for name in ("temporal_samples", "temporal_radius_um"):
                args[name] = list(args[name])
        return args

    def build(self) -> torch.nn.Module:
        return build_lineage_model(**self.constructor_args(), feature_dropout=self.feature_dropout)


@dataclass(frozen=True)
class TrainConfig:
    run_name: str = "run"
    out_dir: Path = Path("outputs/training")
    train_dir: Path = Path("data/train")
    #: Directory with `test/`, whose movies are pinned to the test split.
    competition_dir: Path = Path("data")
    split_manifest: Path | None = None
    split_seed: int = 314159
    split_ratios: tuple[float, float, float] = DEFAULT_RATIOS
    #: Move the unpinned test movies into train.
    fold_test_into_train: bool = True
    seed: int = 0

    epochs: int = 20
    batch_size: int = 2
    accumulation: int = 4
    workers: int = 6
    val_batch_size: int = 2
    val_samples: int = 192

    detection_only_steps: int = 300
    association_ramp_steps: int = 300
    detection_threshold: float = 0.3
    match_radius_um: float = 3.0
    #: `loss` or `loss_parent` (validation, minimised).
    selection_metric: str = "loss"
    #: Stop when `selection_metric` has not improved for this many epochs.
    early_stop_patience: int | None = None
    #: After these epochs (0-based), copy the current `best.pt` to `best-e<epoch>.pt`.
    snapshot_epochs: tuple[int, ...] = ()
    #: Largest share of matched annotated linker nodes moved onto the detector's
    #: own detections (`evaluate.ProposalCoords`), ramped in linearly from
    #: `proposal_start_steps` over `proposal_ramp_steps` optimizer steps.
    proposal_fraction: float = 0.0
    proposal_start_steps: int = 600
    proposal_ramp_steps: int = 2000
    #: Also report linking on matched detections in every validation pass.
    proposal_val: bool = False

    ema_decay: float = 0.999
    ema_warmup: int = 1000
    amp: bool = True
    log_every: int = 20
    #: Start from these weights (a checkpoint written by this trainer or a
    #: shipped model); the model config must match.
    init_checkpoint: Path | None = None

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    weights: LossWeights = field(default_factory=LossWeights)

    def to_dict(self) -> dict:
        out = {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("model", "data", "optim", "weights")
        }
        for key in ("out_dir", "train_dir", "competition_dir"):
            out[key] = str(out[key])
        for key in ("split_manifest", "init_checkpoint"):
            out[key] = str(out[key]) if out[key] else None
        out["split_ratios"] = list(self.split_ratios)
        out["snapshot_epochs"] = list(self.snapshot_epochs)
        out["model"] = asdict(self.model)
        out["model"]["temporal_samples"] = list(self.model.temporal_samples)
        out["model"]["temporal_radius_um"] = list(self.model.temporal_radius_um)
        out["data"] = self.data.to_dict()
        out["optim"] = self.optim.to_dict()
        out["weights"] = self.weights.to_dict()
        return out


def environment() -> dict:
    return {
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
    }


class Trainer:
    def __init__(self, config: TrainConfig, device: torch.device | None = None) -> None:
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        torch.manual_seed(config.seed)
        self.out_dir = Path(config.out_dir) / config.run_name
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.split = _resolve_split(config)
        self.model = config.model.build().to(self.device)
        if config.init_checkpoint is not None:
            load_initial_weights(self.model, config.init_checkpoint, config.model)
        self.optimizer = build_optimizer(self.model, config.optim)
        self.ema = WeightAverage(self.model, config.ema_decay, config.ema_warmup)
        self.amp_dtype = autocast_dtype(self.device) if config.amp else None
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.amp_dtype == torch.float16
        )
        loaders = _build_loaders(config, self.split)
        self.train_loader, self.val_loader, self.train_sampler = loaders
        self.step = 0
        self.history: list[dict] = []
        self.best: float | None = None
        self.best_epoch = -1
        self.write_config()

    def write_config(self) -> None:
        payload = {
            "config": self.config.to_dict(),
            "split": self.split.to_dict(),
            "environment": environment(),
            "parameters": sum(p.numel() for p in self.model.parameters()),
        }
        (self.out_dir / "config.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
        print(
            f"[train] run={self.config.run_name} split={self.split.digest} "
            f"device={self.device} amp={self.amp_dtype} "
            f"params={payload['parameters']:,}",
            flush=True,
        )

    def save(self, name: str, epoch: int, metrics: dict) -> Path:
        """EMA weights merged over the live state, in the format inference loads."""
        state = self.model.state_dict()
        state = {k: (self.ema.shadow[k].to(v.dtype).cpu() if k in self.ema.shadow else v.cpu())
                 for k, v in state.items()}
        path = self.out_dir / name
        torch.save(
            {
                "model": state,
                "config": {"model": self.config.model.constructor_args(),
                           "train": self.config.to_dict()},
                "split_digest": self.split.digest,
                "epoch": epoch,
                "step": self.step,
                "metrics": metrics,
            },
            path,
        )
        return path

    def association_scale(self) -> float:
        """0 during the detection warm-up, then linear to 1 over the ramp."""
        start = self.config.detection_only_steps
        ramp = max(self.config.association_ramp_steps, 1)
        return float(min(max((self.step - start) / ramp, 0.0), 1.0))

    def proposal_scale(self) -> float:
        """Share of matched annotations linked at their detection this step."""
        start = self.config.proposal_start_steps
        ramp = max(self.config.proposal_ramp_steps, 1)
        return self.config.proposal_fraction * float(
            min(max((self.step - start) / ramp, 0.0), 1.0))

    def step_weights(self, scale: float) -> LossWeights:
        base = self.config.weights
        return replace(
            base,
            parent=base.parent * scale,
            division=base.division * scale,
            velocity=base.velocity * scale,
            daughter=base.daughter * scale,
        )

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        self.train_sampler.set_epoch(epoch)
        totals: dict[str, float] = {}
        seen = 0
        started = time.time()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.optimizer.zero_grad(set_to_none=True)

        for index, batch in enumerate(self.train_loader):
            scale = self.association_scale()
            weights = self.step_weights(scale)
            with torch.autocast(
                self.device.type,
                dtype=self.amp_dtype or torch.float32,
                enabled=self.amp_dtype is not None,
            ):
                terms, _dense, stats = batch_losses(
                    self.model, batch, self.device, weights, association=scale > 0,
                    proposal_fraction=self.proposal_scale(),
                    proposal_threshold=self.config.detection_threshold,
                    proposal_match_um=self.config.match_radius_um,
                )
            loss = terms.total / self.config.accumulation
            if loss.requires_grad:
                self.scaler.scale(loss).backward()

            if (index + 1) % self.config.accumulation == 0:
                set_learning_rate(
                    self.optimizer, learning_rate_at(self.step, self.config.optim)
                )
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.optim.grad_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.ema.update(self.model)
                self.step += 1
                _add(totals, "grad_norm", float(grad_norm))

            _add(totals, "loss", float(terms.total.detach()))
            for name, value in terms.parts.items():
                _add(totals, f"loss_{name}", float(value))
            for name, value in stats.values.items():
                _add(totals, name, value)
            seen += 1
            if self.config.log_every and index % self.config.log_every == 0:
                print(
                    f"[train] epoch {epoch} batch {index} step {self.step} "
                    f"loss {float(terms.total):.4f} assoc_scale {scale:.2f} "
                    f"proposal {self.proposal_scale():.2f} "
                    f"lr {self.optimizer.param_groups[0]['lr']:.2e}",
                    flush=True,
                )

        out = {name: value / max(seen, 1) for name, value in totals.items()}
        out["seconds"] = time.time() - started
        out["examples_per_second"] = seen * self.config.batch_size / max(out["seconds"], 1e-6)
        if self.device.type == "cuda":
            out["peak_memory_gib"] = torch.cuda.max_memory_allocated(self.device) / 2**30
        return out

    def run(self) -> dict:
        for epoch in range(self.config.epochs):
            train_metrics = self.train_epoch(epoch)
            with self.ema.evaluated(self.model):
                val_metrics = validate(
                    self.model,
                    self.val_loader,
                    self.device,
                    self.config.weights,
                    detection_threshold=self.config.detection_threshold,
                    match_radius_um=self.config.match_radius_um,
                    proposal_metrics=self.config.proposal_val,
                )
            self.history.append(
                {"epoch": epoch, "step": self.step, "train": train_metrics, "val": val_metrics}
            )
            (self.out_dir / "history.json").write_text(json.dumps(self.history, indent=2) + "\n")
            self.save("last.pt", epoch, val_metrics)
            selected = val_metrics.get(self.config.selection_metric, float("nan"))
            improved = selected == selected and (self.best is None or selected < self.best)
            print(
                f"[train] epoch {epoch} done in {train_metrics['seconds']:.0f}s "
                f"train_loss {train_metrics.get('loss', float('nan')):.4f} "
                f"val_loss {val_metrics.get('loss', float('nan')):.4f} "
                f"{self.config.selection_metric} {selected:.4f} improved {improved} "
                f"det_recall {val_metrics.get('det_recall', float('nan')):.3f} "
                f"node_ratio {val_metrics.get('node_ratio', float('nan')):.3f} "
                f"parent_acc {val_metrics.get('parent_accuracy', float('nan')):.3f}",
                flush=True,
            )
            if improved:
                self.best = selected
                self.best_epoch = epoch
                self.save("best.pt", epoch, val_metrics)
            if epoch in self.config.snapshot_epochs and (self.out_dir / "best.pt").exists():
                shutil.copyfile(self.out_dir / "best.pt", self.out_dir / f"best-e{epoch}.pt")
            patience = self.config.early_stop_patience
            if patience is not None and epoch - self.best_epoch >= patience:
                print(f"[train] early stop after epoch {epoch} (best epoch {self.best_epoch})",
                      flush=True)
                break
        return {"history": self.history, "best": self.best, "best_epoch": self.best_epoch,
                "split_digest": self.split.digest, "out_dir": str(self.out_dir)}


def load_initial_weights(model: torch.nn.Module, checkpoint: Path, model_config: ModelConfig) -> None:
    """Install a checkpoint's weights for fine-tuning; its architecture must match."""
    blob = torch.load(checkpoint, map_location="cpu", weights_only=True)
    recorded = blob["config"]["model"]
    wanted = model_config.constructor_args()
    if recorded != wanted:
        raise ValueError(f"{checkpoint}: model {recorded} != this run's {wanted}")
    model.load_state_dict(blob["model"], strict=True)
    print(f"[train] initialized from {checkpoint}", flush=True)


def _add(totals: dict[str, float], name: str, value: float) -> None:
    if value == value:
        totals[name] = totals.get(name, 0.0) + value


def _resolve_split(config: TrainConfig) -> SplitManifest:
    if config.split_manifest is not None:
        return read_manifest(config.split_manifest)
    manifest = build_split_from_dir(
        config.train_dir,
        config.competition_dir,
        seed=config.split_seed,
        ratios=config.split_ratios,
        fold_test=config.fold_test_into_train,
    )
    manifest.write(Path(config.out_dir) / config.run_name / "split.json")
    return manifest


def _build_loaders(
    config: TrainConfig, split: SplitManifest
) -> tuple[DataLoader, DataLoader, EpochIndexSampler]:
    train_set = LineageCropDataset(
        config.train_dir, list(split.train), config.data, seed=config.seed, train=True
    )
    val_set = LineageCropDataset(
        config.train_dir,
        list(split.val),
        # Validation is ground truth only.
        replace(config.data, samples_per_epoch=config.val_samples, pseudo_dir=None),
        seed=config.seed + 1,
        train=False,
    )
    common = dict(
        collate_fn=collate,
        num_workers=config.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.workers > 0,
        prefetch_factor=4 if config.workers > 0 else None,
    )
    train_sampler = EpochIndexSampler(config.data.samples_per_epoch)
    # The validation crops are the same every epoch.
    val_sampler = EpochIndexSampler(config.val_samples)
    return (
        DataLoader(train_set, batch_size=config.batch_size, sampler=train_sampler, **common),
        DataLoader(val_set, batch_size=config.val_batch_size, sampler=val_sampler, **common),
        train_sampler,
    )
