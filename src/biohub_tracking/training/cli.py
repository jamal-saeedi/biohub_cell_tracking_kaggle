"""Train one lineage model from a recipe: ``biohub-train`` (or ``python -m biohub_tracking.training.cli``).

    biohub-train --recipe recipes/train/EX-ms-lc-s5.json \\
        --train-dir data/train --competition-dir data \\
        --pseudo-dir outputs/pseudo_labels/v11 --out-dir outputs/training

A recipe holds the run's full training configuration plus what it needs from
earlier steps (`requires`: an initial checkpoint and/or a pseudo-label set) and
the digest of its split on the full training set. `--smoke` shrinks any recipe
to a few CPU-minutes that still run every code path.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields, replace
from pathlib import Path

from biohub_tracking.training.augment import AugmentConfig
from biohub_tracking.training.dataset import DataConfig
from biohub_tracking.training.losses import LossWeights
from biohub_tracking.training.optim import OptimConfig
from biohub_tracking.training.trainer import ModelConfig, TrainConfig, Trainer

#: Machine-specific settings that come from the command line, never from a recipe.
PATH_KEYS = ("out_dir", "train_dir", "competition_dir", "split_manifest", "init_checkpoint")


def _build(cls, payload: dict, nested: dict | None = None):
    names = {f.name for f in fields(cls)}
    unknown = set(payload) - names
    if unknown:
        raise KeyError(f"{cls.__name__}: unknown settings {sorted(unknown)}")
    kwargs = dict(payload)
    for key, sub in (nested or {}).items():
        if key in kwargs:
            kwargs[key] = sub(kwargs[key])
    for key, value in kwargs.items():
        if isinstance(value, list):
            kwargs[key] = tuple(value)
    return cls(**kwargs)


def train_config_from_dict(payload: dict) -> TrainConfig:
    """A `TrainConfig` from the nested dict `TrainConfig.to_dict` writes."""
    payload = {k: v for k, v in payload.items() if k not in PATH_KEYS}
    data = lambda d: _build(DataConfig, d, {"augment": lambda a: _build(AugmentConfig, a)})  # noqa: E731
    return _build(TrainConfig, payload, {
        "model": lambda d: _build(ModelConfig, d),
        "data": data,
        "optim": lambda d: _build(OptimConfig, d),
        "weights": lambda d: _build(LossWeights, d),
    })


def apply_sets(payload: dict, items: list[str]) -> dict:
    """`["data.samples_per_epoch=64", "epochs=2"]` applied to a nested config dict."""
    payload = json.loads(json.dumps(payload))
    for item in items:
        key, _, raw = item.partition("=")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        *path, name = key.split(".")
        node = payload
        for part in path:
            node = node[part]
        if name not in node:
            raise KeyError(f"unknown setting {key}")
        node[name] = value
    return payload


def smoke(config: TrainConfig) -> TrainConfig:
    """Small enough for a CPU, still exercising detection, association, pseudo-labels
    and (for recipes that use it) linking on the detector's own detections."""
    return replace(
        config,
        epochs=2,
        batch_size=1,
        accumulation=2,
        workers=0,
        val_samples=2,
        amp=False,
        detection_only_steps=1,
        association_ramp_steps=1,
        proposal_start_steps=1,
        proposal_ramp_steps=1,
        log_every=1,
        model=replace(config.model, stem_channels=4, feature_channels=8, association_hidden=16),
        data=replace(config.data, crop_zyx=(16, 64, 64), samples_per_epoch=4,
                     distractors_per_frame=8),
        optim=replace(config.optim, warmup_steps=1, total_steps=4, learning_rate=1e-3),
    )


def main(argv: list[str] | None = None) -> int:
    import torch

    p = argparse.ArgumentParser(prog="biohub-train", description=__doc__.splitlines()[0])
    p.add_argument("--recipe", type=Path, required=True)
    p.add_argument("--train-dir", type=Path, required=True, help="movies with .zarr + .geff")
    p.add_argument("--competition-dir", type=Path, default=None,
                   help="directory with test/ (pinned to the test split); default: the "
                        "train directory's parent")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/training"))
    p.add_argument("--pseudo-dir", type=Path, default=None)
    p.add_argument("--init-checkpoint", type=Path, default=None)
    p.add_argument("--split-manifest", type=Path, default=None,
                   help="an explicit split (train/val stems) instead of the recipe's seed")
    p.add_argument("--run-name", default=None)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override recipe settings, e.g. epochs=8 data.samples_per_epoch=512")
    p.add_argument("--smoke", action="store_true", help="a few CPU-minutes through every code path")
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    recipe = json.loads(args.recipe.read_text())
    requires = recipe.get("requires", {})
    if requires.get("pseudo_labels") and args.pseudo_dir is None:
        p.error(f"this recipe trains on pseudo-label set {requires['pseudo_labels']!r}: "
                "pass --pseudo-dir")
    if requires.get("init") and args.init_checkpoint is None:
        p.error(f"this recipe fine-tunes {requires['init']!r}: pass --init-checkpoint")

    config = train_config_from_dict(apply_sets(recipe["config"], args.set))
    config = replace(
        config,
        run_name=args.run_name or config.run_name,
        out_dir=args.out_dir,
        train_dir=args.train_dir,
        competition_dir=args.competition_dir or args.train_dir.parent,
        split_manifest=args.split_manifest,
        init_checkpoint=args.init_checkpoint,
        data=replace(config.data, pseudo_dir=str(args.pseudo_dir) if args.pseudo_dir else None),
    )
    if args.smoke:
        config = smoke(config)
    trainer = Trainer(config, torch.device(args.device) if args.device else None)
    expected = recipe.get("split_digest")
    if expected and trainer.split.digest != expected:
        print(f"[train] note: split {trainer.split.digest} differs from the recipe's {expected} "
              "(a different set of training movies)", flush=True)
    result = trainer.run()
    print(json.dumps({k: result[k] for k in ("best", "best_epoch", "split_digest", "out_dir")},
                     indent=2))
    return 0 if result["history"] else 1


if __name__ == "__main__":
    sys.exit(main())
