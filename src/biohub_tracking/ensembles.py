"""Ensemble specifications: which trained runs and re-scorers make up a pipeline.

A spec is a JSON file such as `recipes/ensembles/v11-teacher.json`:

    {
      "runs": ["R3-ft24-noisy-lc-s1", "BX-wide-lc-s2", "CX-deep-lc-s3"],
      "member_tta_views": [4, 4, 4],
      "rescorers": ["MRES-A3-m0", "MRES-A3-m1", "MRES-A3-m2"],
      "overrides": {"solver": {"disappearance_weight": 12.0}}
    }

Runs are primary first and resolve to `<runs_dir>/<run>/best.pt` (or to
`<runs_dir>/<entry>` for an entry such as `B3-v11-ft32-noisy-lc-s31/last.pt`);
re-scorers to `<rescorers_dir>/<name>.npz`. `overrides` change `IsotropicConfig` defaults.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from biohub_tracking.isotropic.config import (
    IsotropicConfig,
    config_from_dict,
    config_to_dict,
)


@dataclass(frozen=True)
class EnsembleSpec:
    runs: tuple[str, ...]
    member_tta_views: tuple[int, ...] = ()
    rescorers: tuple[str, ...] = ()
    overrides: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | str) -> EnsembleSpec:
        payload = json.loads(Path(path).read_text())
        return cls(
            runs=tuple(payload["runs"]),
            member_tta_views=tuple(payload.get("member_tta_views", ())),
            rescorers=tuple(payload.get("rescorers", ())),
            overrides=payload.get("overrides", {}),
        )


def apply_overrides(config: IsotropicConfig, overrides: dict) -> IsotropicConfig:
    """`config` with nested `{"section": {"key": value}}` or top-level values replaced."""
    payload = config_to_dict(config)
    for key, value in overrides.items():
        if isinstance(value, dict):
            if key not in payload or not isinstance(payload[key], dict):
                raise KeyError(f"unknown config section {key!r}")
            for name, item in value.items():
                if name not in payload[key]:
                    raise KeyError(f"unknown setting {key}.{name}")
                payload[key][name] = item
        else:
            if key not in payload:
                raise KeyError(f"unknown setting {key!r}")
            payload[key] = value
    return config_from_dict(payload)


def parse_set(items: list[str]) -> dict:
    """`["solver.disappearance_weight=12", ...]` -> nested overrides (values as JSON)."""
    out: dict = {}
    for item in items:
        key, _, raw = item.partition("=")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        section, _, name = key.partition(".")
        if name:
            out.setdefault(section, {})[name] = value
        else:
            out[section] = value
    return out


def ensemble_config(spec: EnsembleSpec, runs_dir: Path | str,
                    rescorers_dir: Path | str | None = None,
                    base: IsotropicConfig | None = None) -> IsotropicConfig:
    """The inference config of `spec`, with its files resolved."""
    runs_dir = Path(runs_dir)
    checkpoints = [runs_dir / run if run.endswith(".pt") else runs_dir / run / "best.pt"
                   for run in spec.runs]
    for path in checkpoints:
        if not path.is_file():
            raise FileNotFoundError(f"no checkpoint at {path}")
    rescorers: tuple[Path, ...] = ()
    if spec.rescorers:
        if rescorers_dir is None:
            raise ValueError("this ensemble needs its re-scorers: pass the re-scorer directory")
        rescorers = tuple(Path(rescorers_dir) / f"{name}.npz" for name in spec.rescorers)
    config = base or IsotropicConfig()
    config = apply_overrides(config, spec.overrides)
    many = len(checkpoints) > 1
    return config_from_dict({
        **config_to_dict(config),
        "checkpoint": str(checkpoints[0]),
        "ensemble_checkpoints": [str(p) for p in checkpoints[1:]],
        "ensemble_rescorers": [str(p) for p in rescorers] if many else [],
        "edge_rescorer": str(rescorers[0]) if rescorers and not many else None,
        "detection": {**config_to_dict(config)["detection"],
                      "member_tta_views": list(spec.member_tta_views) if many else []},
    })
