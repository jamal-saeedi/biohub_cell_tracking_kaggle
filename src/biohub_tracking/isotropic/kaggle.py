"""Kaggle runtime setup and the run plan (devices, shards, time budget).

This module imports only the standard library at module level: it runs before
the pinned dependency wheels are installed. Facts it encodes:

1. Attached datasets appear at `/kaggle/input/datasets/<owner>/<slug>` or at
   `/kaggle/input/<slug>`; both are probed.
2. A directory in a dataset may arrive as a directory or as `<name>.zip`.
3. numpy and scipy are never reinstalled: they are already imported when a
   notebook cell runs, and replacing them in place breaks later imports.
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "DATASET_CANDIDATES",
    "KaggleEnvironment",
    "RunPlan",
    "SKIP_WHEEL_PREFIXES",
    "bootstrap",
    "find_artifacts_dir",
    "install_offline_wheels",
    "is_kaggle",
    "materialize",
    "plan_run",
    "versions",
]

#: The code dataset, under both mount conventions.
DATASET_CANDIDATES: tuple[str, ...] = (
    "/kaggle/input/datasets/jamalsaeedi/biohub-cell-tracking-kaggle",
    "/kaggle/input/biohub-cell-tracking-kaggle",
)

#: Wheels that must not be reinstalled (point 3 above).
SKIP_WHEEL_PREFIXES: tuple[str, ...] = ("numpy-", "scipy-")


def is_kaggle() -> bool:
    return Path("/kaggle/working").exists()


@dataclass
class KaggleEnvironment:
    """What `bootstrap` resolved and installed."""

    on_kaggle: bool
    artifacts_dir: Path
    staging_root: Path | None = None
    installed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def describe(self) -> str:
        where = "Kaggle" if self.on_kaggle else "local"
        lines = [f"{where}: code at {self.artifacts_dir}"]
        if self.installed:
            lines.append(f"  installed    {len(self.installed)} wheels")
        if self.skipped:
            lines.append(f"  kept baseline {', '.join(self.skipped)}")
        return "\n".join(lines)


def find_artifacts_dir(candidates: tuple[str, ...] = DATASET_CANDIDATES) -> Path:
    """The first existing mount, or the first candidate so an error names a path."""
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    return Path(candidates[0])


def materialize(artifacts_dir: Path, name: str, staging_root: Path) -> Path:
    """`<artifacts_dir>/<name>` as a directory, extracting `<name>.zip` if needed."""
    plain = Path(artifacts_dir) / name
    if plain.is_dir():
        return plain
    archive = Path(artifacts_dir) / f"{name}.zip"
    if not archive.is_file():
        raise FileNotFoundError(
            f"neither {plain} nor {archive} exists — is the dataset attached?"
        )
    destination = Path(staging_root) / name
    if not destination.exists():
        destination.mkdir(parents=True)
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(destination)
    return destination


def install_offline_wheels(
    wheels_dir: Path,
    *,
    skip_prefixes: tuple[str, ...] = SKIP_WHEEL_PREFIXES,
    python: str | None = None,
) -> tuple[list[str], list[str]]:
    """`pip install --no-index --no-deps` every wheel by file path; returns
    `(installed, skipped)` names."""
    wheels_dir = Path(wheels_dir)
    found = sorted(wheels_dir.glob("*.whl")) + sorted(wheels_dir.glob("*.tar.gz"))
    if not found:
        raise FileNotFoundError(f"no wheels in {wheels_dir}")
    installed, skipped = [], []
    for wheel in found:
        if wheel.name.startswith(skip_prefixes):
            skipped.append(wheel.name)
            continue
        subprocess.run(
            [python or sys.executable, "-m", "pip", "install",
             "--no-index", "--no-deps", str(wheel)],
            check=True,
        )
        installed.append(wheel.name)
    return installed, skipped


def bootstrap(artifacts_dir: Path | str | None = None, *, install: bool = True) -> KaggleEnvironment:
    """On Kaggle, install the pinned dependency wheels of the code dataset; a no-op locally.

    The `biohub_tracking` wheel itself is installed by the notebook before this
    module can be imported.
    """
    if not is_kaggle():
        return KaggleEnvironment(
            on_kaggle=False,
            artifacts_dir=Path(artifacts_dir) if artifacts_dir else Path.cwd(),
        )

    resolved = Path(artifacts_dir) if artifacts_dir else find_artifacts_dir()
    staging_root = Path("/kaggle/working/biohub_artifacts")
    staging_root.mkdir(parents=True, exist_ok=True)
    installed: list[str] = []
    skipped: list[str] = []
    if install:
        wheels = materialize(resolved, "wheels", staging_root)
        installed, skipped = install_offline_wheels(wheels)
    return KaggleEnvironment(
        on_kaggle=True,
        artifacts_dir=resolved,
        staging_root=staging_root,
        installed=installed,
        skipped=skipped,
    )


@dataclass
class RunPlan:
    """Device, sharding and time-budget decisions."""

    device: object  # torch.device
    n_gpus: int
    gpu_names: list[str]
    num_shards: int | None
    n_movies: int
    session_deadline_seconds: float | None
    ilp_reserve_fraction: float
    amp_dtype: object | None

    def describe(self) -> str:
        lines = [f"device {self.device}  ({self.n_gpus} GPU(s))"]
        for index, name in enumerate(self.gpu_names):
            lines.append(f"  GPU {index}: {name}")
        lines.append(f"  autocast     {self.amp_dtype}")
        lines.append(
            f"  sharding     {self.num_shards or 1} process(es)"
            + ("" if self.num_shards else "  (single-process path)")
        )
        lines.append(f"  movies       {self.n_movies}")
        if self.session_deadline_seconds:
            lines.append(
                f"  ILP guard    <= {self.ilp_reserve_fraction:.0%} of remaining "
                f"session / remaining movies, deadline "
                f"{self.session_deadline_seconds / 3600:.1f} h"
            )
        return "\n".join(lines)


def plan_run(
    n_movies: int,
    *,
    session_hours: float | None = 12.0,
    ilp_reserve_fraction: float = 0.3,
    max_shards: int | None = None,
) -> RunPlan:
    """One worker process per visible GPU (capped by `n_movies` and `max_shards`;
    `num_shards=None` means the single-process path), and the session deadline the
    per-movie ILP budget and the deadline guard plan against."""
    import torch

    from biohub_tracking.isotropic.predict import inference_autocast_dtype

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    names = [torch.cuda.get_device_properties(i).name for i in range(n_gpus)]

    usable = min(n_gpus, n_movies) if n_movies else n_gpus
    if max_shards is not None:
        usable = min(usable, max_shards)
    num_shards = usable if usable > 1 else None

    return RunPlan(
        device=device,
        n_gpus=n_gpus,
        gpu_names=names,
        num_shards=num_shards,
        n_movies=n_movies,
        session_deadline_seconds=None if session_hours is None else session_hours * 3600.0,
        ilp_reserve_fraction=ilp_reserve_fraction,
        amp_dtype=inference_autocast_dtype("auto", device),
    )


def versions() -> str:
    """Python, CUDA and the version of every package the pipeline imports."""
    import importlib.metadata as metadata
    import platform

    import torch

    names = ("torch", "numpy", "scipy", "pandas", "numba", "polars", "zarr",
             "tracksdata", "ilpy", "pyscipopt", "biohub_tracking")
    found = []
    for name in names:
        try:
            found.append(f"{name} {metadata.version(name)}")
        except metadata.PackageNotFoundError:
            found.append(f"{name} -")
    return (f"python {platform.python_version()}, CUDA {torch.version.cuda}, "
            f"cuDNN {torch.backends.cudnn.version()}\n" + ", ".join(found))
