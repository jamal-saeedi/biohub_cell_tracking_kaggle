"""Input and output paths, read from `SETTINGS.json`, the only place they are set.

`SETTINGS.json` has a ``local`` and a ``kaggle`` section; the one matching the
runtime is used. A value is a path or a list of candidate paths (the first that
exists wins, else the first). Relative paths are relative to the settings file.

Keys:

* ``TEST_DATA_DIR``  -- directory of the test movies (``<stem>.zarr``)
* ``MODEL_DIR``      -- root holding ``<handle>/<version>/`` per model variant
* ``SUBMISSION_DIR`` -- where ``submission.csv`` is written
* ``WORK_DIR``       -- per-movie predictions and shard logs
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

__all__ = ["SETTINGS_FILE", "Settings", "find_settings", "is_kaggle", "load_settings"]

SETTINGS_FILE = "SETTINGS.json"


def is_kaggle() -> bool:
    return Path("/kaggle/working").exists()


@dataclass(frozen=True)
class Settings:
    test_data_dir: Path
    model_dirs: tuple[Path, ...]
    submission_dir: Path
    work_dir: Path
    source: Path

    def describe(self) -> str:
        return "\n".join([
            f"settings     {self.source}",
            f"  test data  {self.test_data_dir}",
            f"  models     {', '.join(map(str, self.model_dirs))}",
            f"  submission {self.submission_dir}",
            f"  work       {self.work_dir}",
        ])


def find_settings(start: Path | str | None = None) -> Path:
    """`BIOHUB_SETTINGS`, else the first `SETTINGS.json` in `start` (default: the
    working directory) or one of its parents."""
    override = os.environ.get("BIOHUB_SETTINGS")
    if override:
        return Path(override)
    here = Path(start or Path.cwd()).resolve()
    for directory in (here, *here.parents):
        if (directory / SETTINGS_FILE).is_file():
            return directory / SETTINGS_FILE
    raise FileNotFoundError(f"no {SETTINGS_FILE} in {here} or its parents; set BIOHUB_SETTINGS")


def load_settings(path: Path | str | None = None) -> Settings:
    path = Path(path) if path else find_settings()
    section = json.loads(path.read_text())["kaggle" if is_kaggle() else "local"]
    base = path.resolve().parent

    def candidates(key: str) -> list[Path]:
        value = section[key]
        values = value if isinstance(value, list) else [value]
        return [p if p.is_absolute() else base / p for p in map(Path, values)]

    def first(key: str) -> Path:
        options = candidates(key)
        return next((p for p in options if p.exists()), options[0])

    return Settings(
        test_data_dir=first("TEST_DATA_DIR"),
        model_dirs=tuple(candidates("MODEL_DIR")),
        submission_dir=first("SUBMISSION_DIR"),
        work_dir=first("WORK_DIR"),
        source=path,
    )
