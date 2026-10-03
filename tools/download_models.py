"""Download the model files from Kaggle Models into MODEL_DIR (SETTINGS.json) and verify them.

    python tools/download_models.py [--variant fin|best|all] [--settings SETTINGS.json]

Needs `pip install kagglehub` (or `pip install -e ".[download]"`); public models need no
Kaggle credentials. Files land in `<MODEL_DIR>/<handle>/<version>/`.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biohub_tracking.recipe import VARIANTS, verify_variant  # noqa: E402
from biohub_tracking.settings import load_settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", choices=[*sorted(VARIANTS), "all"], default="all")
    parser.add_argument("--settings", type=Path, default=None)
    args = parser.parse_args()

    import kagglehub

    root = load_settings(args.settings).model_dirs[0]
    for name in sorted(VARIANTS) if args.variant == "all" else [args.variant]:
        spec = VARIANTS[name]
        target = root / spec.handle / str(spec.version)
        cached = Path(kagglehub.model_download(spec.kaggle_handle))
        for relative in spec.sha256:
            (target / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached / relative, target / relative)
        verify_variant(name, target)
        print(f"{name}: {len(spec.sha256)} files verified in {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
