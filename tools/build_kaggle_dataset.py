"""Assemble the Kaggle code dataset the inference notebook attaches.

    python tools/build_kaggle_dataset.py --wheels DIR [--out outputs/kaggle_dataset]

The dataset holds the `biohub_tracking` wheel (`src/`), the pinned dependency wheels
(`wheels/`, the exact set the submissions installed), `SETTINGS.json` and a
`manifest.json` (git commit, wheel sha256, model handles) and the MIT `LICENSE`. Model
files are not part of it; the notebook attaches them from Kaggle Models. Upload with
`kaggle datasets create -p OUT -r zip --public` (first time) or
`kaggle datasets version -p OUT -r zip -m MSG`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biohub_tracking.recipe import VARIANTS  # noqa: E402

DATASET = "jamalsaeedi/biohub-cell-tracking-kaggle"
GITHUB = "https://github.com/jamal-saeedi/biohub_cell_tracking_kaggle"
KAGGLE_MODEL_URL = "https://www.kaggle.com/models/jamalsaeedi/biohub-cell-tracking"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wheels", type=Path, required=True, help="directory of the pinned wheels")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "kaggle_dataset")
    args = parser.parse_args()

    wheels = sorted(args.wheels.glob("*.whl"))
    if not wheels:
        raise SystemExit(f"no wheels in {args.wheels}")
    if args.out.exists():
        shutil.rmtree(args.out)
    (args.out / "src").mkdir(parents=True)
    (args.out / "wheels").mkdir()
    for wheel in wheels:
        shutil.copy2(wheel, args.out / "wheels" / wheel.name)
    subprocess.run([sys.executable, "-m", "pip", "wheel", str(ROOT), "--no-deps",
                    "-w", str(args.out / "src")], check=True)
    package = next((args.out / "src").glob("biohub_tracking-*.whl"))
    shutil.copy2(ROOT / "SETTINGS.json", args.out / "SETTINGS.json")
    shutil.copy2(ROOT / "LICENSE", args.out / "LICENSE")

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True).stdout.strip() or "unknown"
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                                text=True).stdout.strip())
    manifest = {
        "git_commit": commit,
        "uncommitted_changes": dirty,
        "package_wheel": package.name,
        "package_wheel_sha256": sha256(package),
        "dependency_wheels": [w.name for w in wheels],
        "models": {name: spec.kaggle_handle for name, spec in VARIANTS.items()},
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    (args.out / "dataset-metadata.json").write_text(json.dumps(
        {"title": "biohub cell tracking kaggle", "id": DATASET,
         "subtitle": "Inference code and pinned wheels for the Biohub cell tracking solution",
         "description": (f"Inference package and pinned dependency wheels. MIT licence.\n\n"
                         f"Code: {GITHUB}\n\nModels: {KAGGLE_MODEL_URL}\n"),
         "licenses": [{"name": "other"}]},
        indent=1) + "\n")
    print(f"{args.out}: {package.name} + {len(wheels)} wheels, commit {commit[:12]}"
          f"{' (uncommitted changes)' if dirty else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
