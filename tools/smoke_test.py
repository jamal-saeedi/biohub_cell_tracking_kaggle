"""End-to-end smoke test of training, pseudo-labelling, re-scorer training and inference.

    python tools/smoke_test.py [--out outputs/smoke]

Runs on CPU in a few minutes on synthetic movies, with every model shrunk by
`biohub-train --smoke`. The steps are the real pipeline's, in order:

1. a ground-truth-only model (recipe W-link-s2)
2. pseudo-labels from it (one-model teacher)
3. a student fine-tuned from it on those labels (recipe R3-ft24-noisy-lc-s1)
4. a MultiScale model from scratch on the same labels (recipe EX-ms-lc-s5)
5. member-own edge re-scorers for the two-model ensemble
6. pseudo-labels from the two-model ensemble with its re-scorers
7. inference with that ensemble: `submission.csv`, validated
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def step(title: str) -> None:
    print(f"\n=== {title}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=ROOT / "outputs" / "smoke")
    args = p.parse_args()
    out = args.out.resolve()
    if out.exists():
        shutil.rmtree(out)
    data, runs, labels, trees = out / "data", out / "runs", out / "labels", out / "rescorers"
    recipes = ROOT / "recipes" / "train"

    from biohub_tracking import cli as predict_cli
    from biohub_tracking import labels as labels_cli
    from biohub_tracking.submission import validate_submission
    from biohub_tracking.training import cli as train_cli
    from biohub_tracking.training import rescorer as rescorer_cli
    from biohub_tracking.training.splits import build_split_from_dir

    step("synthetic movies")
    subprocess.run([sys.executable, str(ROOT / "tools" / "make_synthetic_data.py"),
                    "--out", str(data), "--movies", "12"], check=True)
    train_dir = data / "train"
    split = build_split_from_dir(train_dir, data, seed=314159, fold_test=True)
    align = [next(s for s in split.train if s.startswith(c)) for c in ("44b6", "6bba")]
    common = ["--train-dir", str(train_dir), "--competition-dir", str(data),
              "--out-dir", str(runs), "--smoke", "--device", "cpu"]
    # Keep the smoke teacher's low-confidence labels, so the merge paths run.
    keep_all = ["--set", "data.pseudo_min_node_prob=0.0", "data.pseudo_min_edge_prob=0.0"]

    def ensemble(name: str, members: list[str], rescorer_names: list[str]) -> Path:
        path = out / f"{name}.json"
        # A two-epoch model's centre probabilities sit near their initial 0.02: the
        # smoke run detects below that and lets the solver keep such cells.
        spec = {"runs": members, "member_tta_views": [2] * len(members) if len(members) > 1 else [],
                "rescorers": rescorer_names,
                "overrides": {"detection": {"threshold": 0.01, "tta_views": 2},
                              "solver": {"disappearance_weight": 1.0, "node_logit_bias": -5.0}}}
        path.write_text(json.dumps(spec))
        return path

    step("1. ground-truth-only model")
    train_cli.main(["--recipe", str(recipes / "W-link-s2.json"), "--run-name", "teacher", *common])

    step("2. pseudo-labels from the one-model teacher")
    labels_cli.main(["--ensemble", str(ensemble("teacher", ["teacher"], [])),
                     "--runs", str(runs), "--train-dir", str(train_dir),
                     "--competition-dir", str(data), "--out", str(labels / "round1"),
                     "--align-movies", *align])

    step("3. noisy student fine-tuned from the teacher")
    train_cli.main(["--recipe", str(recipes / "R3-ft24-noisy-lc-s1.json"), "--run-name", "student",
                    "--init-checkpoint", str(runs / "teacher" / "best.pt"),
                    "--pseudo-dir", str(labels / "round1"), *common, *keep_all])

    step("4. MultiScale model from scratch on the labels")
    train_cli.main(["--recipe", str(recipes / "EX-ms-lc-s5.json"), "--run-name", "multiscale",
                    "--pseudo-dir", str(labels / "round1"), *common, *keep_all])

    step("5. member-own edge re-scorers")
    from biohub_tracking.ensembles import EnsembleSpec, ensemble_config

    pair = ensemble("pair", ["student", "multiscale"], [])
    config = ensemble_config(EnsembleSpec.load(pair), runs)
    rescorer_cli.train_rescorers(config, train_dir, list(split.val) + align, trees,
                                 ["SMOKE-m0", "SMOKE-m1"], trees=20, jobs=2)

    step("6. pseudo-labels from the ensemble with its re-scorers")
    rescored = ensemble("pair-rescored", ["student", "multiscale"], ["SMOKE-m0", "SMOKE-m1"])
    labels_cli.main(["--ensemble", str(rescored), "--runs", str(runs), "--rescorers", str(trees),
                     "--train-dir", str(train_dir), "--competition-dir", str(data),
                     "--out", str(labels / "round2"), "--align-movies", *align])

    step("7. inference with the ensemble")
    settings = out / "SETTINGS.json"
    settings.write_text(json.dumps({"local": {
        "TEST_DATA_DIR": str(data / "test"), "MODEL_DIR": str(runs),
        "SUBMISSION_DIR": str(out / "submission"), "WORK_DIR": str(out / "work")}}))
    predict_cli.main(["--settings", str(settings), "--ensemble", str(rescored),
                      "--runs", str(runs), "--rescorers", str(trees), "--session-hours", "1"])
    validate_submission(out / "submission" / "submission.csv",
                        sorted(p.stem for p in (data / "test").glob("*.zarr")))
    print(f"\nSMOKE TEST PASSED: {out / 'submission' / 'submission.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
