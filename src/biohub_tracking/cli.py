"""Command-line inference: ``biohub-predict`` (or ``python -m biohub_tracking.cli``).

Runs the same steps as ``notebooks/inference.ipynb``: paths from
`SETTINGS.json`, the shipped config of the chosen variant, a run plan for the
visible GPUs, then `run_pipeline`, which predicts every movie, writes
``submission.csv`` and validates it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

__all__ = ["main", "run", "summarize"]


def run(variant: str, settings, *, movies: list[str] | None = None,
        num_shards: int | None = None, session_hours: float | None = 12.0,
        config=None) -> tuple[Path, dict]:
    """Predict the test movies in `settings.test_data_dir`; return (submission path, timing).

    `config` replaces the shipped variant's (e.g. a retrained ensemble)."""
    import torch

    from biohub_tracking.isotropic.kaggle import plan_run
    from biohub_tracking.isotropic.pipeline import run_pipeline
    from biohub_tracking.recipe import shipped_config

    if config is None:
        config = shipped_config(variant, list(settings.model_dirs))
    data_dir = Path(settings.test_data_dir)
    stems = sorted(movies) if movies else sorted(p.stem for p in data_dir.glob("*.zarr"))
    if not stems:
        raise FileNotFoundError(f"no <stem>.zarr movies in {data_dir}")
    plan = plan_run(len(stems), session_hours=session_hours, ilp_reserve_fraction=0.3,
                    max_shards=num_shards)
    print(plan.describe(), flush=True)
    if not torch.cuda.is_available():
        print("WARNING: no GPU -- this pipeline will not finish on CPU in reasonable time.", flush=True)

    Path(settings.submission_dir).mkdir(parents=True, exist_ok=True)
    timing: dict = {}
    submission = run_pipeline(
        data_dir,
        Path(settings.work_dir) / "predictions",
        Path(settings.submission_dir) / "submission.csv",
        config=config,
        dataset_stems=sorted(movies) if movies else None,
        device=plan.device,
        num_shards=plan.num_shards,
        session_deadline_seconds=plan.session_deadline_seconds,
        ilp_reserve_fraction=plan.ilp_reserve_fraction,
        timing=timing,
    )
    return submission, timing


def summarize(submission: Path) -> str:
    """Nodes, edges and divisions per movie."""
    import pandas as pd

    df = pd.read_csv(submission)
    rows = []
    for stem, group in df.groupby("dataset"):
        edges = group[group["row_type"] == "edge"]
        rows.append({"dataset": stem,
                     "nodes": int((group["row_type"] == "node").sum()),
                     "edges": len(edges),
                     "divisions": int((edges["source_id"].value_counts() == 2).sum())})
    return pd.DataFrame(rows).to_string(index=False)


def main(argv: list[str] | None = None) -> int:
    from biohub_tracking.recipe import DEFAULT_VARIANT, VARIANTS
    from biohub_tracking.settings import Settings, load_settings

    parser = argparse.ArgumentParser(prog="biohub-predict", description=__doc__.splitlines()[0])
    parser.add_argument("--variant", choices=sorted(VARIANTS), default=DEFAULT_VARIANT)
    parser.add_argument("--settings", type=Path, default=None,
                        help="SETTINGS.json (default: BIOHUB_SETTINGS, else the nearest one upward)")
    parser.add_argument("--movies", nargs="+", default=None, metavar="STEM",
                        help="only these movies (default: every <stem>.zarr in TEST_DATA_DIR)")
    parser.add_argument("--num-shards", type=int, default=None,
                        help="at most this many GPU worker processes (default: one per GPU)")
    parser.add_argument("--session-hours", type=float, default=12.0,
                        help="time budget the deadline guard plans against (default 12)")
    parser.add_argument("--ensemble", type=Path, default=None,
                        help="run your own models instead: an ensemble spec (recipes/ensembles)")
    parser.add_argument("--runs", type=Path, default=None, help="with --ensemble: <run>/best.pt root")
    parser.add_argument("--rescorers", type=Path, default=None,
                        help="with --ensemble: directory of <name>.npz")
    args = parser.parse_args(argv)

    settings: Settings = load_settings(args.settings)
    print(settings.describe(), flush=True)
    config = None
    if args.ensemble is not None:
        from biohub_tracking.ensembles import EnsembleSpec, ensemble_config

        if args.runs is None:
            parser.error("--ensemble needs --runs")
        config = ensemble_config(EnsembleSpec.load(args.ensemble), args.runs, args.rescorers)
    submission, timing = run(args.variant, settings, movies=args.movies,
                             num_shards=args.num_shards, session_hours=args.session_hours,
                             config=config)
    print(f"\nSubmission written and validated at: {submission}")
    print(f"  {timing['nodes']} nodes, {timing['edges']} edges, {timing['divisions']} divisions")
    print(f"  predict {timing['predict_seconds'] / 60:.1f} min, total {timing['total_seconds'] / 60:.1f} min")
    print(summarize(submission))
    return 0


if __name__ == "__main__":
    sys.exit(main())
