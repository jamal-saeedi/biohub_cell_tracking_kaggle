"""Multi-GPU prediction: one worker subprocess per GPU, with a deadline guard.

Workers are OS subprocesses (a notebook cell cannot `spawn`) that claim movies
from a filesystem queue (`os.rename` is atomic), so a faster GPU simply takes
more movies. Each worker receives the whole configuration as JSON.

The deadline guard re-picks, before every movie, the richest TTA setting at
which this worker's remaining movies still finish before the session deadline;
a movie that fails is retried at the next cheaper setting, and the parent
re-runs whatever is still missing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from biohub_tracking.isotropic import predict as iso_predict
from biohub_tracking.isotropic.config import IsotropicConfig, config_from_dict, config_to_dict
from biohub_tracking.tracking_io import open_dataset, save_graph

__all__ = ["predict_sharded", "movie_weight"]


def movie_weight(ds_path: Path) -> float:
    """`frames * Z * Y * X` from the zarr's shape metadata (no pixels read); the
    queue hands out the largest movies first."""
    dataset = open_dataset(ds_path, load_image=False)
    weight = 1.0
    for dim in dataset.image_shape or ():
        weight *= float(dim)
    return weight


def _setup_queue(queue_dir: Path, stems: list[str], weights: dict[str, float]) -> None:
    queue_dir.mkdir(parents=True, exist_ok=True)
    order = sorted(stems, key=lambda s: weights.get(s, 0.0), reverse=True)
    width = len(str(max(len(order), 1)))
    for rank, stem in enumerate(order):
        (queue_dir / f"{rank:0{width}d}__{stem}.pending").touch()


def _claim_next(queue_dir: Path) -> str | None:
    """Atomically claim one pending stem: the loser of a rename race gets `OSError`."""
    for path in sorted(queue_dir.glob("*.pending")):
        stem = path.name.split("__", 1)[1][: -len(".pending")]
        try:
            os.rename(path, queue_dir / f"{path.stem}.claimed")
        except OSError:
            continue
        return stem
    return None


#: The deadline guard's ladder: each model's TTA views scaled by these fractions
#: (rounded up, at least 1), richest first; the last rung is always
#: `predict.degraded_config` (1 view per model).
GUARD_FRACTIONS = (1.0, 0.75, 0.5)
#: The guard's margin on its own per-movie cost estimate.
GUARD_SAFETY = 1.05


def guard_ladder(config: IsotropicConfig) -> list[tuple[IsotropicConfig, tuple[int, ...]]]:
    """(config, per-model view counts) from the full recipe down to the 1-view fallback."""
    k = 1 + len(config.ensemble_checkpoints)
    per_member = bool(config.detection.member_tta_views)
    full = tuple(int(v) for v in config.detection.member_tta_views) if per_member \
        else (int(config.detection.tta_views),) * k
    ladder = [(config, full)]
    for fraction in GUARD_FRACTIONS[1:]:
        views = tuple(max(1, math.ceil(v * fraction)) for v in full)
        if views in {v for _, v in ladder} or views == (1,) * k:
            continue
        rung = (config.with_detection(member_tta_views=views) if per_member
                else config.with_detection(tta_views=views[0]))
        ladder.append((rung, views))
    ladder.append((iso_predict.degraded_config(config), (1,) * k))
    return ladder


def guard_cost_ratio(views: tuple[int, ...], full: tuple[int, ...]) -> float:
    """Decode cost of `views` relative to `full`: each model costs about one view of
    fixed work plus one per TTA view, so (sum(views) + K) / (sum(full) + K)."""
    k = len(full)
    return (sum(views) + k) / (sum(full) + k)


def guard_pick(history: list[tuple[int, float, float]], ladder_views: list[tuple[int, ...]],
               budget_per_movie: float, floor: int = 0) -> tuple[int, float]:
    """The richest rung (index >= `floor`) whose estimated per-movie seconds fit
    `budget_per_movie`, else the cheapest; returns (rung, its estimate).

    `history` is (rung used, movie seconds, decode seconds) per finished movie.
    Decode time is normalised to the full recipe by `guard_cost_ratio`; both
    parts are the median of the last 5 movies.
    """
    full = ladder_views[0]
    recent = history[-5:]
    decode_full = float(np.median([d / guard_cost_ratio(ladder_views[r], full) for r, _, d in recent]))
    other = float(np.median([max(0.0, t - d) for _, t, d in recent]))
    estimate = 0.0
    for rung in range(floor, len(ladder_views)):
        estimate = GUARD_SAFETY * (other + decode_full * guard_cost_ratio(ladder_views[rung], full))
        if estimate <= budget_per_movie:
            return rung, estimate
    return len(ladder_views) - 1, estimate


#: Time kept free after the workers for the post stage, the CSV and a re-run of a
#: missing movie: at least 30 min, growing by 6 s a movie past ~250 movies.
POST_RESERVE_MIN = 1800.0
POST_RESERVE_FIXED = 300.0
POST_RESERVE_PER_MOVIE = 6.0


def post_reserve_seconds(n_movies: int) -> float:
    """Seconds kept free before the session deadline for the post stage."""
    return max(POST_RESERVE_MIN, POST_RESERVE_FIXED + POST_RESERVE_PER_MOVIE * n_movies)


def _process_age_seconds() -> float:
    """Seconds since this process started (Linux /proc; 0.0 elsewhere): the
    notebook's setup counts against the same session."""
    try:
        start_ticks = int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19])
        uptime = float(Path("/proc/uptime").read_text().split()[0])
        return max(0.0, uptime - start_ticks / os.sysconf("SC_CLK_TCK"))
    except (OSError, ValueError, IndexError):
        return 0.0


def _live_workers(queue_dir: Path, stale_seconds: float) -> int:
    """Workers whose heartbeat file was touched within `stale_seconds`."""
    now = time.time()
    return sum(1 for f in queue_dir.glob("*.alive") if now - f.stat().st_mtime <= stale_seconds)


def _requeue(queue_dir: Path, stem: str) -> None:
    """Hand a claimed movie back to the queue."""
    for path in queue_dir.glob(f"*__{stem}.claimed"):
        try:
            os.rename(path, path.with_suffix(".pending"))
        except OSError:
            pass


def _threads_per_shard(num_shards: int, cap: int = 16) -> int:
    """At most `cap` cores in total, divided evenly across shards."""
    return max(1, min(cap, os.cpu_count() or 1) // max(1, num_shards))


def predict_sharded(
    data_dir: Path,
    output_dir: Path,
    config: IsotropicConfig,
    dataset_stems: list[str] | None = None,
    max_frames: int | None = None,
    num_shards: int | None = None,
    gpu_ids: list[int] | None = None,
    cpu_cap: int = 16,
    ilp_timeout: float | None = None,
    session_deadline_seconds: float | None = None,
    ilp_reserve_fraction: float = 0.3,
) -> list[Path]:
    """Multi-GPU `predict.predict`: one worker per GPU, single-process when that is 1.

    Returns the `.geff` paths that exist; a movie no worker finished is re-run
    here on the fallback configurations.
    """
    data_dir, output_dir = Path(data_dir), Path(output_dir)

    if dataset_stems is None:
        dataset_stems = sorted(p.stem for p in data_dir.glob("*.zarr"))
    if num_shards is None:
        num_shards = min(len(dataset_stems), torch.cuda.device_count())

    if num_shards <= 1 or len(dataset_stems) <= 1:
        return iso_predict.predict(
            data_dir, output_dir, config,
            dataset_stems=dataset_stems, max_frames=max_frames,
            ilp_timeout=ilp_timeout,
            session_deadline_seconds=session_deadline_seconds,
            ilp_reserve_fraction=ilp_reserve_fraction,
        )

    if gpu_ids is None:
        gpu_ids = list(range(num_shards))
    if len(gpu_ids) < num_shards:
        raise ValueError(f"num_shards={num_shards} but {len(gpu_ids)} gpu_ids given")

    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "_shard_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    queue_dir = output_dir / "_shard_queue"
    _setup_queue(
        queue_dir, list(dataset_stems),
        {stem: movie_weight(data_dir / stem) for stem in dataset_stems},
    )

    config_path = output_dir / "_shard_config.json"
    config_path.write_text(json.dumps(config_to_dict(config), indent=1))
    threads = _threads_per_shard(num_shards, cap=cpu_cap)

    processes, handles = [], []
    for index, gpu_id in enumerate(gpu_ids[:num_shards]):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["OMP_NUM_THREADS"] = str(threads)
        env["MKL_NUM_THREADS"] = str(threads)
        env["NUMBA_NUM_THREADS"] = str(threads)
        argv = [
            sys.executable, "-m", "biohub_tracking.isotropic.shard", "--worker",
            "--data-dir", str(data_dir),
            "--output-dir", str(output_dir),
            "--config", str(config_path),
            "--queue-dir", str(queue_dir),
            "--threads", str(threads),
            "--num-shards", str(num_shards),
        ]
        if max_frames is not None:
            argv += ["--max-frames", str(max_frames)]
        if ilp_timeout is not None:
            argv += ["--ilp-timeout", str(ilp_timeout)]
        if session_deadline_seconds is not None:
            # Every worker plans against the same deadline, net of the time
            # the session already spent.
            argv += ["--session-deadline-seconds",
                     str(session_deadline_seconds - _process_age_seconds())]
        argv += ["--ilp-reserve-fraction", str(ilp_reserve_fraction)]
        handle = (log_dir / f"shard{index}_gpu{gpu_id}.log").open("w")
        handles.append(handle)
        processes.append(
            subprocess.Popen(argv, env=env, stdout=handle, stderr=subprocess.STDOUT)
        )

    failures = [(i, p.wait()) for i, p in enumerate(processes)]
    failures = [(i, code) for i, code in failures if code != 0]
    for handle in handles:
        handle.close()
    if failures:
        detail = ", ".join(f"shard {i} exited {code}" for i, code in failures)
        print(f"predict_sharded: {detail}; logs under {log_dir}", flush=True)

    saved = [output_dir / f"{stem}.geff" for stem in dataset_stems]
    missing = [p.stem for p in saved if not p.exists()]
    if missing:
        print(f"predict_sharded: {len(missing)} movie(s) missing, re-running them degraded: "
              f"{missing}", flush=True)
        rerun_missing(data_dir, output_dir, config, missing, max_frames=max_frames)
    # Anything still missing gets the model-free fallback in `run_pipeline`.
    return sorted(p for p in saved if p.exists())


def rerun_missing(data_dir: Path, output_dir: Path, config: IsotropicConfig,
                  stems: list[str], max_frames: int | None = None) -> list[str]:
    """Re-run each of `stems` on the degraded recipe, then on the primary model alone.

    One movie at a time, so a movie that raises cannot cost the others. Returns the
    stems that still have no `.geff`.
    """
    rungs = (("degraded", iso_predict.degraded_config(config)),
             ("primary-only", iso_predict.primary_only_config(config)))
    still = []
    for stem in stems:
        for label, fallback in rungs:
            try:
                iso_predict.predict(data_dir, output_dir, fallback, dataset_stems=[stem],
                                    max_frames=max_frames, ilp_timeout=60.0)
            except Exception as error:  # noqa: BLE001 -- reported, then the next rung
                print(f"predict_sharded: {label} re-run of {stem} failed ({error!r})", flush=True)
                iso_predict.release_memory()
                continue
            print(f"predict_sharded: {stem} DEGRADED re-run ok ({label})", flush=True)
            break
        if not (Path(output_dir) / f"{stem}.geff").exists():
            still.append(stem)
    if still:
        print(f"predict_sharded: NO MODEL OUTPUT for {still}; the pipeline writes the "
              "model-free FALLBACK for them", flush=True)
    return still


def worker_main(argv: list[str] | None = None) -> None:
    """Child-process loop: claim a movie, predict it, save it, repeat."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--ilp-timeout", type=float, default=None)
    parser.add_argument("--session-deadline-seconds", type=float, default=None)
    parser.add_argument("--ilp-reserve-fraction", type=float, default=0.3)
    parser.add_argument("--post-reserve-seconds", type=float, default=None,
                        help="kept free before the session deadline for the single-process "
                             "post stage and the CSV; default post_reserve_seconds(movies)")
    args = parser.parse_args(argv)

    if args.threads is not None:
        torch.set_num_threads(args.threads)
    config = config_from_dict(json.loads(args.config.read_text()))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(
        f"[iso shard pid={os.getpid()}] CUDA_VISIBLE_DEVICES="
        f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r} device={device} "
        f"threads={args.threads}",
        flush=True,
    )
    loaded = iso_predict.load_for_inference(config, device)

    done = 0
    # The worker's clock starts at its own process start.
    start = time.perf_counter() - _process_age_seconds()
    alive = args.queue_dir / f"worker-{os.getpid()}.alive"
    alive.touch()
    if args.post_reserve_seconds is None:
        queued = len(list(args.queue_dir.glob("*.pending"))) + len(list(args.queue_dir.glob("*.claimed")))
        args.post_reserve_seconds = post_reserve_seconds(queued)
    print(f"[iso shard pid={os.getpid()}] deadline {args.session_deadline_seconds}, post reserve "
          f"{args.post_reserve_seconds:.0f}s", flush=True)
    ladder = guard_ladder(config)
    ladder_views = [views for _, views in ladder]
    history: list[tuple[int, float, float]] = []  # (rung, movie seconds, decode seconds)
    rung, floor, at_floor, dead_movies = 0, 0, 0, 0
    while True:
        stem = _claim_next(args.queue_dir)
        if stem is None:
            break
        pending = len(list(args.queue_dir.glob("*.pending")))
        pace = history[-1][1] if history else 600.0
        live = _live_workers(args.queue_dir, stale_seconds=max(1800.0, 3.0 * pace))
        mine = 1 + pending / max(1, min(args.num_shards, live))
        if args.session_deadline_seconds is not None and history:
            # The richest rung at which this worker's remaining movies, at the
            # recent pace, end before the deadline minus the post-stage reserve.
            left = (args.session_deadline_seconds - args.post_reserve_seconds
                    - (time.perf_counter() - start))
            picked, estimate = guard_pick(history, ladder_views, left / mine, floor)
            if picked != rung:
                print(f"[iso shard pid={os.getpid()}] GUARD rung {rung} -> {picked} (views "
                      f"{'/'.join(map(str, ladder_views[picked]))}): {mine:.1f} movies x "
                      f"{estimate:.0f}s est vs {left:.0f}s left", flush=True)
            rung = picked
        else:
            rung = max(rung, floor)
        if args.session_deadline_seconds is None:
            movie_timeout = args.ilp_timeout
        else:
            # The ILP budget: a share of the remaining time, divided by this
            # worker's expected remaining movies.
            remaining = (args.session_deadline_seconds - args.post_reserve_seconds
                         - (time.perf_counter() - start))
            adaptive = max(30.0, args.ilp_reserve_fraction * remaining / mine)
            movie_timeout = (
                adaptive if args.ilp_timeout is None
                else min(args.ilp_timeout, adaptive)
            )
        t0 = time.perf_counter()
        used = rung
        prediction = None
        while prediction is None:
            try:
                prediction = iso_predict.predict_movie(
                    loaded, args.data_dir / stem, ladder[used][0],
                    max_frames=args.max_frames, ilp_timeout=movie_timeout,
                )
            except Exception as error:  # noqa: BLE001 -- one movie must not cost the run
                error.__traceback__ = None  # its frames hold the failed attempt's GPU tensors
                iso_predict.release_memory()
                if used == len(ladder) - 1:
                    print(f"[iso shard pid={os.getpid()}] {stem} FAILED at the cheapest rung "
                          f"({error!r}); leaving it to the parent", flush=True)
                    break
                print(f"[iso shard pid={os.getpid()}] {stem} FAILED at rung {used} ({error!r}); "
                      f"retrying at rung {used + 1}", flush=True)
                used += 1
                if isinstance(error, torch.cuda.OutOfMemoryError):
                    # Later movies start at the rung that fitted.
                    floor = max(floor, used)
        if prediction is None:
            dead_movies += 1
            if dead_movies >= 2:
                # Two movies in a row failed at every rung: this GPU is broken.
                # Give the movie back and stop; the other workers take the rest.
                _requeue(args.queue_dir, stem)
                print(f"[iso shard pid={os.getpid()}] two movies failed at every rung; "
                      f"requeued {stem} and stopping this worker", flush=True)
                break
            continue
        dead_movies = 0
        if floor and used == floor:
            at_floor += 1
            if at_floor >= 3:  # retry the richer rung after 3 clean movies
                floor, at_floor = floor - 1, 0
        history.append((used, time.perf_counter() - t0, float(prediction.stats["decode_seconds"])))
        alive.touch()
        save_graph(prediction.graph, args.output_dir / f"{stem}.geff")
        done += 1
        print(
            f"[iso shard pid={os.getpid()}] {stem}: "
            f"{prediction.stats['solved_nodes']} nodes, "
            f"{prediction.stats['solved_edges']} edges, "
            f"decode {prediction.stats['decode_seconds']:.0f}s assoc "
            f"{prediction.stats['associate_seconds']:.0f}s rescore "
            f"{prediction.stats.get('rescore_seconds', 0.0):.0f}s ilp "
            f"{prediction.stats['ilp_seconds']:.0f}s"
            f"{f' DEGRADED rung {used} views ' + '/'.join(map(str, ladder_views[used])) if used else ''}, "
            f"{(time.perf_counter() - start) / 60:.1f} min elapsed",
            flush=True,
        )
        del prediction
        iso_predict.release_memory()
    alive.unlink(missing_ok=True)
    print(f"[iso shard pid={os.getpid()}] done, {done} movie(s)", flush=True)


if __name__ == "__main__":
    worker_main()
