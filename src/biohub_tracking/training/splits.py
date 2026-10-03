"""Seeded, movie-disjoint train/validation/test splits.

* The unit is a whole movie.
* The public Kaggle test movies (whatever is in `<competition>/test`) are pinned
  to `test`; they also exist under `train/` with ground truth.
* Stratified by cohort (`44b6` / `6bba`) and by whether a movie has divisions.

A manifest's digest identifies its stem lists; checkpoints record it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from biohub_tracking.training.corpus import MovieSummary, movie_stems, summarize

MANIFEST_VERSION = 1
DEFAULT_RATIOS: tuple[float, float, float] = (0.70, 0.15, 0.15)


@dataclass(frozen=True)
class SplitManifest:
    seed: int
    ratios: tuple[float, float, float]
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]
    pinned_test: tuple[str, ...]
    stats: dict[str, dict[str, float]]

    @property
    def digest(self) -> str:
        """Content hash of the three stem lists -- the identity of this split."""
        payload = json.dumps(
            {
                "train": sorted(self.train),
                "val": sorted(self.val),
                "test": sorted(self.test),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def stems(self, split: str) -> tuple[str, ...]:
        if split not in ("train", "val", "test"):
            raise ValueError(f"unknown split {split!r}")
        return getattr(self, split)

    def to_dict(self) -> dict:
        return {
            "version": MANIFEST_VERSION,
            "seed": self.seed,
            "ratios": list(self.ratios),
            "digest": self.digest,
            "counts": {s: len(self.stems(s)) for s in ("train", "val", "test")},
            "pinned_test": list(self.pinned_test),
            "stats": self.stats,
            "train": list(self.train),
            "val": list(self.val),
            "test": list(self.test),
        }

    def write(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path


def read_manifest(path: Path | str) -> SplitManifest:
    payload = json.loads(Path(path).read_text())
    if payload.get("version") != MANIFEST_VERSION:
        raise ValueError(
            f"split manifest version {payload.get('version')} not supported"
        )
    manifest = SplitManifest(
        seed=int(payload["seed"]),
        ratios=tuple(float(v) for v in payload["ratios"]),  # type: ignore[arg-type]
        train=tuple(payload["train"]),
        val=tuple(payload["val"]),
        test=tuple(payload["test"]),
        pinned_test=tuple(payload.get("pinned_test", ())),
        stats=payload.get("stats", {}),
    )
    if manifest.digest != payload["digest"]:
        raise ValueError("split manifest digest does not match its stem lists")
    return manifest


def pinned_test_stems(competition_dir: Path | str) -> tuple[str, ...]:
    """Stems of the public Kaggle test movies, read from `<competition_dir>/test`."""
    test_dir = Path(competition_dir) / "test"
    if not test_dir.exists():
        return ()
    return tuple(sorted(p.stem for p in test_dir.glob("*.zarr")))


def _stratum(summary: MovieSummary) -> str:
    return f"{summary.cohort}/{'div' if summary.n_divisions else 'nodiv'}"


def _stratum_rng(seed: int, stratum: str) -> np.random.Generator:
    """A per-stratum random stream."""
    tag = int.from_bytes(hashlib.sha256(stratum.encode()).digest()[:8], "big")
    return np.random.default_rng([seed, tag])


def build_split(
    summaries: list[MovieSummary],
    *,
    seed: int,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    pinned_test: tuple[str, ...] = (),
) -> SplitManifest:
    """Assign every movie to exactly one split. Ratios are fractions of the whole
    corpus including the pinned movies; stratum members are spread evenly
    through the draw order (key `(rank + 0.5) / size`)."""
    if len(ratios) != 3 or not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"ratios must be three fractions summing to 1, got {ratios}")
    if any(r < 0 for r in ratios):
        raise ValueError("ratios must be non-negative")

    by_stem = {s.stem: s for s in summaries}
    if len(by_stem) != len(summaries):
        raise ValueError("duplicate stems in summaries")
    pinned = tuple(s for s in pinned_test if s in by_stem)
    missing = sorted(set(pinned_test) - set(by_stem))
    if missing:
        raise ValueError(f"pinned test stems absent from the corpus: {missing}")

    total = len(summaries)
    n_test = max(round(total * ratios[2]), len(pinned))
    n_val = round(total * ratios[1])
    pool = [s for s in summaries if s.stem not in set(pinned)]
    if n_val + (n_test - len(pinned)) > len(pool):
        raise ValueError("val+test targets exceed the number of unpinned movies")

    keyed: list[tuple[float, str, str]] = []
    strata: dict[str, list[str]] = {}
    for summary in pool:
        strata.setdefault(_stratum(summary), []).append(summary.stem)
    for stratum in sorted(strata):
        members = sorted(strata[stratum])
        order = _stratum_rng(seed, stratum).permutation(len(members))
        for rank, position in enumerate(order):
            keyed.append(((rank + 0.5) / len(members), stratum, members[position]))
    keyed.sort()
    ordered = [stem for _, _, stem in keyed]

    take_test = n_test - len(pinned)
    test = tuple(sorted(pinned + tuple(ordered[:take_test])))
    val = tuple(sorted(ordered[take_test : take_test + n_val]))
    train = tuple(sorted(ordered[take_test + n_val :]))

    manifest = SplitManifest(
        seed=seed, ratios=ratios, train=train, val=val, test=test,
        pinned_test=pinned, stats={},
    )
    return replace(manifest, stats=_stats(manifest, by_stem))


def _stats(manifest: SplitManifest, by_stem: dict[str, MovieSummary]) -> dict:
    out: dict[str, dict[str, float]] = {}
    for split in ("train", "val", "test"):
        rows = [by_stem[s] for s in manifest.stems(split)]
        nodes = sum(r.n_nodes for r in rows)
        estimated = sum(r.estimated_true_nodes or 0.0 for r in rows)
        out[split] = {
            "movies": len(rows),
            "nodes": nodes,
            "edges": sum(r.n_edges for r in rows),
            "divisions": sum(r.n_divisions for r in rows),
            "movies_with_divisions": sum(1 for r in rows if r.n_divisions),
            "coverage": (nodes / estimated) if estimated else float("nan"),
            **{
                f"cohort_{c}": sum(1 for r in rows if r.cohort == c)
                for c in sorted({r.cohort for r in by_stem.values()})
            },
        }
    return out


def fold_test_into_train(
    manifest: SplitManifest, by_stem: dict[str, MovieSummary] | None = None
) -> SplitManifest:
    """Move every unpinned test movie into train; `val` is unchanged."""
    pinned = set(manifest.pinned_test)
    moved = tuple(s for s in manifest.test if s not in pinned)
    folded = replace(
        manifest,
        train=tuple(sorted(manifest.train + moved)),
        test=tuple(sorted(s for s in manifest.test if s in pinned)),
        stats={},
    )
    return replace(folded, stats=_stats(folded, by_stem)) if by_stem else folded


def build_split_from_dir(
    train_dir: Path | str,
    competition_dir: Path | str | None = None,
    *,
    seed: int,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    fold_test: bool = False,
) -> SplitManifest:
    """Index `train_dir`, pin whatever is in `<competition_dir>/test`, and split."""
    train_dir = Path(train_dir)
    competition_dir = Path(competition_dir) if competition_dir else train_dir.parent
    summaries = summarize(train_dir, movie_stems(train_dir))
    manifest = build_split(
        summaries,
        seed=seed,
        ratios=ratios,
        pinned_test=pinned_test_stems(competition_dir),
    )
    by_stem = {s.stem: s for s in summaries}
    if fold_test:
        manifest = fold_test_into_train(manifest, by_stem)
    return manifest
