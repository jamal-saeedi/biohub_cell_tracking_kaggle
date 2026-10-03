"""Submission CSV writer and structural validation."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import pandas as pd

SUBMISSION_COLUMNS = ["dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
CSV_COLUMNS = ["id", *SUBMISSION_COLUMNS]

MovieResult = tuple[str, dict[int, dict[str, object]], list[dict[str, object]], dict[str, int]]


def write_test_submission(movies: list[MovieResult], output_path: Path) -> None:
    """Write the `(dataset, nodes_by_id, edges, stats)` movies, in the given order,
    as the competition's CSV. Coordinates are rounded to integer voxels."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row_id = 0
    total_nodes = 0
    total_edges = 0

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for dataset, nodes_by_id, edges, _stats in movies:
            if not nodes_by_id:
                raise AssertionError(f"{dataset}: post-processing removed every node")

            for node_id in sorted(nodes_by_id):
                node = nodes_by_id[node_id]
                writer.writerow(
                    {
                        "id": row_id,
                        "dataset": dataset,
                        "row_type": "node",
                        "node_id": int(node["node_id"]),
                        "t": int(node["t"]),
                        "z": max(0, int(round(float(node["z"])))),
                        "y": max(0, int(round(float(node["y"])))),
                        "x": max(0, int(round(float(node["x"])))),
                        "source_id": -1,
                        "target_id": -1,
                    }
                )
                row_id += 1

            for edge in edges:
                source_id = int(edge["source_id"])
                target_id = int(edge["target_id"])
                if source_id not in nodes_by_id or target_id not in nodes_by_id:
                    raise AssertionError(f"{dataset}: dangling edge after filtering")
                writer.writerow(
                    {
                        "id": row_id,
                        "dataset": dataset,
                        "row_type": "edge",
                        "node_id": -1,
                        "t": -1,
                        "z": -1,
                        "y": -1,
                        "x": -1,
                        "source_id": source_id,
                        "target_id": target_id,
                    }
                )
                row_id += 1

            total_nodes += len(nodes_by_id)
            total_edges += len(edges)

    assert row_id == total_nodes + total_edges, "Internal row counter mismatch"
    assert total_nodes > 0, "No node rows produced"

    header = output_path.open().readline().strip().split(",")
    assert header == CSV_COLUMNS, f"Bad CSV header: {header}"


def validate_submission(csv_path: Path, expected_datasets: list[str]) -> None:
    """Re-read the CSV and check schema, contiguous ids, dataset coverage, node
    times and coordinates, and lineage degrees. Raises `RuntimeError` on any violation."""

    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    frame = pd.read_csv(csv_path)
    if frame.empty or frame.columns.tolist() != CSV_COLUMNS:
        raise RuntimeError("submission schema changed")
    if frame["id"].tolist() != list(range(len(frame))):
        raise RuntimeError("row ids are not contiguous")
    if set(frame["row_type"].unique()) != {"node", "edge"}:
        raise RuntimeError("row types changed")

    datasets = sorted(frame["dataset"].astype(str).unique())
    expected = sorted(expected_datasets)
    if datasets != expected:
        raise RuntimeError({"expected": expected, "actual": datasets})

    for movie, group in frame.groupby("dataset", sort=True):
        nodes = group[group["row_type"].eq("node")]
        edges = group[group["row_type"].eq("edge")]
        if nodes.empty or nodes["t"].lt(0).any():
            raise RuntimeError(f"{movie}: invalid node time")
        if nodes[["z", "y", "x"]].lt(0).any().any():
            raise RuntimeError(f"{movie}: negative coordinate")
        node_time = dict(zip(nodes["node_id"].astype(int), nodes["t"].astype(int)))
        incoming: Counter[int] = Counter()
        outgoing: Counter[int] = Counter()
        for edge in edges.itertuples():
            source = int(edge.source_id)
            target = int(edge.target_id)
            if (
                source not in node_time
                or target not in node_time
                or node_time[target] != node_time[source] + 1
            ):
                raise RuntimeError(f"{movie}: invalid lineage edge")
            incoming[target] += 1
            outgoing[source] += 1
        max_in = max(incoming.values(), default=0)
        max_out = max(outgoing.values(), default=0)
        if max_in > 1 or max_out > 2:
            raise RuntimeError(f"{movie}: invalid lineage degree")
