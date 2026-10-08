#!/usr/bin/env python3
"""Collect result files into the paper's table.

    python benchmarks/summarize.py results

Within a cell, the methods are pooled over the prompts they all completed, so a
speedup is never a ratio between two different workloads; anything a method
dropped is reported rather than quietly left out. Speedup is the wall-clock
token rate over the autoregressive run of the same cell, which means both
numbers come from the same machine and the same session.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

METHOD_ORDER = ["ar", "no-spine-branches", "no-bigram", "no-bypass",
                "no-prefill-harvest", "iso3", "iso5", "goose"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()

    # Sub-directories are separate sweeps (the topology comparison has its own
    # autoregressive and Goose runs), so they stay separate groups instead of
    # overwriting the main table's cells.
    cells: Dict[Tuple[str, str, str], Dict[str, dict]] = defaultdict(dict)
    for path in sorted(args.results_dir.rglob("*.json")):
        with open(path) as handle:
            data = json.load(handle)
        summary = data["summary"]
        group = str(path.parent.relative_to(args.results_dir))
        cells[(group, summary["model"], summary["dataset"])][summary["method"]] = data

    width = max(6, *(len(method) for methods in cells.values() for method in methods))
    print(f"{'group':<10} {'model':<12} {'dataset':<10} {'method':<{width}} {'n':>9} "
          f"{'tau':>6} {'tok/s':>8} {'speedup':>8}  lossless")
    for (group, model, dataset), methods in sorted(cells.items()):
        common = common_prompts(methods)
        if not common:
            print(f"{group:<10} {model:<12} {dataset:<10} "
                  f"no prompt was completed by every method of this cell")
            continue
        baseline = (pooled(methods["ar"], common)["tokens_per_second"]
                    if "ar" in methods else None)

        for method in sorted(methods, key=_rank):
            data = methods[method]
            totals = pooled(data, common)
            summary = data["summary"]
            speedup = (totals["tokens_per_second"] / baseline
                       if baseline else float("nan"))
            completed = summary.get("n_completed", len(data["samples"]))
            dropped = summary.get("n_prompts", 0) - completed
            count = f"{len(common)}/{summary.get('n_prompts', '?')}"
            print(f"{group:<10} {model:<12} {dataset:<10} {method:<{width}} {count:>9} "
                  f"{totals['tau']:>6.2f} {totals['tokens_per_second']:>8.1f} "
                  f"{speedup:>8.2f}  {summary['lossless']}"
                  + (f"  [{dropped} failed]" if dropped > 0 else ""))


def common_prompts(methods: Dict[str, dict]) -> List[str]:
    """Prompts every method of this cell completed."""
    shared = None
    for data in methods.values():
        finished = {r["task_id"] for r in data["samples"] if "error" not in r}
        shared = finished if shared is None else (shared & finished)
    return sorted(shared or [])


def pooled(data: dict, task_ids: List[str]) -> dict:
    """Totals over a fixed set of prompts."""
    wanted = set(task_ids)
    rows = [r for r in data["samples"] if r.get("task_id") in wanted and "error" not in r]
    tokens = sum(r["generated_tokens"] for r in rows)
    calls = sum(r["forward_calls"] for r in rows)
    wall = sum(r["wall_time"] for r in rows)
    return {
        "tau": tokens / max(1, calls),
        "tokens_per_second": tokens / max(1e-6, wall),
    }


def _rank(method: str) -> int:
    return METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER)


if __name__ == "__main__":
    main()
