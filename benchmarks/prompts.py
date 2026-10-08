"""The five benchmarks of the paper (Table 4).

Prompts are raw text: no chat template is applied, so every method sees the
same input and the comparison against the trained draft heads is the
raw-prompt protocol of Appendix D.4.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

# Pinned to a commit, not to a branch: the file this points at defines eighty of
# the prompts the paper reports on, and a branch can be edited under it.
MT_BENCH_COMMIT = "27a05b04a35510afb1d767ae7e5990cbd278f8fe"
MT_BENCH_URL = (
    f"https://raw.githubusercontent.com/lm-sys/FastChat/{MT_BENCH_COMMIT}/"
    "fastchat/llm_judge/data/mt_bench/question.jsonl"
)


@dataclass(frozen=True)
class BenchmarkSpec:
    domain: str
    n_samples: int
    max_new_tokens: int


BENCHMARKS: Dict[str, BenchmarkSpec] = {
    "humaneval": BenchmarkSpec("code", 164, 512),
    "mbpp": BenchmarkSpec("code", 500, 512),
    "classeval": BenchmarkSpec("code", 100, 512),
    "gsm8k": BenchmarkSpec("math", 1319, 1024),
    "mtbench": BenchmarkSpec("dialogue", 80, 1024),
}


def load_prompts(name: str, n_samples: int = -1,
                 data_dir: Path = Path("data")) -> List[Dict[str, str]]:
    """Return ``[{"task_id": ..., "prompt": ...}, ...]`` for one benchmark."""
    if name not in BENCHMARKS:
        raise ValueError(f"unknown benchmark: {name}")

    samples = _LOADERS[name](data_dir)
    expected = BENCHMARKS[name].n_samples
    if len(samples) != expected:
        # The upstream dataset has changed size, so this is no longer the
        # prompt set the paper reports on; say so rather than quietly
        # benchmarking something else.
        print(f"warning: {name} loaded {len(samples)} prompts, the paper used {expected}")
    if n_samples > 0:
        samples = samples[:n_samples]
    return samples


def _load_humaneval(_data_dir: Path) -> List[Dict[str, str]]:
    from datasets import load_dataset

    data = load_dataset("openai/openai_humaneval", split="test")
    return [{"task_id": item["task_id"], "prompt": item["prompt"]} for item in data]


def _load_mbpp(_data_dir: Path) -> List[Dict[str, str]]:
    from datasets import load_dataset

    data = load_dataset("google-research-datasets/mbpp", "full", split="test")
    return [{"task_id": f"mbpp_{item['task_id']}", "prompt": f"# {item['text']}\n"}
            for item in data]


def _load_classeval(_data_dir: Path) -> List[Dict[str, str]]:
    from datasets import load_dataset

    data = load_dataset("FudanSELab/ClassEval", split="test")
    return [{"task_id": item.get("task_id", f"class_{i}"), "prompt": item["skeleton"]}
            for i, item in enumerate(data)]


def _load_gsm8k(_data_dir: Path) -> List[Dict[str, str]]:
    from datasets import load_dataset

    data = load_dataset("openai/gsm8k", "main", split="test")
    return [
        {
            "task_id": f"gsm8k_{i}",
            "prompt": (f"Question: {item['question']}\n\n"
                       "Answer: Let me solve step by step.\n"),
        }
        for i, item in enumerate(data)
    ]


def _load_mtbench(data_dir: Path) -> List[Dict[str, str]]:
    path = data_dir / "mt_bench_question.jsonl"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MT_BENCH_URL, path)

    samples = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            samples.append({"task_id": str(item["question_id"]),
                            "prompt": item["turns"][0]})
    return samples


_LOADERS = {
    "humaneval": _load_humaneval,
    "mbpp": _load_mbpp,
    "classeval": _load_classeval,
    "gsm8k": _load_gsm8k,
    "mtbench": _load_mtbench,
}
