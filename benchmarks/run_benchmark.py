#!/usr/bin/env python3
r"""Run one model on one benchmark and write the per-sample measurements.

    python benchmarks/run_benchmark.py --model llama3-8b --dataset humaneval \
        --methods ar goose --output-dir results

The autoregressive run is always the reference: it fixes the speedup
denominator and the token sequence every other method is checked against, so
put ``ar`` first (the script does it for you).
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.prompts import BENCHMARKS, load_prompts
from goose import GooseConfig, decode, decode_autoregressive, stop_tokens

# Checkpoint and inference dtype per model. Qwen3 runs in BF16: in FP16 its
# forward pass can produce NaN logits on long prompts, and greedy decoding
# does not report that as an error. --dtype float16 gives the paper's
# precision.
MODELS = {
    "vicuna-7b": ("lmsys/vicuna-7b-v1.3", "float16"),
    "vicuna-13b": ("lmsys/vicuna-13b-v1.3", "float16"),
    "vicuna-33b": ("lmsys/vicuna-33b-v1.3", "float16"),
    "llama3-8b": ("meta-llama/Meta-Llama-3-8B-Instruct", "float16"),
    "qwen3-8b": ("Qwen/Qwen3-8B", "bfloat16"),
}

# Goose, the autoregressive reference, and configurations of Goose itself: the
# isotropic control of Table 1 and the four ablations of Table 2. The other
# methods compared in the paper are not included.
METHODS = {
    "ar": None,
    "goose": GooseConfig(),
    "iso3": GooseConfig(topology="isotropic", branching_factor=3),
    "iso5": GooseConfig(topology="isotropic", branching_factor=5),
    "no-spine-branches": GooseConfig(spine_branch_ratio=0.0),
    "no-bigram": GooseConfig(use_bigram=False),
    "no-bypass": GooseConfig(use_confidence=False),
    "no-prefill-harvest": GooseConfig(use_prefill_harvest=False),
}

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=sorted(MODELS),
                        help="model key; pass --model-path to use another checkpoint")
    parser.add_argument("--model-path", default=None,
                        help="override the Hugging Face id or local path")
    parser.add_argument("--dataset", required=True, choices=sorted(BENCHMARKS))
    parser.add_argument("--methods", nargs="+", default=["ar", "goose"],
                        choices=sorted(METHODS))
    parser.add_argument("--num-samples", type=int, default=-1,
                        help="default: the full benchmark (Table 4)")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="default: the per-benchmark limit of Table 4")
    parser.add_argument("--dtype", default=None, choices=sorted(DTYPES),
                        help="default: float16, or bfloat16 for qwen3-8b")
    parser.add_argument("--device-map", default="auto",
                        help='"auto" shards across the visible GPUs, as the 33B model needs')
    parser.add_argument("--stop-tokens", default="model", choices=["model", "tokenizer"],
                        help="'model' stops on every id the checkpoint declares; "
                             "'tokenizer' stops only on tokenizer.eos_token_id, which "
                             "is what the paper's runs did")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    spec = BENCHMARKS[args.dataset]
    max_new_tokens = args.max_new_tokens or spec.max_new_tokens
    methods = sorted(set(args.methods), key=lambda m: (m != "ar", m))
    if "ar" not in methods and methods:
        print("warning: without the 'ar' arm nothing checks that the speculative "
              "output matches greedy decoding, and there is no speedup denominator")

    default_path, default_dtype = MODELS[args.model]
    model_path = args.model_path or default_path
    dtype = args.dtype or default_dtype
    print(f"loading {model_path} [{dtype}]")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # torch_dtype, not dtype: the newer spelling raises TypeError on the
    # declared transformers floor, where this one still works.
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=DTYPES[dtype], device_map=args.device_map)
    model.eval()

    samples = load_prompts(args.dataset, args.num_samples, args.data_dir)
    print(f"{args.model} x {args.dataset}: {len(samples)} prompts, "
          f"max_new_tokens={max_new_tokens}, methods={methods}")

    warmup = tokenizer("Hello", return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        model(warmup)

    stops = (tokenizer.eos_token_id if args.stop_tokens == "tokenizer"
             else stop_tokens(model, tokenizer))
    print(f"stopping on {sorted(stops) if not isinstance(stops, int) else [stops]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference: Dict[str, List[int]] = {}

    for method in methods:
        results = run_method(model, tokenizer, method, samples, max_new_tokens,
                             reference, stops)
        summary = summarize(results, args.model, args.dataset, method, dtype,
                            model_path, max_new_tokens, len(samples))
        summary["stop_tokens"] = args.stop_tokens
        path = args.output_dir / f"{args.model}_{args.dataset}_{method}.json"
        with open(path, "w") as handle:
            json.dump({"summary": summary, "samples": results}, handle, indent=2)
        print(f"  {method:5s} tau={summary['tau']:.2f} "
              f"tok/s={summary['tokens_per_second']:.1f} "
              f"lossless={summary['lossless']} -> {path}")


def run_method(model, tokenizer, method: str, samples: List[Dict[str, str]],
               max_new_tokens: int, reference: Dict[str, List[int]],
               stops) -> List[Dict]:
    config = METHODS[method]
    results = []

    for i, sample in enumerate(samples):
        input_ids = tokenizer(sample["prompt"], return_tensors="pt").input_ids.to(model.device)
        prompt_length = input_ids.shape[1]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            if method == "ar":
                output, stats = decode_autoregressive(
                    model, tokenizer, input_ids, max_new_tokens=max_new_tokens,
                    eos_token_id=stops)
            else:
                output, stats = decode(
                    model, tokenizer, input_ids, max_new_tokens=max_new_tokens,
                    eos_token_id=stops, config=config)
        except Exception as error:
            # One bad prompt should not cost the whole run; the sample is
            # recorded as failed and excluded from the aggregate.
            print(f"    [{i + 1}/{len(samples)}] {method} failed: "
                  f"{type(error).__name__}: {error}")
            traceback.print_exc()
            results.append({"task_id": sample["task_id"], "error": repr(error)})
            gc.collect()
            torch.cuda.empty_cache()
            continue
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

        generated = output[0, prompt_length:].tolist()
        record = {
            "task_id": sample["task_id"],
            "prompt_length": prompt_length,
            "tau": round(stats.compression_ratio, 4),
            # The decoder's own clock, which covers allocating its draft
            # sources as well as decoding; `elapsed` adds only this loop's
            # tokenisation and bookkeeping and is kept to show that gap.
            "tokens_per_second": round(stats.tokens_per_second, 2),
            "wall_time_measured_here": round(elapsed, 4),
            **asdict(stats),
        }
        if method == "ar":
            reference[sample["task_id"]] = generated
        else:
            matches = compare_to_reference(reference, sample["task_id"], generated)
            if matches is not None:
                record["lossless"] = matches
        results.append(record)

        if (i + 1) % 25 == 0 or (i + 1) == len(samples):
            print(f"    [{i + 1}/{len(samples)}] {method}")
        if (i + 1) % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    return results


def compare_to_reference(reference: Dict[str, List[int]], task_id: str,
                         generated: List[int]) -> Optional[bool]:
    """Whether this sample reproduced the autoregressive token sequence."""
    expected = reference.get(task_id)
    if expected is None:
        return None
    return expected == generated


def summarize(results: List[Dict], model: str, dataset: str, method: str,
              dtype: str, model_path: str, max_new_tokens: int,
              n_prompts: int) -> Dict:
    valid = [r for r in results if "error" not in r]
    tokens = sum(r["generated_tokens"] for r in valid)
    calls = sum(r["forward_calls"] for r in valid)
    wall = sum(r["wall_time"] for r in valid)
    checked = [r["lossless"] for r in valid if "lossless" in r]

    return {
        "model": model,
        "model_path": model_path,
        "dataset": dataset,
        "method": method,
        "dtype": dtype,
        "max_new_tokens": max_new_tokens,
        "n_prompts": n_prompts,
        "n_completed": len(valid),
        "n_failed": len(results) - len(valid),
        "generated_tokens": tokens,
        "forward_calls": calls,
        "wall_time": round(wall, 2),
        "tau": round(tokens / max(1, calls), 3),
        "tokens_per_second": round(tokens / max(1e-6, wall), 2),
        # "reference" is the autoregressive arm itself; a speculative arm with
        # no reference to compare against says so rather than borrowing the word.
        "lossless": ("reference" if method == "ar"
                     else f"{sum(checked)}/{len(checked)}" if checked
                     else "not checked (no ar arm in this run)"),
    }


if __name__ == "__main__":
    main()
