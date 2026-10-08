#!/usr/bin/env bash
# Goose across the paper's grid: five models x five benchmarks against the
# autoregressive reference, then the equal-budget topology control of Table 1.
# The published baselines are other systems and are run from their own
# repositories, not from here.
#
#   bash benchmarks/run_paper_experiments.sh [output_dir]
#
# One model is resident at a time; the 33B model needs two GPUs and is left to
# the device map. Runtime is dominated by the autoregressive reference, which
# every speedup is measured against.
set -uo pipefail   # not -e: one cell that does not fit in memory should not
                   # take the remaining cells and the summary down with it

OUT=${1:-results}
PYTHON=${PYTHON:-python}
# The paper's runs stopped only on tokenizer.eos_token_id, so this script does
# too. Drop the flag to stop on every id the checkpoint declares, which is the
# decoder's default.
STOPPING="--stop-tokens tokenizer"
MODELS=(vicuna-7b llama3-8b qwen3-8b vicuna-13b vicuna-33b)
DATASETS=(classeval gsm8k humaneval mbpp mtbench)

for model in "${MODELS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    $PYTHON benchmarks/run_benchmark.py \
      --model "$model" --dataset "$dataset" \
      --methods ar goose $STOPPING \
      --output-dir "$OUT" || echo "FAILED: $model x $dataset (main grid)"
  done
done

# Anisotropic against isotropic at the same node budget (Table 1). The
# autoregressive arm comes along so these cells are lossless-checked too.
for model in llama3-8b qwen3-8b; do
  for dataset in "${DATASETS[@]}"; do
    $PYTHON benchmarks/run_benchmark.py \
      --model "$model" --dataset "$dataset" \
      --methods ar iso3 iso5 goose $STOPPING \
      --output-dir "$OUT/topology" || echo "FAILED: $model x $dataset (topology)"
  done
done

$PYTHON benchmarks/summarize.py "$OUT"
