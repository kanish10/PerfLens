#!/usr/bin/env bash
# Full PerfLens Tune experiment. Run on the GPU box (a VM instance with
# Nsight Compute counter access, same setup as the M6 case study), from the
# PerfLens checkout, with gpu-engine checked out as a sibling (../gpu-engine,
# same convention as suite.yaml).
#
#   scripts/run_tune_experiment.sh            # full matrix
#   QUICK=1 scripts/run_tune_experiment.sh    # batch 8 only, 2 seeds (~half the time)
#
# What it runs, per batch size:
#   1. exhaustive search for every kernel (ground truth; also warms the build
#      cache, so later strategies don't pay nvcc time twice)
#   2. random / evo / agent on bias_gelu, at an equal budget, for each seed.
#      Only bias_gelu: add_layernorm and argmax have 12-config spaces, smaller
#      than a useful budget, so exhaustive search is simply the answer there.
#   3. Nsight Compute re-measurement of default vs. exhaustive-best per kernel
# then writes results/tune/{summary.csv,REPORT.md,ncu_verify.csv}.
#
# Env overrides: ENGINE_DIR, DB, BUDGET, SEEDS, BATCHES, MODEL, NCU.
set -euo pipefail

ENGINE_DIR=${ENGINE_DIR:-../gpu-engine}
DB=${DB:-results/tune/tune.db}
OUT=$(dirname "$DB")
BUDGET=${BUDGET:-20}
NCU=${NCU:-ncu}
if [[ "${QUICK:-0}" == "1" ]]; then
  SEEDS=${SEEDS:-"1 2"}
  BATCHES=${BATCHES:-"8"}
else
  SEEDS=${SEEDS:-"1 2 3"}
  BATCHES=${BATCHES:-"8 32"}
fi
MODEL_ARGS=()
[[ -n "${MODEL:-}" ]] && MODEL_ARGS=(--model "$MODEL")

die() { echo "error: $*" >&2; exit 1; }
command -v perflens >/dev/null || die "perflens not on PATH (pip install -e .)"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found -- this needs the GPU box"
[[ -x "$ENGINE_DIR/bench/build_candidate.sh" ]] || die "$ENGINE_DIR/bench/build_candidate.sh missing"
[[ -d /usr/local/cuda/bin ]] && export PATH="/usr/local/cuda/bin:$PATH"
command -v nvcc >/dev/null || die "nvcc not on PATH (CUDA toolkit installed?)"

STRATEGIES=(random evo agent)
if [[ -z "${ANTHROPIC_API_KEY:-}" && -z "${GROQ_API_KEY:-}" ]]; then
  echo "warning: no ANTHROPIC_API_KEY or GROQ_API_KEY -- skipping the agent strategy" >&2
  STRATEGIES=(random evo)
fi
HAVE_NCU=1
command -v "$NCU" >/dev/null || { echo "warning: $NCU not found -- skipping Nsight verification" >&2; HAVE_NCU=0; }

mkdir -p "$OUT"
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv | tee "$OUT/gpu.txt"
git -C "$ENGINE_DIR" rev-parse HEAD | tee "$OUT/engine_sha.txt"

for batch in $BATCHES; do
  for kernel in bias_gelu add_layernorm argmax; do
    perflens tune run --kernel "$kernel" --batch "$batch" --strategy exhaustive \
      --db "$DB" --engine-dir "$ENGINE_DIR"
  done
  for seed in $SEEDS; do
    for strategy in "${STRATEGIES[@]}"; do
      perflens tune run --kernel bias_gelu --batch "$batch" --strategy "$strategy" \
        --budget "$BUDGET" --seed "$seed" --db "$DB" --engine-dir "$ENGINE_DIR" "${MODEL_ARGS[@]}"
    done
  done
  if [[ "$HAVE_NCU" == "1" ]]; then
    for kernel in bias_gelu add_layernorm argmax; do
      perflens tune verify --kernel "$kernel" --batch "$batch" --db "$DB" \
        --engine-dir "$ENGINE_DIR" --ncu-path "$NCU" --out "$OUT"
    done
  fi
done

perflens tune report --db "$DB" --out "$OUT"
echo
echo "Done. Commit $OUT/ (summary.csv, REPORT.md, ncu_verify.csv, gpu.txt, engine_sha.txt)."
echo "*.db is gitignored; add the database too with: git add -f $DB"
