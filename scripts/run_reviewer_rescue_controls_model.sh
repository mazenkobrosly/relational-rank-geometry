#!/usr/bin/env bash
set -euo pipefail

# Run reviewer-rescue prompt controls with the same relational-arity pipeline as
# the main paper. This script is intended for a GPU pod with the repository
# copied to /workspace/plucker_sign_entropy, but PROJECT_DIR can override it.
#
# Examples:
#   MODEL_SHORT=8B  bash scripts/run_reviewer_rescue_controls_model.sh
#   MODEL_SHORT=70B bash scripts/run_reviewer_rescue_controls_model.sh
#   MODEL_SHORT=405B bash scripts/run_reviewer_rescue_controls_model.sh
#
# To run only the highest-value controls first:
#   CONTROL_BANKS="arity_nonce_predicate_r3_r6.jsonl arity_query_swap_nonce_r3_r6.jsonl" \
#   MODEL_SHORT=8B bash scripts/run_reviewer_rescue_controls_model.sh

PROJECT_DIR="${PROJECT_DIR:-/workspace/plucker_sign_entropy}"
OUT_ROOT="${OUT_ROOT:-/workspace/reviewer_rescue_controls}"
HF_HOME="${HF_HOME:-/workspace/hf_home}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
MODEL_SHORT="${MODEL_SHORT:-8B}"
ANALYSIS_WORKERS="${ANALYSIS_WORKERS:-32}"
LIVE_EVERY="${LIVE_EVERY:-10}"
PROJ_DIM="${PROJ_DIM:-64}"
PROJECTION_SEED="${PROJECTION_SEED:-42}"
CONTROL_TUPLE_BUDGET="${CONTROL_TUPLE_BUDGET:-20}"
N_CANDIDATES="${N_CANDIDATES:-24000}"
N_HUB_TOKENS="${N_HUB_TOKENS:-70}"
MAX_LENGTH="${MAX_LENGTH:-768}"
CONTROL_BANKS="${CONTROL_BANKS:-arity_nonce_predicate_r3_r6.jsonl arity_query_swap_nonce_r3_r6.jsonl arity_template_generalization_r3_r6.jsonl arity_generic_predicate_r3_r6.jsonl}"

export HF_HOME TRANSFORMERS_CACHE HF_HUB_CACHE TOKENIZERS_PARALLELISM

cd "$PROJECT_DIR"
mkdir -p "$OUT_ROOT/logs" "$HF_HOME"

case "$MODEL_SHORT" in
  8B)
    RUNNER="analysis/run_8b_relational_arity_benchmark.py"
    DIAG="analysis/run_8b_arity_diagonal_dominance.py"
    LAYERS="${LAYERS:-0,5,10,15,20,25,30,31}"
    RANKS="${RANKS:-1,2,3,4,5,6,7}"
    MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"
    ;;
  70B)
    RUNNER="analysis/run_70b_relational_arity_benchmark.py"
    DIAG="analysis/run_70b_arity_diagonal_dominance.py"
    LAYERS="${LAYERS:-20,30,40,50,55,60,70}"
    RANKS="${RANKS:-1,2,3,4,5,6,7}"
    MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.3-70B-Instruct}"
    ;;
  405B)
    RUNNER="analysis/run_405b_relational_arity_benchmark.py"
    DIAG="analysis/run_405b_arity_diagonal_dominance.py"
    LAYERS="${LAYERS:-20,30,40,60,80,90}"
    RANKS="${RANKS:-1,2,3,4,5,6,7}"
    MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.1-405B-Instruct}"
    ;;
  *)
    echo "Unknown MODEL_SHORT=$MODEL_SHORT; expected 8B, 70B, or 405B" >&2
    exit 2
    ;;
esac

python3 analysis/build_reviewer_rescue_controls.py --out-dir data/reviewer_rescue

echo "[rescue] model=$MODEL_SHORT model_id=$MODEL_ID"
echo "[rescue] layers=$LAYERS ranks=$RANKS out=$OUT_ROOT"
echo "[rescue] banks=$CONTROL_BANKS"

for bank in $CONTROL_BANKS; do
  slug="${bank%.jsonl}"
  task_bank="data/reviewer_rescue/$bank"
  cache_dir="$OUT_ROOT/cache_${MODEL_SHORT}_${slug}"
  results_dir="$OUT_ROOT/results_${MODEL_SHORT}_${slug}"
  figures_dir="$OUT_ROOT/figures_${MODEL_SHORT}_${slug}"
  mkdir -p "$cache_dir" "$results_dir" "$figures_dir"

  echo "[rescue] BEGIN $MODEL_SHORT $slug capture $(date -u)"
  python3 -u "$RUNNER" \
    --mode capture \
    --task-bank "$task_bank" \
    --model-id "$MODEL_ID" \
    ${MODEL_LOAD_PATH:+--model-load-path "$MODEL_LOAD_PATH"} \
    ${CACHE_DIR:+--cache-dir "$CACHE_DIR"} \
    --layers "$LAYERS" \
    --ranks "$RANKS" \
    --max-length "$MAX_LENGTH" \
    --output-cache-dir "$cache_dir" \
    --skip-existing \
    --proj-dim "$PROJ_DIM" \
    --projection-seed "$PROJECTION_SEED" \
    --control-tuple-budget "$CONTROL_TUPLE_BUDGET" \
    --n-candidates "$N_CANDIDATES" \
    --n-hub-tokens "$N_HUB_TOKENS" \
    --analysis-workers "$ANALYSIS_WORKERS" \
    --results-dir "$results_dir" \
    --figures-dir "$figures_dir" \
    --live-every "$LIVE_EVERY" \
    2>&1 | tee "$OUT_ROOT/logs/${MODEL_SHORT}_${slug}_capture.log"

  echo "[rescue] BEGIN $MODEL_SHORT $slug selector-analysis $(date -u)"
  python3 -u "$RUNNER" \
    --mode analyze \
    --task-bank "$task_bank" \
    --layers "$LAYERS" \
    --ranks "$RANKS" \
    --output-cache-dir "$cache_dir" \
    --proj-dim "$PROJ_DIM" \
    --projection-seed "$PROJECTION_SEED" \
    --control-tuple-budget "$CONTROL_TUPLE_BUDGET" \
    --n-candidates "$N_CANDIDATES" \
    --n-hub-tokens "$N_HUB_TOKENS" \
    --analysis-workers "$ANALYSIS_WORKERS" \
    --results-dir "$results_dir" \
    --figures-dir "$figures_dir" \
    --live-every "$LIVE_EVERY" \
    2>&1 | tee "$OUT_ROOT/logs/${MODEL_SHORT}_${slug}_selector_analysis.log"

  echo "[rescue] BEGIN $MODEL_SHORT $slug diagonal-dominance $(date -u)"
  python3 -u "$DIAG" \
    --cache-dir "$cache_dir" \
    --results-dir "$results_dir" \
    --figures-dir "$figures_dir" \
    --layers "$LAYERS" \
    --ranks "$RANKS" \
    --proj-dim "$PROJ_DIM" \
    --projection-seed "$PROJECTION_SEED" \
    --workers "$ANALYSIS_WORKERS" \
    --live-every "$LIVE_EVERY" \
    2>&1 | tee "$OUT_ROOT/logs/${MODEL_SHORT}_${slug}_diagonal.log"

  echo "[rescue] DONE $MODEL_SHORT $slug $(date -u)"
done

python3 analysis/summarize_reviewer_rescue_controls.py \
  --run-root "$OUT_ROOT" \
  --out-dir "$OUT_ROOT/summary" \
  2>&1 | tee "$OUT_ROOT/logs/${MODEL_SHORT}_summary.log"

echo "[rescue] ALL DONE $(date -u)"
