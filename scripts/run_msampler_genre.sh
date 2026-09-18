#!/usr/bin/env bash
# =============================================================================
# P5 idea-1: M-as-neighbor-sampler — genre experiments at paper width d=784.
#
# Run in YOUR OWN terminal (persists for hours; harness-managed bg tasks get reaped):
#     bash run_msampler_genre.sh
#
# Checkpoint/resume is ON by default. If a run is interrupted, just re-run this
# script: each config continues from saved_models/<run_name>.pt, and already-finished
# configs short-circuit in seconds. Per-epoch val/test also land in MLflow live.
#
# Variants (single TGNv2 head; M only selects/weights neighbours -> not an ensemble):
#   V1  = --neighbor_sampler memory                    (M-affinity neighbour sampling)
#   V2  = V1 + --m_edge_feature                        (+ per-edge M affinity to attention)
# Neighbour-budget sweep k_m in {5,10,20}. Baseline recency/TGNv2 (~0.469) is commented.
# Bar: beat TGNv2 0.469. (Note: paper genre uses 50 epochs/3 seeds; bump EPOCHS/SEEDS for the final number.)
# =============================================================================
set -u
cd /Users/aleksandrpanysev/Documents/GitHub/2Q_2026_tgn_user_item || exit 1
PY=/Users/aleksandrpanysev/miniconda3/envs/tgb/bin/python
mkdir -p logs/runs

EPOCHS=10
SEED=1
COMMON="--dataset tgbn-genre --epochs ${EPOCHS} --global_hidden_dims 784 --num_last_neighbours 30 \
        --batch_size 200 --lr 1e-4 --learning_scheduler constant --experiment m-sampler --seed ${SEED}"

run () {  # $1 = short label (for the log filename), rest = extra args
  local label="$1"; shift
  echo "================ $(date '+%Y-%m-%d %H:%M:%S')  ${label}  (seed ${SEED}) ================"
  $PY train-tgbn-nodeproppred.py $COMMON "$@" 2>&1 \
    | tee "logs/runs/${label}-s${SEED}.log" \
    | grep -aE "RESUMED|training Epoch|val/ndcg|test/ndcg|best (val|test) score"
}

# --- baseline (TGNv2 number already known ~0.469; uncomment to reproduce at this config) ---
# run "recency" --neighbor_sampler recency

# --- neighbour-budget sweep: k_m in {5,10,20} x {V1, V2} ---
for KM in 20; do
  # run "mem-km${KM}"        --neighbor_sampler memory --kappa_select 25 --k_m "${KM}"
  run "mem-km${KM}-medge"  --neighbor_sampler memory --kappa_select 25 --k_m "${KM}" --m_edge_feature
done

echo "================ ALL DONE — browse: mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db ================"
