#!/usr/bin/env bash
# =============================================================================
# P5 idea-2: distillation on M-scores — genre, paper width d=784.
# KD aux loss on the ~98% target-less batches; teacher = dense per-(user,item) M.
# Two variants:
#   V1 = WITH the M-sampler  (--neighbor_sampler memory --k_m 10): teacher reuses the selection M.
#   V2 = WITH the default recency sampler (--neighbor_sampler recency): a SEPARATE teacher M.
# Baselines to compare against (no distill) already live in the `m-sampler` experiment:
#   V1 baseline = mem-km10 ; V2 baseline = recency (plain TGNv2).
#
# Run in YOUR OWN terminal (persists; harness reaps background tasks):  bash scripts/run_distill_genre.sh
# Checkpoint/resume is ON: re-run to continue from saved_models/<run_name>.pt.
#
# COST: KD adds a GNN forward/backward on every distill_stride-th target-less batch.
#   --distill_stride 50 ≈ supervised batch count ≈ ~2x epoch time (so ~28 min/epoch at d=784).
#   Raise the stride (cheaper, less KD signal) or lower it (more) to taste. Watch that val NDCG
#   EXCEEDS the M teacher (~0.52), not just approaches it (too-strong KD pins the student to it).
# =============================================================================
set -u
cd /Users/aleksandrpanysev/Documents/GitHub/2Q_2026_tgn_user_item || exit 1
PY=/Users/aleksandrpanysev/miniconda3/envs/tgb/bin/python
mkdir -p logs/runs

EPOCHS=10
SEED=1
LAMBDA=1.0
STRIDE=50      # diagnosis: stride=1 (KD on every target-less batch) HURTS — keep KD steps <= CE steps
COMMON="--dataset tgbn-genre --epochs ${EPOCHS} --global_hidden_dims 784 --num_last_neighbours 30 \
        --batch_size 200 --lr 1e-4 --learning_scheduler constant --experiment distill --seed ${SEED} \
        --distill --distill_lambda ${LAMBDA} --distill_stride ${STRIDE} --kd_grad head"
# --kd_grad head: KD updates only node_pred (detaches the embedding) so it can't destabilize the TGN
# memory that eval reads — the fix for the student capping below the ~0.52 teacher.

run () {  # $1 = short label, rest = extra args
  local label="$1"; shift
  echo "================ $(date '+%Y-%m-%d %H:%M:%S')  ${label}  (seed ${SEED}, lambda ${LAMBDA}, stride ${STRIDE}) ================"
  $PY train-tgbn-nodeproppred.py $COMMON "$@" 2>&1 \
    | tee "logs/runs/${label}-s${SEED}.log" \
    | grep -aE "RESUMED|training Epoch|train_kd_loss|val/ndcg|test/ndcg|best (val|test) score"
}

# V1 — distillation WITH the M-sampler (teacher reuses the selection M, k_m=10; kappa 40 = teacher optimum 0.522)
run "distill-mem-km10" --neighbor_sampler memory --k_m 10 --kappa_select 40

# V2 — distillation WITH the default recency sampler (separate teacher M, kappa_teacher inherits 40)
run "distill-recency"  --neighbor_sampler recency --kappa_select 40

echo "================ ALL DONE — browse: mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db ================"
