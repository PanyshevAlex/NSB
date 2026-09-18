#!/usr/bin/env bash
# =============================================================================
# FAST tgbn-reddit iteration: train each epoch only on the LAST 30% of the train split
# (--train_tail_frac 0.3; the label pointer is fast-forwarded automatically). ~3x faster
# epochs on the 27M-edge reddit. Numbers are for RELATIVE A/B only — NOT paper-comparable.
#
# Arms: recency baseline vs CountLoader (memory sampler, k_m=10, kappa 40 = reddit ranker optimum).
# Run in YOUR OWN terminal:  bash scripts/run_reddit_iter.sh
# Checkpoint/resume is ON — re-run to continue (ttf runs get their own checkpoints/slug).
# =============================================================================
set -u
cd /Users/aleksandrpanysev/Documents/GitHub/2Q_2026_tgn_user_item || exit 1
PY=/Users/aleksandrpanysev/miniconda3/envs/tgb/bin/python
mkdir -p logs/runs

EPOCHS=8
SEED=1
TTF=0.3
COMMON="--dataset tgbn-reddit --epochs ${EPOCHS} --global_hidden_dims 128 --num_last_neighbours 30 \
        --batch_size 200 --lr 1e-4 --learning_scheduler constant --experiment reddit-iter --seed ${SEED} \
        --train_tail_frac ${TTF} --eval_every 2"

run () {  # $1 = label, rest = extra args
  local label="$1"; shift
  echo "================ $(date '+%Y-%m-%d %H:%M:%S')  ${label}  (ttf=${TTF}, seed ${SEED}) ================"
  $PY train-tgbn-nodeproppred.py $COMMON "$@" 2>&1 \
    | tee "logs/runs/reddit-iter-${label}-s${SEED}.log" \
    | grep -aE "RESUMED|train-tail|training Epoch|val/ndcg|test/ndcg|best (val|test) score"
}

run "recency"   --neighbor_sampler recency
run "mem-km10"  --neighbor_sampler memory --k_m 10 --kappa_select 40

echo "================ DONE — experiment 'reddit-iter' in mlflow ================"
