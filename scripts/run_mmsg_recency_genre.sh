#!/usr/bin/env bash
# =============================================================================
# --m_message_feature with the DEFAULT recency LastNeighborLoader (genre).
# The neighbours are the most-recent partners (plain TGNv2 sampling); the M-affinity enters ONLY
# as a message-like feature column (memory GRU + GNN edge_attr), scored by a lockstep M-carrier
# (score-only MSampler at --kappa_select; without it the column would be constant zero).
# Isolates "M as FEATURE" from "M as SELECTOR" (the m-sampler runs).
#
# log1p squash per the diagnosis (notebooks/p4_mmsg_forward_diag.ipynb): raw M is O(100-2500) and
# saturates GRU gates on genre.
# Run in YOUR OWN terminal: bash scripts/run_mmsg_recency_genre.sh   (resume is on)
# =============================================================================
set -u
cd /Users/aleksandrpanysev/Documents/GitHub/2Q_2026_tgn_user_item || exit 1
PY=/Users/aleksandrpanysev/miniconda3/envs/tgb/bin/python
mkdir -p logs/runs
COMMON="--dataset tgbn-genre --neighbor_sampler recency --num_last_neighbours 30 \
        --global_hidden_dims 784 --epochs 50 --lr 1e-4 --batch_size 200 \
        --learning_scheduler constant --experiment m-msg --seed 1"

run () {  # $1 = label, rest = extra args
  local label="$1"; shift
  echo "================ $(date '+%Y-%m-%d %H:%M:%S')  ${label} ================"
  $PY train-tgbn-nodeproppred.py $COMMON "$@" 2>&1 \
    | tee "logs/runs/${label}-s1.log" \
    | grep -aE "RESUMED|training Epoch|val/ndcg|test/ndcg|best (val|test) score"
}

# feature arm: recency selection + M as message feature (carrier at kappa 40, log1p squash)
run "recency-mmsg-log1p" --m_message_feature --m_msg_transform log1p --kappa_select 40

# control arm: plain recency TGNv2 — identical config minus the feature (one-bit A/B)
run "recency-control"

echo "================ DONE — experiment 'm-msg' ================"
