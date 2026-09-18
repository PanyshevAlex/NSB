#!/usr/bin/env bash
# =============================================================================
# DECISIVE clean A/B for --m_message_feature on genre, in the NO-OVERFIT regime (d=128).
# Why d=128: at d=784 every arm (control included) peaks ~epoch 3 then overfits down to ~0.43,
# so best-val sits on a noisy declining curve and the +-0.006 deltas are fragile / single-seed.
# At d=128 the distill-iter runs rose MONOTONICALLY for 20 epochs to ~0.49 (no collapse) -> best-val
# is stable and the feature's true effect is not confounded by overfit.
#
# Arms (matched: memory sampler k_m=10, kappa 40 -> so the ONLY difference is the message column):
#   control     : countloader, no message feature
#   mmsg-log1p  : + M as message feature, log1p squash (current default; genre test 0.4895 @1 seed)
#   mmsg-ecdf   : + M as message feature, frozen train-ECDF (stationary; was only run 2 epochs before)
# Run in YOUR OWN terminal:  bash scripts/run_mmsg_iter_genre.sh    (resume on; per-arm slug)
# For the real answer set SEEDS="1 2 3" and report mean+-std of test@best-val.
# =============================================================================
set -u
cd /Users/aleksandrpanysev/Documents/GitHub/2Q_2026_tgn_user_item || exit 1
PY=/Users/aleksandrpanysev/miniconda3/envs/tgb/bin/python
mkdir -p logs/runs
SEEDS="1"    # <- set to "1 2 3" for mean+-std (each seed ~20-30 min)
BASE="--dataset tgbn-genre --neighbor_sampler memory --k_m 10 --kappa_select 40 \
      --global_hidden_dims 128 --num_last_neighbours 30 --epochs 20 --lr 1e-4 --batch_size 200 \
      --learning_scheduler constant --experiment mmsg-iter"

run () {  # $1 = label, $2 = seed, rest = extra args
  local label="$1" seed="$2"; shift 2
  echo "================ $(date '+%Y-%m-%d %H:%M:%S')  ${label} seed ${seed} ================"
  $PY train-tgbn-nodeproppred.py $BASE --seed "$seed" "$@" 2>&1 \
    | tee "logs/runs/mmsg-iter-${label}-s${seed}.log" \
    | grep -aE "RESUMED|ECDF score reference|training Epoch|val/ndcg|test/ndcg|best (val|test) score"
}

for s in $SEEDS; do
  run "control"     "$s"
  run "mmsg-log1p"  "$s" --m_message_feature --m_msg_transform log1p
  run "mmsg-ecdf"   "$s" --m_message_feature --m_msg_transform ecdf
done
echo "================ DONE — compare test@best_val across arms in experiment 'mmsg-iter' ================"
