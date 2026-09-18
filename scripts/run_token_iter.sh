#!/usr/bin/env bash
# =============================================================================
# CHEAP tgbn-token iteration: last 5% of train, first 5% of val, NO test stream, small model (d=128).
#   --train_tail_frac 0.05 : train each epoch on the LAST 5% of train (label pointer fast-forwarded,
#                            e_id counters rebased -> features stay correct)
#   --val_frac 0.05        : evaluate on the FIRST 5% of val (head continues the train stream)
#   --no-eval_test         : test split never streamed (required with val_frac<1; best-epoch by val)
# Numbers are RELATIVE-ONLY (arm vs arm) — never compare to paper numbers.
#
# Arms:
#   default : recency LastNeighborLoader baseline (plain TGNv2)
#   method  : CountLoader (--neighbor_sampler memory, k_m=10, kappa 16 = token optimum)
#             + M as message feature (--m_message_feature, log1p squash)
#
# Run in YOUR OWN terminal:  bash scripts/run_token_iter.sh   (resume on; own _ttf/_vf/_notest slug)
# NB: MSampler on token allocates the dense M (61756x1001 float64 ~ 0.5 GB + buffers) — fine on 16GB.
# =============================================================================
set -u
cd /Users/aleksandrpanysev/Documents/GitHub/2Q_2026_tgn_user_item || exit 1
PY=/Users/aleksandrpanysev/miniconda3/envs/tgb/bin/python
mkdir -p logs/runs
SEED=1
EPOCHS=10
COMMON="--dataset tgbn-token --global_hidden_dims 128 --num_last_neighbours 10 --batch_size 200 \
        --lr 1e-4 --learning_scheduler constant --epochs ${EPOCHS} --seed ${SEED} \
        --train_tail_frac 0.05 --val_frac 0.05 --no-eval_test --experiment token-iter"

run () {  # $1 = label, rest = extra args
  local label="$1"; shift
  echo "================ $(date '+%Y-%m-%d %H:%M:%S')  ${label}  (token 5%/5%, no test, seed ${SEED}) ================"
  $PY train-tgbn-nodeproppred.py $COMMON "$@" 2>&1 \
    | tee "logs/runs/token-iter-${label}-s${SEED}.log" \
    | grep -aE "RESUMED|train-tail|training Epoch|val/ndcg|best val score|best validation epoch|Traceback"
}

run "default" --neighbor_sampler recency
run "method"  --neighbor_sampler memory --k_m 10 --kappa_select 16 \
              --m_message_feature --m_msg_transform log1p

echo "================ DONE — compare best_val_ndcg in experiment 'token-iter' ================"
