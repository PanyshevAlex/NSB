#!/usr/bin/env bash
# FAST distillation ablation (d=128, converges in-session) to find the best KD variant.
# Diagnosis (2026-07-01): the teacher is STRONG (~0.52 at kappa 25-40, sweep-verified — NOT 0.46).
# The student capped ~0.49 because KD (a) destabilized the TGN memory it backprops through on ~98%
# of batches, and (b) took its own optimizer.step() ~55x more than CE. This ablation isolates the fix:
#   base-nodistill : M-sampler, no KD (reference)
#   kd-full        : current KD (updates memory+GNN+head)  -> expect ~0.49 (reproduces the problem)
#   kd-gnnhead     : KD detaches memory (trains GNN+head)   -> memory protected
#   kd-head        : KD detaches embedding (trains head)    -> memory+GNN protected (cleanest)
# Watch val/ndcg per epoch: the fix should climb toward the ~0.52 teacher, and MORE KD should stop hurting.
set -u
cd /Users/aleksandrpanysev/Documents/GitHub/2Q_2026_tgn_user_item || exit 1
PY=/Users/aleksandrpanysev/miniconda3/envs/tgb/bin/python
mkdir -p logs/runs

# d=128 (converges in ~1-2 min/epoch vs ~14 min at d=784); kappa 40 = sweep optimum (teacher 0.522).
BASE="--dataset tgbn-genre --epochs 20 --global_hidden_dims 128 --num_last_neighbours 30 \
      --batch_size 200 --lr 1e-4 --learning_scheduler constant --experiment distill-iter --seed 1 \
      --neighbor_sampler memory --k_m 10 --kappa_select 40"
DIST="--distill --distill_lambda 1.0 --distill_stride 50"

run () {  # $1 = label, rest = extra args
  local label="$1"; shift
  echo "================ $(date '+%Y-%m-%d %H:%M:%S')  ${label} ================"
  $PY train-tgbn-nodeproppred.py $BASE "$@" 2>&1 \
    | tee "logs/runs/iter-${label}.log" \
    | grep -aE "RESUMED|training Epoch|train_kd_loss|val/ndcg|test/ndcg|best (val|test) score"
}

# run "base-nodistill"
# run "kd-full"     $DIST --kd_grad full
# run "kd-gnnhead"  $DIST --kd_grad gnn_head
run "kd-head"     $DIST --kd_grad head
echo "================ DONE — compare best_val/best_test in experiment 'distill-iter' ================"
