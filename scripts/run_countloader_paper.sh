#!/usr/bin/env bash
# =============================================================================
# CountLoader (dense per-(user,item) affinity M-sampler, --neighbor_sampler memory) INSTEAD of the
# recency LastNeighborLoader, on ALL 4 tgbn datasets, at the paper Table-2 hyperparameters.
# Samples at most 10 neighbours per query user (--k_m 10). No distillation.
#
# Compare best_test_ndcg (experiment 'countloader') vs the paper TGNv2 (recency LastLoader) baseline:
#     trade 0.735 | genre 0.469 | reddit 0.507 | token 0.294
#
# Run in YOUR OWN terminal (persists; the Claude harness reaps background tasks). Checkpoint/resume
# is ON — re-run this script to continue from saved_models/<run_name>.pt.
#
# ⚠️ Full paper config on CPU is HEAVY: token = 50 epochs @ d=1024 over ~72.9M edges/epoch; trade =
#    750 epochs. Expect many hours→days per dataset. Comment out datasets / cut --epochs as needed;
#    for the paper protocol also loop SEED over 1 2 3.
# =============================================================================
set -u
cd /Users/aleksandrpanysev/Documents/GitHub/2Q_2026_tgn_user_item || exit 1
PY=/Users/aleksandrpanysev/miniconda3/envs/tgb/bin/python
mkdir -p logs/runs
SEED=1
# CountLoader = the dense-affinity memory sampler; k_m=10 = sample <=10 highest-affinity items.
COMMON="--neighbor_sampler memory --k_m 10 --batch_size 200 --experiment countloader --seed ${SEED}"

run () {  # $1 = dataset, rest = per-dataset paper hyperparameters
  local name="$1"; shift
  echo "================ $(date '+%Y-%m-%d %H:%M:%S')  ${name}  (CountLoader k_m=10, seed ${SEED}) ================"
  $PY train-tgbn-nodeproppred.py --dataset "$name" $COMMON "$@" 2>&1 \
    | tee "logs/runs/countloader-${name}-s${SEED}.log" \
    | grep -aE "RESUMED|training Epoch|val/ndcg|test/ndcg|best (val|test) score"
}

# --- paper Table-2 per-dataset config (lr / epochs / d / scheduler); k_m=10 overrides the sampler size ---
run tgbn-trade  --lr 1e-3 --epochs 750 --global_hidden_dims 784  --num_last_neighbours 25 \
                --learning_scheduler step_lr --step_lr_gamma 0.5 --step_lr_step_size 250
run tgbn-genre  --lr 1e-4 --epochs 50  --global_hidden_dims 784  --num_last_neighbours 30 --learning_scheduler constant
run tgbn-reddit --lr 1e-4 --epochs 50  --global_hidden_dims 784  --num_last_neighbours 30 --learning_scheduler constant
run tgbn-token  --lr 1e-4 --epochs 50  --global_hidden_dims 1024 --num_last_neighbours 10 --learning_scheduler constant

echo "================ DONE — browse: mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db (experiment 'countloader') ================"
