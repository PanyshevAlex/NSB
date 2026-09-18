#!/usr/bin/env bash
# =============================================================================
# --m_message_feature with the CATEGORICAL-EMBEDDING transform (--m_msg_transform ecdf_emb):
# the affinity score is quantized into --m_msg_bins equal-frequency ECDF categories (bin 0 = empty/
# cold) and each category is encoded via a learned torch.nn.Embedding of dim --m_msg_emb_dim, in BOTH
# channels (memory GRU message + GNN edge_attr). Designed to fix the token failure mode measured in
# notebooks/p4_mmsg_score_after_transform.ipynb: the raw/log1p score column is bimodal + zero-inflated
# (15.5% near-zero on the predicted users) -> the "no history" case gets its OWN vector instead of a
# gate-saturating number.
#
# Regime = d=128 (NO overfit — at d=784 every arm peaks ~e3 then decays; d=128 rises monotonically to
# ~0.49, so best-val is stable and the A/B is clean). Run in YOUR OWN terminal; resume on; per-arm slug.
# For the real answer set SEEDS="1 2 3" and report mean+-std of test@best-val.
# =============================================================================
set -u
cd /Users/aleksandrpanysev/Documents/GitHub/2Q_2026_tgn_user_item || exit 1
PY=/Users/aleksandrpanysev/miniconda3/envs/tgb/bin/python
mkdir -p logs/runs
SEEDS="1"    # <- "1 2 3" for mean+-std
BINS=16; EMB=8
BASE="--dataset tgbn-genre --neighbor_sampler memory --k_m 10 --kappa_select 40 \
      --global_hidden_dims 128 --num_last_neighbours 30 --epochs 20 --lr 1e-4 --batch_size 200 \
      --learning_scheduler constant --experiment mmsg-emb"

run () {  # $1 = label, $2 = seed, rest = extra args
  local label="$1" seed="$2"; shift 2
  echo "================ $(date '+%Y-%m-%d %H:%M:%S')  ${label} seed ${seed} ================"
  $PY train-tgbn-nodeproppred.py $BASE --seed "$seed" "$@" 2>&1 \
    | tee "logs/runs/mmsg-emb-${label}-s${seed}.log" \
    | grep -aE "RESUMED|ECDF score reference|training Epoch|val/ndcg|test/ndcg|best (val|test) score"
}

for s in $SEEDS; do
  run "control"    "$s"                                                              # no message feature
  run "ecdf"       "$s" --m_message_feature --m_msg_transform ecdf                   # scalar reference
  run "ecdf-emb"   "$s" --m_message_feature --m_msg_transform ecdf_emb --m_msg_bins $BINS --m_msg_emb_dim $EMB
done
echo "================ DONE — compare test@best_val across arms in experiment 'mmsg-emb' ================"

# ---- single commands for the datasets where ecdf_emb is most motivated (token/reddit), paper config ----
# token (extreme bimodal/zero-inflated column):
#   python train-tgbn-nodeproppred.py --dataset tgbn-token  --neighbor_sampler memory --k_m 10 \
#     --kappa_select 16 --global_hidden_dims 1024 --num_last_neighbours 10 --epochs 50 --lr 1e-4 \
#     --learning_scheduler constant --experiment mmsg-emb --seed 1 \
#     --m_message_feature --m_msg_transform ecdf_emb --m_msg_bins 16 --m_msg_emb_dim 8
# reddit:
#   python train-tgbn-nodeproppred.py --dataset tgbn-reddit --neighbor_sampler memory --k_m 10 \
#     --kappa_select 40 --global_hidden_dims 784  --num_last_neighbours 30 --epochs 50 --lr 1e-4 \
#     --learning_scheduler constant --experiment mmsg-emb --seed 1 \
#     --m_message_feature --m_msg_transform ecdf_emb --m_msg_bins 16 --m_msg_emb_dim 8
