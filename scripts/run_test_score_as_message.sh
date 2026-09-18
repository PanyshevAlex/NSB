# mmsg arm — ECDF squash (diagnosis v2, notebooks/p4_mmsg_score_after_transform.ipynb: log1p fixed
# the SCALE but not the per-epoch ramp + train->eval drift -> all mmsg arms peak @e2-3 then decay;
# frozen train-ECDF makes the column Uniform[0,1] and stationary — val drift measured 0).
python train-tgbn-nodeproppred.py --dataset tgbn-genre --neighbor_sampler memory --k_m 10 \
--kappa_select 40 --global_hidden_dims 784 --epochs 50 --lr 1e-4 --experiment m-msg \
--m_message_feature --m_msg_transform ecdf --learning_scheduler constant --seed 1

# control arm: SAME config minus the message feature — run it LONG (>=16 epochs): the audit found
# NO d784 genre control past epoch 5, so the late-epoch decline attribution needs this curve.
python train-tgbn-nodeproppred.py --dataset tgbn-genre --neighbor_sampler memory --k_m 10 \
--kappa_select 40 --global_hidden_dims 784 --epochs 50 --lr 1e-4 --experiment m-msg \
--learning_scheduler constant --seed 1

# (optional) raw-score arm — the one diagnosed to hurt genre; uncomment only for the ablation table
# python train-tgbn-nodeproppred.py --dataset tgbn-genre --neighbor_sampler memory --k_m 10 \
# --kappa_select 40 --global_hidden_dims 784 --epochs 50 --lr 1e-4 --experiment m-msg \
# --m_message_feature --learning_scheduler constant --seed 1
