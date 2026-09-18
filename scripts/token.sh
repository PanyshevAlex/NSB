python train-tgbn-nodeproppred.py --dataset tgbn-token --neighbor_sampler memory \
--k_m 10 --kappa_select 40 --global_hidden_dims 128 --epochs 50 \
--lr 1e-4 --experiment m-msg --m_message_feature --m_msg_transform log1p \
--learning_scheduler constant --seed 1 --train_tail_frac 0.05 --val_frac 0.05 --no-eval_test