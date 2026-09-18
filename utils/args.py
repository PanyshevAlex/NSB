import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='parsing command line arguments as hyperparameters')
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='random seed to use')
    parser.add_argument('--dataset', type=str, default='tgbn-trade', help="The dataset to train on.")
    parser.add_argument('--experiment', type=str, default='baselines',
                        help="MLflow experiment name = research thread (e.g. baselines, user-item-matrix).")
    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs.")
    parser.add_argument('--batch_size', type=int, default=200, help="Batch Size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning Rate.")
    parser.add_argument("--global_hidden_dims", type=int, default=512, help="Hidden dims for intermediate embeddings.")
    parser.add_argument("--num_last_neighbours", type=int, default=10, help="No. of last neighbours in GNN component.")
    parser.add_argument("--use_tgnv2", default=True, action=argparse.BooleanOptionalAction, help="Whether to use TGNv2 or not.")
    # --- M-as-neighbor-sampler (P5 idea 1) ---
    parser.add_argument("--neighbor_sampler", type=str, default="recency", choices=["recency", "memory"],
                        help="Neighbor sampler: 'recency' (LastNeighborLoader) or 'memory' (dense per-(user,item) affinity M).")
    parser.add_argument("--use_gnn", default=True, action=argparse.BooleanOptionalAction,
                        help="Use the GNN embedding; --no-use_gnn ablates it (TGN memory + MLP only).")
    parser.add_argument("--k_m", type=int, default=-1,
                        help="Read-budget K_M for the memory sampler (top-K_M items by affinity; -1 = num_last_neighbours).")
    parser.add_argument("--kappa_select", type=float, default=25.0,
                        help="Selection half-life in DAYS for the memory sampler's affinity decay.")
    parser.add_argument("--m_edge_feature", default=False, action=argparse.BooleanOptionalAction,
                        help="Feed per-edge M affinity as a GNN edge feature (variant V2; requires --neighbor_sampler memory).")
    parser.add_argument("--m_msg_transform", type=str, default="raw", choices=["raw", "log1p", "ecdf", "ecdf_emb"],
                        help="Squash for the --m_message_feature score column (BOTH channels). 'raw' = as-is (trade: magnitude IS the signal); 'log1p' = tames the O(100-2500) scale (fixes GRU-gate saturation but NOT the per-epoch ramp / train->eval drift); 'ecdf' = frozen train-ECDF -> Uniform[0,1], stationary; 'ecdf_emb' = quantize the ECDF into --m_msg_bins equal-frequency CATEGORIES (bin 0 = empty/cold) and encode each via a learned torch.nn.Embedding of dim --m_msg_emb_dim -> a distinct vector for the 'no history' case, no magnitude, stationary bins (targets token's bimodal/zero-inflated column). Own checkpoint slugs (_mmsglog/_mmsgecdf/_mmsgemb).")
    parser.add_argument("--m_msg_bins", type=int, default=16,
                        help="ecdf_emb: number of equal-frequency ECDF bins the score is quantized into (bin 0 collects empty/near-zero rows).")
    parser.add_argument("--m_msg_emb_dim", type=int, default=8,
                        help="ecdf_emb: torch.nn.Embedding dim per score bin (memory + GNN channels each own one; the message widens by this instead of by 1).")
    parser.add_argument("--m_message_feature", default=False, action=argparse.BooleanOptionalAction,
                        help="Treat the CountLoader affinity M[u,i] as a FIRST-CLASS message-like feature: it rides as an extra column INSIDE raw_msg, so it flows through EncodeIndexModule -> the memory GRU update AND the GNN edge_attr exactly like raw_msg (widens msg_dim by 1). Score source: the memory sampler's M, or (recency sampler) a lockstep score-only M-carrier built at --kappa_select — so the feature is REAL under both samplers. Default off => byte-identical. Distinct from --m_edge_feature (GNN-only, per-user-normalized).")
    parser.add_argument("--resume", default=True, action=argparse.BooleanOptionalAction,
                        help="Auto-resume from saved_models/<run_name>.pt if it exists (per-epoch checkpoint; survives background-task reaps).")
    # --- Distillation on M-scores (P5 idea 2): KD aux loss on the ~98% target-less batches ---
    parser.add_argument("--distill", default=False, action=argparse.BooleanOptionalAction,
                        help="Distill the head on M soft-targets over target-less batches. Teacher = the memory sampler's M (variant 1) or a separate MSampler under the recency sampler (variant 2). Single head at inference (not an ensemble).")
    parser.add_argument("--distill_stride", type=int, default=50,
                        help="Run the KD forward/backward on every Nth TARGET-LESS batch (cost knob). genre has ~88k target-less batches/epoch: stride=50 ≈ supervised count (~2x epoch time); stride=1 ≈ 55x (prohibitive at d=784 on CPU).")
    parser.add_argument("--distill_lambda", type=float, default=1.0,
                        help="KD loss weight (per-user-mean cross-entropy of the soft target; same scale as the supervised CE).")
    parser.add_argument("--kappa_teacher", type=float, default=-1.0,
                        help="Teacher M decay half-life in DAYS; -1 inherits --kappa_select. Only decouples in variant 2 (recency, separate teacher M); variant 1 always reuses the selection M.")
    parser.add_argument("--eval_every", type=int, default=1,
                        help="Run val+test only every N epochs (the final epoch is always evaluated). Cuts the heavy eval streams on big datasets; best-epoch selection then sees only evaluated epochs.")
    parser.add_argument("--val_frac", type=float, default=1.0,
                        help="Evaluate on only the FIRST fraction of the val split (the head continues the train stream, so causality/contiguity hold). Cheap-iteration knob; requires --no-eval_test when < 1 (the skipped val edges would break the e_id == global-index invariant for the test stream). Relative numbers only.")
    parser.add_argument("--eval_test", default=True, action=argparse.BooleanOptionalAction,
                        help="Stream the test split at eval epochs. --no-eval_test skips it entirely (best-epoch selection is by val anyway; cheap iteration).")
    parser.add_argument("--train_tail_frac", type=float, default=1.0,
                        help="Train each epoch only on the LAST fraction of the train split (faster iteration on heavy datasets like reddit); labels before the tail are skipped. 1.0 = full train. Numbers are for RELATIVE A/B only, not comparable to paper runs.")
    parser.add_argument("--kd_grad", type=str, default="full", choices=["full", "gnn_head", "head"],
                        help="Which student params the KD loss updates: 'full' (memory+GNN+head, current), 'gnn_head' (detach memory — KD trains GNN+head only), 'head' (detach embedding — KD trains only node_pred). 'head'/'gnn_head' protect the sequential TGN memory that eval reads.")
    parser.add_argument("--dump_eval_preds", default=False, action=argparse.BooleanOptionalAction,
                        help="Dump per-(user,day) test NDCG@10 of the best-val epoch to mlruns/_eval_preds_<dataset>.parquet (for error/headroom analysis).")
    parser.add_argument("--learning_scheduler", type=str, default='constant')
    parser.add_argument("--cosine_annealing_ratio", type=float, default=0.2)
    parser.add_argument("--step_lr_gamma", type=float, default=0.5)
    parser.add_argument("--step_lr_step_size", type=int, default=250)
    parser.add_argument("--label_aggregator_window", type=int, default=7)
    parser.add_argument("--label_aggregator_noise_factor", type=float, default=0.01)
    parser.add_argument("--label_aggregator_mode", type=str, default='moving_average')
    parser.add_argument("--dataset_size", type=float, default=1.)
    
    return parser
