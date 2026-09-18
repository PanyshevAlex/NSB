# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## What this is

Research implementation of **TGNv2** ([arxiv 2411.03596](https://arxiv.org/abs/2411.03596)) for the **node property prediction** task on [TGB](https://tgb.complexdatalab.com/) (Temporal Graph Benchmark) `tgbn-*` datasets. The code starts from the PyG/TGB reference implementations (see the source-link comments at the top of `models/mtgn.py`, `train-tgbn-nodeproppred.py`, and `modules/heuristics.py`) and extends classic TGN into TGNv2.

## Research goal

This is the **baseline / starting code** for a research project targeting an **A\*-venue paper**. The TGNv2 pipeline here is the foundation to build the new contribution on top of.

The core idea under development: for **node affinity prediction**, the `tgbn-*` graphs have an **item–user** (bipartite) structure where items are far fewer than users (`N_item << N_user`). We want to exploit this asymmetry. Because items are few, we can afford to materialize and maintain dense **`N_user × N_item`** state (e.g. per-user × per-item affinity/memory matrices). In the general temporal-graph case this is intractable, since fully-relational state scales as **`N_user × N_user`** — too expensive. The asymmetry is what makes the dense-matrix approach feasible, and is the lever the new method is meant to pull.

### Hard constraint: NO ensembles (of any kind)

**Ensembles are strictly forbidden in this project — in every form, including hidden/implicit ones.** Do not propose, design, or implement any idea that combines multiple predictors to produce a result. This ban covers, non-exhaustively:

- Explicit ensembles: bagging, boosting, stacking, blending, voting/averaging over multiple models, snapshot ensembles, model soups, weight averaging (EMA/SWA), seed-averaging of predictions.
- **Hidden / implicit ensembles**: multiple prediction heads whose outputs are merged, mixture-of-experts, multi-branch models combined at the end, test-time augmentation averaged into one prediction, multi-view/multi-encoder fusion that is effectively an averaged committee, or any "single model" that is an ensemble in disguise.

If a proposed approach reduces to averaging/combining several models or heads, it is out of scope — reject it and find a single-model formulation instead.

### Compute: experiments run on CPU

All experiments are run on **CPU**, not GPU. The code falls back to CPU automatically (`device = cuda if available else cpu`), so just run on a machine without CUDA — or force it via `CUDA_VISIBLE_DEVICES=""`. Keep model sizes / batch sizes / dataset choices realistic for CPU-bound runtimes when designing experiments.

### Research process, roadmap & related work

The active research lives in four checked-in Markdown docs — read them before proposing or implementing any method change:

- **Process / phases — `framework.md`.** The work runs through phases **P0–P7**: P0 baselines → P1 EDA → P2 error analysis → P3 hypothesis synthesis → P5 experiment program. The flow is **analysis (P1–P2) → hypotheses (P3) → `backlog.md`**, and the `notebooks/` map 1:1 to these phases.
- **Active roadmap — `backlog.md`.** Prioritized tasks **T1–T9** (each tied to a falsifiable hypothesis H1–H5 with an exit gate); start order is T1→T2→T3. Key empirical anchor *beyond* paper Table 1: a **single-model, M-only, continuous-time per-pair memory** (dense `M[u,i]`, half-life κ≈40 days, parameter-free `rank/ECDF` write transform) already reaches **genre test NDCG@10 ≈ 0.526–0.530** — beating the *learned* TGNv2 (0.469) and the best static heuristic MovAvg(L) (0.509) — and also beats TGNv2 on reddit (0.560) and token (0.449) on the same protocol (`notebooks/p3_perpair_memory.ipynb`). The paper baseline is a floor, not a ceiling.
- **Related work / positioning — `lit_review_cold_start_user_item.md`, `adjacent_fields_dense_memory.md`.** The closest competitor is **NAViS** ([arxiv 2510.06940](https://arxiv.org/abs/2510.06940), Oct 2025): like this project it fixes the proven TGN-can't-express-persistence gap (the paper's Thm 1) via *state design*, but with a **global/virtual SSM state** — whereas our lever is a **per-(user,item) dense matrix** (per-pair EMA = per-pair SSM), feasible only because `N_item ≪ N_user`. New methods must benchmark against **both NAViS and TGNv2**. Dense-memory update-rule theory (prefer delta-rule / erase-then-add over pure additive EMA; GLA-style per-row decay) is in `adjacent_fields_dense_memory.md`.

### Current direction (what the contribution is now)

**Cold-start turned out to be a dead end.** The P4 diagnostic (`notebooks/p4_coldstart.ipynb`) showed warmth-based "cold" users are <1% of genre/reddit predictions (fixing them cannot move overall NDCG), popularity-backoff is useless on every dataset, and the real M-only headroom is *ranking within the row support*, not coverage. The direction pivoted to using the dense per-(user,item) memory **`M` *inside* the learned TGNv2 model** (one head at inference ⇒ not an ensemble), two ideas:

1. **`M` as the neighbour sampler** — replaces the recency `LastNeighborLoader`; for each query user the GNN is fed its **top-`k_m` items by decayed affinity `M[u,:]`** instead of its most-recent partners (`models/msampler.py`). ✅ **It works:** beats the recency-TGNv2 baseline and the 0.469 benchmark on genre. **Optimum at `--k_m 10`** (matches NDCG@10). The `--m_edge_feature` variant (V2 — also feed per-edge `M` affinity into the attention) gave **no gain over plain sampling (V1)**. Pre-experiment analysis: `notebooks/p4_m_as_sampler.ipynb`; learned A/B lives in the `m-sampler` MLflow experiment. See "M-as-neighbour-sampler" under Commands.
2. **Distillation on `M`-scores** (`--distill`) — a KD aux loss on the ~98% target-less batches (no label that day ⇒ no gradient today): the single head is distilled toward `row-normalize(M[u,:])` (soft target, **not** `softmax(M/T)`) for the warm user nodes. Pre-experiment analysis `notebooks/p4_m_distill_head.ipynb` cleared all gates (GO). **Implemented**, two variants: V1 = with the M-sampler (teacher reuses the selection `M`), V2 = with the default recency sampler (a separate teacher `M`). Run via `scripts/run_distill_genre.sh`; learned A/B result pending. `M` stays a train-only teacher (single head at inference ⇒ not an ensemble).

Both are single-model uses of `M` (selector / training teacher; `M` never appears at the inference output). **Index fact** (genre/reddit/token/trade): item node-ids occupy `[0, num_classes)` and users `[num_classes, num_nodes)` — so an `M` column index *is* the item node id, and `M[u,:]` ranks neighbour nodes directly (no remap).

## Commands

Run everything inside the **`tgb` conda environment** (`conda activate tgb`). All commands below assume that environment is active.

```bash
# Install (torch wheels are pinned to CUDA 11.8 — +cu118; adjust for CPU/other CUDA)
# The importable `tgb` package comes from `py-tgb==0.9.2` in requirements.txt — the `TGB/`
# entry in .gitignore is only an optional local clone of the source repo, not a build dependency.
pip install -r requirements.txt

# Quick smoke test — NOT the paper config (see "TGNv2 reference hyperparameters" below).
# --experiment names the MLflow research thread this run belongs to
python train-tgbn-nodeproppred.py --dataset tgbn-trade --epochs 5 --batch_size 200 --lr 1e-3 --experiment baselines

# Train classic TGN instead of TGNv2 (--use_tgnv2 defaults ON; this disables the index-encoding message module)
python train-tgbn-nodeproppred.py --dataset tgbn-trade --no-use_tgnv2 --experiment baselines

# Run the moving-average / persistent-forecast baseline (edit the `name` var in the file to pick the dataset)
python persistent_forecast_messages.py

# Browse runs (local MLflow UI backed by the SQLite store)
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

There is no test suite, linter, or build step — this is a research training script.

All flags live in `utils/args.py`. Key ones: `--dataset` (`tgbn-trade|tgbn-genre|tgbn-reddit|tgbn-token`), `--experiment` (MLflow research thread), `--global_hidden_dims` (one dim shared by memory/embedding/time/idx encoders), `--num_last_neighbours`, `--learning_scheduler` (`constant|cosine_annealing|step_lr`) with sub-flags `--step_lr_gamma`/`--step_lr_step_size` (for `step_lr`) and `--cosine_annealing_ratio` (for `cosine_annealing`), `--dump_eval_preds` (off by default; dumps per-(user,day) test NDCG@10 of the best-val epoch to `mlruns/_eval_preds_<dataset>.parquet` for error/headroom analysis), `--seed`.

**M-as-sampler flags** (the current contribution; see "Current direction"): `--neighbor_sampler` (`recency` [default, `LastNeighborLoader`] | `memory` [`MSampler`, dense affinity]), `--k_m` (neighbour budget for the memory sampler — top-`k_m` items by `M[u,:]`; `-1` = use `--num_last_neighbours`), `--kappa_select` (selection-decay **half-life in days** for the affinity, default `25`; small → recency-like, large/∞ → lifetime affinity), `--m_edge_feature` (V2: also feed per-edge `M` affinity into the GNN attention as a 1-dim **normalized** edge column; needs `--neighbor_sampler memory`), `--m_message_feature` (feed the affinity `M[u,i]` as a **first-class message-like feature** — the raw decayed score rides as one extra column *inside* `raw_msg`, so it flows through `EncodeIndexModule` → the memory GRU update **and** the GNN `edge_attr` exactly like the message, widening `msg_feat_dim` by 1; score source: the memory sampler's own `M`, or — under the **recency** sampler — a lockstep score-only **M-carrier** (`MSampler` at `--kappa_select`, inserted in `process_edges`/reset per epoch; GNN-channel score recovered via the sampled edges' global `(src,dst)[e_id]`), so the feature is REAL under both samplers (`scripts/run_mmsg_recency_genre.sh` = the "M as feature, recency as selector" A/B — isolates M-as-FEATURE from M-as-SELECTOR); default off ⇒ byte-identical, own checkpoint slug `_mmsg`. Implemented purely as a widened `raw_msg` column, so `msgmodule.py`/`mtgn.py`/`embmodule.py` are untouched — only the dim knob + score concat at the two consumption points [`process_edges` for memory, `_gnn_msg` for the GNN]), `--m_msg_transform` (`raw`|`log1p`, squash for that column in BOTH channels; **diagnosis 2026-07-10, `notebooks/p4_mmsg_forward_diag.ipynb`**: the RAW column is O(100–2500) next to O(1) features and non-stationary (+89% p95 train→test since M resets per train epoch but grows through val/test), the trained net does NOT shrink its weight ⇒ **GRU-gate saturation 9.5% vs 0.8%**, one column = 24.5% of the pre-activation norm ⇒ raw HURTS genre (equal-epoch val 0.4887 < 0.4917) while HELPING trade (there magnitude≈label) — so use **`log1p` on genre**, keep `raw` on trade; slug `_mmsglog`. Also `ecdf` (frozen train-ECDF → Uniform[0,1], stationary; one cheap pre-pass fits the reference) and **`ecdf_emb`** (quantize the ECDF into `--m_msg_bins` equal-frequency categories [bin 0 = empty/cold] and encode each via a learned `torch.nn.Embedding` of dim `--m_msg_emb_dim`, in BOTH channels — targets token's **bimodal + zero-inflated** column [measured: 15.5% near-zero on predicted users, Sarle 0.59–0.69], giving the "no history" case its own vector instead of a gate-saturating number). NB `ecdf_emb` DOES touch `models/msgmodule.py` + `models/embmodule.py` — each grows an `nn.Embedding` that expands the trailing bin-index column (the bin index, not the vector, is stored in the message store; the Embedding runs in `EncodeIndexModule`/GNN with current weights — a stored vector would carry a stale cross-iteration graph). Genre A/B `scripts/run_mmsg_emb_genre.sh`; slug `_mmsgemb{bins}x{dim}`), `--use_gnn`/`--no-use_gnn` (ablate the GNN = memory+MLP only; default on — note this was previously hard-coded `True`), `--resume`/`--no-resume` (auto-resume from a per-epoch checkpoint `saved_models/<run_name>.pt`; default on — see "M-as-neighbour-sampler" below).

**Fast-iteration flags**: `--train_tail_frac` (default `1.0`) — train each epoch only on the LAST fraction of the train split; the TGB label pointer is fast-forwarded past the skipped days automatically, and the neighbor-loader/teacher edge counters are **rebased to the tail's global offset** so `e_id == global data index` still holds (`data.t[e_id]`/`data.msg[e_id]` stay correct in train AND the continuing val/test streams — bug found+fixed 2026-07-09; memory/loader are rebuilt each epoch anyway, so this only shortens warmup + supervised-day count). `--eval_every N` (default `1`) — run the heavy val+test streams only every Nth epoch (the final epoch is always evaluated; best-epoch selection sees only evaluated epochs; `test_ndcgs` is an epoch-keyed dict in the checkpoint — old list-format checkpoints are auto-converted on resume). `--val_frac` (default `1.0`) — evaluate on only the FIRST fraction of val (the head continues the train stream ⇒ causal/contiguous); **requires `--no-eval_test` when <1** (the skipped val edges would break the `e_id == global index` invariant for the test stream — asserted at startup). `--eval_test`/`--no-eval_test` — skip the test stream entirely (best-epoch is by val anyway; `best_test_ndcg` then absent from the run summary). Cheapest-possible setup (e.g. token: `scripts/run_token_iter.sh` = `ttf 0.05` + `val_frac 0.05` + `no-eval_test` + `d=128`). Use for quick A/Bs on heavy datasets (reddit: `ttf 0.3` + `eval_every 2` ≈ 4-5× faster iteration, `scripts/run_reddit_iter.sh`); numbers are relative-only, NOT paper-comparable. Note: the rank/ECDF φ and the `ttf` slug/checkpoint then derive from the sliced tail.

**Distillation flags** (idea 2; off by default): `--distill` (master switch — KD aux loss on target-less batches; teacher = the memory sampler's `M` [variant 1] or a separate `MSampler` under the recency sampler [variant 2]), `--distill_stride` (run KD on every Nth target-less batch — cost knob; default `50` ≈ supervised count; **do NOT use `1`** — see below), `--distill_lambda` (KD weight, default `1.0`), `--kappa_teacher` (teacher decay half-life in days; `-1` = inherit `--kappa_select`; only decouples in variant 2), `--kd_grad` (`full`|`gnn_head`|`head`, default `full`) — how far the KD gradient flows: `head` detaches the embedding so KD trains **only `node_pred`**, `gnn_head` detaches the memory output (KD trains GNN+head), protecting the sequential TGN memory that eval reads.

**Distillation diagnosis (2026-07-01, first runs underperformed at ~0.49 vs teacher ~0.52).** Root causes were NOT a weak teacher (a κ sweep confirmed the in-harness teacher is 0.504 @κ=0 → **0.522 @κ=40** → 0.520 @κ=100 — strong; κ=0 is the *worst*, so any "use κ=0" advice is wrong). The real issues: (1) the KD `optimizer.step()` fires ~55× more than CE at `--distill_stride 1`, starving CE and polluting Adam → **keep stride ≥ ~50** (KD steps ≤ CE steps; `stride 1 test 0.453 < stride 50 0.487`); (2) full-pipeline KD backprops through the TGN memory on ~98% of batches and destabilizes the state eval reads → use **`--kd_grad head`** (or `gnn_head`); (3) the d=784 runs were reaped at 2-4 epochs (undertrained) → iterate at `d=128` (`scripts/run_distill_iter.sh`, the `head`/`gnn_head`/`full` ablation) to converge in-session, then measure at `d=784`. Recommended config: `--distill --distill_stride 50 --distill_lambda 1.0 --kd_grad head --kappa_select 40`.

### TGNv2 reference hyperparameters (paper Table 2 — the config for *measuring* quality)

These are the **canonical hyperparameters from the TGNv2 paper** ([arxiv 2411.03596](https://arxiv.org/abs/2411.03596), Appendix C / Table 2). Use them whenever a run is meant to produce a **quality number comparable to the paper**. They are deliberately **NOT** the config for fast hypothesis testing — for quick iteration use a lighter setup (see "Measurement vs hypothesis testing" below) and spend the full paper config only on measurement runs.

The paper uses the **same hyperparameters for TGN and TGNv2** (its "TGN (tuned)" baseline), toggled by `--use_tgnv2` / `--no-use_tgnv2`. The architecture is already fixed in code to match the paper: GRU memory updater, `LastAggregator` (last message in a batch), one `TransformerConv` embedding with **2 heads + dropout 0.1** (`models/embmodule.py:23`), 2-layer MLP+ReLU decoder (`models/decoder.py`), `cos(Linear(·))` time/index encoders. Optimizer is **Adam**. The single `--global_hidden_dims` is the paper's shared `d` (`= d_memory = d_embedding = d_decoder = d_time = d_node`).

⚠️ The `utils/args.py` **defaults do not match the paper** (`global_hidden_dims=512`, `epochs=50`, `num_last_neighbours=10`) — you must pass the per-dataset flags explicitly.

| flag | tgbn-trade | tgbn-genre | tgbn-reddit | tgbn-token |
|---|---|---|---|---|
| `--lr` | 1e-3 | 1e-4 | 1e-4 | 1e-4 |
| `--batch_size` | 200 | 200 | 200 | 200 |
| `--epochs` | 750 | 50 | 50 | 50 |
| `--global_hidden_dims` (d) | 784 | 784 | 784 | 1024 |
| `--num_last_neighbours` (x) | 25 | 30 | 30 | 10 |
| `--learning_scheduler` | `step_lr` (γ=0.5 every 250 ep) | `constant` | `constant` | `constant` |

```bash
# tgbn-trade — LR decays ×0.5 every 250 epochs
python train-tgbn-nodeproppred.py --dataset tgbn-trade \
  --lr 1e-3 --batch_size 200 --epochs 750 --global_hidden_dims 784 --num_last_neighbours 25 \
  --learning_scheduler step_lr --step_lr_gamma 0.5 --step_lr_step_size 250 \
  --experiment baselines --seed 1

# tgbn-genre / tgbn-reddit — identical flags
python train-tgbn-nodeproppred.py --dataset tgbn-genre  --lr 1e-4 --batch_size 200 --epochs 50 \
  --global_hidden_dims 784  --num_last_neighbours 30 --learning_scheduler constant --experiment baselines --seed 1
python train-tgbn-nodeproppred.py --dataset tgbn-reddit --lr 1e-4 --batch_size 200 --epochs 50 \
  --global_hidden_dims 784  --num_last_neighbours 30 --learning_scheduler constant --experiment baselines --seed 1

# tgbn-token — larger d, fewer neighbours
python train-tgbn-nodeproppred.py --dataset tgbn-token  --lr 1e-4 --batch_size 200 --epochs 50 \
  --global_hidden_dims 1024 --num_last_neighbours 10 --learning_scheduler constant --experiment baselines --seed 1

# Paper's "TGN (tuned)" baseline = any command above + --no-use_tgnv2
```

**Protocol (paper):** repeat each config over **3 seeds**, pick the best epoch on validation, report **mean ± std NDCG@10** on val and test (NDCG `k=10` is hardcoded in TGB's `nodeproppred/evaluate.py`).

**Reference test NDCG@10 — the bar to reproduce / beat** (paper Table 1):

| | trade | genre | reddit | token |
|---|---|---|---|---|
| **TGNv2** (this model) | **0.735** | **0.469** | **0.507** | **0.294** |
| TGN (tuned) | 0.374 | 0.367 | 0.315 | 0.169 |
| Moving Avg over messages (M) | 0.777 | 0.472 | 0.411 | 0.415 |
| Moving Avg over labels (L) | 0.823 | 0.509 | 0.559 | 0.508 |

The (M)/(L) heuristics consume ground-truth messages/labels and set the absolute ceiling; **TGNv2 is the strongest *learned* model** and is the baseline the new method must beat.

**Measurement vs hypothesis testing.** All four Table-2 configs above — **including `tgbn-trade`** — are *measurement* configs: run them (paper protocol, 3 seeds) only to produce a quality number comparable to the paper. Do **not** use them for day-to-day hypothesis testing.

**Primary hypothesis-testing dataset: `tgbn-genre`.** It is the smallest dataset that actually exhibits the item–user asymmetry the new method exploits (`N_user 974` ≈ 2× `N_item 513`). `tgbn-trade` is symmetric (≈254×254), so it does *not* exercise the dense `N_user × N_item` lever and is reserved for paper-quality measurement; reddit/token are too heavy for fast CPU iteration (and at full `d` may be impractical to reproduce on CPU at all). Iterate on genre with a lighter config — smaller `d`, far fewer epochs — e.g. a starting point:

```bash
python train-tgbn-nodeproppred.py --dataset tgbn-genre \
  --lr 1e-4 --batch_size 200 --epochs 10 --global_hidden_dims 128 --num_last_neighbours 30 \
  --learning_scheduler constant --experiment user-item-matrix --seed 1
```

Tune `--epochs` / `--global_hidden_dims` to your CPU budget; switch to the full Table-2 genre config (and the 3-seed paper protocol) only when reporting a number.

### M-as-neighbour-sampler runs (the current contribution)

The whole genre sweep is wrapped in **`scripts/run_msampler_genre.sh`** (all shell run-scripts live in `scripts/`): the neighbour-budget sweep `k_m ∈ {10,5,20} × {V1, V2}` on genre at paper width (`d=784`, 10 epochs, experiment `m-sampler`). Edit the `EPOCHS`/`SEED` vars at the top — for the **paper number use `EPOCHS=50` and seeds 1,2,3**. A single config is just:

```bash
# V1 — M as the neighbour sampler (optimum k_m=10)
python train-tgbn-nodeproppred.py --dataset tgbn-genre --epochs 10 --global_hidden_dims 784 \
  --num_last_neighbours 30 --batch_size 200 --lr 1e-4 --learning_scheduler constant \
  --experiment m-sampler --seed 1 --neighbor_sampler memory --k_m 10 --kappa_select 25
# V2 — add per-edge M affinity feature (gave no gain over V1)
#   ... same line + --m_edge_feature
# M-only ablation (sanity bar): + --no-use_gnn
```

**Checkpoint/resume (built in).** Each epoch saves `saved_models/<run_name>.pt` (model + optimizer + scheduler + epoch + best-val + the MLflow `run_id`); on relaunch with the **identical** args (`--resume` is on by default) it continues from the last epoch and **re-attaches the same MLflow run**. So an interrupted run is restarted by just re-running the same command; a finished config short-circuits in seconds. `saved_models/` is git-ignored.

⚠️ **Run long jobs in your own terminal, not via a Codex background task** — this environment **reaps background (`run_in_background`) tasks** across context-compaction / idle boundaries, and a `d=784` epoch (~14 min) exceeds the 10-min foreground cap, so these runs cannot complete reliably from inside the harness. Checkpoint/resume makes a user-terminal run robust to any interruption.

### Datasets (already downloaded — do NOT re-download)

The `tgbn-*` datasets are **already on disk**, bundled inside the installed `py-tgb` package:

```
/Users/aleksandrpanysev/miniconda3/envs/tgb/lib/python3.13/site-packages/tgb/datasets/
  ├── tgbn_trade/    tgbn_genre/    tgbn_reddit/    tgbn_token/   (each: <name>_edgelist.csv, <name>_node_labels.csv, <name>.zip)
```

**Path resolution gotcha:** TGB computes `root = PROJ_DIR + root`, where `PROJ_DIR` is the installed `tgb/` package directory. Both scripts pass `root="datasets"`, so it resolves to `<tgb-pkg>/datasets/<name_with_underscores>/` — **not** a `./datasets` folder in the cwd. The files are present, so `dataset.download()` returns early ("files found") and nothing is fetched from the network. Because `root` is always prefixed by `PROJ_DIR`, you **cannot** point it at an arbitrary absolute path — keep `root="datasets"` as-is; that already uses the local copies above.

### Experiment tracking: local MLflow + SQLite

All run tracking goes through a **local MLflow** instance backed by **SQLite** — no wandb, no remote account, no network.

- **Backend store**: `sqlite:///mlruns/mlflow.db`. **Artifact store**: `./mlruns/`. Both are git-ignored.
- Set once at process start: `mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")` → `mlflow.set_experiment(args.experiment)` → wrap the whole train/eval loop in `with mlflow.start_run(run_name=...)` (the context manager marks crashed runs `FAILED` instead of leaving them `RUNNING`).
- Browse results: `mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`.

**Philosophy:** the run *name* is a short human label; the *source of truth* is `params` + `tags`, which is what you filter and group by in the UI. Do **not** encode hyperparameters into the name.

#### Naming rules

- **Experiment = a research thread**, not a dataset or a config. Examples: `baselines`, `user-item-matrix`. Set via `--experiment`. The dataset and model are recorded as params/tags, so one experiment can hold "my method vs baselines across all datasets" — the view you want for the paper.
- **Run name = short, readable**: `{model}-{dataset}-s{seed}` (e.g. `tgnv2-trade-s1`). The full hyperparameter slug (the old `run_name`) goes into the **tag `slug`**, not the name; it also stays as the `logs/tidy/<slug>` text-log filename.

#### Logging rules (do these in every run)

1. **Params**: `mlflow.log_params(vars(args))` once at the start — the full config lives in params, never in the name.
2. **Tags** (`mlflow.set_tags`): `model` (`tgn`/`tgnv2`), `dataset`, `seed`, `slug`, `device` (`cpu`), git commit + dirty flag, hostname, torch/python versions — everything you filter by but don't tune.
3. **Metrics — two tiers.** (a) *Streaming training curve*, one point per **label-timestamp**: keys `train/loss` and `train/<eval_metric>` (e.g. `train/ndcg`), logged with `step` = a monotonic integer `global_step` that `train()` threads through and returns so it spans all epochs. (b) *Per-epoch summaries* (`step=epoch`): `epoch/train_loss`, `epoch/train_<eval_metric>`, `epoch/lr` from `train()`, plus `val/<eval_metric>` / `test/<eval_metric>` from `test()`. `step` is always an **int** (the old fractional `train/epoch` metric is gone).
4. **Summary metrics** at the end of the run: `best_val_ndcg`, `best_test_ndcg`, `best_epoch`. These are the headline numbers for the paper — they must be in MLflow, not only in the text log.
5. **Artifacts**: the run attaches the `logs/tidy/<slug>` text log (`mlflow.log_artifact`), plus — only when `--dump_eval_preds` is set — `mlruns/_eval_preds_<dataset>.parquet` (per-(user,day) test NDCG of the best-val epoch). An args dump / env snapshot / checkpoint are *not* currently written; the full config already lives in `params`.
6. **Seeds**: always log `seed` (param + tag); run each config over multiple seeds and report mean±std aggregated by `slug` (the slug is seed-independent only if you strip the seed — keep a separate `config_id` tag if you need a seed-free grouping key).

The custom `Logger` (text logs under `logs/tidy/`) stays — it is useful offline and is attached to the run as an artifact.

The wandb→MLflow migration history is in `migration_wandb_to_mlflow.md` (explains why git history and some pins still reference wandb).

## Architecture

The pipeline is **TGN memory → GNN embedding → MLP node predictor**, wired together in `main()` of `train-tgbn-nodeproppred.py`.

- **`models/mtgn.py`** — the core. `MTGNMemory` is a modified PyG `TGNMemory`: per-node memory vectors updated by a GRU from aggregated messages, plus a `last_update` timestamp buffer and per-node message stores. Also defines `LastAggregator` (keeps the most recent message per node via `scatter_argmax` over timestamps), `TimeEncoder` (`cos(Linear(t))`), and `LastNeighborLoader` (ever-growing dense `[num_nodes, size]` neighbor table for GNN sampling).
- **`models/msgmodule.py`** — `EncodeIndexModule` is **the TGNv2 addition**. The message = `[z_src, z_dst, raw_msg, src_enc, dst_enc, t_enc]`, where `src_enc`/`dst_enc` come from encoding the node **indices** through a `TimeEncoder` (`idx_enc` in `MTGNMemory`). When `--use_tgnv2` is off, `idx_dim` becomes `-1`, index encoding is skipped, and the message module is PyG's `IdentityMessage` — i.e. plain TGN.
- **`models/embmodule.py`** — `MGraphAttentionEmbedding`: a single `TransformerConv` (2 heads) over sampled neighbors, with edge features `[rel_time_enc, msg]`. With the V2 variant (`--m_edge_feature`) it also accepts an optional per-edge `m_e` affinity concatenated into `edge_attr` (`edge_dim += m_edge_dim`); `m_e=None` (recency sampler) is the no-op default.
- **`models/msampler.py`** — `MSampler`, **the M-as-neighbour-sampler addition** (idea 1). A drop-in for `LastNeighborLoader` returning the same `(n_id, mem_edge_index, e_id)` triple (+ a 4th per-edge affinity `m_e` for V2). It maintains a dense per-(user,item) Hawkes affinity `M[N,C]` (rank/ECDF write transform, `kappa_select`-day decay) plus a `last_eid[N,C]` buffer so a selected (u,i) edge can recover its global `data.t`/`data.msg` (the e_id == global-index invariant matches the recency loader). Per query user it emits its `min(#seen, k_m)` highest-affinity items — **variable size, never padding zero-affinity slots**. Parameter-free / non-differentiable: it only selects (and optionally weights via `m_e`) which item memories the single head sees. It also serves as the **distillation teacher** (idea 2): `soft_target(users)` returns the row-normalized decayed `M[users]` (+ a warm mask) as the KD soft label.
- **`models/decoder.py`** — `NodePredictor`: 2-layer MLP → class logits (loss is `CrossEntropyLoss`; no softmax in the model).
- **`modules/heuristics.py`** — `MovingAverageMessages`, the non-learned baseline (mean of last `k` messages per (src,dst), used by `persistent_forecast_messages.py`).

### The train/test loop (the non-obvious part)

`tgbn-*` labels are emitted **per "day"** (label timestamp), not per edge. The loop in `train()`/`test()` (`train-tgbn-nodeproppred.py`) handles this:

1. Iterate temporal edge batches. When a batch's last timestamp crosses the current `label_t` (`query_t > label_t`), a label boundary was hit.
2. Fetch that day's node labels via `dataset.get_node_label(query_t)`.
3. Split the batch with `previous_day_mask`: edges *before* the boundary are written into memory/neighbor-loader now; edges *after* are held and processed at the end of the iteration so they're not leaked into the prediction.
4. For the labeled source nodes: sample neighbors (`LastNeighborLoader`), read their memory, run the GNN, map back to query nodes via the `assoc` index table, predict, compute loss + the dataset's `eval_metric` (NDCG).
5. After predicting, `process_edges` updates memory/neighbors with the ground-truth edges, then `memory.detach()` truncates backprop-through-time across batches.

Other subtleties:
- `MTGNMemory.forward` returns *freshly recomputed* memory during training but the *stored* memory during eval; `update_state` also reorders memory-update vs message-store-write between train and eval. `train(mode=False)` flushes the message store into memory.
- `assoc` (and the loader's internal `_assoc`) are scratch tensors that relabel sparse global node ids to dense local indices for each GNN call — they hold no state between calls.
- Best test score is selected by the epoch with the best **validation** NDCG (`best_test_idx`); all epochs still run. `dataset.reset_label_time()` is called at the end of every epoch.
- `train()`/`test()` both take an `epoch` arg (to tag MLflow metrics); `train()` also threads and returns the monotonic `global_step`. With `--dump_eval_preds`, `test(..., collect=True)` returns `(metric_dict, recs)` and records one `(query_t, user_id, NDCG@10)` row per labeled user with ≥1 positive label (via sklearn `ndcg_score(..., k=10)`); only the **best-val epoch's** `recs` are written to `mlruns/_eval_preds_<dataset>.parquet` (schema `[ts, user, ndcg]`) and logged as an artifact. Without the flag, `recs` stays empty.
- The `neighbor_loader` is swappable: `LastNeighborLoader` (recency) or `MSampler` (affinity), chosen by `--neighbor_sampler`. `process_edges(...)` updates the TGN memory **and** the sampler in lockstep on the strict-causal previous-day edges (`MSampler.insert` takes `t,msg` to build `M`/`last_eid`; `LastNeighborLoader.insert` ignores them). The sampler call returns a 4-tuple for `MSampler` (with `m_e`) and a 3-tuple for the recency loader — the call sites in `train()`/`test()` handle both. `use_gnn` is read from `--use_gnn` (no longer hard-coded).
- **Checkpoint/resume**: `main()` saves `saved_models/<run_name>.pt` after every epoch and auto-resumes from it on relaunch (continuing the loop and re-attaching the same MLflow run via the stored `run_id`); `--no-resume` forces a fresh run. The per-node TGN memory and the neighbour loader are reset+rebuilt each epoch anyway, so the checkpoint only carries learnable weights + optimizer/scheduler + epoch + best-val tracking.
- **Distillation step (`--distill`)**: the `if query_t > label_t:` (supervised) branch now has an `else:` for target-less batches — it reads the teacher soft target `teacher.soft_target(users)` and runs the shared `_student_logits(...)` forward on the warm user nodes (`src >= num_classes`) **before** `process_edges` writes the batch (read-before-write causality), then `kd_loss = λ · −Σ soft·log_softmax(student)` (CE form, NaN-safe on the sparse teacher), its own `backward()` + `optimizer.step()` (the supervised step never runs on these batches), all before `memory.detach()`. Both the supervised and KD paths call the single `_student_logits` helper so they cannot drift. `process_edges(..., teacher)` updates the separate variant-2 teacher in lockstep (no-op when teacher *is* the sampler). `test()` is untouched — `M` never enters inference. KD loss is logged as `train/kd_loss` / `epoch/train_kd_loss`.

## Jupyter notebooks

**Drive notebooks through the Jupyter MCP** (`jupyter-mcp-server`) — execute cells against a live kernel, don't just hand-edit `.ipynb` JSON. The kernel must be the **`tgb` conda env** (so torch/tgb/mlflow/polars/plotly are importable).

**Hard ban: never build or edit notebooks out-of-band.** No `nbformat`, no writing/patching `.ipynb` JSON by hand, no Python scripts that emit notebooks. Create a notebook with `use_notebook(mode="create")` and populate it cell-by-cell via the MCP (`insert_cell` for markdown, `insert_execute_code_cell` for code, `overwrite_cell_source` to revise). The only acceptable reason to touch raw JSON is if the MCP server is genuinely unavailable — say so first.

Setup in this repo (already provisioned):

- A tgb-env Jupyter Lab server is rooted at the repo: `http://localhost:8890`, token `tgn2q2026`. Start it (if not running) with:
  ```bash
  /Users/aleksandrpanysev/miniconda3/envs/tgb/bin/jupyter-lab --no-browser --port 8890 \
    --IdentityProvider.token tgn2q2026 \
    --ServerApp.root_dir=/Users/aleksandrpanysev/Documents/GitHub/2Q_2026_tgn_user_item
  ```
  Check with `jupyter server list`.
- The MCP is wired via project-local **`.mcp.json`** (git-ignored, holds the token), pointing `DOCUMENT_URL`/`RUNTIME_URL` at `:8890` and `DOCUMENT_ID` at `experiments/analysis.ipynb`. Codex loads `.mcp.json` **on connect** — after creating/changing it you must reconnect the session for the `jupyter` MCP tools to appear (otherwise only `mcp__ide__executeCode`, which needs a notebook open in the IDE, and the built-in `NotebookEdit` are available).
- **Notebooks live in `notebooks/`** — one per research phase (see `framework.md`), named `p{phase}_{slug}.ipynb` so they sort in research order and map 1:1 to the framework phases (on disk: `p1_eda_user_item.ipynb`, `p2_error_analysis.ipynb`, `p2b_cold_user_headroom.ipynb`, `p3_perpair_memory.ipynb`, `p3_user_item_matrix.ipynb`, `p3_write_rule_probe.ipynb`, `p4_coldstart.ipynb` [cold-start dead-end diagnostic], `p4_m_as_sampler.ipynb` [idea 1 pre-experiment analysis — GO, k_m≈10], `p4_m_distill_head.ipynb` [idea 2 pre-experiment analysis — GO], plus a non-conforming `scratch.ipynb` for ephemeral work; P0 baseline reproduction lives in MLflow — there is no `p0_baselines.ipynb`). See `notebooks/README.md`. Drive any of them via `use_notebook(notebook_path="notebooks/<file>.ipynb")` — `DOCUMENT_ID` in `.mcp.json` is only the default landing doc, not a limit. (`experiments/analysis.ipynb` is the old scratch notebook, superseded by `notebooks/`.)

Content conventions for every notebook:

- **Visualization: use Plotly** (`plotly.express` / `plotly.graph_objects`), not matplotlib/seaborn.
- **DataFrames: use Polars** (`import polars as pl`), not pandas. Convert at the boundary only if a library hard-requires pandas (e.g. `.to_pandas()` for Plotly Express).
- **Language: write in Russian** — markdown cells, prose, and code comments should be in Russian.
- **Read and analyse every cell's output as you go.** After executing a cell, actually inspect the result before adding the next one — sanity-check the numbers and plots. If something looks off (suspiciously large/small, a value that contradicts another cell, an artifact of a wrong baseline), stop and fix that cell (`overwrite_cell_source` + re-run) instead of stacking more cells on a shaky result. A headline number that gates a research decision must be double-checked, not taken at face value.

## Conventions

- One shared `--global_hidden_dims` sets `memory_dim`, `embedding_dim`, `time_dim`, and `idx_dim` together — don't assume they can diverge without code changes.
- `device` is module-global (`cuda` if available else `cpu`); tensors are moved per-batch.
- `.gitignore` excludes the `TGB` clone, the MLflow store (`mlruns/`, `mlflow.db`), the Jupyter MCP config `.mcp.json` (holds the local token), `experiments/.ipynb_checkpoints/`, `wandb/`, `saved_models/`, `saved_results/`, `slurm-log/`, `*.job`, and all `__pycache__`/`*.pyc`. All of `logs/` is git-ignored (run logs go to `logs/tidy/<run_name>`, whose `run_name` encodes every hyperparameter). `datasets/` is not ignored but is auto-populated.
