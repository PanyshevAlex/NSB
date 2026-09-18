import timeit
import torch
import mlflow
import os
import math
import platform
import socket
import subprocess
import utils

from tqdm import tqdm
from models.mtgn import MTGNMemory, LastAggregator, LastNeighborLoader
from models.msampler import MSampler
from models.embmodule import MGraphAttentionEmbedding
from models.msgmodule import EncodeIndexModule
from models.decoder import NodePredictor
from modules.labels import LabelAggregator
from logger.logger import Logger

from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
)

from tgb.nodeproppred.dataset_pyg import PyGNodePropPredDataset, TemporalData
from tgb.nodeproppred.evaluate import Evaluator
from tgb.utils.utils import set_random_seed

from torch.optim.lr_scheduler import LRScheduler

import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Starting Point:
# https://github.com/shenyangHuang/TGB/blob/main/examples/nodeproppred/tgbn-genre/tgn.py

def _git_commit_and_dirty():
    """Best-effort git commit SHA and dirty flag, logged as MLflow tags for provenance."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        ) != 0
        return sha, dirty
    except Exception:
        return None, None

SCORE_REF = None   # frozen train-ECDF reference for --m_msg_transform ecdf/ecdf_emb (sorted scores; set in main)
SCORE_BINS = 16    # number of ECDF categories for ecdf_emb (set from --m_msg_bins in main)


def _score_tf(sc, use_msg_score):
    """Transform of the --m_message_feature score column (same in BOTH channels). For 'ecdf_emb'
    it returns the integer BIN INDEX (as float); the learned Embedding is applied downstream in the
    message module / GNN (must run there with current weights — a stored embedding vector would carry
    a stale cross-iteration graph through the TGN message store)."""
    if use_msg_score == "log1p":
        return torch.log1p(sc)
    if use_msg_score in ("ecdf", "ecdf_emb"):  # -> Uniform[0,1] vs the frozen train reference (stationary)
        u = torch.searchsorted(SCORE_REF, sc.contiguous(), right=True).to(sc.dtype) / SCORE_REF.numel()
        if use_msg_score == "ecdf_emb":         # -> equal-frequency bin index in [0, SCORE_BINS-1]
            return (u * SCORE_BINS).floor().clamp_(0, SCORE_BINS - 1)
        return u
    return sc


def process_edges(memory, src, dst, t, msg, neighbor_loader, teacher=None, use_msg_score=False,
                  score_src=None):
    if src.nelement() > 0:
        mem_msg = msg
        if use_msg_score:  # --m_message_feature: append the affinity as an extra raw_msg column
            # Score source: the memory sampler's own M, else the lockstep M-carrier (recency sampler).
            msrc = neighbor_loader if hasattr(neighbor_loader, "score_for") else score_src
            if msrc is not None:  # read BEFORE insert -> pre-edge affinity (causal)
                sc = msrc.score_for(src, dst, t).view(-1, 1)
            else:  # no M anywhere -> neutral 0 column (keeps dims aligned)
                sc = torch.zeros(src.numel(), 1, dtype=torch.float32, device=device)
            sc = _score_tf(sc, use_msg_score)  # log1p (scale) / ecdf (scale + stationarity)
            mem_msg = torch.cat([msg.to(torch.float32), sc], dim=-1)
        memory.update_state(src, dst, t, mem_msg)
        # t/msg are ignored by LastNeighborLoader, used by MSampler to build M / last_eid.
        # insert() gets the ORIGINAL msg (M is built from msg[:,0] only; keep it un-widened).
        neighbor_loader.insert(src, dst, t, msg)
        # Lockstep M-carrier (recency + --m_message_feature): score-only MSampler, kept in sync.
        if score_src is not None and score_src is not neighbor_loader and score_src is not teacher:
            score_src.insert(src, dst, t, msg)
        # Distillation teacher: a SEPARATE MSampler under the recency sampler (variant 2) kept in
        # lockstep on the same past edges. No-op when teacher IS the memory sampler (variant 1).
        if teacher is not None and teacher is not neighbor_loader:
            teacher.insert(src, dst, t, msg)


def _gnn_msg(e_id, m_e_raw, data, use_msg_score, score_src=None):
    """The message tensor fed to the GNN edge_attr. With --m_message_feature the raw sampled-edge
    affinity rides as an extra column INSIDE msg (aligned 1:1 with e_id), exactly like raw_msg.
    Memory sampler: m_e_raw comes from MSampler.__call__. Recency sampler: the score is read from
    the lockstep M-carrier via the sampled edges' GLOBAL endpoints (e_id == global index invariant),
    decayed to read-time t_now — the same convention as m_e_raw."""
    gmsg = data.msg[e_id].to(device)
    if use_msg_score:
        if m_e_raw is not None:
            sc = m_e_raw.view(-1, 1).to(gmsg.dtype)
        elif score_src is not None:  # recency sampler + M-carrier
            sc = score_src.score_for(data.src[e_id], data.dst[e_id]).view(-1, 1).to(gmsg.dtype)
        else:  # no M anywhere -> neutral 0 column (keeps edge_dim aligned)
            sc = torch.zeros(gmsg.size(0), 1, dtype=gmsg.dtype, device=device)
        sc = _score_tf(sc, use_msg_score)  # same transform as the memory channel (one convention)
        gmsg = torch.cat([gmsg, sc], dim=-1)
    return gmsg


def _student_logits(n_id, memory, gnn, node_pred, neighbor_loader, data, assoc, use_gnn,
                    kd_grad="full", use_msg_score=False, score_src=None):
    """Shared student forward (sample neighbours -> memory -> GNN -> head). Called by BOTH the
    supervised path (kd_grad='full') and the KD distillation path so the two cannot drift (esp. m_e).

    kd_grad controls how far the KD gradient flows: 'full' = all params; 'gnn_head' = detach the
    memory output (KD trains GNN+head only, TGN memory frozen); 'head' = detach the embedding
    (KD trains only node_pred). 'head'/'gnn_head' protect the sequential memory that eval reads."""
    sample = neighbor_loader(n_id)
    n_id_neighbors, mem_edge_index, e_id = sample[0], sample[1], sample[2]
    m_e = sample[3] if len(sample) > 3 else None       # per-edge affinity, normalized (V2)
    m_e_raw = sample[4] if len(sample) > 4 else None   # per-edge affinity, raw (message-like)
    assoc[n_id_neighbors] = torch.arange(n_id_neighbors.size(0), device=device)
    z, last_update = memory(n_id_neighbors)
    if kd_grad == "gnn_head":
        z = z.detach()                      # freeze TGN memory for KD
    if use_gnn:
        z = gnn(z, last_update, mem_edge_index,
                data.t[e_id].to(device), _gnn_msg(e_id, m_e_raw, data, use_msg_score, score_src), m_e=m_e)
    z = z[assoc[n_id]]
    if kd_grad == "head":
        z = z.detach()                      # freeze memory + GNN for KD (head-only)
    return node_pred(z)


def train(
        epoch,
        global_step,
        memory,
        gnn,
        node_pred,
        lr_scheduler: LRScheduler,
        dataset: PyGNodePropPredDataset,
        data: TemporalData,
        evaluator: Evaluator,
        neighbor_loader,
        train_loader,
        optimizer,
        assoc,
        use_gnn=True,
        teacher=None,
        distill_lambda=0.0,
        distill_stride=50,
        kd_grad="full",
        train_start_t=None,
        use_msg_score=False,
        e_id_offset=0,
        score_src=None):
    eval_metric = dataset.eval_metric

    memory.train()
    gnn.train()
    node_pred.train()

    criterion = torch.nn.CrossEntropyLoss()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.
    # --train_tail_frac < 1 streams only the train tail, but e_id must stay == GLOBAL data index
    # (data.t[e_id]/data.msg[e_id] index the full stream) -> rebase the edge counters to the
    # tail's global start; the counter then reaches n_train exactly at the val boundary.
    neighbor_loader.cur_e_id = e_id_offset
    if teacher is not None and teacher is not neighbor_loader:
        teacher.reset_state()  # separate distillation teacher (variant 2): fresh M each epoch
        teacher.cur_e_id = e_id_offset
    if score_src is not None and score_src is not neighbor_loader and score_src is not teacher:
        score_src.reset_state()  # recency M-carrier (--m_message_feature): fresh M each epoch
        score_src.cur_e_id = e_id_offset

    total_loss = 0
    total_kd_loss = 0  # distillation (idea 2)
    num_kd = 0
    tl_idx = 0  # counter over target-less batches (for --distill_stride)
    distill_stride = max(int(distill_stride), 1)  # guard: avoid modulo-by-zero
    label_t = dataset.get_label_time()  # check when does the first label start

    # --train_tail_frac < 1: the stream starts mid-train, so fast-forward the TGB label
    # pointer past the skipped days (otherwise early days get scored on an empty memory).
    if train_start_t is not None:
        skipped = 0
        while label_t < train_start_t:
            if dataset.get_node_label(label_t) is None:
                break
            label_t = dataset.get_label_time()
            skipped += 1
        if skipped:
            print(f"[train-tail] skipped {skipped} label days before tail start")
    num_label_ts = 0
    total_score = 0

    train_loader_length = len(train_loader)
    count = 0
    for batch in tqdm(train_loader):
        count += 1
        batch = batch.to(device)
        optimizer.zero_grad()
        src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        query_t = batch.t[-1]
        # check if this batch moves to the next day
        if query_t > label_t:
            
            # find the node labels from the past day
            label_tuple = dataset.get_node_label(query_t)
            label_ts, label_srcs, labels = (
                label_tuple[0],
                label_tuple[1],
                label_tuple[2],
            )
            
            label_aggregator.update(label_srcs.to(device), labels.to(device))
            
            label_t = dataset.get_label_time()
            label_srcs = label_srcs.to(device)

            # Process all edges that are still in the past day
            previous_day_mask = batch.t < label_t
            process_edges(
                memory,
                src[previous_day_mask],
                dst[previous_day_mask],
                t[previous_day_mask],
                msg[previous_day_mask],
                neighbor_loader,
                teacher,
                use_msg_score=use_msg_score,
                score_src=score_src,
            )

            # Reset edges to be the edges from tomorrow so they can be used later
            src, dst, t, msg = (
                src[~previous_day_mask],
                dst[~previous_day_mask],
                t[~previous_day_mask],
                msg[~previous_day_mask],
            )

            # Supervised student forward (shared helper with the KD distillation path below).
            pred = _student_logits(label_srcs, memory, gnn, node_pred, neighbor_loader, data, assoc, use_gnn,
                                   use_msg_score=use_msg_score, score_src=score_src)

            loss = criterion(pred, labels.to(device))
            # np_pred = pred.cpu().detach().numpy()
            # np_true = labels.cpu().detach().numpy()

            # input_dict = {
            #     "y_true": np_true,
            #     "y_pred": np_pred,
            #     "eval_metric": [eval_metric],
            # }
            # result_dict = evaluator.eval(input_dict)
            # score = result_dict[eval_metric]
            # total_score += score
            num_label_ts += 1

            loss.backward()
            optimizer.step()
            total_loss += float(loss)

            # Streaming training curve: one point per label-timestamp,
            # indexed by a monotonic global step that spans all epochs.
            mlflow.log_metrics(
                {
                    "train/loss": total_loss / num_label_ts,
                    f"train/{eval_metric}": total_score / num_label_ts,
                    "train/kd_loss": total_kd_loss / max(num_kd, 1),
                },
                step=global_step,
            )
            global_step += 1

        else:
            # Target-less batch (no label today): distill the single head on M soft-targets (idea 2).
            # Reads teacher M + student memory on PAST edges only — before this batch is written below.
            tl_idx += 1
            if teacher is not None and distill_lambda > 0 and (tl_idx % distill_stride == 0):
                kd_users = torch.unique(src[src >= teacher.C])  # user nodes (items are ids < num_classes)
                if kd_users.numel() > 0:
                    soft, warm = teacher.soft_target(kd_users)
                    kd_users, soft = kd_users[warm], soft[warm]  # drop cold (all-zero) rows
                    if kd_users.numel() > 0:
                        student_logits = _student_logits(kd_users, memory, gnn, node_pred,
                                                         neighbor_loader, data, assoc, use_gnn, kd_grad=kd_grad,
                                                         use_msg_score=use_msg_score, score_src=score_src)
                        # KL(teacher || student) up to a constant = cross-entropy of the soft target;
                        # the CE form avoids 0*log0 NaNs on the sparse row-normalized teacher.
                        kd_loss = distill_lambda * -(soft * torch.log_softmax(student_logits, dim=1)).sum(dim=1).mean()
                        kd_loss.backward()
                        optimizer.step()  # KD owns its step (the supervised step never runs here)
                        total_kd_loss += float(kd_loss)
                        num_kd += 1

        # Update memory and neighbor loader with ground-truth state.
        process_edges(memory, src, dst, t, msg, neighbor_loader, teacher, use_msg_score=use_msg_score,
                      score_src=score_src)
        memory.detach()

    lr_scheduler.step()
    metrics = {
        "epoch/train_loss": total_loss / num_label_ts,
        f"epoch/train_{eval_metric}": total_score / num_label_ts,
        "epoch/train_kd_loss": total_kd_loss / max(num_kd, 1),
        "epoch/lr": float(lr_scheduler.get_lr()[0]),
    }
    mlflow.log_metrics(metrics, step=epoch)
    return metrics, global_step


@torch.no_grad()
def test(
        memory,
        gnn,
        node_pred,
        epoch,
        dataset: PyGNodePropPredDataset,
        data: TemporalData,
        evaluator: Evaluator,
        neighbor_loader,
        loader,
        assoc,
        split:str,
        use_gnn=True,
        collect=False,
        use_msg_score=False,
        score_src=None):

    eval_metric = dataset.eval_metric

    memory.eval()
    gnn.eval()
    node_pred.eval()
    total_score = 0
    label_t = dataset.get_label_time()  # check when does the first label start
    num_label_ts = 0
    recs = []  # per-(user, day) NDCG when collect=True (for error/headroom analysis)

    for batch in tqdm(loader):
        batch = batch.to(device)
        src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        query_t = batch.t[-1]
        if query_t > label_t:
            label_tuple = dataset.get_node_label(query_t)
            if label_tuple is None:
                break
            label_ts, label_srcs, labels = (
                label_tuple[0],
                label_tuple[1],
                label_tuple[2],
            )
            label_t = dataset.get_label_time()
            label_srcs = label_srcs.to(device)

            # Process all edges that are still in the past day
            previous_day_mask = batch.t < label_t
            process_edges(
                memory,
                src[previous_day_mask],
                dst[previous_day_mask],
                t[previous_day_mask],
                msg[previous_day_mask],
                neighbor_loader,
                use_msg_score=use_msg_score,
                score_src=score_src,
            )
            # Reset edges to be the edges from tomorrow so they can be used later
            src, dst, t, msg = (
                src[~previous_day_mask],
                dst[~previous_day_mask],
                t[~previous_day_mask],
                msg[~previous_day_mask],
            )

            """
            modified for node property prediction
            1. sample neighbors from the neighbor loader for all nodes to be predicted
            2. extract memory from the sampled neighbors and the nodes
            3. run gnn with the extracted memory embeddings and the corresponding time and message
            """
            n_id = label_srcs
            sample = neighbor_loader(n_id)
            n_id_neighbors, mem_edge_index, e_id = sample[0], sample[1], sample[2]
            m_e = sample[3] if len(sample) > 3 else None       # per-edge affinity, normalized (V2)
            m_e_raw = sample[4] if len(sample) > 4 else None   # per-edge affinity, raw (message-like)
            assoc[n_id_neighbors] = torch.arange(n_id_neighbors.size(0), device=device)

            z,  last_update = memory(n_id_neighbors)
            if use_gnn:
                z = gnn(
                    z,
                    last_update,
                    mem_edge_index,
                    data.t[e_id].to(device),
                    _gnn_msg(e_id, m_e_raw, data, use_msg_score, score_src),
                    m_e=m_e,
                )
            z = z[assoc[n_id]]

            # loss and metric computation
            pred = node_pred(z)
            np_pred = pred.cpu().detach().numpy()
            np_true = labels.cpu().detach().numpy()

            input_dict = {
                "y_true": np_true,
                "y_pred": np_pred,
                "eval_metric": [eval_metric],
            }
            result_dict = evaluator.eval(input_dict)
            score = result_dict[eval_metric]
            total_score += score
            num_label_ts += 1

            if collect:
                from sklearn.metrics import ndcg_score
                us = n_id.cpu().numpy()
                qt = int(query_t)
                for i in range(len(us)):
                    if np_true[i].sum() > 0:
                        recs.append((qt, int(us[i]),
                                     float(ndcg_score(np_true[i:i+1], np_pred[i:i+1], k=10))))

        process_edges(memory, src, dst, t, msg, neighbor_loader, use_msg_score=use_msg_score,
                      score_src=score_src)

    metric_dict = {
        f"{split}/{eval_metric}": total_score / num_label_ts
    }
    mlflow.log_metrics(metric_dict, step=epoch)
    return metric_dict, recs


def main(args):
    # setting random seed
    seed = int(args.seed)  # 1,2,3,4,5
    torch.manual_seed(seed)
    set_random_seed(seed)

    LOG_DIR = 'logs/tidy/'
    os.makedirs(LOG_DIR, exist_ok=True)

    name = args.dataset
    dataset = PyGNodePropPredDataset(name=name, root="datasets")
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask

    num_classes = dataset.num_classes
    data = dataset.get_TemporalData()
    data = data.to(device)

    evaluator = Evaluator(name=name)

    train_data = data[train_mask]
    val_data = data[val_mask]
    test_data = data[test_mask]

    # Fast-iteration mode: keep only the tail of the train split (val/test untouched).
    # Memory/loader are rebuilt from scratch each epoch anyway, so this just shortens the
    # warmup + supervised-day count; train() fast-forwards the TGB label pointer to match.
    train_start_t = None
    e_id_offset = 0  # global data-index of the first streamed train edge (e_id == global index invariant)
    if args.train_tail_frac < 1.0:
        n_ev = train_data.src.size(0)
        e_id_offset = int(n_ev * (1.0 - args.train_tail_frac))  # train is the prefix of `data`
        keep = torch.zeros(n_ev, dtype=torch.bool)
        keep[e_id_offset:] = True
        train_data = train_data[keep]
        train_start_t = float(train_data.t[0])

    # Cheap-iteration: evaluate on only the HEAD fraction of val (continues the train stream, so
    # causality/e_id contiguity hold). The tail of val is then never inserted -> the test stream's
    # e_ids would misalign, hence the hard requirement of --no-eval_test.
    if args.val_frac < 1.0:
        assert not args.eval_test, \
            "--val_frac < 1 requires --no-eval_test (skipped val edges break the e_id invariant for test)"
        n_va = val_data.src.size(0)
        keep_v = torch.zeros(n_va, dtype=torch.bool)
        keep_v[:int(n_va * args.val_frac)] = True
        val_data = val_data[keep_v]

    # hyperparameters
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    all_hidden_dims = args.global_hidden_dims
    last_neighbour = args.num_last_neighbours

    train_loader = TemporalDataLoader(train_data, batch_size=batch_size)
    val_loader = TemporalDataLoader(val_data, batch_size=batch_size)
    test_loader = TemporalDataLoader(test_data, batch_size=batch_size)

    use_gnn = args.use_gnn

    memory_dim = all_hidden_dims
    idx_dim = all_hidden_dims if args.use_tgnv2 else -1
    embedding_dim = all_hidden_dims
    time_dim = all_hidden_dims
    raw_msg_dim = data.msg.size(-1)
    # --m_message_feature: the CountLoader affinity rides as one extra column INSIDE raw_msg, so it
    # flows through EncodeIndexModule/GRU AND the GNN edge_attr exactly like msg. One dim knob:
    # use_msg_score: False when off, else the score transform ('raw' | 'log1p') — truthy, threaded everywhere.
    use_msg_score = args.m_msg_transform if args.m_message_feature else False
    score_dim = 1 if args.m_message_feature else 0
    msg_feat_dim = raw_msg_dim + score_dim  # widened message width threaded to every construction site
    # ecdf_emb: the trailing score column is a BIN INDEX expanded to a learned embedding INSIDE the
    # message module / GNN (score_emb_dim replaces the 1 column). 0 for scalar transforms / off.
    score_emb_dim = args.m_msg_emb_dim if (args.m_message_feature and args.m_msg_transform == "ecdf_emb") else 0
    global SCORE_BINS
    SCORE_BINS = args.m_msg_bins

    # Neighbor sampler: recency (LastNeighborLoader) or dense per-(user,item) affinity (MSampler).
    if args.neighbor_sampler == "memory":
        k_m = args.k_m if args.k_m > 0 else last_neighbour
        # rank/ECDF write-transform fit on the OFFICIAL TGB train split (train_mask). NB: the p3/p4
        # offline probe fit on data.train_val_test_split(0.15,0.15), so in-harness M magnitudes /
        # selection ties need not be bit-identical to the 0.526 offline anchor — the official split
        # is the right causal choice here; compare the in-harness M-only run as the reference, not 0.526.
        sorted_train_w = torch.sort(train_data.msg[:, 0]).values
        alpha_select = math.log(2) / (args.kappa_select * 86400.0)
        neighbor_loader = MSampler(data.num_nodes, num_classes, k_m, alpha_select,
                                   sorted_train_w, device=device)
        variant_tag = 'mem-km{}{}'.format(k_m, '-medge' if args.m_edge_feature else '')
    else:
        neighbor_loader = LastNeighborLoader(data.num_nodes, size=last_neighbour, device=device)
        variant_tag = 'recency'
    if not use_gnn:
        variant_tag += '-nognn'
    if args.train_tail_frac < 1.0:
        variant_tag += '-ttf{}'.format(args.train_tail_frac)
    if args.m_message_feature:
        _mmsg_sfx = {'raw': '', 'log1p': 'log', 'ecdf': 'ecdf',
                     'ecdf_emb': 'emb{}x{}'.format(args.m_msg_bins, args.m_msg_emb_dim)}[args.m_msg_transform]
        variant_tag += '-mmsg' + _mmsg_sfx
    # Per-edge M affinity feature (V2) only makes sense with the memory sampler.
    m_edge_dim = 1 if (args.m_edge_feature and args.neighbor_sampler == "memory") else 0

    # Distillation teacher (idea 2): variant 1 reuses the memory sampler's M (single matrix);
    # variant 2 (recency) gets a separate MSampler used only for its M (k_m unused -> 1). None if off.
    teacher = None
    if args.distill:
        if isinstance(neighbor_loader, MSampler):
            teacher = neighbor_loader
        else:
            kappa_t = args.kappa_teacher if args.kappa_teacher > 0 else args.kappa_select
            teacher = MSampler(data.num_nodes, num_classes, 1, math.log(2) / (kappa_t * 86400.0),
                               torch.sort(train_data.msg[:, 0]).values, device=device)
        variant_tag += '-distill'

    # Score source for --m_message_feature: the memory sampler's own M; under the RECENCY sampler —
    # a lockstep score-only M-carrier (else the column would be constant-zero = placebo). Reuses the
    # distill teacher when present (already lockstep), otherwise builds one at --kappa_select.
    score_src = None
    if args.m_message_feature:
        if args.neighbor_sampler == "memory":
            score_src = neighbor_loader
        elif teacher is not None:
            score_src = teacher
        else:
            score_src = MSampler(data.num_nodes, num_classes, 1,
                                 math.log(2) / (args.kappa_select * 86400.0),
                                 torch.sort(train_data.msg[:, 0]).values, device=device)

    # --m_msg_transform ecdf: fit the FROZEN train-ECDF reference of scores in one cheap pre-pass
    # (throwaway MSampler streamed over train; score read BEFORE insert, subsampled every ~37th batch).
    # Mirrors the rank/ECDF phi write-transform; makes the column Uniform[0,1] and stationary.
    if args.m_message_feature and args.m_msg_transform in ("ecdf", "ecdf_emb"):
        global SCORE_REF
        _pre = MSampler(data.num_nodes, num_classes, 1, math.log(2) / (args.kappa_select * 86400.0),
                        torch.sort(train_data.msg[:, 0]).values, device=device)
        _ref = []
        for _i, _b in enumerate(train_loader):
            if _i % 37 == 0:
                _ref.append(_pre.score_for(_b.src, _b.dst, _b.t))
            _pre.insert(_b.src, _b.dst, _b.t, _b.msg)
        SCORE_REF = torch.sort(torch.cat(_ref)).values
        del _pre
        print(f"[mmsg] frozen ECDF score reference fitted: {SCORE_REF.numel():,} values "
              f"(p50={SCORE_REF[SCORE_REF.numel() // 2]:.1f}, max={SCORE_REF.max():.0f})")
    msg_module = EncodeIndexModule(idx_dim, msg_feat_dim, memory_dim, time_dim,
                                   score_emb_dim=score_emb_dim, score_bins=args.m_msg_bins) if \
        args.use_tgnv2 else IdentityMessage(msg_feat_dim, memory_dim, time_dim)

    aggregator_module = LastAggregator(msg_module.out_channels)

    memory = MTGNMemory(
        data.num_nodes,
        msg_feat_dim,
        memory_dim,
        time_dim,
        idx_dim,
        message_module=msg_module,
        aggregator_module=aggregator_module,
    ).to(device)

    gnn = (
        MGraphAttentionEmbedding(
            in_channels=memory_dim,
            out_channels=embedding_dim,
            msg_dim=msg_feat_dim,
            time_enc=memory.time_enc,
            m_edge_dim=m_edge_dim,
            score_emb_dim=score_emb_dim,
            score_bins=args.m_msg_bins,
        )
        .to(device)
        .float()
    )

    node_pred = NodePredictor(in_dim=embedding_dim, out_dim=num_classes).to(device)

    optimizer = torch.optim.Adam(
        set(memory.parameters()) | set(gnn.parameters()) | set(node_pred.parameters()),
        lr=lr,
    )

    if args.learning_scheduler == 'constant':
        lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
        scheduler_desc = ''
    elif args.learning_scheduler == 'cosine_annealing':
        ratio = args.cosine_annealing_ratio
        t_max = int(math.ceil(args.epochs / ratio))
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
        scheduler_desc = '_ratio_{}'.format(ratio)
    elif args.learning_scheduler == 'step_lr':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_lr_step_size, gamma=args.step_lr_gamma)
        scheduler_desc = '_gamma_{}_step_size_{}'.format(args.step_lr_gamma, args.step_lr_step_size)
    else:
        raise ValueError

    model_name = 'tgnv2' if args.use_tgnv2 else 'tgn'
    if args.neighbor_sampler == "memory":
        sampler_desc = '_sampler_memory_km_{}_kappa_{}{}'.format(
            (args.k_m if args.k_m > 0 else last_neighbour), args.kappa_select,
            '_medge' if args.m_edge_feature else '')
    else:
        sampler_desc = ''
    if not use_gnn:
        sampler_desc += '_nognn'
    if args.train_tail_frac < 1.0:  # fast-iteration runs get their own checkpoints/slug
        sampler_desc += '_ttf{}'.format(args.train_tail_frac)
    if args.val_frac < 1.0:
        sampler_desc += '_vf{}'.format(args.val_frac)
    if not args.eval_test:
        sampler_desc += '_notest'
    if args.m_message_feature:  # message-feature grows dims -> separate checkpoint from the off-runs
        sampler_desc += '_mmsg' + {'raw': '', 'log1p': 'log', 'ecdf': 'ecdf',
                                   'ecdf_emb': 'emb{}x{}'.format(args.m_msg_bins, args.m_msg_emb_dim)}[args.m_msg_transform]
    if args.distill:  # keep --distill / --no-distill on separate checkpoints
        sampler_desc += '_distill_l{}_str{}{}{}'.format(
            args.distill_lambda, args.distill_stride,
            '' if args.kd_grad == 'full' else '_kd' + args.kd_grad,
            # kappa_teacher only affects variant 2 (recency); a no-op under the memory sampler
            '_kt{}'.format(args.kappa_teacher)
            if (args.kappa_teacher > 0 and args.neighbor_sampler != 'memory') else '')
    run_name = 'scheduler_{}_{}_dataset_{}_bs_{}_lr_{}_epochs_{}_last_neighbour_{}_global_dims_{}{}_seed_{}'.format(
        args.learning_scheduler + scheduler_desc, model_name, name, batch_size, lr, epochs, last_neighbour, all_hidden_dims, sampler_desc, seed)

    # --- Local MLflow tracking: SQLite backend store, ./mlruns artifact store ---
    mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
    mlflow.set_experiment(args.experiment)
    short_name = '{}-{}-{}-s{}'.format(model_name, name.replace('tgbn-', ''), variant_tag, seed)  # human label; full slug goes in a tag

    log_full_path = os.path.join(LOG_DIR, run_name)
    logger = Logger(log_full_path)

    # Helper vector to map global node indices to local ones.
    assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)

    max_val_score = 0  # find the best test score based on validation score
    best_epoch = 0
    test_ndcgs = {}  # epoch -> test score (dict: with --eval_every > 1 not every epoch is evaluated)
    best_recs = []  # per-(user,day) test NDCG of the best-val epoch (when --dump_eval_preds)
    global_step = 0  # monotonic step for the streaming (per-label-timestamp) training curve

    eval_metric = dataset.eval_metric
    git_sha, git_dirty = _git_commit_and_dirty()

    # --- Checkpoint/resume: per-epoch state so a reaped background run continues where it left off. ---
    os.makedirs('saved_models', exist_ok=True)
    ckpt_path = os.path.join('saved_models', f'{run_name}.pt')
    start_epoch = 1
    resume_run_id = None
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        memory.load_state_dict(ckpt['memory'])
        gnn.load_state_dict(ckpt['gnn'])
        node_pred.load_state_dict(ckpt['node_pred'])
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        start_epoch = ckpt['epoch'] + 1
        max_val_score = ckpt['max_val_score']
        tn = ckpt['test_ndcgs']  # older checkpoints stored a per-epoch list + best_test_idx
        test_ndcgs = {i + 1: v for i, v in enumerate(tn)} if isinstance(tn, list) else tn
        best_epoch = ckpt.get('best_epoch', ckpt.get('best_test_idx', -1) + 1)
        global_step = ckpt['global_step']
        resume_run_id = ckpt.get('mlflow_run_id')
        logger.log_and_write(f"RESUMED from {ckpt_path} at epoch {start_epoch} (best_val={max_val_score:.4f})")

    run_ctx = (mlflow.start_run(run_id=resume_run_id) if resume_run_id
               else mlflow.start_run(run_name=short_name))
    with run_ctx:
        if resume_run_id is None:
            # Source of truth: full config as params, filterable metadata as tags (first launch only).
            mlflow.log_params(vars(args))
        tags = {
            "model": model_name,
            "dataset": name,
            "seed": seed,
            "slug": run_name,
            "device": str(device),
            "hostname": socket.gethostname(),
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
        }
        if git_sha is not None:
            tags["git_commit"] = git_sha
            tags["git_dirty"] = git_dirty
        mlflow.set_tags(tags)

        for epoch in range(start_epoch, epochs + 1):
            start_time = timeit.default_timer()
            train_dict, global_step = train(
                epoch=epoch,
                global_step=global_step,
                memory=memory,
                gnn=gnn,
                node_pred=node_pred,
                lr_scheduler=lr_scheduler,
                dataset=dataset,
                data=data,
                evaluator=evaluator,
                neighbor_loader=neighbor_loader,
                train_loader=train_loader,
                optimizer=optimizer,
                assoc=assoc,
                use_gnn=use_gnn,
                teacher=teacher,
                distill_lambda=args.distill_lambda,
                distill_stride=args.distill_stride,
                kd_grad=args.kd_grad,
                train_start_t=train_start_t,
                use_msg_score=use_msg_score,
                e_id_offset=e_id_offset,
                score_src=score_src,
            )
            logger.log_and_write("------------------------------------")
            logger.log_and_write(f"training Epoch: {epoch:02d}")
            logger.log_and_write(train_dict)
            logger.log_and_write("Training takes--- %s seconds ---" % (timeit.default_timer() - start_time))

            # --eval_every N: run the heavy val/test streams only every Nth epoch (final epoch always).
            do_eval = (epoch % args.eval_every == 0) or (epoch == epochs)
            if do_eval:
                start_time = timeit.default_timer()
                val_dict, _ = test(
                    memory=memory,
                    gnn=gnn,
                    node_pred=node_pred,
                    epoch=epoch,
                    dataset=dataset,
                    data=data,
                    evaluator=evaluator,
                    neighbor_loader=neighbor_loader,
                    assoc=assoc,
                    loader=val_loader,
                    use_gnn=use_gnn,
                    split='val',
                    use_msg_score=use_msg_score,
                    score_src=score_src,
                )

                logger.log_and_write(val_dict)
                val_ndcg = val_dict[f'val/{eval_metric}']
                if val_ndcg > max_val_score:
                    max_val_score = val_ndcg
                    best_epoch = epoch
                logger.log_and_write("Validation takes--- %s seconds ---" % (timeit.default_timer() - start_time))

            if do_eval and args.eval_test:
                start_time = timeit.default_timer()
                test_dict, test_recs = test(
                    memory=memory,
                    gnn=gnn,
                    node_pred=node_pred,
                    epoch=epoch,
                    dataset=dataset,
                    data=data,
                    evaluator=evaluator,
                    neighbor_loader=neighbor_loader,
                    assoc=assoc,
                    loader=test_loader,
                    use_gnn=use_gnn,
                    split='test',
                    collect=args.dump_eval_preds,
                    use_msg_score=use_msg_score,
                    score_src=score_src,
                )
                test_ndcgs[epoch] = test_dict[f'test/{eval_metric}']
                # keep the test-split per-(user,day) NDCG of the best-val epoch
                if args.dump_eval_preds and best_epoch == epoch:
                    best_recs = test_recs

                logger.log_and_write(test_dict)
                logger.log_and_write("Test takes--- %s seconds ---" % (timeit.default_timer() - start_time))
            logger.log_and_write("------------------------------------")
            dataset.reset_label_time()

            # Per-epoch checkpoint so a reaped/killed run resumes here (this env reaps background tasks).
            torch.save({
                'epoch': epoch,
                'memory': memory.state_dict(),
                'gnn': gnn.state_dict(),
                'node_pred': node_pred.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'max_val_score': max_val_score,
                'best_epoch': best_epoch,
                'test_ndcgs': test_ndcgs,
                'global_step': global_step,
                'mlflow_run_id': mlflow.active_run().info.run_id,
            }, ckpt_path)

        max_test_score = test_ndcgs.get(best_epoch) if test_ndcgs else None  # None with --no-eval_test
        logger.log_and_write("------------------------------------")
        logger.log_and_write("------------------------------------")
        logger.log_and_write("best val score: {}".format(max_val_score))
        logger.log_and_write("best validation epoch   : {}".format(best_epoch))
        logger.log_and_write("best test score: {}".format(max_test_score if max_test_score is not None
                                                          else "SKIPPED (--no-eval_test)"))

        # Headline numbers for the paper — summary metrics on the run.
        summary = {"best_val_ndcg": max_val_score, "best_epoch": best_epoch}
        if max_test_score is not None:
            summary["best_test_ndcg"] = max_test_score
        mlflow.log_metrics(summary)
        # Attach the (now complete) text log as a run artifact.
        mlflow.log_artifact(log_full_path)

        # Dump per-(user,day) test NDCG@10 of the best-val epoch for error/headroom analysis.
        if args.dump_eval_preds:
            import polars as pl
            dump_path = os.path.join("mlruns", f"_eval_preds_{name}.parquet")
            (pl.DataFrame(best_recs, schema=["ts", "user", "ndcg"], orient="row")
               .write_parquet(dump_path))
            mlflow.log_artifact(dump_path)
            logger.log_and_write(f"dumped {len(best_recs)} per-(user,day) test NDCG rows -> {dump_path}")


if __name__ == '__main__':
    parser = utils.get_parser()
    args = parser.parse_args()
    main(args)
