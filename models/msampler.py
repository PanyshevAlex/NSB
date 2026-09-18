"""Dense per-(user, item) Hawkes-affinity neighbor sampler (P5 idea 1).

Drop-in replacement for `LastNeighborLoader`: same `(n_id, mem_edge_index, e_id)` contract
consumed by the GNN in `train-tgbn-nodeproppred.py`, plus a 4th return `m_e` — the per-edge
per-user-NORMALIZED affinity (V2 `--m_edge_feature`) — and a 5th return `m_e_raw` — the RAW
decayed affinity at read time, aligned 1:1 with `e_id` (`--m_message_feature`).

Instead of the most-recent `size` events, it feeds the GNN, per query user `u`, its
**top-K_M items by decayed affinity** `M[u, :] > 0` (variable size — never pads zero-affinity
items, so the "sample K but only m<K nonzero -> K-m noise" case cannot occur). `e_id` indexes
the global `data.t` / `data.msg` exactly like the recency loader, recovered from a dense
`last_eid[N, C]` buffer (M itself carries no event id).

The affinity is the same continuous-time per-pair memory validated in
`notebooks/p3_perpair_memory.ipynb` / `p4_m_as_sampler.ipynb`:
    M[u, i](t) = sum_{e in (u,i), t_e < t} phi(w_e) * exp(-alpha * (t - t_e)),
with `phi` = rank/ECDF write transform fit on TRAIN weights only (causal, parameter-free).
It is parameter-free / non-differentiable: it only SELECTS which item memories the single
GNN head attends to (and optionally weights them via `m_e`), never emits a prediction.
"""
import math

import torch


class MSampler:
    def __init__(self, num_nodes, num_classes, size, alpha, sorted_train_w, device=None):
        self.N = int(num_nodes)
        self.C = int(num_classes)
        self.size = int(size)            # read-budget K_M
        self.alpha = float(alpha)        # selection decay rate (= ln2 / half_life)
        self.device = device
        self.sorted_w = sorted_train_w.to(device=device, dtype=torch.float64)
        self.nw = max(int(self.sorted_w.numel()), 1)
        self.M = torch.zeros(self.N, self.C, dtype=torch.float64, device=device)
        self.tlast = torch.zeros(self.N, self.C, dtype=torch.float64, device=device)
        self.last_eid = torch.empty(self.N, self.C, dtype=torch.long, device=device)
        self._assoc = torch.empty(self.N, dtype=torch.long, device=device)
        self.reset_state()

    def reset_state(self):
        """Fresh state at the start of each train pass (mirrors LastNeighborLoader.reset_state)."""
        self.M.zero_()
        self.tlast.zero_()
        self.last_eid.fill_(-1)
        self.cur_e_id = 0          # running global edge id; matches data.t/msg index order
        self.t_now = 0.0           # latest inserted timestamp (read-decay reference)

    def _phi(self, w):
        """rank/ECDF write transform on TRAIN weights (causal, parameter-free), in (0, 1]."""
        return torch.searchsorted(self.sorted_w, w.to(torch.float64), right=True).to(torch.float64) / self.nw

    def insert(self, src, dst, t, msg):
        """Write a batch of edges into M / last_eid (called from process_edges on PAST edges).

        `t` (timestamps) and `msg` (raw features; col 0 = weight) are required — the uniform
        process_edges interface always passes them (LastNeighborLoader ignores them).
        """
        if src.numel() == 0:
            return
        # Flat (user, item) cell id assumes items in [0, C) and user/item id disjointness.
        assert int(dst.max()) < self.C, "items must be node ids in [0, num_classes) for flat indexing"
        E = src.numel()
        tb = float(t.max())
        self.t_now = max(self.t_now, tb)
        w = msg[:, 0].contiguous()  # column view is non-contiguous; avoid per-call copy in searchsorted
        flat = src.long() * self.C + dst.long()                 # [E] flat (user, item) cell id
        uniq, inv = torch.unique(flat, return_inverse=True)
        add = torch.zeros(uniq.numel(), dtype=torch.float64, device=self.device)
        add.scatter_add_(0, inv, self._phi(w))                  # sum phi(w) per cell
        eids = torch.arange(self.cur_e_id, self.cur_e_id + E, device=self.device)
        emax = torch.full((uniq.numel(),), -1, dtype=torch.long, device=self.device)
        emax.scatter_reduce_(0, inv, eids, reduce="amax", include_self=True)  # latest event per cell
        Mf, Tf, Ef = self.M.view(-1), self.tlast.view(-1), self.last_eid.view(-1)
        if self.alpha > 0:
            decay = torch.exp(-self.alpha * (tb - Tf[uniq]).clamp(min=0))
            Mf[uniq] = Mf[uniq] * decay + add
        else:
            Mf[uniq] = Mf[uniq] + add
        Tf[uniq] = tb
        Ef[uniq] = emax
        self.cur_e_id += E

    def __call__(self, n_id):
        """Return (n_id_neighbors, mem_edge_index, e_id, m_e, m_e_raw) for query users `n_id`.

        Per user: its <=K_M highest-affinity (M>0) items. Edges are (item -> user) in local
        indices over the returned unique node set, matching LastNeighborLoader's contract.
        `m_e` = per-user-normalized affinity in (0, 1] (top neighbour = 1), float32.
        `m_e_raw` = RAW decayed affinity at read time t_now, float32, aligned 1:1 with `e_id`
        (note: includes the cell total up to now, i.e. the last event's own contribution —
        unlike score_for-before-insert in the memory channel, which is strictly pre-edge)."""
        dev = self.device
        Q = n_id.size(0)
        K = min(self.size, self.C)
        rows = self.M[n_id].clone()
        if self.alpha > 0:
            dt = (self.t_now - self.tlast[n_id]).clamp(min=0)
            rows = rows * torch.exp(-self.alpha * dt)           # decay to read time
        vals, cols = rows.topk(K, dim=1)                        # [Q, K] descending
        valid = vals > 0                                        # drop zero-affinity slots (no padding)
        qrow = torch.arange(Q, device=dev).view(-1, 1).expand(-1, K)
        qi = qrow[valid]
        items = cols[valid]
        users = n_id[qi]
        e_id = self.last_eid[users, items]
        maxper = vals[:, 0].clamp(min=1e-12)                    # per-user top affinity
        m_e = (vals[valid] / maxper[qi]).to(torch.float32)      # normalized affinity edge feature (V2)
        m_e_raw = vals[valid].to(torch.float32)                 # RAW decayed affinity, aligned 1:1 with e_id

        n_id_out = torch.cat([n_id, items]).unique()
        self._assoc[n_id_out] = torch.arange(n_id_out.size(0), device=dev)
        mem_edge_index = torch.stack([self._assoc[items], self._assoc[users]])  # [item_local; user_local]
        return n_id_out, mem_edge_index, e_id, m_e, m_e_raw

    def score_for(self, src, dst, t=None):
        """Raw decayed affinity M[src, dst] per edge (float32, shape [E]) — the message-like score.
        Read STRICTLY BEFORE these edges are inserted (causal: affinity = history before the edge).
        `t` (per-edge timestamps) decays each cell to the edge's own time; None -> decay to t_now."""
        if src.numel():
            assert int(dst.max()) < self.C, "score_for: dst must be item ids < num_classes (same layout as insert)"
        flat = src.to(torch.long) * self.C + dst.to(torch.long)
        vals = self.M.view(-1)[flat].clone()
        if self.alpha > 0:
            tref = self.t_now if t is None else t.to(self.tlast.dtype)
            vals = vals * torch.exp(-self.alpha * (tref - self.tlast.view(-1)[flat]).clamp(min=0))
        return vals.to(torch.float32)

    def soft_target(self, n_id):
        """Distillation teacher (idea 2): row-normalized decayed affinity `M[users]` as a soft label.

        Returns `(soft [Q, C] float32, warm [Q] bool)`. `soft` puts 100% of the mass on the user's
        seen items (row-normalize, NOT softmax(M/T) — raw |M| are large/heterogeneous so a fixed-T
        softmax is fragile; row-normalize gives ~37 effective classes). Columns align 1:1 with the
        node_pred class axis (item-node == class). `warm` flags non-empty rows so the caller drops
        cold (all-zero) users that would divide by zero. `M` is parameter-free ⇒ the target is detached.
        """
        rows = self.M[n_id].clone()
        if self.alpha > 0:
            dt = (self.t_now - self.tlast[n_id]).clamp(min=0)
            rows = rows * torch.exp(-self.alpha * dt)        # decay to read time (causal)
        s = rows.sum(dim=1)
        warm = s > 0
        soft = torch.zeros(n_id.size(0), self.C, dtype=torch.float32, device=self.device)
        soft[warm] = (rows[warm] / s[warm].unsqueeze(1)).to(torch.float32)
        return soft, warm
