"""
GNN-based modules used in the architecture of MP-TG models

"""

import math

import torch_geometric.nn
from torch_geometric.nn import TransformerConv
import torch


class MGraphAttentionEmbedding(torch.nn.Module):
    """
    Reference:
    - https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py
    """
    def __init__(self, in_channels, out_channels, msg_dim, time_enc, m_edge_dim=0,
                 score_emb_dim=0, score_bins=0):
        super().__init__()
        self.time_enc = time_enc
        self.m_edge_dim = m_edge_dim  # >0 => per-edge M affinity is concatenated into edge_attr (V2)

        # --m_msg_transform ecdf_emb: the LAST msg column is a score BIN INDEX -> embed it here.
        self.score_emb_dim = score_emb_dim
        eff_msg = msg_dim
        if score_emb_dim > 0:
            self.score_emb = torch.nn.Embedding(score_bins, score_emb_dim)
            eff_msg = (msg_dim - 1) + score_emb_dim
        edge_dim = eff_msg + time_enc.out_channels + m_edge_dim
        self.conv = TransformerConv(in_channels, out_channels // 2, heads=2, dropout=0.1, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg, m_e=None):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        if self.score_emb_dim > 0:  # expand the trailing bin-index column into its learned embedding
            emb = self.score_emb(msg[:, -1].long().clamp_(0, self.score_emb.num_embeddings - 1))
            msg = torch.cat([msg[:, :-1], emb], dim=-1)
        feats = [rel_t_enc, msg]
        if self.m_edge_dim > 0:
            if m_e is None:  # safety: recency sampler carries no affinity -> neutral 0
                m_col = torch.zeros(edge_index.size(1), self.m_edge_dim, dtype=x.dtype, device=x.device)
            else:
                m_col = m_e.view(-1, 1).to(x.dtype)
            feats.append(m_col)
        edge_attr = torch.cat(feats, dim=-1)

        res = self.conv(x, edge_index, edge_attr)
        return res

