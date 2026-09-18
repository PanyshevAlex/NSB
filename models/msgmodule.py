import torch


class EncodeIndexModule(torch.nn.Module):
    def __init__(self, idx_dim: int, raw_msg_dim: int, memory_dim: int, time_dim: int,
                 score_emb_dim: int = 0, score_bins: int = 0):
        super().__init__()
        # --m_msg_transform ecdf_emb: the LAST raw_msg column is a score BIN INDEX; embed it here
        # (with current weights) into a score_emb_dim vector instead of passing the scalar through.
        self.score_emb_dim = score_emb_dim
        if score_emb_dim > 0:
            self.score_emb = torch.nn.Embedding(score_bins, score_emb_dim)
            self.out_channels = (raw_msg_dim - 1) + score_emb_dim + 2 * memory_dim + 2 * idx_dim + time_dim
        else:
            self.out_channels = raw_msg_dim + 2 * memory_dim + 2 * idx_dim + time_dim

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor, raw_msg: torch.Tensor,
                t_enc: torch.Tensor, src_enc: torch.Tensor, dst_enc: torch.Tensor):
        if self.score_emb_dim > 0:  # expand the trailing bin-index column into its learned embedding
            emb = self.score_emb(raw_msg[:, -1].long().clamp_(0, self.score_emb.num_embeddings - 1))
            raw_msg = torch.cat([raw_msg[:, :-1], emb], dim=-1)
        return torch.cat([z_src, z_dst, raw_msg, src_enc, dst_enc, t_enc], dim=-1)