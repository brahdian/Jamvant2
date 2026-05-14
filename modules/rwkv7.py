import torch
import torch.nn as nn
from typing import Optional, Tuple

class RWKV7HeadQK(nn.Module):
    def __init__(self, d_model: int, n_head: int = 8, head_size: int = 64):
        super().__init__()
        self.d_model, self.n_head, self.head_size = d_model, n_head, head_size
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.g_proj = nn.Linear(d_model, d_model, bias=False)
        self.w_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = nn.LayerNorm(head_size)
        self.k_norm = nn.LayerNorm(head_size)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, state: Optional[torch.Tensor] = None):
        """Supports state passing for State-Streaming."""
        B, L, D = x.shape
        H, S = self.n_head, self.head_size
        q = self.q_norm(self.q_proj(x).view(B, L, H, S))
        k = self.k_norm(self.k_proj(x).view(B, L, H, S))
        v = self.v_proj(x).view(B, L, H, S)
        g, w = torch.sigmoid(self.g_proj(x)).view(B, L, H, S), torch.sigmoid(self.w_proj(x)).view(B, L, H, S)

        if state is None: state = torch.zeros(B, H, S, S, device=x.device, dtype=x.dtype)
        y = []
        for t in range(L):
            qt, kt, vt, gt, wt = q[:, t].unsqueeze(-1), k[:, t].unsqueeze(-2), v[:, t].unsqueeze(-1), g[:, t].unsqueeze(-1), w[:, t].unsqueeze(-1)
            state = state * (1 - gt @ kt) + (vt @ kt) * wt
            y.append((state @ qt).squeeze(-1).view(B, 1, D))
        return self.out_proj(torch.cat(y, dim=1)), state
