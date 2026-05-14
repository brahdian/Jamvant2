import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class Mamba3MIMOCore(nn.Module):
    def __init__(self, d_model: int, d_state: int = 64, expand: int = 2, n_groups: int = 4):
        super().__init__()
        self.d_model, self.d_state, self.n_groups = d_model, d_state, n_groups
        self.d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.x_proj = nn.Linear(self.d_inner, n_groups * (d_state * 2 + 1), bias=False)
        self.dt_proj = nn.Linear(n_groups, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).repeat(self.d_inner, 1).float()))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.q_norm = nn.LayerNorm(d_state)
        self.k_norm = nn.LayerNorm(d_state)

    def forward(self, x, state: Optional[torch.Tensor] = None):
        """Supports state passing for State-Streaming."""
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        mimo_params = self.x_proj(x).view(B, L, self.n_groups, -1)
        B_p, C_p, dt_p = torch.split(mimo_params, [self.d_state, self.d_state, 1], dim=-1)
        B_p = self.q_norm(B_p).mean(dim=2)
        C_p = self.k_norm(C_p).mean(dim=2)
        dt = F.softplus(self.dt_proj(dt_p.squeeze(-1)))
        A = -torch.exp(self.A_log.float())

        # Initialize or use passed state
        h = state if state is not None else torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        y = []
        for t in range(L):
            xt, dtt = x[:, t, :].unsqueeze(-1), dt[:, t, :].unsqueeze(-1)
            bt, ct = B_p[:, t, :].unsqueeze(1), C_p[:, t, :].unsqueeze(-1)
            h = h * torch.exp(A * dtt) + xt * (bt * dtt)
            y.append(torch.matmul(h, ct).squeeze(-1).unsqueeze(1))

        return torch.cat(y, dim=1) * F.silu(z), h
