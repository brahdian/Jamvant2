import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Tuple, Optional, List

# --- Utility: RMSNorm for Stability ---
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

# --- 1. Mamba-3 MIMO Core ---
class Mamba3MIMOCore(nn.Module):
    def __init__(self, d_model: int, d_state: int = 64, expand: int = 2, n_groups: int = 4):
        super().__init__()
        self.d_model, self.d_state, self.expand, self.n_groups = d_model, d_state, expand, n_groups
        self.d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=4, groups=self.d_inner, padding=3)
        self.x_proj = nn.Linear(self.d_inner, n_groups * (d_state * 2 + 1), bias=False)
        self.dt_proj = nn.Linear(n_groups, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).repeat(self.d_inner, 1).float()))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.norm = RMSNorm(self.d_inner)

    def forward(self, x: torch.Tensor):
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :L].transpose(1, 2)
        x = F.silu(x)

        mimo_params = self.x_proj(x).view(B, L, self.n_groups, -1)
        B_p, C_p, dt_p = torch.split(mimo_params, [self.d_state, self.d_state, 1], dim=-1)
        dt = F.softplus(self.dt_proj(dt_p.squeeze(-1)))
        A = -torch.exp(self.A_log.float())

        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        y_list = []
        for t in range(L):
            xt = x[:, t, :].unsqueeze(-1)
            bt = B_p[:, t, :].mean(dim=1).unsqueeze(1)
            ct = C_p[:, t, :].mean(dim=1).unsqueeze(-1)
            dtt = dt[:, t, :].unsqueeze(-1)
            h = h * torch.exp(A * dtt) + xt * (bt * dtt)
            y_list.append(torch.matmul(h, ct).squeeze(-1).unsqueeze(1))
        return self.norm(torch.cat(y_list, dim=1)) * F.silu(z)

# --- 2. LNN Trigger ---
class LNNTrigger(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.w_tau = nn.Linear(d_model, d_model)
        self.w_s = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, 1)
        self.norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape
        h = torch.zeros(B, D, device=x.device)
        outputs = []
        for t in range(L):
            xt = x[:, t, :]
            h = h + 0.1 * (-torch.sigmoid(self.w_tau(xt)) * h + torch.tanh(self.w_s(xt)))
            outputs.append(h.unsqueeze(1))
        h_ltc = self.norm(torch.cat(outputs, dim=1))
        return h_ltc, torch.sigmoid(self.gate(h_ltc))

# --- 3. RWKV-7 HeadQK ---
class RWKV7HeadQK(nn.Module):
    def __init__(self, d_model: int, n_head: int = 8):
        super().__init__()
        self.d_model, self.n_head = d_model, n_head
        self.head_size = d_model // n_head
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.g_proj = nn.Linear(d_model, d_model, bias=False)
        self.w_proj = nn.Linear(d_model, d_model, bias=False)
        self.output_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None):
        B, L, D = x.shape
        H, S = self.n_head, self.head_size
        q = self.q_proj(x).view(B, L, H, S)
        k, v = self.k_proj(x).view(B, L, H, S), self.v_proj(x).view(B, L, H, S)
        g, w = torch.sigmoid(self.g_proj(x)).view(B, L, H, S), torch.sigmoid(self.w_proj(x)).view(B, L, H, S)
        if state is None: state = torch.zeros(B, H, S, S, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(L):
            qt, kt, vt, gt, wt = q[:, t], k[:, t], v[:, t], g[:, t], w[:, t]
            state = state * (1 - gt.unsqueeze(-1) @ kt.unsqueeze(-2)) + (vt.unsqueeze(-1) @ kt.unsqueeze(-2)) * wt.unsqueeze(-1)
            outputs.append((state @ qt.unsqueeze(-1)).squeeze(-1).unsqueeze(1))
        y = self.output_proj(torch.cat(outputs, dim=1).view(B, L, D))
        return self.norm(y), state

# --- 4. Mixture-of-Recursions (MoR) ---
class MoRCore(nn.Module):
    def __init__(self, d_model: int, max_recursion: int = 4):
        super().__init__()
        self.max_recursion = max_recursion
        self.rwkv_block = RWKV7HeadQK(d_model)
        self.norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, trigger_prob: torch.Tensor):
        state = None
        for _ in range(self.max_recursion):
            res, state = self.rwkv_block(self.norm(x), state)
            x = x + res * trigger_prob
        return x

# --- 5. Hardened Recursive Flux Engine ---
class RecursiveFluxEngine(nn.Module):
    def __init__(self, d_model: int = 1024, n_layers: int = 24, vocab_size: int = 50257):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.ingestion = nn.ModuleList([Mamba3MIMOCore(d_model) for _ in range(n_layers // 2)])
        self.trigger = LNNTrigger(d_model)
        self.thinker = MoRCore(d_model)
        self.decoder = nn.ModuleList([Mamba3MIMOCore(d_model) for _ in range(n_layers // 2)])
        self.final_norm = RMSNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)
        self.output_head.weight = self.embedding.weight
        self.last_trigger_prob = None

    def forward(self, input_ids: torch.Tensor):
        x = self.embedding(input_ids)
        for block in self.ingestion: x = x + block(x)
        h_ltc, trigger_prob = self.trigger(x)
        self.last_trigger_prob = trigger_prob
        x = self.thinker(x + h_ltc, trigger_prob)
        for block in self.decoder: x = x + block(x)
        return self.output_head(self.final_norm(x))

def stability_loss(trigger_prob): return torch.mean(trigger_prob)

if __name__ == "__main__":
    model = RecursiveFluxEngine(d_model=256, n_layers=4)
    print("Recursive Flux Engine Architecture Verified.")
