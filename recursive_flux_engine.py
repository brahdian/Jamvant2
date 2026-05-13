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
        norm_x = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.weight * x_normed

# --- 1. Hardened Mamba-3 MIMO Core ---
# Implements Input-Dependent Selection Mechanism and MIMO logic
class Mamba3MIMOCore(nn.Module):
    def __init__(self, d_model: int, d_state: int = 64, expand: int = 2, n_groups: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = d_model * expand
        self.n_groups = n_groups

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=4, groups=self.d_inner, padding=3)

        # MIMO Projection: Maps inner dim to groups of SSM parameters (B, C, dt)
        self.x_proj = nn.Linear(self.d_inner, self.n_groups * (d_state * 2 + 1), bias=False)
        self.dt_proj = nn.Linear(self.n_groups, self.d_inner, bias=True)

        # Initialization: Small values for A to ensure stable recurrence
        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.d_state + 1).repeat(self.d_inner, 1).float()))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.norm = RMSNorm(self.d_inner)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        # Convolution
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :L]
        x = x.transpose(1, 2)
        x = F.silu(x)

        # 1. Selective Mechanism: derive B, C, dt from input
        # MIMO logic: Project to multiple parameter groups
        mimo_params = self.x_proj(x) # (B, L, G * (2*S + 1))
        mimo_params = mimo_params.view(B, L, self.n_groups, -1)

        # Split into B, C, and raw_dt
        B_p, C_p, dt_p = torch.split(mimo_params, [self.d_state, self.d_state, 1], dim=-1)

        # Aggregate groups for the state update (MIMO reduction)
        B_p = B_p.mean(dim=2) # (B, L, S)
        C_p = C_p.mean(dim=2) # (B, L, S)
        dt_p = dt_p.squeeze(-1) # (B, L, G)
        dt = F.softplus(self.dt_proj(dt_p)) # (B, L, d_inner)

        # 2. SSM Recurrence (Scan)
        # Optimized for PyTorch via vectorized update where possible,
        # but maintaining O(L) logic for the prototype's clarity.
        A = -torch.exp(self.A_log.float()) # (d_inner, d_state)

        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        y_list = []

        # Parallel associative scan would replace this in a full-scale CUDA implementation
        for t in range(L):
            xt = x[:, t, :].unsqueeze(-1) # (B, d_inner, 1)
            bt = B_p[:, t, :].unsqueeze(1) # (B, 1, d_state)
            ct = C_p[:, t, :].unsqueeze(-1) # (B, d_state, 1)
            dtt = dt[:, t, :].unsqueeze(-1) # (B, d_inner, 1)

            # Discretize A and B (Euler)
            # A_bar = exp(A * dt)
            # B_bar = B * dt
            a_bar = torch.exp(A * dtt)
            b_bar = bt * dtt

            h = h * a_bar + xt * b_bar

            # y = C * h
            yt = torch.matmul(h, ct).squeeze(-1) # (B, d_inner)
            y_list.append(yt.unsqueeze(1))

        y = torch.cat(y_list, dim=1)
        y = self.norm(y)
        y = y * F.silu(z)
        return self.dropout(y)

# --- 2. Optimized LNN Trigger ---
class LNNTrigger(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.w_tau = nn.Linear(d_model, d_model)
        self.w_s = nn.Linear(d_model, d_model)
        nn.init.orthogonal_(self.w_tau.weight)

        self.gate = nn.Linear(d_model, 1)
        self.norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape
        h = torch.zeros(B, D, device=x.device)
        dt = 0.1

        outputs = []
        for t in range(L):
            xt = x[:, t, :]
            f_x = torch.sigmoid(self.w_tau(xt))
            s_x = torch.tanh(self.w_s(xt))
            h = h + dt * (-f_x * h + s_x)
            outputs.append(h.unsqueeze(1))

        h_ltc = torch.cat(outputs, dim=1)
        h_ltc = self.norm(h_ltc)
        trigger_prob = torch.sigmoid(self.gate(h_ltc))
        return h_ltc, trigger_prob

# --- 3. Hardened RWKV-7 HeadQK ---
class RWKV7HeadQK(nn.Module):
    def __init__(self, d_model: int, n_head: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_size = d_model // n_head

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.g_proj = nn.Linear(d_model, d_model, bias=False)
        self.w_proj = nn.Linear(d_model, d_model, bias=False)

        self.output_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None):
        B, L, D = x.shape
        H, S = self.n_head, self.head_size

        q = self.q_proj(x).view(B, L, H, S)
        k = self.k_proj(x).view(B, L, H, S)
        v = self.v_proj(x).view(B, L, H, S)
        g = torch.sigmoid(self.g_proj(x)).view(B, L, H, S)
        w = torch.sigmoid(self.w_proj(x)).view(B, L, H, S)

        if state is None:
            state = torch.zeros(B, H, S, S, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(L):
            qt, kt, vt, gt, wt = q[:, t], k[:, t], v[:, t], g[:, t], w[:, t]

            kt = kt.unsqueeze(-2) # (B, H, 1, S)
            vt = vt.unsqueeze(-1) # (B, H, S, 1)
            gt = gt.unsqueeze(-1) # (B, H, S, 1)
            wt = wt.unsqueeze(-1) # (B, H, S, 1)

            # Generalized Delta Rule update
            state = state * (1 - gt @ kt) + (vt @ kt) * wt

            yt = (state @ qt.unsqueeze(-1)).squeeze(-1)
            outputs.append(yt.unsqueeze(1))

        y = torch.cat(outputs, dim=1).view(B, L, D)
        y = self.output_proj(y)
        return self.dropout(self.norm(y)), state

# --- 4. Mixture-of-Recursions (MoR) ---
class MoRCore(nn.Module):
    def __init__(self, d_model: int, max_recursion: int = 4):
        super().__init__()
        self.max_recursion = max_recursion
        self.rwkv_block = RWKV7HeadQK(d_model)
        self.layer_norm = RMSNorm(d_model)

    def _recursive_step(self, x, trigger_prob, state):
        res, state = self.rwkv_block(self.layer_norm(x), state)
        x = x + res * trigger_prob
        return x, state

    def forward(self, x: torch.Tensor, trigger_prob: torch.Tensor):
        state = None
        for _ in range(self.max_recursion):
            if self.training:
                def checkpoint_fn(curr_x, curr_state):
                    return self._recursive_step(curr_x, trigger_prob, curr_state)
                x, state = checkpoint(checkpoint_fn, x, state, use_reentrant=False)
            else:
                x, state = self._recursive_step(x, trigger_prob, state)
        return x

# --- 5. Hardened Recursive Flux Engine ---
class RecursiveFluxEngine(nn.Module):
    def __init__(self, d_model: int = 1024, vocab_size: int = 50257, d_state: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.ingestion = Mamba3MIMOCore(d_model, d_state)
        self.trigger = LNNTrigger(d_model)
        self.thinker = MoRCore(d_model)
        self.decoder = Mamba3MIMOCore(d_model, d_state)

        self.output_head = nn.Linear(d_model, vocab_size, bias=False)
        self.final_norm = RMSNorm(d_model)
        self.last_trigger_prob = None

        self.output_head.weight = self.embedding.weight

    def forward(self, input_ids: torch.Tensor):
        x = self.embedding(input_ids)
        x = self.ingestion(x)
        h_ltc, trigger_prob = self.trigger(x)
        self.last_trigger_prob = trigger_prob
        x = x + h_ltc
        x = self.thinker(x, trigger_prob)
        x = self.decoder(x)
        x = self.final_norm(x)
        return self.output_head(x)

def stability_loss(trigger_prob):
    return torch.mean(trigger_prob)

if __name__ == "__main__":
    model = RecursiveFluxEngine(d_model=256, vocab_size=1000)
    input_data = torch.randint(0, 1000, (2, 16))
    logits = model(input_data)
    print(f"Logits shape: {logits.shape}")
    print("Mamba-3 Selection Mechanism & MoR Core Hardened.")
