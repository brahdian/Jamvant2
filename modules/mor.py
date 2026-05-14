import torch
import torch.nn as nn
from typing import Optional, Tuple, List

class MoRThinker(nn.Module):
    def __init__(self, shared_block: nn.Module, max_recursion: int = 10, d_model: int = 1024):
        super().__init__()
        self.shared_block = shared_block
        self.max_recursion = max_recursion
        self.halting_head = nn.Linear(d_model, 1)

    def forward(self, x, trigger_prob, state: Optional[torch.Tensor] = None):
        """Supports state passing for recursive thinking in streaming."""
        B, L, D = x.shape
        p = torch.zeros(B, L, self.max_recursion, device=x.device)
        cum_p = torch.zeros(B, L, device=x.device)
        y = torch.zeros_like(x)
        hidden = x
        curr_state = state

        for n in range(self.max_recursion):
            # The shared block (FluxBlock) now returns (output, next_state)
            hidden, curr_state = self.shared_block(hidden, curr_state)
            lambda_n = torch.sigmoid(self.halting_head(hidden)).squeeze(-1)
            # Bias halting by LNN trigger
            lambda_n = lambda_n * trigger_prob.squeeze(-1)

            p_n = (1.0 - cum_p) if n == self.max_recursion - 1 else lambda_n * (1.0 - cum_p)
            p[:, :, n] = p_n
            cum_p += p_n
            y += p_n.unsqueeze(-1) * hidden
        return y, p, curr_state
