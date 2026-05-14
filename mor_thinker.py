import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class MoRWrapper(nn.Module):
    """
    Mixture-of-Recursions (MoR):
    - Uses a recursive stack of shared layers.
    - Governed by PonderNet halting probability logic.
    - Easy tokens exit early, hard tokens recurse up to max_recursion steps.
    """
    def __init__(self, shared_block: nn.Module, max_recursion: int = 10, d_model: int = 1024):
        super().__init__()
        self.shared_block = shared_block
        self.max_recursion = max_recursion
        self.d_model = d_model

        # PonderNet Halting Logit Generator
        self.halting_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, trigger_prob: torch.Tensor):
        # x: (B, L, D)
        B, L, D = x.shape

        # Initialize
        p = torch.zeros(B, L, self.max_recursion, device=x.device) # Halting probabilities
        cum_p = torch.zeros(B, L, device=x.device)
        y = torch.zeros_like(x)

        hidden = x

        # In a real MoR, the trigger_prob from LNN initializes or biases the halting
        for n in range(self.max_recursion):
            # 1. Shared layer pass (Mamba-3 + RWKV-7)
            hidden = self.shared_block(hidden)

            # 2. Halting probability calculation (PonderNet)
            # We bias the halting by the LNN trigger: hard tokens (low trigger_prob for 'easy')
            # are forced to recurse more.
            # lambda_n: probability of halting at step n
            lambda_n = torch.sigmoid(self.halting_head(hidden)).squeeze(-1) # (B, L)

            # Adjust lambda_n for the last step
            if n == self.max_recursion - 1:
                p_n = 1.0 - cum_p
            else:
                p_n = lambda_n * (1.0 - cum_p)

            p[:, :, n] = p_n
            cum_p += p_n

            # 3. Accumulate output weighted by halting probability
            y += p_n.unsqueeze(-1) * hidden

            # Optimization: could break if cum_p is close to 1 for all tokens
            # but for parallel processing on GPU, we usually run the max steps.

        return y, p # Return output and halting distribution for loss
