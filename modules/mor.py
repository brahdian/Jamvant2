import torch
import torch.nn as nn

class MoRThinker(nn.Module):
    """
    Mixture-of-Recursions Wrapper.
    Adaptive depth via PonderNet halting logic.
    Integrated with LNN trigger probability.
    """
    def __init__(self, shared_block: nn.Module, max_recursion: int = 10, d_model: int = 1024):
        super().__init__()
        self.shared_block = shared_block
        self.max_recursion = max_recursion
        self.halting_head = nn.Linear(d_model, 1)

    def forward(self, x, trigger_prob):
        B, L, D = x.shape
        p = torch.zeros(B, L, self.max_recursion, device=x.device)
        cum_p = torch.zeros(B, L, device=x.device)
        y = torch.zeros_like(x)
        hidden = x

        for n in range(self.max_recursion):
            # Pass through shared thinking block (Mamba-3 + RWKV-7)
            hidden = self.shared_block(hidden)

            # PonderNet halting logic biased by LNN trigger_prob
            # trigger_prob comes from LTC state monitor
            lambda_n = torch.sigmoid(self.halting_head(hidden)).squeeze(-1)

            # Combine PonderNet halting with LNN saliency detection
            # If saliency is low (trigger_prob), we halt earlier
            lambda_n = lambda_n * trigger_prob.squeeze(-1)

            p_n = (1.0 - cum_p) if n == self.max_recursion - 1 else lambda_n * (1.0 - cum_p)
            p[:, :, n] = p_n
            cum_p += p_n

            y += p_n.unsqueeze(-1) * hidden

        return y, p
