import torch
import torch.nn as nn
from ncps.torch import LTC

class LiquidDetector(nn.Module):
    """
    Liquid LTC Layer:
    - Acts as a state monitor for Mamba-3 hidden states.
    - Maps discrete state to continuous-time ODE.
    - Saliency Trigger: magnitude of LTC state crosses threshold to trigger recursion.
    """
    def __init__(self, d_model: int, units: int = 128, threshold: float = 0.5):
        super().__init__()
        self.d_model = d_model
        self.ltc = LTC(d_model, units)
        self.threshold = threshold
        self.trigger_gate = nn.Linear(units, 1)

    def forward(self, x: torch.Tensor):
        # x: (Batch, Seq, D) from Mamba-3
        # ncps LTC expects (Seq, Batch, D) or (Batch, Seq, D) depending on config
        # Default LTC is (Batch, Seq, D)

        ltc_out, _ = self.ltc(x) # (B, L, units)

        # Saliency Trigger Logic:
        # Map the continuous hidden state to a trigger probability
        # We use a learned projection of the magnitude
        trigger_logits = self.trigger_gate(ltc_out) # (B, L, 1)
        trigger_prob = torch.sigmoid(trigger_logits)

        # Binary trigger for MoR activation (can use Gumbel-Softmax in training)
        is_hard_token = (trigger_prob > self.threshold).float()

        return ltc_out, trigger_prob, is_hard_token
