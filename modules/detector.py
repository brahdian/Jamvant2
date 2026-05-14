import torch
import torch.nn as nn
from ncps.torch import LTC
from typing import Optional, Tuple

class LiquidDetector(nn.Module):
    def __init__(self, d_model: int, units: int = 128):
        super().__init__()
        self.ltc = LTC(d_model, units)
        self.trigger_gate = nn.Linear(units, 1)
        # Projection to match d_model for residual connection
        self.out_proj = nn.Linear(units, d_model)

    def forward(self, x, state: Optional[torch.Tensor] = None):
        """Supports state passing for continuous-time streaming."""
        ltc_out, h_next = self.ltc(x, hx=state)
        trigger_prob = torch.sigmoid(self.trigger_gate(ltc_out))
        # Project ltc_out to d_model for residual
        ltc_res = self.out_proj(ltc_out)
        return ltc_res, trigger_prob, h_next
