import torch
import torch.nn as nn
from ncps.torch import LTC

class LiquidDetector(nn.Module):
    """
    Liquid LTC state monitor (Official ncps-based).
    Maps Mamba hidden states to continuous-time ODE for intent detection.
    """
    def __init__(self, d_model: int, units: int = 128):
        super().__init__()
        self.ltc = LTC(d_model, units)
        self.trigger_gate = nn.Linear(units, 1)

    def forward(self, x):
        ltc_out, _ = self.ltc(x)
        trigger_prob = torch.sigmoid(self.trigger_gate(ltc_out))
        return ltc_out, trigger_prob
