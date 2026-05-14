import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict
from modules.mamba3 import Mamba3MIMOCore
from modules.rwkv7 import RWKV7HeadQK
from modules.detector import LiquidDetector
from modules.mor import MoRThinker

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

class RecursiveFluxBlock(nn.Module):
    """Shared block for MoR: Mamba-3 + RWKV-7 with state-passing."""
    def __init__(self, d_model: int):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.mamba = Mamba3MIMOCore(d_model)
        self.norm2 = RMSNorm(d_model)
        self.rwkv = RWKV7HeadQK(d_model, n_head=8, head_size=d_model // 8)

    def forward(self, x: torch.Tensor, state: Optional[Tuple] = None):
        s1, s2 = state if state is not None else (None, None)

        m_out, s1_next = self.mamba(self.norm1(x), s1)
        x = x + m_out

        r_out, s2_next = self.rwkv(self.norm2(x), s2)
        x = x + r_out

        return x, (s1_next, s2_next)

class RecursiveFluxEngine(nn.Module):
    def __init__(self, d_model: int = 2560, n_layers: int = 32, vocab_size: int = 50257):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.ingestion = Mamba3MIMOCore(d_model)
        self.detector = LiquidDetector(d_model)
        self.shared_block = RecursiveFluxBlock(d_model)
        self.thinker = MoRThinker(self.shared_block, max_recursion=8, d_model=d_model)
        self.decoder = Mamba3MIMOCore(d_model)
        self.final_norm = RMSNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)
        self.output_head.weight = self.embedding.weight
        self.last_trigger_prob = None

    def forward(self, input_ids: torch.Tensor, states: Optional[Dict] = None):
        x = self.embedding(input_ids)
        s = states if states is not None else {}

        x, s_ing = self.ingestion(x, s.get('ing'))
        ltc_res, trigger_prob, s_det = self.detector(x, s.get('det'))
        self.last_trigger_prob = trigger_prob

        x, p_dist, s_think = self.thinker(x + ltc_res, trigger_prob, s.get('think'))

        x, s_dec = self.decoder(x, s.get('dec'))
        logits = self.output_head(self.final_norm(x))

        next_states = {
            'ing': s_ing,
            'det': s_det,
            'think': s_think,
            'dec': s_dec
        }
        return logits, p_dist, next_states

def stability_loss(p_dist: torch.Tensor):
    steps = torch.arange(p_dist.size(-1), device=p_dist.device).float() + 1
    return torch.mean(torch.sum(p_dist * steps, dim=-1))
