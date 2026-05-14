import torch
import torch.nn as nn
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
    def __init__(self, d_model: int):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.mamba = Mamba3MIMOCore(d_model)
        self.norm2 = RMSNorm(d_model)
        self.rwkv = RWKV7HeadQK(d_model)
    def forward(self, x):
        x = x + self.mamba(self.norm1(x))
        x = x + self.rwkv(self.norm2(x))
        return x

def stability_loss(p_dist: torch.Tensor):
    steps = torch.arange(p_dist.size(-1), device=p_dist.device).float() + 1
    return torch.mean(torch.sum(p_dist * steps, dim=-1))

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

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.ingestion(x)
        _, trigger_prob = self.detector(x)
        self.last_trigger_prob = trigger_prob
        x, p_dist = self.thinker(x, trigger_prob)
        x = self.decoder(x)
        return self.output_head(self.final_norm(x)), p_dist

if __name__ == "__main__":
    model = RecursiveFluxEngine(d_model=512, n_layers=4)
    print("Modular Recursive Flux Engine Initialized.")
