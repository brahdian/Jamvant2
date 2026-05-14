import torch
import torch.nn as nn
from mamba3_mimo import Mamba3MIMOCore
from liquid_detector import LiquidDetector
from mor_thinker import MoRWrapper
from rwkv7_headqk import RWKV7HeadQK

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

class RecursiveFluxBlock(nn.Module):
    """Shared block for MoR: Mamba-3 + RWKV-7"""
    def __init__(self, d_model: int):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.mamba = Mamba3MIMOCore(d_model)
        self.norm2 = RMSNorm(d_model)
        self.rwkv = RWKV7HeadQK(d_model)

    def forward(self, x: torch.Tensor):
        x = x + self.mamba(self.norm1(x))
        x = x + self.rwkv(self.norm2(x))
        return x

def stability_loss(p_dist: torch.Tensor):
    """
    PonderNet stability loss to encourage early halting and sparsity.
    Calculates KL-Divergence between the halting distribution and a Geometric distribution.
    """
    # Simplified version: penalize long recursion
    steps = torch.arange(p_dist.size(-1), device=p_dist.device).float() + 1
    expected_steps = torch.sum(p_dist * steps, dim=-1)
    return torch.mean(expected_steps)

class RecursiveFluxEngine(nn.Module):
    """
    Recursive Flux Engine (3B Architecture):
    - Streamer: Mamba-3 MIMO
    - Detector: Liquid LTC
    - Thinker: MoR + PonderNet
    - Recall: RWKV-7 HeadQK
    """
    def __init__(self, d_model: int = 2560, n_layers: int = 32, vocab_size: int = 50257):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Ingestion (Streamer)
        self.ingestion = Mamba3MIMOCore(d_model)

        # Detector (Liquid LTC)
        self.detector = LiquidDetector(d_model)

        # Thinker (MoR Wrapper with Shared Block)
        self.shared_block = RecursiveFluxBlock(d_model)
        self.thinker = MoRWrapper(self.shared_block, max_recursion=8, d_model=d_model)

        # Output Decoder (Mamba-3)
        self.decoder = Mamba3MIMOCore(d_model)

        self.final_norm = RMSNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)
        self.output_head.weight = self.embedding.weight

        # Attribute to store the last trigger probability from LNN
        self.last_trigger_prob = None

    def forward(self, input_ids: torch.Tensor):
        x = self.embedding(input_ids)

        # 1. Ingestion
        x = self.ingestion(x)

        # 2. Semantic Trigger Detection
        _, trigger_prob, _ = self.detector(x)
        self.last_trigger_prob = trigger_prob

        # 3. Recursive Thinking
        x, p_dist = self.thinker(x, trigger_prob)

        # 4. Decoder & Output
        x = self.decoder(x)
        logits = self.output_head(self.final_norm(x))

        return logits, p_dist

def get_model_size(model):
    return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    model = RecursiveFluxEngine(d_model=512, n_layers=4)
    print(f"Recursive Flux Engine Architecture Verified.")
    print(f"Total Parameters: {get_model_size(model) / 1e6:.2f}M")
