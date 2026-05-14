import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
from modules.mamba3 import Mamba3MIMOCore
from modules.rwkv7 import RWKV7HeadQK
from modules.detector import LiquidDetector
from modules.mor import MoRThinker

class StateBuffer:
    """Manages the pipelined states for State-Streaming (The Matios Fix)."""
    def __init__(self):
        self.mamba_state: Optional[torch.Tensor] = None
        self.lnn_state: Optional[torch.Tensor] = None
        self.mor_state: Optional[torch.Tensor] = None
        self.rwkv_state: Optional[torch.Tensor] = None

class StreamingFluxEngine(nn.Module):
    """
    Recursive Flux Engine with State-Streaming (The Matios Fix):
    Allows overlapping Ingestion (t+1) with Recursive Thinking (t).
    """
    def __init__(self, d_model: int = 2560, n_layers: int = 32, vocab_size: int = 50257):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.ingestion = Mamba3MIMOCore(d_model)
        self.detector = LiquidDetector(d_model)

        # Shared Thinking Block
        self.rwkv = RWKV7HeadQK(d_model)
        self.mamba_inner = Mamba3MIMOCore(d_model)

        # Thinking Wrapper
        self.thinker = MoRThinker(self.shared_thinking_step, max_recursion=8, d_model=d_model)

        self.decoder = Mamba3MIMOCore(d_model)
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)

    def shared_thinking_step(self, x, state=None):
        # x: chunk output from ingestion
        # state is managed by MoRThinker internally or passed here
        x = x + self.mamba_inner(x)
        x = self.rwkv(x, state)
        return x

    def forward_pipelined(self, chunks: List[torch.Tensor]):
        """
        Implements State-Streaming:
        Chunk t+1 Ingestion overlaps with Chunk t Thinking.
        """
        B, L, D = chunks[0].shape
        buffer = StateBuffer()
        outputs = []

        # Initial chunk
        x_t = self.embedding(chunks[0])
        # Stage 1: Ingestion t
        x_ingested_t = self.ingestion(x_t)
        _, trigger_t = self.detector(x_ingested_t)

        for i in range(1, len(chunks)):
            # --- OVERLAP START ---
            # 1. Start Ingestion for Chunk i (t+1)
            # In a real production setup, this would be on a separate CUDA stream
            x_next = self.embedding(chunks[i])
            x_ingested_next = self.ingestion(x_next)
            _, trigger_next = self.detector(x_ingested_next)

            # 2. Parallel Thinking for Chunk i-1 (t)
            # MoR is "thinking" about the previous chunk
            x_thought_prev, _ = self.thinker(x_ingested_t, trigger_t)

            # --- OVERLAP END ---

            # Decode and Store output for i-1
            out_prev = self.output_head(self.decoder(x_thought_prev))
            outputs.append(out_prev)

            # Shift pipeline
            x_ingested_t = x_ingested_next
            trigger_t = trigger_next

        # Process the final chunk's thinking
        x_thought_final, _ = self.thinker(x_ingested_t, trigger_t)
        outputs.append(self.output_head(self.decoder(x_thought_final)))

        return torch.stack(outputs) # (Chunks, B, L, Vocab)
