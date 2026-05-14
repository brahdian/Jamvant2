import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from recursive_flux_engine import RecursiveFluxEngine, stability_loss

class FluxTrainer:
    def __init__(self, model: RecursiveFluxEngine, lr: float = 3e-4):
        self.model = model
        self.optimizer = optim.AdamW(model.parameters(), lr=lr)
        self.scaler = GradScaler()

    def train_step(self, x, y):
        self.model.train()
        self.optimizer.zero_grad()
        with autocast():
            # Correct tuple unpacking
            logits, p_dist = self.model(x)
            ce_loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), y.view(-1))
            reg_loss = stability_loss(p_dist)
            total_loss = ce_loss + 0.01 * reg_loss
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return total_loss.item()
