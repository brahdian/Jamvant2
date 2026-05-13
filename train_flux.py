import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from recursive_flux_engine import RecursiveFluxEngine, stability_loss

class FluxTrainer:
    def __init__(
        self,
        model: RecursiveFluxEngine,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        max_norm: float = 1.0
    ):
        self.model = model
        self.max_norm = max_norm
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95)
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100000, eta_min=1e-5
        )
        self.scaler = GradScaler()

    def train_step(self, x, y):
        self.model.train()
        self.optimizer.zero_grad()

        with autocast():
            logits = self.model(x)
            ce_loss = nn.CrossEntropyLoss()(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )
            # Accessing trigger_prob for regularization
            reg_loss = stability_loss(self.model.last_trigger_prob)
            total_loss = ce_loss + 0.01 * reg_loss

        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        return total_loss.item(), grad_norm.item()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RecursiveFluxEngine(d_model=512).to(device)
    trainer = FluxTrainer(model)
    print(f"Flux Engine Trainer active on {device}.")

    # Example Step
    dummy_x = torch.randint(0, 50000, (2, 32)).to(device)
    dummy_y = torch.randint(0, 50000, (2, 32)).to(device)
    loss, gnorm = trainer.train_step(dummy_x, dummy_y)
    print(f"Initial Loss: {loss:.4f}, GradNorm: {gnorm:.4f}")

if __name__ == "__main__":
    main()
