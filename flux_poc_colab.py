import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from recursive_flux_engine import RecursiveFluxEngine, stability_loss
import time

def generate_synthetic_data(batch_size, seq_len, vocab_size):
    context = torch.randint(10, vocab_size - 1, (batch_size, seq_len // 4))
    gap = torch.zeros((batch_size, seq_len // 2), dtype=torch.long)
    target = context[:, 0:1]
    input_ids = torch.cat([context, gap, torch.ones((batch_size, 1), dtype=torch.long) * 5], dim=1)
    labels = torch.cat([context[:, 1:], gap, torch.ones((batch_size, 1), dtype=torch.long) * 5, target], dim=1)
    return input_ids, labels

def run_flux_poc():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    # POC scale: 8*64 = 512 d_model
    model = RecursiveFluxEngine(d_model=512, n_layers=12).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler()

    model.train()
    for i in range(10):
        input_ids, labels = generate_synthetic_data(4, 32, 50257)
        input_ids, labels = input_ids.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast():
            # Correctly handle 3-tuple return
            logits, p_dist, _ = model(input_ids)
            ce_loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss = ce_loss + 0.01 * stability_loss(p_dist)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        print(f"Step {i} | Loss: {loss.item():.4f}")

if __name__ == "__main__":
    run_flux_poc()
