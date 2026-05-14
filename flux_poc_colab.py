import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from recursive_flux_engine import RecursiveFluxEngine, stability_loss
import time

# --- Synthetic Logic & Recall Dataset ---
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

    # POC scale
    model = RecursiveFluxEngine(d_model=512, n_layers=12).to(device)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler()

    model.train()
    for i in range(50):
        start_time = time.time()
        input_ids, labels = generate_synthetic_data(4, 128, 50257)
        input_ids, labels = input_ids.to(device), labels.to(device)

        optimizer.zero_grad()
        with autocast():
            # Correctly handle tuple return (logits, p_dist)
            logits, p_dist = model(input_ids)
            ce_loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
            # stability_loss on p_dist
            reg_loss = stability_loss(p_dist)
            loss = ce_loss + 0.01 * reg_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if i % 5 == 0:
            print(f"Step {i} | Loss: {loss.item():.4f} | Time: {time.time()-start_time:.2f}s")

    model.eval()
    with torch.no_grad():
        test_input, test_labels = generate_synthetic_data(1, 128, 50257)
        test_input = test_input.to(device)
        logits, _ = model(test_input)
        predicted_id = torch.argmax(logits[0, -1, :]).item()
        actual_id = test_labels[0, -1].item()
        print(f"\nTarget: {actual_id} | Pred: {predicted_id}")

if __name__ == "__main__":
    run_flux_poc()
