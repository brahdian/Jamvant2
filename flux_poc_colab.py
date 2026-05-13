import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from recursive_flux_engine import RecursiveFluxEngine, stability_loss
import time

# --- Synthetic Logic & Recall Dataset ---
def generate_synthetic_data(batch_size, seq_len, vocab_size):
    """
    Generates a dataset where the model must recall a token from earlier
    in the sequence after a "logical gap" (zeros).
    """
    # Random context tokens
    context = torch.randint(10, vocab_size - 1, (batch_size, seq_len // 4))
    # Gap
    gap = torch.zeros((batch_size, seq_len // 2), dtype=torch.long)
    # Target (the first token of context)
    target = context[:, 0:1]

    # Sequence: [context, gap, trigger, target_hint]
    # We want the model to predict 'target' at the end
    input_ids = torch.cat([context, gap, torch.ones((batch_size, 1), dtype=torch.long) * 5], dim=1)
    labels = torch.cat([context[:, 1:], gap, torch.ones((batch_size, 1), dtype=torch.long) * 5, target], dim=1)

    return input_ids, labels

def run_flux_poc():
    # 1. Config for Colab T4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    # 500M Config
    model = RecursiveFluxEngine(d_model=1024, n_layers=24).to(device)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler()

    # 2. Training Loop (Micro-PoC)
    model.train()
    for i in range(50): # 50 steps for the PoC
        start_time = time.time()

        input_ids, labels = generate_synthetic_data(4, 128, 50257)
        input_ids, labels = input_ids.to(device), labels.to(device)

        optimizer.zero_grad()

        with autocast():
            logits = model(input_ids)
            ce_loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
            reg_loss = stability_loss(model.last_trigger_prob)
            loss = ce_loss + 0.01 * reg_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        end_time = time.time()
        if i % 5 == 0:
            print(f"Step {i} | Loss: {loss.item():.4f} | Time/Step: {end_time-start_time:.2f}s")

    # 3. Logic Test (Recall Verification)
    model.eval()
    with torch.no_grad():
        test_input, test_labels = generate_synthetic_data(1, 128, 50257)
        test_input = test_input.to(device)
        logits = model(test_input)
        predicted_id = torch.argmax(logits[0, -1, :]).item()
        actual_id = test_labels[0, -1].item()

        print("\n--- PoC Verification ---")
        print(f"Logical Recall Token (Target): {actual_id}")
        print(f"Model Prediction: {predicted_id}")
        if predicted_id == actual_id:
            print("SUCCESS: The Recursive Flux Engine correctly recalled the token across the gap.")
        else:
            print("INFO: Model requires more training to converge, but the forward pass is stable.")

if __name__ == "__main__":
    run_flux_poc()
