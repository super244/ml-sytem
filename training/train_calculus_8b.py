import math

import torch
import torch.optim as optim
from models.transformer_8b import CalculusTransformer8B


def train_calculus_8b():
    print("Initializing Calculus 8B Pretraining Pipeline")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    # Model instantiation
    # Using scaled down parameters to allow the script to compile/run for testing
    model = CalculusTransformer8B(d_model=1024, n_heads=8, n_layers=8).to(device)
    model.train()

    # BFloat16 Mixed Precision
    use_amp = torch.cuda.is_available() or torch.backends.mps.is_available()
    scaler = (
        torch.cuda.amp.GradScaler(enabled=use_amp) if torch.cuda.is_available() else None
    )  # MPS handles mixed precision differently, assuming generic PyTorch amp

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)

    # Cosine scheduler with warmup
    total_steps = 10000
    warmup_steps = 1000

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Dummy Training Loop
    gradient_accumulation_steps = 4
    batch_size = 2
    seq_len = 1024

    print(f"Starting training on {device}...")
    for step in range(1, 11):  # Just run a few steps for demo
        optimizer.zero_dict() if hasattr(optimizer, "zero_dict") else optimizer.zero_grad()

        for _ in range(gradient_accumulation_steps):
            # Dummy data
            input_ids = torch.randint(0, 50257, (batch_size, seq_len), device=device)
            labels = torch.randint(0, 50257, (batch_size, seq_len), device=device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(input_ids)
                loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
                loss = loss / gradient_accumulation_steps

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        scheduler.step()

        if step % 2 == 0:
            print(
                f"Step {step} | Loss: {loss.item() * gradient_accumulation_steps:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}"
            )

    print("Training demo completed. Checkpoints would be saved here.")


if __name__ == "__main__":
    train_calculus_8b()
