import torch
import torch.nn as nn
import numpy as np
from src.model import Titanic

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_model(
    X_train_t,
    y_train_t,
    X_val_t=None,
    y_val_t=None,
    *,
    epochs: int = 600,
    weight_decay: float = 1e-4,
    seed: int = 33,
    depth: int = 2,
    lr: float = 3e-4,
    hidden: int = 32,
    dropout: float = 0.3,
    log_every: int = 100,
):
    set_seed(seed)

    input_dim = X_train_t.shape[1]
    model = Titanic(input_dim=input_dim, hidden=hidden, depth=depth, dropout=dropout)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    history = []
    for epoch in range(epochs + 1):
        model.train()
        logits = model(X_train_t)
        loss = loss_fn(logits, y_train_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (X_val_t is not None) and (y_val_t is not None) and (epoch % log_every == 0):
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_t)
                val_loss = loss_fn(val_logits, y_val_t).item()
            history.append((epoch, float(loss.item()), float(val_loss)))

            gap = (val_loss - float(loss.item())) / (float(loss.item()) + 1e-12) * 100
            print(
                f"EPOCHS = {epoch:4d} | "
                f"Train Loss = {loss.item():.4f} | "
                f"Val Loss = {val_loss:.4f} | "
                f"Gap = {gap:5.2f} %"
            )

    return model, history