"""The project's network: a configurable MLP regressor trained on sqrt(rented_bike_count),
with early stopping on validation RMSE (original scale), restoring the best weights.
"""
import copy

import numpy as np
import torch
import torch.nn as nn

from regression_common import regression_metrics, sqrt_inverse


class MLPRegressor(nn.Module):
    """1 linear output unit; sqrt-scale target, trained with MSE on that scale."""

    def __init__(self, n_features: int, hidden_sizes: list, dropout: float = 0.0, batchnorm: bool = False):
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def train_model(n_features, hidden_sizes, X_train, y_train, X_val, y_val, *,
                 dropout=0.0, batchnorm=False, lr=1e-3, weight_decay=0.0,
                 batch_size=128, max_epochs=300, patience=20, seed=42):
    """Full training loop: Adam, early stopping on validation RMSE (original
    scale), best weights restored. Returns (model, history) where history has
    per-epoch train/val loss (sqrt-scale MSE) and val RMSE (original scale)."""
    torch.manual_seed(seed)
    model = MLPRegressor(n_features, hidden_sizes, dropout=dropout, batchnorm=batchnorm)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(np.sqrt(y_train), dtype=torch.float32)
    Xv = torch.tensor(X_val, dtype=torch.float32)
    yv_sqrt = torch.tensor(np.sqrt(y_val), dtype=torch.float32)

    n = Xt.shape[0]
    history = {"train_loss": [], "val_loss": [], "val_rmse": []}
    best_val_rmse = np.inf
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    rng = np.random.default_rng(seed)

    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = rng.permutation(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = Xt[idx], yt[idx]
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        history["train_loss"].append(epoch_loss / n)

        model.eval()
        with torch.no_grad():
            val_pred_sqrt = model(Xv)
            val_loss = loss_fn(val_pred_sqrt, yv_sqrt).item()
            val_rmse = regression_metrics(y_val, sqrt_inverse(val_pred_sqrt.numpy()))["rmse"]
        history["val_loss"].append(val_loss)
        history["val_rmse"].append(val_rmse)

        if val_rmse < best_val_rmse - 1e-6:
            best_val_rmse = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    history["stopped_epoch"] = epoch
    history["best_val_rmse"] = best_val_rmse
    return model, history


def predict(model, X) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred_sqrt = model(torch.tensor(X, dtype=torch.float32)).numpy()
    return sqrt_inverse(pred_sqrt)
