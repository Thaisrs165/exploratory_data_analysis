"""The project's network: a configurable MLP binary classifier.

Output is 1 unit, returning logits (no sigmoid applied inside the model) so
that BCEWithLogitsLoss's fused, numerically-stable log-sigmoid is used, as
required for binary classification. Class imbalance (11.7% positive) is
handled with pos_weight = n_negative / n_positive in the training fold, not
by resampling - it changes the loss's gradient, not the data the network sees,
so validation batches don't need matching treatment.
"""
import copy

import numpy as np
import torch
import torch.nn as nn

from classification_common import classification_metrics


class MLPClassifier(nn.Module):
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
        return self.net(x).squeeze(-1)  # logits

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def train_model(n_features, hidden_sizes, X_train, y_train, X_val, y_val, *,
                 dropout=0.0, batchnorm=False, lr=1e-3, weight_decay=0.0,
                 batch_size=256, max_epochs=200, patience=15, seed=42):
    """Adam + BCEWithLogitsLoss(pos_weight=...), early stopping on validation
    PR-AUC (threshold-independent, appropriate for 11.7% positive data), best
    weights restored. Returns (model, history)."""
    torch.manual_seed(seed)
    model = MLPClassifier(n_features, hidden_sizes, dropout=dropout, batchnorm=batchnorm)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - n_pos)
    pos_weight = torch.tensor(n_neg / n_pos, dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)
    Xv = torch.tensor(X_val, dtype=torch.float32)

    n = Xt.shape[0]
    history = {"train_loss": [], "val_loss": [], "val_pr_auc": []}
    best_val_pr_auc = -np.inf
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    rng = np.random.default_rng(seed)
    val_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    yv_t = torch.tensor(y_val, dtype=torch.float32)

    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = rng.permutation(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = Xt[idx], yt[idx]
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        history["train_loss"].append(epoch_loss / n)

        model.eval()
        with torch.no_grad():
            val_logits = model(Xv)
            val_loss = val_loss_fn(val_logits, yv_t).item()
            val_proba = torch.sigmoid(val_logits).numpy()
            val_pr_auc = classification_metrics(y_val, val_proba, threshold=0.5)["pr_auc"]
        history["val_loss"].append(val_loss)
        history["val_pr_auc"].append(val_pr_auc)

        if val_pr_auc > best_val_pr_auc + 1e-6:
            best_val_pr_auc = val_pr_auc
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
    history["best_val_pr_auc"] = best_val_pr_auc
    history["pos_weight"] = float(pos_weight.item())
    return model, history


def predict_proba(model, X) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        return torch.sigmoid(logits).numpy()
