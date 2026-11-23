"""Generic CNN training loop for whatever network a data type provides (D7)."""
import copy

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def train_cnn(model: nn.Module, X: np.ndarray, y: np.ndarray, seed: int, max_epochs: int = 30,
              patience: int = 5, batch_size: int = 64, lr: float = 1e-3) -> nn.Module:
    torch.manual_seed(seed)
    n_val = max(int(0.1 * len(y)), len(np.unique(y)))  # stratified split needs >= one sample per class
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=n_val, stratify=y, random_state=seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)), batch_size=batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(seed))
    X_va, y_va = torch.from_numpy(X_va), torch.from_numpy(y_va)
    opt, loss_fn = torch.optim.Adam(model.parameters(), lr=lr), nn.CrossEntropyLoss()
    best, best_state, bad = float("inf"), copy.deepcopy(model.state_dict()), 0
    for _ in range(max_epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(X_va), y_va).item()
        if v < best:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model.eval()


def cnn_proba(model: nn.Module, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        out = [torch.softmax(model(torch.from_numpy(X[i:i + batch_size])), 1) for i in range(0, len(X), batch_size)]
    return torch.cat(out).numpy()
