"""Training pipeline: features -> standardize -> train -> calibrate -> save artifacts.

Splits are grouped by student so the model never sees the same author in both
train and test — the leakage that inflates most published detector numbers.
Saved artifacts include per-class feature statistics so scan-time reports can
cite concrete evidence, and a temperature + abstention thresholds so scores
are calibrated probabilities rather than raw network confidence.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch import nn

from ..features.extract import ALL_FEATURE_NAMES, FeatureExtractor
from .network import DetectorNet, mc_predict


def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))
    return records


def _fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=50)
    loss_fn = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = loss_fn(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().item())


def train(
    data_path: str | Path,
    out_dir: str | Path,
    epochs: int = 60,
    lr: float = 1e-3,
    seed: int = 7,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(data_path)
    codes = [r["code"] for r in records]
    labels = np.array([r["label"] for r in records])
    groups = np.array([r["student_id"] for r in records])

    # group-aware split: an author appears in exactly one split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    train_idx, rest_idx = next(gss.split(codes, labels, groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_rel, test_rel = next(gss2.split(
        [codes[i] for i in rest_idx], labels[rest_idx], groups[rest_idx]))
    val_idx, test_idx = rest_idx[val_rel], rest_idx[test_rel]

    extractor = FeatureExtractor().fit_lm([codes[i] for i in train_idx])
    X = extractor.extract_batch(codes)
    y = labels

    mean = X[train_idx].mean(axis=0)
    std = X[train_idx].std(axis=0) + 1e-8
    Xs = (X - mean) / std

    def tensors(idx):
        return (torch.tensor(Xs[idx], dtype=torch.float32),
                torch.tensor(y[idx], dtype=torch.long))

    Xtr, ytr = tensors(train_idx)
    Xva, yva = tensors(val_idx)
    Xte, yte = tensors(test_idx)

    model = DetectorNet(in_dim=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    best_val, best_state, patience = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(Xtr), ytr)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(Xva), yva))
        if val_loss < best_val - 1e-4:
            best_val, best_state, patience = val_loss, model.state_dict(), 0
        else:
            patience += 1
            if patience >= 12:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        temperature = _fit_temperature(model(Xva), yva)

    p_mean, p_std = mc_predict(model, Xte, temperature=temperature)
    preds = (p_mean >= 0.5).long().numpy()
    metrics = {
        "test_accuracy": float(accuracy_score(yte.numpy(), preds)),
        "test_auc": float(roc_auc_score(yte.numpy(), p_mean.numpy())),
        "temperature": temperature,
        "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
        "n_features": X.shape[1],
        "epochs_ran": epoch + 1,
    }

    # abstention accounting on the test split
    conclusive = ((p_mean >= 0.85) | (p_mean <= 0.15)) & (p_std <= 0.08)
    metrics["abstention_rate"] = float((~conclusive).float().mean())
    if conclusive.any():
        idx = conclusive.numpy()
        metrics["conclusive_accuracy"] = float(
            accuracy_score(yte.numpy()[idx], preds[idx]))

    # per-class feature stats for evidence reporting
    human_mask, ai_mask = y[train_idx] == 0, y[train_idx] == 1
    stats = {
        "feature_names": ALL_FEATURE_NAMES,
        "scaler_mean": mean.tolist(), "scaler_std": std.tolist(),
        "human_mean": X[train_idx][human_mask].mean(axis=0).tolist(),
        "human_std": (X[train_idx][human_mask].std(axis=0) + 1e-8).tolist(),
        "ai_mean": X[train_idx][ai_mask].mean(axis=0).tolist(),
        "temperature": temperature,
        "thresholds": {"ai": 0.85, "human": 0.15, "max_std": 0.08},
        "metrics": metrics,
    }

    torch.save({"state_dict": model.state_dict(), "in_dim": X.shape[1]},
               out_dir / "model.pt")
    with open(out_dir / "stats.json", "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    with open(out_dir / "lm.pkl", "wb") as fh:
        pickle.dump(extractor.lm, fh)

    return metrics


def load_artifacts(model_dir: str | Path):
    """Return (model, extractor, stats) ready for scanning."""
    model_dir = Path(model_dir)
    with open(model_dir / "stats.json", encoding="utf-8") as fh:
        stats = json.load(fh)
    with open(model_dir / "lm.pkl", "rb") as fh:
        lm = pickle.load(fh)
    ckpt = torch.load(model_dir / "model.pt", weights_only=False)
    model = DetectorNet(in_dim=ckpt["in_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, FeatureExtractor(lm=lm), stats
