import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import ECGDataset


def _make_loaders(train_df, val_df, feature_cols, signals_path, batch_size, num_workers, idx_col, label_col):
    labels = train_df[label_col].astype(int).to_numpy()
    pos, neg = max(int(labels.sum()), 1), max(int((labels == 0).sum()), 1)
    weights = np.where(labels == 1, 0.5 / pos, 0.5 / neg)
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(labels), replacement=True)
    kw = dict(num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    train_loader = DataLoader(ECGDataset(train_df, feature_cols, signals_path, idx_col=idx_col, label_col=label_col),
                              batch_size=batch_size, sampler=sampler, drop_last=True, **kw)
    val_loader = DataLoader(ECGDataset(val_df, feature_cols, signals_path, idx_col=idx_col, label_col=label_col),
                            batch_size=batch_size, shuffle=False, **kw)
    return train_loader, val_loader


def train_model(model_cls, train_df, val_df, feature_cols, signals_path, save_dir,
                model_kwargs=None, checkpoint_name=None,
                epochs=100, batch_size=64, lr=1e-4, num_workers=4,
                idx_col="norm_idx", label_col="chagas_label"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_loader, val_loader = _make_loaders(
        train_df, val_df, feature_cols, signals_path, batch_size, num_workers, idx_col, label_col)

    model = model_cls(n_tab_features=len(feature_cols), **(model_kwargs or {})).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    n_total = epochs * len(train_loader)
    n_warmup = 5 * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: min(s / max(n_warmup, 1), 0.5 * (1 + math.cos(math.pi * (s - n_warmup) / max(n_total - n_warmup, 1))))
        if s >= n_warmup else s / max(n_warmup, 1)
    )

    start_epoch = 1
    if checkpoint_name is not None:
        ckpt = torch.load(save_dir / checkpoint_name, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scaler.is_enabled() and ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        if ckpt.get("scheduler"):
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt["epoch"]) + 1

    train_losses, val_losses, val_aucs, best_auc = [], [], [], -1.0

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        s, n, skipped = 0.0, 0, 0
        for sig, tab, lab in train_loader:
            sig = torch.nan_to_num(sig.to(device, non_blocking=True), nan=0.0, posinf=0.0, neginf=0.0)
            tab = torch.nan_to_num(tab.to(device, non_blocking=True), nan=0.0, posinf=0.0, neginf=0.0)
            lab = lab.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                loss = nn.functional.binary_cross_entropy_with_logits(
                    model(sig, tab), lab * 0.9 + 0.05)
            if not torch.isfinite(loss):
                skipped += 1
                continue
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            s += loss.detach().item() * sig.size(0)
            n += sig.size(0)
        train_losses.append(s / max(n, 1))

        model.eval()
        s, n, probs, labs = 0.0, 0, [], []
        with torch.no_grad():
            for sig, tab, lab in val_loader:
                sig = torch.nan_to_num(sig.to(device, non_blocking=True), nan=0.0, posinf=0.0, neginf=0.0)
                tab = torch.nan_to_num(tab.to(device, non_blocking=True), nan=0.0, posinf=0.0, neginf=0.0)
                lab = lab.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    out = model(sig, tab)
                    loss = nn.functional.binary_cross_entropy_with_logits(out, lab)
                if torch.isfinite(loss):
                    s += loss.detach().item() * sig.size(0)
                    n += sig.size(0)
                probs.append(torch.sigmoid(out.float()).cpu().numpy().ravel())
                labs.append(lab.cpu().numpy().ravel())

        val_losses.append(s / max(n, 1))
        probs = np.concatenate(probs)
        labs = np.concatenate(labs)
        mask = np.isfinite(probs) & np.isfinite(labs)
        val_auc = float(roc_auc_score(labs[mask], probs[mask])) if len(np.unique(labs[mask])) > 1 else float("nan")
        val_aucs.append(val_auc)

        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(),
                 "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                 "epoch": epoch, "val_auc": val_auc}
        torch.save(state, save_dir / f"epoch_{epoch:03d}.pt")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_auc": val_auc, "feature_cols": list(feature_cols)},
                       save_dir / "best.pt")

        print(f"epoch {epoch:03d} | train={train_losses[-1]:.4f} | val={val_losses[-1]:.4f} | "
              f"auc={val_auc:.4f} | best={best_auc:.4f} | lr={scheduler.get_last_lr()[0]:.2e} | skipped={skipped}")

    return {"train_losses": train_losses, "val_losses": val_losses,
            "val_aucs": val_aucs, "best_auc": best_auc, "model": model}


def model_predict(model, df, feature_cols, signals_path,
                  batch_size=64, num_workers=4, idx_col="norm_idx", label_col="chagas_label"):
    device = next(model.parameters()).device
    loader = DataLoader(ECGDataset(df, feature_cols, signals_path, idx_col=idx_col, label_col=label_col),
                        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    model.eval()
    probs, labs = [], []
    with torch.no_grad():
        for sig, tab, lab in loader:
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                out = model(sig.to(device, non_blocking=True), tab.to(device, non_blocking=True))
            probs.append(torch.sigmoid(out.float()).cpu().numpy().ravel())
            labs.append(lab.numpy().ravel())
    return np.concatenate(probs), np.concatenate(labs).astype(int)
