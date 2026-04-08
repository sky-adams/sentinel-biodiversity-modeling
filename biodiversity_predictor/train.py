from pathlib import Path
import json
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
import plotly.express as px
import pandas as pd

from .dataset import SentinelBIITileDataset
from .model import BIIRegressor


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    x, y = zip(*batch)
    return torch.stack(x), torch.stack(y)


def finite_batch(x, y):
    x_mask = torch.isfinite(x).all(dim=(1, 2, 3))
    if y.ndim == 1:
        y_mask = torch.isfinite(y)
    else:
        y_mask = torch.isfinite(y).all(dim=1)
    mask = x_mask & y_mask
    return x[mask], y[mask]


def masked_metrics(targets, preds):
    targets = np.asarray(targets).reshape(-1)
    preds = np.asarray(preds).reshape(-1)
    mask = np.isfinite(targets) & np.isfinite(preds)
    targets = targets[mask]
    preds = preds[mask]
    if len(targets) == 0:
        return {"rmse": None, "mae": None, "r2": None, "n": 0}
    rmse = float(mean_squared_error(targets, preds) ** 0.5)
    mae = float(mean_absolute_error(targets, preds))
    r2 = float(r2_score(targets, preds)) if len(targets) > 1 else None
    return {"rmse": rmse, "mae": mae, "r2": r2, "n": int(len(targets))}


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    losses = []
    for batch in tqdm(loader, leave=False):
        if batch is None:
            continue
        x, y = batch
        x = x.to(device).float()
        y = y.to(device).float()

        x, y = finite_batch(x, y)
        if x.numel() == 0:
            continue

        pred = model(x).squeeze(-1)
        if y.ndim > 1:
            y = y.squeeze(-1)

        mask = torch.isfinite(pred) & torch.isfinite(y)
        if mask.sum() == 0:
            continue

        loss = loss_fn(pred[mask], y[mask])
        if torch.isnan(loss) or torch.isinf(loss):
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

    return float(np.mean(losses)) if losses else None


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    for batch in loader:
        if batch is None:
            continue
        x, y = batch
        x = x.to(device).float()
        y = y.to(device).float()

        x, y = finite_batch(x, y)
        if x.numel() == 0:
            continue

        pred = model(x).squeeze(-1)
        if y.ndim > 1:
            y = y.squeeze(-1)

        all_preds.append(pred.detach().cpu().numpy())
        all_targets.append(y.detach().cpu().numpy())

    preds = np.concatenate(all_preds, axis=0) if all_preds else np.array([])
    targets = np.concatenate(all_targets, axis=0) if all_targets else np.array([])
    return masked_metrics(targets, preds), targets, preds


def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    set_seed(42)
    config = {
        "tif_path": "data/santa_barbara_sentinel_bii.tif",
        "patch_size": 64,
        "stride": 64,
        "batch_size": 8,
        "epochs": 20,
        "lr": 1e-3,
        "nan_threshold": 0.10,
        "output_dir": "outputs",
    }

    out = Path(config["output_dir"])
    out.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = SentinelBIITileDataset(
        config["tif_path"],
        patch_size=config["patch_size"],
        stride=config["stride"],
        nan_threshold=config["nan_threshold"],
    )

    print(
        f"Dataset windows: total={ds.skip_stats['total']}, "
        f"kept={ds.skip_stats['kept']}, skipped={ds.skip_stats['skipped']}"
    )

    n = len(ds)
    idx = np.arange(n)
    rng = np.random.default_rng(42)
    rng.shuffle(idx)

    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    print(f"Train split indices: {len(train_idx)}")
    print(f"Val split indices: {len(val_idx)}")
    print(f"Test split indices: {len(test_idx)}")

    loader_kwargs = dict(batch_size=config["batch_size"], num_workers=0, collate_fn=collate_skip_none)

    train_loader = DataLoader(Subset(ds, train_idx), shuffle=True, **loader_kwargs)
    val_loader = DataLoader(Subset(ds, val_idx), shuffle=False, **loader_kwargs)
    test_loader = DataLoader(Subset(ds, test_idx), shuffle=False, **loader_kwargs)

    sample = ds[0]
    while sample is None:
        sample = ds[rng.integers(0, len(ds))]
    sample_x, _ = sample
    model = BIIRegressor(in_channels=sample_x.shape[0]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    best_path = out / "best_model.pt"
    history = []

    for epoch in range(config["epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics, _, _ = evaluate(model, val_loader, device)

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            **val_metrics,
        }
        history.append(row)

        if val_metrics["rmse"] is not None and val_metrics["rmse"] < best_val:
            best_val = val_metrics["rmse"]
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics, targets, preds = evaluate(model, test_loader, device)

    print(f"Target stats: min={float(np.min(targets)):.6f}, max={float(np.max(targets)):.6f}, mean={float(np.mean(targets)):.6f}, std={float(np.std(targets)):.6f}")
    print(f"Prediction stats: min={float(np.min(preds)):.6f}, max={float(np.max(preds)):.6f}, mean={float(np.mean(preds)):.6f}, std={float(np.std(preds)):.6f}")
    print("First 10 actual vs predicted:")
    for a, p in list(zip(targets[:10], preds[:10])):
        print(f"actual={float(a):.6f}, predicted={float(p):.6f}")

    with open(out / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    with open(out / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    df = pd.DataFrame({"actual": targets, "predicted": preds})
    fig = px.scatter(df, x="actual", y="predicted", title="Predicted vs Actual")
    fig.add_shape(
        type="line",
        x0=df["actual"].min(),
        y0=df["actual"].min(),
        x1=df["actual"].max(),
        y1=df["actual"].max(),
        line=dict(dash="dash", color="red"),
    )
    
    try:
        fig.write_image(str(out / "pred_vs_actual.png"))
    except Exception as e:
        print(e)
    print(test_metrics)


if __name__ == "__main__":
    main()