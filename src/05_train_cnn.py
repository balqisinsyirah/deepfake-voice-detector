"""Step 5: Train a simple CNN baseline on log-Mel spectrograms.

The model is intentionally small: four convolution blocks, global average pooling,
and a two-class linear head. It uses the speaker-disjoint split from Step 4,
training-only normalization statistics, class weights, light SpecAugment, and early
stopping on validation ROC-AUC.

Usage:
    python src/05_train_cnn.py --smoke-test
    python src/05_train_cnn.py
    python src/05_train_cnn.py --epochs 20 --batch-size 32

Outputs:
    outputs/models/05_cnn_baseline.pt
    outputs/logs/05_cnn_history.csv
    outputs/logs/05_cnn_metrics.json
    outputs/logs/05_cnn_val_predictions.csv
    outputs/figures/05_cnn_training_curves.png
    outputs/figures/05_cnn_confusion_matrix.png
    outputs/figures/05_cnn_roc_curve.png
    outputs/figures/05_cnn_probability_distribution.png
"""

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEL_DIR = PROJECT_ROOT / "data" / "processed" / "mel"
FEATURES_PATH = MEL_DIR / "mel_spectrograms.npy"
LABELS_PATH = MEL_DIR / "labels.npy"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
TRAIN_CSV = SPLIT_DIR / "train.csv"
VAL_CSV = SPLIT_DIR / "val.csv"
TRAIN_STATS_PATH = SPLIT_DIR / "train_stats.json"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"

CLASS_NAMES = ["Real", "Fake"]
COLORS = {0: "#2A9D8F", 1: "#E76F51"}
PALETTE = {"Real": COLORS[0], "Fake": COLORS[1]}

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.unicode_minus": False,
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


class MelDataset(Dataset):
    """Read normalized log-Mel features and optionally apply SpecAugment."""

    def __init__(
        self,
        features_path: Path,
        labels_path: Path,
        indices: np.ndarray,
        mean: float,
        std: float,
        augment: bool = False,
        seed: int = 42,
    ):
        self.features = np.load(features_path, mmap_mode="r")
        self.labels = np.load(labels_path)
        self.indices = np.asarray(indices)
        self.mean = float(mean)
        self.std = float(std) if std > 0 else 1.0
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.indices)

    def _spec_augment(self, x: np.ndarray) -> np.ndarray:
        """Mask a few random time and frequency bands (in standardized space)."""
        n_freq, n_time = x.shape
        for _ in range(2):
            width = int(self.rng.integers(1, 11))
            start = int(self.rng.integers(0, max(n_time - width, 1)))
            x[:, start : start + width] = 0.0
        for _ in range(2):
            width = int(self.rng.integers(1, 11))
            start = int(self.rng.integers(0, max(n_freq - width, 1)))
            x[start : start + width, :] = 0.0
        return x

    def __getitem__(self, position: int):
        row = int(self.indices[position])
        x = np.array(self.features[row], dtype=np.float32, copy=True)
        x = (x - self.mean) / self.std
        if self.augment:
            x = self._spec_augment(x)
        x = torch.from_numpy(x).unsqueeze(0)
        y = torch.tensor(int(self.labels[row]), dtype=torch.long)
        return x, y


class SimpleCNN(nn.Module):
    """A compact four-block CNN baseline for (1, 128, 126) log-Mel inputs."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.4):
        super().__init__()

        def block(in_channels: int, out_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
            )

        self.features = nn.Sequential(
            block(1, 32),
            block(32, 64),
            block(64, 128),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Give each DataLoader worker a deterministic independent RNG."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    info = torch.utils.data.get_worker_info()
    if info is not None:
        info.dataset.rng = np.random.default_rng(worker_seed)


def build_loaders(batch_size: int, workers: int, seed: int):
    train_meta = pd.read_csv(TRAIN_CSV)
    val_meta = pd.read_csv(VAL_CSV)
    stats = json.loads(TRAIN_STATS_PATH.read_text(encoding="utf-8"))
    mean = float(stats["feature_mean"])
    std = float(stats["feature_std"])

    train_indices = np.sort(train_meta["row_index"].to_numpy())
    val_indices = np.sort(val_meta["row_index"].to_numpy())

    train_dataset = MelDataset(
        FEATURES_PATH, LABELS_PATH, train_indices, mean, std, augment=True, seed=seed
    )
    val_dataset = MelDataset(
        FEATURES_PATH, LABELS_PATH, val_indices, mean, std, augment=False, seed=seed
    )

    common = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        **common,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    return train_loader, val_loader, train_meta, val_meta, stats


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    losses = []
    probs = []
    labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            losses.append(float(loss.item()) * len(y))
            prob = torch.softmax(logits, dim=1)[:, 1]
            probs.append(prob.cpu().numpy())
            labels.append(y.cpu().numpy())

    probs = np.concatenate(probs)
    labels = np.concatenate(labels)
    preds = (probs >= 0.5).astype(int)
    metrics = {
        "loss": float(np.sum(losses) / len(labels)),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probs)),
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1]).tolist(),
    }
    return metrics, probs, labels


def plot_training_curves(history: pd.DataFrame, best_epoch: int, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    axes[0].plot(history["epoch"], history["train_loss"], marker="o", label="Train loss")
    axes[0].plot(history["epoch"], history["val_loss"], marker="o", label="Validation loss")
    axes[0].axvline(best_epoch, color="grey", ls="--", lw=1, label=f"Best epoch {best_epoch}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss curves")
    axes[0].legend()

    axes[1].plot(history["epoch"], history["val_f1"], marker="o", label="Validation F1")
    axes[1].plot(history["epoch"], history["val_roc_auc"], marker="o", label="Validation ROC-AUC")
    axes[1].axvline(best_epoch, color="grey", ls="--", lw=1, label=f"Best epoch {best_epoch}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("Validation metrics")
    axes[1].legend()

    fig.suptitle("Simple CNN baseline: training history", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, output_path: Path) -> None:
    row_sums = cm.sum(axis=1, keepdims=True)
    percentages = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100
    annotations = np.array(
        [[f"{cm[i, j]:,}\n({percentages[i, j]:.1f}%)" for j in range(2)] for i in range(2)]
    )
    fig, ax = plt.subplots(figsize=(5.8, 5))
    sns.heatmap(
        cm,
        annot=annotations,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Simple CNN baseline: validation confusion matrix")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_roc(labels: np.ndarray, probs: np.ndarray, auc: float, output_path: Path) -> None:
    fpr, tpr, _ = roc_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(5.8, 5))
    ax.plot(fpr, tpr, color=COLORS[1], lw=2, label=f"CNN baseline (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Simple CNN baseline: validation ROC curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_probabilities(labels: np.ndarray, probs: np.ndarray, output_path: Path) -> None:
    plot_df = pd.DataFrame(
        {
            "Probability of fake": probs,
            "Class": [CLASS_NAMES[int(label)] for label in labels],
        }
    )
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.histplot(
        data=plot_df,
        x="Probability of fake",
        hue="Class",
        hue_order=["Real", "Fake"],
        palette=PALETTE,
        bins=50,
        stat="density",
        common_norm=False,
        element="step",
        fill=True,
        alpha=0.35,
        ax=ax,
    )
    threshold_line = ax.axvline(0.5, color="black", ls="--", lw=1, label="Decision threshold 0.5")
    ax.set_xlabel("Predicted probability of fake")
    ax.set_ylabel("Density")
    ax.set_title("Validation probability distribution")
    # histplot's hue legend is replaced below, so rebuild the class entries by hand
    class_handles = [
        Patch(facecolor=PALETTE[name], edgecolor=PALETTE[name], alpha=0.35, label=name)
        for name in CLASS_NAMES
    ]
    ax.legend(handles=class_handles + [threshold_line], loc="upper center", ncol=3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true", help="Run one batch and exit")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing outputs")
    args = parser.parse_args()

    seed_everything(args.seed)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = LOG_DIR / "05_cnn_metrics.json"
    if metrics_path.exists() and not args.overwrite and not args.smoke_test:
        raise SystemExit(f"{metrics_path} already exists. Use --overwrite to retrain.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader, val_loader, train_meta, val_meta, stats = build_loaders(
        args.batch_size, args.workers, args.seed
    )
    model = SimpleCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
    print(f"Train batches: {len(train_loader):,}; validation batches: {len(val_loader):,}")

    if args.smoke_test:
        xb, yb = next(iter(train_loader))
        xb = xb.to(device)
        logits = model(xb)
        print(f"Smoke test input: {tuple(xb.shape)} | logits: {tuple(logits.shape)}")
        print(f"Smoke test labels: {torch.bincount(yb, minlength=2).tolist()}")
        return

    class_weights = torch.tensor(stats["class_weights"], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    train_indices = np.sort(train_meta["row_index"].to_numpy())
    val_indices = np.sort(val_meta["row_index"].to_numpy())
    history = []
    best_auc = -1.0
    best_epoch = 0
    best_metrics = None
    best_probs = None
    best_labels = None
    epochs_without_improvement = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item()) * len(y)
            train_count += len(y)

        train_loss /= train_count
        val_metrics, val_probs, val_labels = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_metrics["roc_auc"])
        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_roc_auc": val_metrics["roc_auc"],
            "lr": current_lr,
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | acc={val_metrics['accuracy']:.4f} | "
            f"f1={val_metrics['f1']:.4f} | auc={val_metrics['roc_auc']:.4f} | lr={current_lr:.2e}"
        )

        if val_metrics["roc_auc"] > best_auc + 1e-4:
            best_auc = val_metrics["roc_auc"]
            best_epoch = epoch
            best_metrics = val_metrics
            best_probs = val_probs
            best_labels = val_labels
            epochs_without_improvement = 0
            torch.save(
                {
                    "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "config": {
                        "architecture": "SimpleCNN",
                        "input_shape": [1, 128, 126],
                        "num_classes": 2,
                        "dropout": 0.4,
                        "feature_mean": stats["feature_mean"],
                        "feature_std": stats["feature_std"],
                        "class_weights": stats["class_weights"],
                        "class_names": CLASS_NAMES,
                        "seed": args.seed,
                        "batch_size": args.batch_size,
                        "epoch": epoch,
                    },
                },
                MODEL_DIR / "05_cnn_baseline.pt",
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs).")
            break

    elapsed = time.time() - start_time
    history_df = pd.DataFrame(history)
    history_df.to_csv(LOG_DIR / "05_cnn_history.csv", index=False)

    val_lookup = val_meta.set_index("row_index").loc[val_indices]
    predictions = pd.DataFrame(
        {
            "row_index": val_indices,
            "file": val_lookup["file"].to_numpy(),
            "speaker": val_lookup["speaker"].to_numpy(),
            "label": val_lookup["label"].to_numpy(),
            "label_id": val_lookup["label_id"].to_numpy(),
            "prob_fake": best_probs,
            "pred_label": (best_probs >= 0.5).astype(int),
        }
    )
    predictions["pred_class"] = predictions["pred_label"].map({0: "Real", 1: "Fake"})
    predictions.to_csv(LOG_DIR / "05_cnn_val_predictions.csv", index=False)

    result = {
        "model": "SimpleCNN",
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "elapsed_sec": round(elapsed, 2),
        "trainable_parameters": n_params,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "feature_mean": stats["feature_mean"],
        "feature_std": stats["feature_std"],
        "class_weights": stats["class_weights"],
        "val_metrics": best_metrics,
    }
    (LOG_DIR / "05_cnn_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    plot_training_curves(history_df, best_epoch, FIG_DIR / "05_cnn_training_curves.png")
    plot_confusion_matrix(
        np.array(best_metrics["confusion_matrix"]), FIG_DIR / "05_cnn_confusion_matrix.png"
    )
    plot_roc(best_labels, best_probs, best_metrics["roc_auc"], FIG_DIR / "05_cnn_roc_curve.png")
    plot_probabilities(best_labels, best_probs, FIG_DIR / "05_cnn_probability_distribution.png")

    print("\nBest epoch:", best_epoch)
    print(json.dumps(best_metrics, indent=2))
    print(f"Elapsed: {elapsed/60:.1f} min")
    print(f"Model: {MODEL_DIR / '05_cnn_baseline.pt'}")
    print(f"Metrics: {LOG_DIR / '05_cnn_metrics.json'}")


if __name__ == "__main__":
    main()
