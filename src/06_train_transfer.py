"""Step 6: Fine-tune ImageNet-pretrained ResNet18 and EfficientNet-B0.

The 1-channel log-Mel is mapped to [0, 1], resized to 224x224, repeated to three
channels, and normalized with ImageNet statistics. The pretrained backbone is
fine-tuned with a smaller learning rate than the new classification head.

Usage:
    python src/06_train_transfer.py --model resnet18 --smoke-test
    python src/06_train_transfer.py --model resnet18
    python src/06_train_transfer.py --model efficientnet_b0
    python src/06_train_transfer.py --model both

Outputs:
    outputs/models/06_{model}_best.pt
    outputs/logs/06_{model}_history.csv
    outputs/logs/06_{model}_metrics.json
    outputs/logs/06_{model}_val_predictions.csv
    outputs/figures/06_{model}_training_curves.png
    outputs/figures/06_{model}_confusion_matrix.png
    outputs/figures/06_{model}_roc_curve.png
    outputs/figures/06_{model}_probability_distribution.png
"""

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
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
from torchvision.models import EfficientNet_B0_Weights, ResNet18_Weights

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

IMAGE_SIZE = 224
MEL_MIN_DB = -80.0
CLASS_NAMES = ["Real", "Fake"]
COLORS = {0: "#2A9D8F", 1: "#E76F51"}
PALETTE = {"Real": COLORS[0], "Fake": COLORS[1]}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

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


class TransferMelDataset(Dataset):
    """Log-Mel -> [0,1] -> 224x224 -> 3 channels -> ImageNet normalization."""

    def __init__(
        self,
        features_path: Path,
        labels_path: Path,
        indices: np.ndarray,
        augment: bool = False,
        seed: int = 42,
    ):
        self.features = np.load(features_path, mmap_mode="r")
        self.labels = np.load(labels_path)
        self.indices = np.asarray(indices)
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.indices)

    def _spec_augment(self, x: torch.Tensor) -> torch.Tensor:
        """Mask random time/frequency bands after ImageNet normalization.

        Masked cells are set to 0, i.e. the ImageNet channel mean (not the data mean).
        """
        _, n_freq, n_time = x.shape
        for _ in range(2):
            width = int(self.rng.integers(1, 17))
            start = int(self.rng.integers(0, max(n_time - width, 1)))
            x[:, :, start : start + width] = 0.0
        for _ in range(2):
            width = int(self.rng.integers(1, 17))
            start = int(self.rng.integers(0, max(n_freq - width, 1)))
            x[:, start : start + width, :] = 0.0
        return x

    def __getitem__(self, position: int):
        row = int(self.indices[position])
        mel = np.array(self.features[row], dtype=np.float32, copy=True)
        # Saved values are dB in [-80, 0]; map to [0, 1] for ImageNet-style input
        x01 = np.clip((mel - MEL_MIN_DB) / (-MEL_MIN_DB), 0.0, 1.0)
        x = torch.from_numpy(x01).unsqueeze(0)
        x = F.interpolate(
            x.unsqueeze(0), size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False
        ).squeeze(0)
        x = x.repeat(3, 1, 1)
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        if self.augment:
            x = self._spec_augment(x)
        y = torch.tensor(int(self.labels[row]), dtype=torch.long)
        return x, y


def build_model(name: str, pretrained: bool = True):
    """Build a two-class model and return backbone/head parameter groups."""
    if name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = torchvision.models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 2)
        head_modules = [model.fc]
    elif name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = torchvision.models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        head_modules = [model.classifier]
    else:
        raise ValueError(f"Unknown model: {name}")

    head_param_ids = {id(p) for module in head_modules for p in module.parameters()}
    head_params = [p for p in model.parameters() if id(p) in head_param_ids]
    backbone_params = [
        p for p in model.parameters() if id(p) not in head_param_ids and p.requires_grad
    ]
    return model, backbone_params, head_params


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    info = torch.utils.data.get_worker_info()
    if info is not None:
        info.dataset.rng = np.random.default_rng(worker_seed)


def build_loaders(batch_size: int, workers: int, seed: int):
    train_meta = pd.read_csv(TRAIN_CSV)
    val_meta = pd.read_csv(VAL_CSV)
    train_indices = np.sort(train_meta["row_index"].to_numpy())
    val_indices = np.sort(val_meta["row_index"].to_numpy())

    train_dataset = TransferMelDataset(FEATURES_PATH, LABELS_PATH, train_indices, True, seed)
    val_dataset = TransferMelDataset(FEATURES_PATH, LABELS_PATH, val_indices, False, seed)

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
    return train_loader, val_loader, train_meta, val_meta


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
            probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
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


def plot_training_curves(history: pd.DataFrame, best_epoch: int, model_name: str, path: Path) -> None:
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

    fig.suptitle(f"{model_name}: training history", fontsize=14)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, model_name: str, path: Path) -> None:
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
    ax.set_title(f"{model_name}: validation confusion matrix")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_roc(labels: np.ndarray, probs: np.ndarray, auc: float, model_name: str, path: Path) -> None:
    fpr, tpr, _ = roc_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(5.8, 5))
    ax.plot(fpr, tpr, color=COLORS[1], lw=2, label=f"{model_name} (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{model_name}: validation ROC curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_probabilities(labels: np.ndarray, probs: np.ndarray, model_name: str, path: Path) -> None:
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
    ax.axvline(0.5, color="black", ls="--", lw=1, label="Decision threshold 0.5")
    ax.set_xlabel("Predicted probability of fake")
    ax.set_ylabel("Density")
    ax.set_title(f"{model_name}: validation probability distribution")
    ax.legend(loc="upper center")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run_training(model_name: str, args, device: torch.device) -> dict:
    print(f"\n=== Training {model_name} ===")
    train_loader, val_loader, train_meta, val_meta = build_loaders(
        args.batch_size, args.workers, args.seed
    )
    model, backbone_params, head_params = build_model(model_name, pretrained=True)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    stats = json.loads(TRAIN_STATS_PATH.read_text(encoding="utf-8"))
    class_weights = torch.tensor(stats["class_weights"], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr_backbone},
            {"params": head_params, "lr": args.lr_head},
        ],
        weight_decay=args.weight_decay,
    )
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
        current_lr_backbone = optimizer.param_groups[0]["lr"]
        current_lr_head = optimizer.param_groups[1]["lr"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "val_roc_auc": val_metrics["roc_auc"],
                "lr_backbone": current_lr_backbone,
                "lr_head": current_lr_head,
            }
        )
        print(
            f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | acc={val_metrics['accuracy']:.4f} | "
            f"f1={val_metrics['f1']:.4f} | auc={val_metrics['roc_auc']:.4f} | "
            f"lr={current_lr_backbone:.2e}/{current_lr_head:.2e}"
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
                        "architecture": model_name,
                        "pretrained": True,
                        "input_shape": [3, IMAGE_SIZE, IMAGE_SIZE],
                        "num_classes": 2,
                        "class_weights": stats["class_weights"],
                        "class_names": CLASS_NAMES,
                        "seed": args.seed,
                        "batch_size": args.batch_size,
                        "epoch": epoch,
                    },
                },
                MODEL_DIR / f"06_{model_name}_best.pt",
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    elapsed = time.time() - start_time
    history_df = pd.DataFrame(history)
    history_df.to_csv(LOG_DIR / f"06_{model_name}_history.csv", index=False)

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
    predictions.to_csv(LOG_DIR / f"06_{model_name}_val_predictions.csv", index=False)

    result = {
        "model": model_name,
        "pretrained": True,
        "image_size": IMAGE_SIZE,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "elapsed_sec": round(elapsed, 2),
        "trainable_parameters": n_params,
        "batch_size": args.batch_size,
        "lr_backbone": args.lr_backbone,
        "lr_head": args.lr_head,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "class_weights": stats["class_weights"],
        "val_metrics": best_metrics,
    }
    (LOG_DIR / f"06_{model_name}_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    plot_training_curves(history_df, best_epoch, model_name, FIG_DIR / f"06_{model_name}_training_curves.png")
    plot_confusion_matrix(
        np.array(best_metrics["confusion_matrix"]), model_name, FIG_DIR / f"06_{model_name}_confusion_matrix.png"
    )
    plot_roc(best_labels, best_probs, best_metrics["roc_auc"], model_name, FIG_DIR / f"06_{model_name}_roc_curve.png")
    plot_probabilities(
        best_labels, best_probs, model_name, FIG_DIR / f"06_{model_name}_probability_distribution.png"
    )

    print(f"Best epoch: {best_epoch}")
    print(json.dumps(best_metrics, indent=2))
    print(f"Elapsed: {elapsed/60:.1f} min")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["resnet18", "efficientnet_b0", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-backbone", type=float, default=1e-4)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model_names = ["resnet18", "efficientnet_b0"] if args.model == "both" else [args.model]
    if args.smoke_test:
        train_loader, _, _, _ = build_loaders(args.batch_size, args.workers, args.seed)
        xb, yb = next(iter(train_loader))
        print(f"Smoke input: {tuple(xb.shape)} | labels: {torch.bincount(yb, minlength=2).tolist()}")
        for name in model_names:
            model, _, _ = build_model(name, pretrained=True)
            model = model.to(device)
            with torch.no_grad():
                logits = model(xb.to(device))
            print(f"{name}: logits {tuple(logits.shape)}")
            del model, logits
            torch.cuda.empty_cache()
        return

    for name in model_names:
        metrics_path = LOG_DIR / f"06_{name}_metrics.json"
        if metrics_path.exists() and not args.overwrite:
            raise SystemExit(f"{metrics_path} exists. Use --overwrite to retrain.")
        run_training(name, args, device)


if __name__ == "__main__":
    main()
