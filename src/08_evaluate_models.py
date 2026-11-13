"""Step 8: Compare all trained models on the validation split.

Metrics: Accuracy, Precision, Recall, Specificity, F1, ROC-AUC, and Equal Error
Rate (EER). EER is the point where the false positive rate equals the false
negative rate; lower is better.

Usage:
    python src/08_evaluate_models.py

Outputs:
    outputs/logs/08_model_comparison.csv
    outputs/logs/08_model_comparison.json
    outputs/logs/08_model_comparison.md
    outputs/logs/08_per_speaker_breakdown.csv
    outputs/figures/08_metric_comparison.png
    outputs/figures/08_roc_comparison.png
    outputs/figures/08_eer_curves.png
    outputs/figures/08_confusion_matrices.png
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import brentq
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"

MODEL_SPECS = [
    {
        "model": "Simple CNN",
        "predictions": "05_cnn_val_predictions.csv",
        "metrics": "05_cnn_metrics.json",
        "color": "#4C72B0",
    },
    {
        "model": "ResNet18",
        "predictions": "06_resnet18_val_predictions.csv",
        "metrics": "06_resnet18_metrics.json",
        "color": "#55A868",
    },
    {
        "model": "EfficientNet-B0",
        "predictions": "06_efficientnet_b0_val_predictions.csv",
        "metrics": "06_efficientnet_b0_metrics.json",
        "color": "#C44E52",
    },
    {
        "model": "CNN+BiGRU",
        "predictions": "07_cnn_bigru_val_predictions.csv",
        "metrics": "07_cnn_bigru_metrics.json",
        "color": "#8172B2",
    },
    {
        "model": "CNN+BiLSTM",
        "predictions": "07_cnn_bilstm_val_predictions.csv",
        "metrics": "07_cnn_bilstm_metrics.json",
        "color": "#CCB974",
    },
]

CLASS_NAMES = ["Real", "Fake"]
MIN_FAKE_PER_SPEAKER = 20  # ignore speakers with too few fake clips for a stable recall

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


def compute_eer(labels: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    """Return the equal error rate and the probability threshold nearest to it."""
    # roc_curve already returns FPR in non-decreasing order, so no re-sorting is needed
    # (an unstable sort would make ties in FPR pick an arbitrary FNR).
    fpr, tpr, thresholds = roc_curve(labels, probs)
    fnr = 1.0 - tpr

    unique_fpr, first_index = np.unique(fpr, return_index=True)
    unique_fnr = fnr[first_index]
    try:
        eer = float(brentq(lambda x: np.interp(x, unique_fpr, unique_fnr) - x, 0.0, 1.0))
    except ValueError:
        closest = int(np.argmin(np.abs(fpr - fnr)))
        eer = float((fpr[closest] + fnr[closest]) / 2.0)

    closest = int(np.argmin(np.abs(fpr - eer)))
    threshold = float(thresholds[closest])
    return eer, threshold


def evaluate_predictions(pred_df: pd.DataFrame) -> dict:
    labels = pred_df["label_id"].to_numpy()
    probs = pred_df["prob_fake"].to_numpy()
    preds = (probs >= 0.5).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    eer, eer_threshold = compute_eer(labels, probs)

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "specificity": float(specificity),
        "balanced_accuracy": float((specificity + recall_score(labels, preds, zero_division=0)) / 2.0),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probs)),
        "eer": float(eer),
        "eer_percent": float(eer * 100.0),
        "eer_threshold": float(eer_threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "confusion_matrix": cm.tolist(),
    }


def per_speaker_breakdown(pred_df: pd.DataFrame, model: str) -> pd.DataFrame:
    """Per-speaker error counts, fake recall, and real specificity at threshold 0.5."""
    df = pred_df.copy()
    df["pred"] = (df["prob_fake"] >= 0.5).astype(int)
    rows = []
    for speaker, g in df.groupby("speaker"):
        fake = g[g.label_id == 1]
        real = g[g.label_id == 0]
        fn = int((fake.pred == 0).sum())
        fp = int((real.pred == 1).sum())
        rows.append(
            {
                "model": model,
                "speaker": speaker,
                "n_real": len(real),
                "n_fake": len(fake),
                "fn": fn,
                "fp": fp,
                "fake_recall": (1 - fn / len(fake)) if len(fake) else np.nan,
                "real_specificity": (1 - fp / len(real)) if len(real) else np.nan,
                "mean_prob_fake_real": float(real.prob_fake.mean()) if len(real) else np.nan,
                "mean_prob_fake_fake": float(fake.prob_fake.mean()) if len(fake) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_metric_comparison(df: pd.DataFrame, path: Path) -> None:
    plot_df = df.sort_values("eer", ascending=True).reset_index(drop=True)
    x = np.arange(len(plot_df))
    color_map = {spec["model"]: spec["color"] for spec in MODEL_SPECS}

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1.5, 1]})

    heat = plot_df.set_index("model")[["accuracy", "f1", "roc_auc"]].T
    heat.index = ["Accuracy", "F1", "ROC-AUC"]
    sns.heatmap(
        heat,
        annot=True,
        fmt=".4f",
        cmap="YlGnBu",
        vmin=0.975,
        vmax=1.0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Score (color scale 0.975-1.0)"},
        ax=axes[0],
    )
    axes[0].set_title("Validation metrics (higher is better)")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("")
    axes[0].tick_params(axis="x", rotation=20)

    bars = axes[1].bar(
        x, plot_df["eer_percent"], color=[color_map[name] for name in plot_df["model"]], width=0.65
    )
    for bar in bars:
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.03,
            f"{bar.get_height():.3f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(plot_df["model"], rotation=20, ha="right")
    axes[1].set_ylabel("EER (%)")
    axes[1].set_title("Equal Error Rate (lower is better)")
    axes[1].set_ylim(0, max(plot_df["eer_percent"].max() * 1.25, 0.5))

    fig.suptitle("Validation comparison of all models", fontsize=14)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_roc_comparison(curves: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for spec in MODEL_SPECS:
        data = curves[spec["model"]]
        label = f"{spec['model']} (AUC={data['auc']:.4f}, EER={data['eer']*100:.2f}%)"
        axes[0].plot(data["fpr"], data["tpr"], color=spec["color"], lw=1.8, label=label)
        axes[1].plot(data["fpr"], data["tpr"], color=spec["color"], lw=1.8)

    axes[0].plot([0, 1], [0, 1], color="grey", ls="--", lw=1, label="Chance")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.01)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("Full ROC curve")
    axes[0].legend(loc="lower right", fontsize=7)

    axes[1].set_xlim(0, 0.08)
    axes[1].set_ylim(0.92, 1.001)
    axes[1].set_xlabel("False positive rate")
    axes[1].set_ylabel("True positive rate")
    axes[1].set_title("Zoomed near the operating region")
    fig.suptitle("ROC curves on the speaker-disjoint validation set", fontsize=14)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_eer_curves(curves: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for spec in MODEL_SPECS:
        data = curves[spec["model"]]
        ax.plot(data["fpr"], data["fnr"], color=spec["color"], lw=1.8,
                label=f"{spec['model']} (EER={data['eer']*100:.2f}%)")
        ax.scatter(
            [data["eer"]],
            [data["eer"]],
            color=spec["color"],
            s=55,
            edgecolor="black",
            linewidth=0.7,
            zorder=5,
        )

    ax.plot([0, 0.08], [0, 0.08], color="grey", ls="--", lw=1, label="FPR = FNR")
    ax.set_xlim(0, 0.08)
    ax.set_ylim(0, 0.08)
    ax.set_xlabel("False positive rate (FPR)")
    ax.set_ylabel("False negative rate (FNR)")
    ax.set_title("FPR vs FNR (DET-style, linear axes): EER is where FPR = FNR")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_confusion_matrices(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for ax, (_, row) in zip(axes.ravel(), df.iterrows()):
        cm = np.array(row["confusion_matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            ax=ax,
        )
        ax.set_title(f"{row['model']}\nAcc={row['accuracy']:.4f}, EER={row['eer_percent']:.2f}%")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    for ax in axes.ravel()[len(df) :]:
        ax.axis("off")
    fig.suptitle("Validation confusion matrices (threshold = 0.5)", fontsize=14)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_markdown(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Step 8 model comparison",
        "",
        "All metrics are computed on the same speaker-disjoint validation split.",
        "EER is the equal error rate; lower is better.",
        "The validation split was also used for early stopping and model selection, so these numbers are mildly optimistic.",
        "",
        "| Model | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | EER (%) | EER threshold | Hardest speaker (fake recall) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['model']} | {row['accuracy']:.4f} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['specificity']:.4f} | {row['f1']:.4f} | "
            f"{row['roc_auc']:.4f} | {row['eer_percent']:.3f} | {row['eer_threshold']:.4f} | "
            f"{row['hardest_speaker']} ({row['hardest_speaker_fake_recall']:.1%}) |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    speaker_frames = []
    curves = {}
    reference_labels = None
    reference_indices = None

    for spec in MODEL_SPECS:
        pred_path = LOG_DIR / spec["predictions"]
        metrics_path = LOG_DIR / spec["metrics"]
        if not pred_path.exists() or not metrics_path.exists():
            raise FileNotFoundError(f"Missing predictions or metrics for {spec['model']}")

        pred_df = pd.read_csv(pred_path)
        labels = pred_df["label_id"].to_numpy()
        probs = pred_df["prob_fake"].to_numpy()
        indices = pred_df["row_index"].to_numpy()

        if reference_labels is None:
            reference_labels = labels
            reference_indices = indices
        else:
            if not np.array_equal(indices, reference_indices):
                raise ValueError(f"Validation row indices differ for {spec['model']}")
            if not np.array_equal(labels, reference_labels):
                raise ValueError(f"Validation labels differ for {spec['model']}")

        result = evaluate_predictions(pred_df)
        speaker_df = per_speaker_breakdown(pred_df, spec["model"])
        speaker_frames.append(speaker_df)
        # Hardest speaker = lowest fake recall among speakers with enough fake clips
        eligible = speaker_df[speaker_df.n_fake >= MIN_FAKE_PER_SPEAKER]
        worst = eligible.sort_values(["fake_recall", "fn"], ascending=[True, False]).iloc[0]
        metric_json = json.loads(metrics_path.read_text(encoding="utf-8"))
        result.update(
            {
                "mean_prob_fake_real": float(pred_df.loc[pred_df.label_id == 0, "prob_fake"].mean()),
                "mean_prob_fake_fake": float(pred_df.loc[pred_df.label_id == 1, "prob_fake"].mean()),
                "hardest_speaker": worst["speaker"],
                "hardest_speaker_fake_recall": float(worst["fake_recall"]),
                "hardest_speaker_fn": int(worst["fn"]),
                "model": spec["model"],
                "trainable_parameters": metric_json.get("trainable_parameters"),
                "best_epoch": metric_json.get("best_epoch"),
                "train_minutes": round(metric_json.get("elapsed_sec", 0.0) / 60.0, 2),
            }
        )
        rows.append(result)

        fpr, tpr, _ = roc_curve(labels, probs)
        curves[spec["model"]] = {
            "fpr": fpr,
            "tpr": tpr,
            "fnr": 1.0 - tpr,
            "auc": result["roc_auc"],
            "eer": result["eer"],
        }

    df = pd.DataFrame(rows)
    df["eer_rank"] = df["eer"].rank(method="min").astype(int)
    df = df[
        [
            "model",
            "accuracy",
            "precision",
            "recall",
            "specificity",
            "balanced_accuracy",
            "f1",
            "roc_auc",
            "eer",
            "eer_percent",
            "eer_threshold",
            "eer_rank",
            "mean_prob_fake_real",
            "mean_prob_fake_fake",
            "hardest_speaker",
            "hardest_speaker_fake_recall",
            "hardest_speaker_fn",
            "tp",
            "fp",
            "tn",
            "fn",
            "confusion_matrix",
            "trainable_parameters",
            "best_epoch",
            "train_minutes",
        ]
    ]
    df.to_csv(LOG_DIR / "08_model_comparison.csv", index=False)
    (LOG_DIR / "08_model_comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_markdown(df, LOG_DIR / "08_model_comparison.md")

    speakers = pd.concat(speaker_frames, ignore_index=True)
    speakers.to_csv(LOG_DIR / "08_per_speaker_breakdown.csv", index=False)

    print(df[["model", "accuracy", "precision", "recall", "f1", "roc_auc", "eer_percent", "eer_rank"]].round(4).to_string(index=False))
    print("\nHardest speaker per model (lowest fake recall, >= "
          f"{MIN_FAKE_PER_SPEAKER} fake clips):")
    print(df[["model", "hardest_speaker", "hardest_speaker_fake_recall", "hardest_speaker_fn", "fn",
              "mean_prob_fake_real", "mean_prob_fake_fake"]].round(4).to_string(index=False))
    print("\nFake recall by validation speaker:")
    print(speakers.pivot(index="speaker", columns="model", values="fake_recall")
          [[spec["model"] for spec in MODEL_SPECS]].round(3).to_string())

    plot_metric_comparison(df, FIG_DIR / "08_metric_comparison.png")
    plot_roc_comparison(curves, FIG_DIR / "08_roc_comparison.png")
    plot_eer_curves(curves, FIG_DIR / "08_eer_curves.png")
    plot_confusion_matrices(df, FIG_DIR / "08_confusion_matrices.png")

    print(f"\nComparison CSV: {LOG_DIR / '08_model_comparison.csv'}")
    print(f"Comparison Markdown: {LOG_DIR / '08_model_comparison.md'}")
    print(f"Figures: {FIG_DIR}")


if __name__ == "__main__":
    main()
