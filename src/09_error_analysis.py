"""Step 9: Error analysis across all trained models.

Questions answered:
  1. Do the five models make the same mistakes? (error overlap)
  2. Which validation speakers cause the errors?
  3. Are errors related to the shortcuts found in EDA / preprocessing
     (wrap-padding amount, original duration, original loudness)?
  4. Why is Adam Driver so hard? (loudness, duration, and spectral profile of his
     clips compared with the training set)
  5. How does the decision threshold trade missed fakes against false alarms?
  6. Which individual clips are the most confident mistakes? (for manual listening)

Inputs: validation predictions from Steps 5-7, split files from Step 4, and the
log-Mel features from Step 3. No model is retrained or re-run.

Usage:
    python src/09_error_analysis.py

Outputs:
    outputs/logs/09_error_summary.json
    outputs/logs/09_error_overlap.csv
    outputs/logs/09_error_overlap_by_model.csv
    outputs/logs/09_error_by_bin.csv
    outputs/logs/09_threshold_analysis.csv
    outputs/logs/09_speaker_profile.csv
    outputs/logs/09_hard_examples.csv
    outputs/figures/09_error_overlap.png
    outputs/figures/09_error_by_confounder.png
    outputs/figures/09_adam_driver_profile.png
    outputs/figures/09_threshold_tradeoff.png
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEL_DIR = PROJECT_ROOT / "data" / "processed" / "mel"
FEATURES_PATH = MEL_DIR / "mel_spectrograms.npy"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"

MODEL_SPECS = [
    ("Simple CNN", "05_cnn_val_predictions.csv", "#4C72B0"),
    ("ResNet18", "06_resnet18_val_predictions.csv", "#55A868"),
    ("EfficientNet-B0", "06_efficientnet_b0_val_predictions.csv", "#C44E52"),
    ("CNN+BiGRU", "07_cnn_bigru_val_predictions.csv", "#8172B2"),
    ("CNN+BiLSTM", "07_cnn_bilstm_val_predictions.csv", "#CCB974"),
]
MODEL_NAMES = [name for name, _, _ in MODEL_SPECS]
MODEL_COLORS = {name: color for name, _, color in MODEL_SPECS}
BEST_MODEL = "EfficientNet-B0"
HARD_SPEAKER = "Adam Driver"
THRESHOLD = 0.5
SEED = 42
PROFILE_SAMPLES = 1500  # clips per training class for the spectral profile
HIGH_BAND_START = 96  # Mel bands 96-127 are roughly the top of the 8 kHz range

CLASS_COLORS = {"Real": "#2A9D8F", "Fake": "#E76F51"}

BINS = {
    "pad_sec": ([-0.001, 0.0, 1.0, 2.0, 3.0, 4.0], ["0 (no pad)", "0-1 s", "1-2 s", "2-3 s", "3-4 s"]),
    "duration_original_sec": ([0, 2, 3, 4, 6, 1e9], ["<2 s", "2-3 s", "3-4 s", "4-6 s", ">6 s"]),
    "rms_before_dbfs": ([-200, -35, -30, -25, -20, 0], ["< -35", "-35 to -30", "-30 to -25", "-25 to -20", "> -20"]),
}
BIN_TITLES = {
    "pad_sec": "Wrap-padding added (s)",
    "duration_original_sec": "Original duration",
    "rms_before_dbfs": "Original loudness (dBFS, before normalization)",
}

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


# --------------------------------------------------------------------------- data


def load_predictions(val_meta: pd.DataFrame) -> pd.DataFrame:
    """Return one row per validation clip with every model's fake probability."""
    merged = val_meta.copy()
    for name, filename, _ in MODEL_SPECS:
        pred = pd.read_csv(LOG_DIR / filename)[["row_index", "label_id", "prob_fake"]]
        check = merged[["row_index", "label_id"]].merge(pred, on="row_index", how="left", suffixes=("", "_p"))
        if check["prob_fake"].isna().any() or not (check["label_id"] == check["label_id_p"]).all():
            raise ValueError(f"Predictions for {name} do not match the validation split")
        merged[f"prob_{name}"] = check["prob_fake"].to_numpy()
        merged[f"err_{name}"] = (merged[f"prob_{name}"] >= THRESHOLD).astype(int) != merged["label_id"]
    merged["n_models_wrong"] = merged[[f"err_{n}" for n in MODEL_NAMES]].sum(axis=1)
    return merged


# ------------------------------------------------------------------ 1. overlap


def error_overlap(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    table = (
        df.groupby(["class_name", "n_models_wrong"]).size().unstack(fill_value=0)
        .reindex(columns=range(len(MODEL_NAMES) + 1), fill_value=0)
    )
    any_wrong = df[df.n_models_wrong > 0]
    all_wrong = df[df.n_models_wrong == len(MODEL_NAMES)]
    info = {
        "clips_wrong_for_any_model": int(len(any_wrong)),
        "clips_wrong_for_all_models": int(len(all_wrong)),
        "all_wrong_by_class": all_wrong.class_name.value_counts().to_dict(),
        "all_wrong_by_speaker": all_wrong.speaker.value_counts().to_dict(),
        "any_wrong_by_speaker": any_wrong.speaker.value_counts().to_dict(),
    }
    return table, info


OVERLAP_GROUPS = [
    ("Only this model", lambda n: n == 1, "#F4A261"),
    ("Shared with 1-3 other models", lambda n: (n >= 2) & (n <= 4), "#E76F51"),
    ("Shared by all 5 models", lambda n: n == 5, "#6A040F"),
]


def per_model_overlap(df: pd.DataFrame) -> pd.DataFrame:
    """For each model, split its errors by how many models share them."""
    rows = []
    for name in MODEL_NAMES:
        wrong = df[df[f"err_{name}"]]
        row = {"model": name}
        for label, rule, _ in OVERLAP_GROUPS:
            row[label] = int(rule(wrong.n_models_wrong).sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def plot_error_overlap(table: pd.DataFrame, by_model: pd.DataFrame, note: str, path: Path) -> None:
    n_models = len(MODEL_NAMES)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.6), gridspec_kw={"width_ratios": [1, 1.1]})

    # (a) Clips grouped by how many models get them wrong
    ax = axes[0]
    data = table.drop(columns=0)
    x = np.arange(data.shape[1])
    width = 0.38
    for i, cls in enumerate(["Real", "Fake"]):
        values = data.loc[cls].to_numpy() if cls in data.index else np.zeros(len(x))
        bars = ax.bar(x + (i - 0.5) * width, values, width, color=CLASS_COLORS[cls], label=f"True {cls}")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1, str(int(value)), ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{int(c)} of {n_models}\nmodels wrong" if c < n_models else f"All {n_models}\nmodels wrong" for c in data.columns]
    )
    ax.set_xlabel("How many of the 5 models misclassify the clip (a count, not a model ID)")
    ax.set_ylabel("Number of validation clips")
    ax.set_title("(a) Clips grouped by how many models get them wrong")
    ax.legend()

    # (b) Each model's errors split by how widely they are shared
    ax = axes[1]
    y = np.arange(len(by_model))
    left = np.zeros(len(by_model))
    for label, _, color in OVERLAP_GROUPS:
        values = by_model[label].to_numpy()
        ax.barh(y, values, left=left, color=color, label=label)
        for yi, (lft, value) in enumerate(zip(left, values)):
            if value >= 4:
                ax.text(lft + value / 2, yi, str(int(value)), ha="center", va="center", color="white", fontsize=9)
        left += values
    for yi, total in enumerate(left):
        ax.text(total + 1.5, yi, f"{int(total)} errors", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(by_model.index)
    ax.invert_yaxis()
    ax.set_xlim(0, left.max() * 1.18)
    ax.set_xlabel("Number of misclassified validation clips (FN + FP)")
    ax.set_title("(b) Each model's errors: unique or shared with other models?")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=9, frameon=False)

    fig.suptitle("Do the five models make the same mistakes? (threshold 0.5)", fontsize=14)
    fig.text(
        0.5,
        -0.06,
        "Models: " + ", ".join(MODEL_NAMES) + ". " + note,
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------- 2/3. error by bin


def error_by_bins(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column, (edges, labels) in BINS.items():
        binned = pd.cut(df[column], bins=edges, labels=labels)
        for cls in ["Real", "Fake"]:
            for label in labels:
                mask = (binned == label) & (df.class_name == cls)
                n = int(mask.sum())
                for name in MODEL_NAMES:
                    errors = int(df.loc[mask, f"err_{name}"].sum())
                    rows.append(
                        {
                            "variable": column,
                            "bin": label,
                            "class_name": cls,
                            "model": name,
                            "n": n,
                            "errors": errors,
                            "error_rate": errors / n if n else np.nan,
                            "errors_excl_hard_speaker": int(
                                df.loc[mask & (df.speaker != HARD_SPEAKER), f"err_{name}"].sum()
                            ),
                            "n_excl_hard_speaker": int((mask & (df.speaker != HARD_SPEAKER)).sum()),
                        }
                    )
    out = pd.DataFrame(rows)
    out["error_rate_excl_hard_speaker"] = out.errors_excl_hard_speaker / out.n_excl_hard_speaker.replace(0, np.nan)
    return out


def plot_error_by_confounder(bins: pd.DataFrame, path: Path) -> None:
    variables = list(BINS)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), sharey="row")
    for col, variable in enumerate(variables):
        labels = BINS[variable][1]
        for row, cls in enumerate(["Fake", "Real"]):
            ax = axes[row, col]
            sub = bins[(bins.variable == variable) & (bins.class_name == cls)]
            for name in MODEL_NAMES:
                line = sub[sub.model == name].set_index("bin").reindex(labels)
                ax.plot(labels, line.error_rate * 100, marker="o", color=MODEL_COLORS[name], label=name)
            counts = sub[sub.model == MODEL_NAMES[0]].set_index("bin").reindex(labels)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels([f"{lab}\n(n={int(n)})" for lab, n in zip(labels, counts.n)], fontsize=8)
            if col == 0:
                ax.set_ylabel("Missed fakes (FNR, %)" if cls == "Fake" else "False alarms on real (FPR, %)")
            ax.set_title(f"{BIN_TITLES[variable]} - true {cls}", fontsize=10)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Validation error rate by padding, duration, and original loudness (threshold 0.5)", fontsize=14)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------ 4. hard speaker profile


def band_profile(features: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Mean log-Mel energy per band with each clip's overall mean removed (as in EDA)."""
    rows = np.sort(rows)
    profile = np.zeros(features.shape[1], dtype=np.float64)
    for start in range(0, len(rows), 500):
        block = np.asarray(features[rows[start : start + 500]], dtype=np.float64)
        per_band = block.mean(axis=2)
        per_band -= per_band.mean(axis=1, keepdims=True)
        profile += per_band.sum(axis=0)
    return profile / len(rows)


def speaker_profile(df: pd.DataFrame, train: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    features = np.load(FEATURES_PATH, mmap_mode="r")
    rng = np.random.default_rng(SEED)

    groups = {}
    for cls, label_id in [("Real", 0), ("Fake", 1)]:
        rows = train.loc[train.label_id == label_id, "row_index"].to_numpy()
        groups[f"Train {cls}"] = rng.choice(rows, size=min(PROFILE_SAMPLES, len(rows)), replace=False)
        hard = df[(df.speaker == HARD_SPEAKER) & (df.label_id == label_id)]
        groups[f"{HARD_SPEAKER} {cls}"] = hard.row_index.to_numpy()
        other = df[(df.speaker != HARD_SPEAKER) & (df.label_id == label_id)]
        groups[f"Other val {cls}"] = rng.choice(other.row_index.to_numpy(), size=min(PROFILE_SAMPLES, len(other)), replace=False)

    profiles = {name: band_profile(features, rows) for name, rows in groups.items()}

    def dist(a: str, b: str) -> float:
        return float(np.sqrt(np.mean((profiles[a] - profiles[b]) ** 2)))

    def high_band(name: str) -> float:
        return float(profiles[name][HIGH_BAND_START:].mean())

    hard_fake = f"{HARD_SPEAKER} Fake"
    spectral = {
        "profile_rmse_dB": {
            f"{hard_fake} vs Train Fake": dist(hard_fake, "Train Fake"),
            f"{hard_fake} vs Train Real": dist(hard_fake, "Train Real"),
            "Other val Fake vs Train Fake": dist("Other val Fake", "Train Fake"),
            "Other val Fake vs Train Real": dist("Other val Fake", "Train Real"),
        },
        "high_band_relative_energy_dB": {name: high_band(name) for name in profiles},
    }

    # Clip-level metadata comparison
    meta_all = pd.concat(
        [
            train.assign(group=np.where(train.label_id == 1, "Train Fake", "Train Real")),
            df.assign(
                group=np.where(
                    df.speaker == HARD_SPEAKER,
                    np.where(df.label_id == 1, f"{HARD_SPEAKER} Fake", f"{HARD_SPEAKER} Real"),
                    np.where(df.label_id == 1, "Other val Fake", "Other val Real"),
                )
            ),
        ],
        ignore_index=True,
    )
    table = (
        meta_all.groupby("group")
        .agg(
            n=("row_index", "size"),
            median_duration_sec=("duration_original_sec", "median"),
            share_padded=("pad_sec", lambda s: float((s > 0).mean())),
            median_rms_before_dbfs=("rms_before_dbfs", "median"),
        )
        .reindex(list(profiles))
    )
    table["high_band_relative_energy_dB"] = [high_band(name) for name in table.index]

    hard_fake_rms = df.loc[(df.speaker == HARD_SPEAKER) & (df.label_id == 1), "rms_before_dbfs"].median()
    train_fake = train[train.label_id == 1]
    spectral["train_fakes_louder_than_hard_speaker_fake_median"] = float(
        (train_fake.rms_before_dbfs >= hard_fake_rms).mean()
    )
    spectral["train_speakers_with_loud_fakes"] = (
        train_fake[train_fake.rms_before_dbfs >= hard_fake_rms].speaker.value_counts().head(5).to_dict()
    )
    return table.reset_index(), spectral, profiles


def plot_hard_speaker(df: pd.DataFrame, train: pd.DataFrame, profiles: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Original loudness of fake clips
    groups = {
        "Train Fake": train.loc[train.label_id == 1, "rms_before_dbfs"],
        "Train Real": train.loc[train.label_id == 0, "rms_before_dbfs"],
        f"{HARD_SPEAKER} Fake": df.loc[(df.speaker == HARD_SPEAKER) & (df.label_id == 1), "rms_before_dbfs"],
    }
    colors = {"Train Fake": "#E76F51", "Train Real": "#2A9D8F", f"{HARD_SPEAKER} Fake": "#6A040F"}
    for name, values in groups.items():
        sns.kdeplot(values, ax=axes[0], label=f"{name} (median {values.median():.1f})", color=colors[name], fill=True, alpha=0.2, common_norm=False)
    axes[0].set_xlabel("Original RMS loudness before normalization (dBFS)")
    axes[0].set_title("(a) Adam Driver's fake clips are much louder\nthan typical training fakes")
    axes[0].legend(fontsize=8)

    # (b) Spectral profile difference
    bands = np.arange(len(next(iter(profiles.values()))))
    ref = profiles["Train Fake"]
    for name, color, ls in [
        ("Train Real", "#2A9D8F", "-"),
        (f"{HARD_SPEAKER} Fake", "#6A040F", "-"),
        ("Other val Fake", "#E76F51", "--"),
    ]:
        axes[1].plot(bands, profiles[name] - ref, color=color, ls=ls, label=name)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].axvspan(HIGH_BAND_START, bands[-1], color="grey", alpha=0.12, label="High Mel bands")
    axes[1].set_xlabel("Mel band (low -> high frequency)")
    axes[1].set_ylabel("Relative energy minus Train Fake (dB)")
    axes[1].set_title("(b) Average spectral shape relative to training fakes")
    axes[1].legend(fontsize=8)

    # (c) Fake probability for Adam Driver's fake clips
    hard = df[(df.speaker == HARD_SPEAKER) & (df.label_id == 1)]
    for name in MODEL_NAMES:
        values = np.clip(hard[f"prob_{name}"].to_numpy(), 1e-6, 1)
        sns.ecdfplot(values, ax=axes[2], label=name, color=MODEL_COLORS[name], log_scale=True)
    axes[2].axvline(THRESHOLD, color="black", ls="--", lw=1, label="Threshold 0.5")
    axes[2].set_xlabel("Predicted probability of fake (log scale)")
    axes[2].set_ylabel("Share of Adam Driver fake clips")
    axes[2].set_title("(c) Most of his fakes get very low fake scores")
    axes[2].legend(fontsize=8)

    fig.suptitle(f"Why is {HARD_SPEAKER} the hardest validation speaker?", fontsize=14)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------ 5. thresholds


def threshold_analysis(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y = df.label_id.to_numpy()
    hard_fake = ((df.speaker == HARD_SPEAKER) & (df.label_id == 1)).to_numpy()
    for name in MODEL_NAMES:
        p = df[f"prob_{name}"].to_numpy()
        fpr, tpr, thr = roc_curve(y, p)
        operating_points = {"threshold 0.5": THRESHOLD}
        for target in [0.001, 0.01, 0.05]:
            ok = np.where(fpr <= target)[0]
            operating_points[f"FPR <= {target:.1%}"] = float(thr[ok[-1]])
        for label, t in operating_points.items():
            pred = p >= t
            rows.append(
                {
                    "model": name,
                    "operating_point": label,
                    "threshold": min(float(t), 1.0),
                    "recall": float(pred[y == 1].mean()),
                    "fpr": float(pred[y == 0].mean()),
                    "missed_fakes": int((~pred & (y == 1)).sum()),
                    "false_alarms": int((pred & (y == 0)).sum()),
                    "hard_speaker_fake_recall": float(pred[hard_fake].mean()),
                }
            )
    return pd.DataFrame(rows)


def plot_threshold_tradeoff(df: pd.DataFrame, path: Path) -> None:
    thresholds = np.logspace(-5, 0, 200)
    y = df.label_id.to_numpy()
    hard_fake = ((df.speaker == HARD_SPEAKER) & (df.label_id == 1)).to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    for name in MODEL_NAMES:
        p = df[f"prob_{name}"].to_numpy()
        recall = [(p[y == 1] >= t).mean() for t in thresholds]
        fpr = [(p[y == 0] >= t).mean() for t in thresholds]
        hard = [(p[hard_fake] >= t).mean() for t in thresholds]
        axes[0].plot(thresholds, np.array(recall) * 100, color=MODEL_COLORS[name], label=name)
        axes[1].plot(thresholds, np.array(fpr) * 100, color=MODEL_COLORS[name], label=name)
        axes[2].plot(thresholds, np.array(hard) * 100, color=MODEL_COLORS[name], label=name)
    titles = ["Recall on all fakes (%)", "False alarm rate on real clips, FPR (%)", f"Recall on {HARD_SPEAKER} fakes (%)"]
    for ax, title in zip(axes, titles):
        ax.set_xscale("log")
        ax.axvline(THRESHOLD, color="black", ls="--", lw=1)
        ax.set_xlabel("Decision threshold on fake probability (log scale)")
        ax.set_title(title)
    axes[0].set_ylim(90, 100.5)
    axes[1].set_ylim(0, 10)
    axes[0].legend(fontsize=8)
    fig.suptitle("Threshold trade-off: lowering the threshold catches more fakes but raises false alarms", fontsize=14)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------- 6. hard examples


def hard_examples(df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """Most confident mistakes of the best model, plus clips every model gets wrong."""
    p = df[f"prob_{BEST_MODEL}"]
    fn = df[(df.label_id == 1) & (p < THRESHOLD)].assign(error_type="FN", confidence=lambda d: 1 - d[f"prob_{BEST_MODEL}"])
    fp = df[(df.label_id == 0) & (p >= THRESHOLD)].assign(error_type="FP", confidence=lambda d: d[f"prob_{BEST_MODEL}"])
    picked = pd.concat([fn.nlargest(k, "confidence"), fp.nlargest(k, "confidence")])
    cols = ["error_type", "file", "speaker", "class_name", "source_path", "duration_original_sec", "pad_sec",
            "rms_before_dbfs", "n_models_wrong"] + [f"prob_{n}" for n in MODEL_NAMES]
    return picked[cols].reset_index(drop=True)


# ------------------------------------------------------------------- main


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    val = pd.read_csv(SPLIT_DIR / "val.csv")
    train = pd.read_csv(SPLIT_DIR / "train.csv")
    df = load_predictions(val)

    overlap_table, overlap_info = error_overlap(df)
    overlap_table.to_csv(LOG_DIR / "09_error_overlap.csv")
    by_model = per_model_overlap(df)
    by_model.to_csv(LOG_DIR / "09_error_overlap_by_model.csv")
    all_wrong = df[df.n_models_wrong == len(MODEL_NAMES)]
    note = f"{len(all_wrong)} clips are misclassified by all {len(MODEL_NAMES)} models" + (
        f"; all of them are {all_wrong.speaker.iloc[0]} {all_wrong.class_name.iloc[0].lower()} clips."
        if len(all_wrong) and all_wrong.speaker.nunique() == 1 and all_wrong.class_name.nunique() == 1
        else "."
    )
    plot_error_overlap(overlap_table, by_model, note, FIG_DIR / "09_error_overlap.png")

    bins = error_by_bins(df)
    bins.to_csv(LOG_DIR / "09_error_by_bin.csv", index=False)
    plot_error_by_confounder(bins, FIG_DIR / "09_error_by_confounder.png")

    profile_table, spectral, profiles = speaker_profile(df, train)
    profile_table.to_csv(LOG_DIR / "09_speaker_profile.csv", index=False)
    plot_hard_speaker(df, train, profiles, FIG_DIR / "09_adam_driver_profile.png")

    thresholds = threshold_analysis(df)
    thresholds.to_csv(LOG_DIR / "09_threshold_analysis.csv", index=False)
    plot_threshold_tradeoff(df, FIG_DIR / "09_threshold_tradeoff.png")

    examples = hard_examples(df)
    examples.to_csv(LOG_DIR / "09_hard_examples.csv", index=False)

    errors_per_model = {
        name: {
            "fn": int((df[f"err_{name}"] & (df.label_id == 1)).sum()),
            "fp": int((df[f"err_{name}"] & (df.label_id == 0)).sum()),
            "fn_from_hard_speaker": int((df[f"err_{name}"] & (df.label_id == 1) & (df.speaker == HARD_SPEAKER)).sum()),
        }
        for name in MODEL_NAMES
    }
    summary = {
        "threshold": THRESHOLD,
        "hard_speaker": HARD_SPEAKER,
        "errors_per_model": errors_per_model,
        "overlap": overlap_info,
        "hard_speaker_spectral": spectral,
    }
    (LOG_DIR / "09_error_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    pd.set_option("display.width", 200)
    print("Errors per model:")
    print(pd.DataFrame(errors_per_model).T.to_string())
    print("\nError overlap (rows: true class, columns: number of models wrong):")
    print(overlap_table.to_string())
    print("\nEach model's errors by how many models share them:")
    print(by_model.to_string())
    print(json.dumps(overlap_info, indent=2))
    print("\nError rate by bin (best model, with/without the hard speaker):")
    best = bins[bins.model == BEST_MODEL]
    print(best[["variable", "bin", "class_name", "n", "errors", "error_rate", "errors_excl_hard_speaker",
                "error_rate_excl_hard_speaker"]].round(4).to_string(index=False))
    print("\nError rate by bin (all models, excluding the hard speaker, summed errors):")
    agg = bins.groupby(["variable", "bin", "class_name"], sort=False).agg(
        n=("n_excl_hard_speaker", "first"), errors_all_models=("errors_excl_hard_speaker", "sum")
    )
    agg["mean_error_rate"] = agg.errors_all_models / (agg.n * len(MODEL_NAMES))
    print(agg.round(4).to_string())
    print("\nSpeaker profile:")
    print(profile_table.round(3).to_string(index=False))
    print(json.dumps(spectral, indent=2))
    print("\nThreshold analysis:")
    print(thresholds.round(4).to_string(index=False))
    print("\nHard examples (best model):")
    print(examples.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
