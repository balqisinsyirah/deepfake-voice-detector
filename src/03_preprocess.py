"""Step 3: Preprocess audio and extract log-Mel spectrogram features.

Pipeline per clip:
  1. Load and resample to 16 kHz mono.
  2. Standardize duration to 4 seconds (center crop; wrap-pad short clips).
  3. Normalize loudness to -23 dBFS RMS (optional; enabled by default).
  4. Extract a 128-band log-Mel spectrogram with librosa.
  5. Save float32 features to a single .npy array plus labels and metadata.

Usage:
    python src/03_preprocess.py
    python src/03_preprocess.py --limit 100 --output-dir data/processed/mel_debug
    python src/03_preprocess.py --no-normalize --output-dir data/processed/mel_raw

Outputs (default):
    data/processed/mel/mel_spectrograms.npy
    data/processed/mel/labels.npy
    data/processed/mel/metadata.csv
    data/processed/mel/feature_config.json
    data/processed/mel/preprocessing_summary.txt
    outputs/figures/03_*.png
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = PROJECT_ROOT / "data" / "raw" / "dataset_index.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "mel"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"

SR = 16000
DURATION = 4.0
TARGET_SAMPLES = int(SR * DURATION)
N_MELS = 128
N_FFT = 2048
HOP = 512
FMIN = 20
FMAX = SR // 2
TARGET_RMS_DBFS = -23.0
PEAK_LIMIT = 0.99
TOP_DB = 80.0
N_FRAMES = 1 + TARGET_SAMPLES // HOP

CLASS_LABELS = {"real": "Real", "fake": "Fake"}
COLORS = {"real": "#2A9D8F", "fake": "#E76F51"}
PALETTE = {CLASS_LABELS[key]: COLORS[key] for key in COLORS}

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


def rms_dbfs(y: np.ndarray) -> float:
    """Return RMS level in dBFS, with a floor to avoid log(0)."""
    if y.size == 0:
        return -100.0
    rms = float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))
    return 20.0 * np.log10(max(rms, 1e-10))


def fix_length(y: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Center-crop long clips and wrap-pad short clips to exactly DURATION seconds.

    Wrap-padding repeats the signal instead of adding silence, which avoids
    introducing a class-correlated silence pattern.
    """
    if len(y) >= TARGET_SAMPLES:
        start = (len(y) - TARGET_SAMPLES) // 2
        y_fixed = y[start : start + TARGET_SAMPLES]
        return y_fixed, 0.0, start / SR

    pad_seconds = (TARGET_SAMPLES - len(y)) / SR
    repeats = int(np.ceil(TARGET_SAMPLES / max(len(y), 1)))
    y_fixed = np.tile(y, repeats)[:TARGET_SAMPLES]
    return y_fixed, pad_seconds, 0.0


def normalize_loudness(y: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Scale the clip to TARGET_RMS_DBFS while avoiding peak clipping."""
    before = rms_dbfs(y)
    gain_db = TARGET_RMS_DBFS - before
    y = y * (10.0 ** (gain_db / 20.0))

    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > PEAK_LIMIT:
        scale = PEAK_LIMIT / peak
        y = y * scale
        gain_db += 20.0 * np.log10(scale)

    return y, gain_db, rms_dbfs(y)


def make_mel(y: np.ndarray) -> np.ndarray:
    """Compute a log-Mel spectrogram normalized to a per-clip maximum of 0 dB."""
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=TOP_DB)
    if not np.isfinite(mel_db).all():
        mel_db = np.nan_to_num(mel_db, nan=-TOP_DB, posinf=0.0, neginf=-TOP_DB)
    return mel_db.astype(np.float32)


def process_one(task: tuple) -> tuple:
    """Worker: load one clip, preprocess it, and return its Mel plus metadata."""
    row_index, path, file, speaker, label, label_id, normalize = task
    try:
        y, _ = librosa.load(path, sr=SR, mono=True)
        original_duration = len(y) / SR
        y, pad_sec, crop_start_sec = fix_length(y)
        rms_before = rms_dbfs(y)

        if normalize:
            y, gain_db, rms_after = normalize_loudness(y)
        else:
            gain_db = 0.0
            rms_after = rms_before

        peak_after = float(np.max(np.abs(y))) if y.size else 0.0
        mel = make_mel(y)

        meta = {
            "row_index": row_index,
            "file": file,
            "speaker": speaker,
            "label": label,
            "label_id": label_id,
            "class_name": CLASS_LABELS[label],
            "source_path": path,
            "duration_original_sec": original_duration,
            "duration_used_sec": DURATION,
            "crop_start_sec": crop_start_sec,
            "pad_sec": pad_sec,
            "rms_before_dbfs": rms_before,
            "rms_after_dbfs": rms_after,
            "gain_db": gain_db,
            "peak_after": peak_after,
            "mel_rows": mel.shape[0],
            "mel_cols": mel.shape[1],
            "error": "",
        }
        return row_index, mel, meta
    except Exception as exc:
        meta = {
            "row_index": row_index,
            "file": file,
            "speaker": speaker,
            "label": label,
            "label_id": label_id,
            "class_name": CLASS_LABELS.get(label, label),
            "source_path": path,
            "error": str(exc),
        }
        return row_index, None, meta


def save_config(output_dir: Path, n_samples: int, normalize: bool) -> None:
    config = {
        "sample_rate": SR,
        "duration_sec": DURATION,
        "target_samples": TARGET_SAMPLES,
        "n_mels": N_MELS,
        "n_fft": N_FFT,
        "hop_length": HOP,
        "fmin": FMIN,
        "fmax": FMAX,
        "target_rms_dbfs": TARGET_RMS_DBFS,
        "peak_limit": PEAK_LIMIT,
        "top_db": TOP_DB,
        "feature_shape": [N_MELS, N_FRAMES],
        "n_samples": n_samples,
        "normalize_loudness": normalize,
        "duration_policy": "center crop; wrap-pad short clips",
        "mel_reference": "per-clip max (0 dB)",
        "label_mapping": {"real": 0, "fake": 1},
    }
    (output_dir / "feature_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def plot_loudness_before_after(meta: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), sharey=True)
    for ax, column, title in [
        (axes[0], "rms_before_dbfs", "Before normalization"),
        (axes[1], "rms_after_dbfs", "After normalization"),
    ]:
        sns.histplot(
            data=meta,
            x=column,
            hue="class_name",
            hue_order=["Real", "Fake"],
            palette=PALETTE,
            bins=60,
            stat="density",
            common_norm=False,
            element="step",
            fill=True,
            alpha=0.35,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("RMS loudness (dBFS)")
        ax.set_ylabel("Density")
        for label in ["real", "fake"]:
            median = meta.loc[meta.label == label, column].median()
            ax.axvline(median, color=COLORS[label], ls="--", lw=1.2)
        legend = ax.get_legend()
        if legend is not None:
            legend.set_title("Class")
    fig.suptitle("Loudness shortcut before and after per-clip RMS normalization", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_loudness_before_after.png")
    plt.close(fig)


def plot_padding(meta: pd.DataFrame, pad_auc: float, output_dir: Path) -> None:
    plot_df = meta[meta.pad_sec > 0]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    if plot_df.empty:
        ax.text(0.5, 0.5, "No clips required padding", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        sns.histplot(
            data=plot_df,
            x="pad_sec",
            hue="class_name",
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
        ax.set_xlabel("Wrap-padding added to reach 4 seconds (seconds)")
        ax.set_ylabel("Density")
        ax.set_title("Residual duration signal: how much padding each class needs")
        ax.text(
            0.98,
            0.95,
            f"Padding-based AUC = {pad_auc:.3f}\n(0.5 = no class signal)",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="none"),
        )
        legend = ax.get_legend()
        if legend is not None:
            legend.set_title("Class")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_padding_by_class.png")
    plt.close(fig)


def plot_examples(meta: pd.DataFrame, features_path: Path, output_dir: Path) -> None:
    """Plot normalized waveforms and final log-Mel inputs for one real/fake pair."""
    pair = meta[meta.file.isin(["818.wav", "14458.wav"])].copy()
    if len(pair) != 2:
        candidates = meta[meta.duration_original_sec >= DURATION]
        pair = pd.concat(
            [
                candidates[candidates.label == "real"].head(1),
                candidates[candidates.label == "fake"].head(1),
            ]
        )
    pair = pair.sort_values("label", ascending=False)  # Real first
    features = np.load(features_path, mmap_mode="r")

    # Use a dedicated colorbar axis so it never covers the Mel panels
    fig = plt.figure(figsize=(13.5, 7))
    gs = fig.add_gridspec(
        2, 3, width_ratios=[1, 1, 0.045], hspace=0.28, wspace=0.18
    )
    wave_axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    mel_real_ax = fig.add_subplot(gs[1, 0])
    mel_fake_ax = fig.add_subplot(gs[1, 1], sharey=mel_real_ax)
    mel_axes = [mel_real_ax, mel_fake_ax]
    cax = fig.add_subplot(gs[:, 2])

    t = np.arange(TARGET_SAMPLES) / SR
    for col, (_, row) in enumerate(pair.iterrows()):
        y, _ = librosa.load(row.source_path, sr=SR, mono=True)
        y, _, _ = fix_length(y)
        if NORMALIZE:
            y, _, _ = normalize_loudness(y)

        wave_axes[col].plot(t, y, color=COLORS[row.label], lw=0.6)
        wave_axes[col].set_ylim(-1, 1)
        wave_axes[col].set_xlim(0, DURATION)
        wave_axes[col].set_ylabel("Amplitude")
        wave_axes[col].set_title(
            f"{row.class_name} | {row.file} | {row.speaker} | RMS {row.rms_after_dbfs:.1f} dBFS",
            fontsize=10,
        )

        mel = features[row.row_index]
        img = librosa.display.specshow(
            mel,
            sr=SR,
            hop_length=HOP,
            x_axis="time",
            y_axis="mel",
            fmax=FMAX,
            vmin=-TOP_DB,
            vmax=0,
            cmap="magma",
            ax=mel_axes[col],
        )
        mel_axes[col].set_title(f"{row.class_name} log-Mel input")
        mel_axes[col].set_xlabel("Time (seconds)")
        mel_axes[col].set_ylabel("Mel frequency")

    fig.colorbar(img, cax=cax, format="%+2.0f dB", label="Energy (dB)")
    fig.suptitle("Final model inputs after duration standardization and loudness normalization", fontsize=14)
    fig.savefig(FIG_DIR / "03_preprocessed_examples.png", bbox_inches="tight")
    plt.close(fig)


def write_summary(meta: pd.DataFrame, features_path: Path, output_dir: Path, pad_auc: float) -> None:
    labels = meta.label_id.to_numpy()
    rms_auc = roc_auc_score(labels, -meta.rms_after_dbfs.to_numpy())
    lines = [
        "Step 3 preprocessing summary",
        "",
        f"Features: {features_path}",
        f"Shape: {tuple(np.load(features_path, mmap_mode='r').shape)}",
        f"Normalize loudness: {NORMALIZE}",
        f"Target RMS: {TARGET_RMS_DBFS} dBFS",
        f"Duration: {DURATION} seconds (center crop; wrap-pad short clips)",
        f"Mel: n_mels={N_MELS}, n_fft={N_FFT}, hop={HOP}, fmin={FMIN}, fmax={FMAX}, top_db={TOP_DB}",
        "",
        "Label counts:",
        meta.class_name.value_counts().to_string(),
        "",
        "RMS loudness by class (median, dBFS):",
        meta.groupby("class_name")["rms_after_dbfs"].median().round(3).to_string(),
        "",
        "RMS-based shortcut AUC after normalization (0.5 = removed):",
        f"{rms_auc:.4f}",
        "",
        "Clips shorter than the target duration (wrap-padded):",
        f"All: {(meta.pad_sec > 0).mean():.1%}",
        (meta.pad_sec > 0).groupby(meta.class_name).mean().map("{:.1%}".format).to_string(),
        "",
        "Padding by class (seconds; count = all clips, including unpadded):",
        meta.groupby("class_name")["pad_sec"].agg(["count", "mean", "median", "max"]).round(3).to_string(),
        "",
        "Crop start by class (seconds):",
        meta.groupby("class_name")["crop_start_sec"].agg(["mean", "median", "max"]).round(3).to_string(),
        "",
        "Residual padding-based shortcut AUC:",
        f"{pad_auc:.4f}",
        "",
        "Notes:",
        "- Features are log-Mel dB values with per-clip maximum at 0 dB.",
        "- Normalization statistics for the neural network should be computed on the training split only.",
    ]
    (output_dir / "preprocessing_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    global NORMALIZE
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N clips (for testing)")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker processes")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for features and metadata")
    parser.add_argument("--no-normalize", action="store_true", help="Disable RMS loudness normalization (control run)")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty output directory")
    parser.add_argument("--figures-only", action="store_true", help="Regenerate Step 3 figures and summary from existing features")
    args = parser.parse_args()

    NORMALIZE = not args.no_normalize
    output_dir = args.output_dir or (DEFAULT_OUTPUT_DIR if args.limit is None else PROJECT_ROOT / "data" / "processed" / "mel_debug")

    if args.figures_only:
        meta_path = output_dir / "metadata.csv"
        features_path = output_dir / "mel_spectrograms.npy"
        config_path = output_dir / "feature_config.json"
        if not meta_path.exists() or not features_path.exists():
            raise SystemExit(f"Missing features or metadata in {output_dir}.")
        if config_path.exists():
            NORMALIZE = json.loads(config_path.read_text(encoding="utf-8")).get("normalize_loudness", NORMALIZE)
        meta = pd.read_csv(meta_path)
        pad_auc = roc_auc_score(meta.label_id.to_numpy(), meta.pad_sec.to_numpy())
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        plot_loudness_before_after(meta, output_dir)
        plot_padding(meta, pad_auc, output_dir)
        plot_examples(meta, features_path, output_dir)
        write_summary(meta, features_path, output_dir, pad_auc)
        print(f"Figures regenerated in {FIG_DIR}")
        print(f"Summary rewritten: {output_dir / 'preprocessing_summary.txt'}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output directory is not empty: {output_dir}. Use --overwrite or choose another directory.")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(INDEX_PATH)
    if args.limit is not None:
        index = index.head(args.limit).copy()
    index = index.reset_index(drop=True)

    tasks = [
        (i, row.path, row.file, row.speaker, row.label, row.label_id, NORMALIZE)
        for i, row in enumerate(index.itertuples(index=False))
    ]

    n = len(tasks)
    features_tmp = output_dir / "mel_spectrograms.tmp.npy"
    features_path = output_dir / "mel_spectrograms.npy"
    features = np.lib.format.open_memmap(
        features_tmp, mode="w+", dtype="float32", shape=(n, N_MELS, N_FRAMES)
    )
    labels = np.empty(n, dtype=np.int8)
    metas = []
    errors = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for row_index, mel, meta in tqdm(
            executor.map(process_one, tasks, chunksize=16),
            total=n,
            desc="Extracting features",
        ):
            metas.append(meta)
            if mel is None:
                errors.append(meta)
                continue
            features[row_index] = mel
            labels[row_index] = meta["label_id"]

    features.flush()
    del features

    if errors:
        pd.DataFrame(errors).to_csv(output_dir / "errors.csv", index=False)
        raise SystemExit(f"{len(errors)} clips failed. See {output_dir / 'errors.csv'}.")

    os.replace(features_tmp, features_path)
    np.save(output_dir / "labels.npy", labels)
    meta = pd.DataFrame(metas).sort_values("row_index").reset_index(drop=True)
    meta.to_csv(output_dir / "metadata.csv", index=False)
    save_config(output_dir, n, NORMALIZE)

    pad_auc = roc_auc_score(meta.label_id.to_numpy(), meta.pad_sec.to_numpy())
    write_summary(meta, features_path, output_dir, pad_auc)

    print(f"\nFeature matrix: {features_path} shape={tuple(np.load(features_path, mmap_mode='r').shape)}")
    print(f"Labels: {output_dir / 'labels.npy'}")
    print(f"Metadata: {output_dir / 'metadata.csv'}")
    print(meta.groupby("class_name")[["rms_before_dbfs", "rms_after_dbfs", "pad_sec"]].median().round(3).to_string())
    print(f"Padding shortcut AUC: {pad_auc:.4f}")

    if args.limit is None:
        plot_loudness_before_after(meta, output_dir)
        plot_padding(meta, pad_auc, output_dir)
        plot_examples(meta, features_path, output_dir)
        print(f"Figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
