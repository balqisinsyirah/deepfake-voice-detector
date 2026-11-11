"""Step 4: Speaker-disjoint 70/30 split and PyTorch DataLoaders.

The split is made at the speaker level, so no speaker appears in both training and
validation. A fixed random seed and a small search over speaker permutations are
used to keep the validation set close to 30% of clips and close to the global
real/fake ratio.

Feature normalization statistics are computed on the training split only.

Usage:
    python src/04_split_dataloader.py
    python src/04_split_dataloader.py --workers 4 --batch-size 32

Outputs:
    data/splits/train.csv
    data/splits/val.csv
    data/splits/split_summary.json
    data/splits/train_stats.json
    data/splits/split_summary.txt
    outputs/figures/04_split_composition.png
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEL_DIR = PROJECT_ROOT / "data" / "processed" / "mel"
FEATURES_PATH = MEL_DIR / "mel_spectrograms.npy"
LABELS_PATH = MEL_DIR / "labels.npy"
METADATA_PATH = MEL_DIR / "metadata.csv"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"

SEED = 42
VAL_FRACTION = 0.30
N_TRIALS = 5000
BATCH_SIZE = 32

CLASS_LABELS = {0: "Real", 1: "Fake"}
COLORS = {0: "#2A9D8F", 1: "#E76F51"}
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


class MelSpectrogramDataset(Dataset):
    """Load log-Mel features lazily from a memmap and return (image, label)."""

    def __init__(self, features_path: Path, labels_path: Path, indices: np.ndarray, mean: float, std: float):
        self.features = np.load(features_path, mmap_mode="r")
        self.labels = np.load(labels_path)
        self.indices = np.asarray(indices)
        self.mean = float(mean)
        self.std = float(std) if std > 0 else 1.0

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        row = int(self.indices[position])
        # Standardize with training-set statistics, then add a channel dimension
        x = np.asarray(self.features[row], dtype=np.float32)
        x = (x - self.mean) / self.std
        x = torch.from_numpy(x).unsqueeze(0)
        y = torch.tensor(int(self.labels[row]), dtype=torch.long)
        return x, y


def choose_speaker_split(meta: pd.DataFrame, seed: int, val_fraction: float, n_trials: int) -> tuple[list[str], dict]:
    """Search random speaker permutations for a balanced, speaker-disjoint split."""
    speaker_stats = (
        meta.groupby("speaker")
        .agg(
            n=("file", "size"),
            n_real=("label_id", lambda s: int((s == 0).sum())),
            n_fake=("label_id", lambda s: int((s == 1).sum())),
        )
        .reset_index()
    )
    stats = speaker_stats.set_index("speaker").to_dict("index")
    speakers = speaker_stats["speaker"].to_numpy()
    total = len(meta)
    global_fake_share = float((meta.label_id == 1).mean())
    target = val_fraction * total

    rng = np.random.default_rng(seed)
    best = None
    for trial in range(n_trials):
        order = rng.permutation(speakers)
        val_speakers = []
        val_n = 0
        val_fake = 0
        for speaker in order:
            if val_n >= target:
                break
            val_speakers.append(str(speaker))
            val_n += stats[speaker]["n"]
            val_fake += stats[speaker]["n_fake"]

        val_real = val_n - val_fake
        if val_real < 100 or val_fake < 100:
            continue
        val_fake_share = val_fake / val_n
        # Prefer a validation share near 30% and a fake share near the global ratio
        objective = abs(val_n / total - val_fraction) * 8.0 + abs(val_fake_share - global_fake_share) * 4.0
        if best is None or objective < best["objective"]:
            best = {
                "objective": objective,
                "trial": trial,
                "val_speakers": val_speakers,
                "val_n": val_n,
                "val_real": val_real,
                "val_fake": val_fake,
                "val_fake_share": val_fake_share,
            }

    if best is None:
        raise RuntimeError("Could not find a valid speaker split.")

    info = {
        "seed": seed,
        "n_trials": n_trials,
        "best_trial": best["trial"],
        "objective": best["objective"],
        "target_val_fraction": val_fraction,
        "global_fake_share": global_fake_share,
        "val_fraction_actual": best["val_n"] / total,
        "val_fake_share": best["val_fake_share"],
    }
    return best["val_speakers"], info


def compute_train_stats(features_path: Path, train_indices: np.ndarray, chunk_size: int = 2000) -> tuple[float, float]:
    """Compute global mean/std over training rows only, in chunks."""
    features = np.load(features_path, mmap_mode="r")
    total_sum = 0.0
    total_sq = 0.0
    count = 0
    for start in range(0, len(train_indices), chunk_size):
        rows = train_indices[start : start + chunk_size]
        block = np.asarray(features[rows], dtype=np.float64)
        total_sum += float(block.sum())
        total_sq += float(np.square(block).sum())
        count += block.size
    mean = total_sum / count
    variance = max(total_sq / count - mean**2, 0.0)
    return mean, float(np.sqrt(variance))


def plot_split(train: pd.DataFrame, val: pd.DataFrame, output_path: Path) -> None:
    counts = pd.DataFrame(
        {
            "Split": ["Train", "Validation"],
            "Real": [len(train[train.label_id == 0]), len(val[val.label_id == 0])],
            "Fake": [len(train[train.label_id == 1]), len(val[val.label_id == 1])],
        }
    )
    counts["Total"] = counts["Real"] + counts["Fake"]
    counts["Fake share"] = counts["Fake"] / counts["Total"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.3, 1]})
    axes[0].bar(counts["Split"], counts["Real"], color=COLORS[0], label="Real")
    axes[0].bar(counts["Split"], counts["Fake"], bottom=counts["Real"], color=COLORS[1], label="Fake")
    for i, row in counts.iterrows():
        axes[0].text(i, row["Total"] + 300, f"{int(row['Total']):,}", ha="center", fontsize=9)
        axes[0].text(i, row["Real"] / 2, f"{int(row['Real']):,}", ha="center", va="center", color="white", fontsize=9)
        axes[0].text(i, row["Real"] + row["Fake"] / 2, f"{int(row['Fake']):,}", ha="center", va="center", color="white", fontsize=9)
    axes[0].set_ylabel("Number of clips")
    axes[0].set_title("Speaker-disjoint 70/30 split")
    axes[0].legend(title="Class", loc="upper right")
    axes[0].set_ylim(0, counts["Total"].max() * 1.12)

    axes[1].bar(counts["Split"], counts["Fake share"] * 100, color="#6C757D")
    global_share = (len(train[train.label_id == 1]) + len(val[val.label_id == 1])) / (len(train) + len(val))
    axes[1].axhline(global_share * 100, color="black", ls="--", lw=1, label=f"Global {global_share*100:.1f}%")
    for i, value in enumerate(counts["Fake share"] * 100):
        axes[1].text(i, value + 0.8, f"{value:.1f}%", ha="center", fontsize=9)
    axes[1].set_ylabel("Fake share (%)")
    axes[1].set_title("Class balance is preserved")
    axes[1].set_ylim(0, 55)
    axes[1].legend(loc="upper right")

    fig.suptitle(
        f"Train: {len(train):,} clips / {train.speaker.nunique()} speakers    "
        f"Validation: {len(val):,} clips / {val.speaker.nunique()} speakers",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def write_text_summary(path: Path, summary: dict) -> None:
    lines = [
        "Step 4 split summary",
        "",
        f"Random seed: {summary['seed']}",
        f"Speaker overlap: {summary['speaker_overlap']}",
        f"Train clips: {summary['n_train']:,} ({summary['n_train']/summary['n_total']:.2%})",
        f"Validation clips: {summary['n_val']:,} ({summary['n_val']/summary['n_total']:.2%})",
        f"Train speakers: {summary['n_train_speakers']}",
        f"Validation speakers: {summary['n_val_speakers']}",
        f"Train fake share: {summary['train_fake_share']:.2%}",
        f"Validation fake share: {summary['val_fake_share']:.2%}",
        f"Global fake share: {summary['global_fake_share']:.2%}",
        f"Class weights (CrossEntropy): {summary['class_weights']}",
        f"Train feature mean: {summary['feature_mean']:.4f}",
        f"Train feature std: {summary['feature_std']:.4f}",
        "",
        "Validation speakers:",
        ", ".join(summary["val_speakers"]),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def smoke_test_dataloaders(train_loader: DataLoader, val_loader: DataLoader, output_path: Path) -> None:
    xb, yb = next(iter(train_loader))
    xv, yv = next(iter(val_loader))
    lines = [
        "DataLoader smoke test",
        "",
        f"Train batches: {len(train_loader)}",
        f"Validation batches: {len(val_loader)}",
        f"Train batch shape: {tuple(xb.shape)}",
        f"Validation batch shape: {tuple(xv.shape)}",
        f"Train batch dtype: {xb.dtype}",
        f"Train batch label counts: {torch.bincount(yb, minlength=2).tolist()}",
        f"Validation batch label counts: {torch.bincount(yv, minlength=2).tolist()}",
        f"Train batch mean: {xb.mean().item():.4f}",
        f"Train batch std: {xb.std().item():.4f}",
        "Validation batch mean: {:.4f}".format(xv.mean().item()),
        "Validation batch std: {:.4f}".format(xv.std().item()),
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for split and shuffling")
    parser.add_argument("--trials", type=int, default=N_TRIALS, help="Number of speaker permutations to search")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing split files")
    args = parser.parse_args()

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if (SPLIT_DIR / "train.csv").exists() and not args.overwrite:
        raise SystemExit(f"{SPLIT_DIR / 'train.csv'} already exists. Use --overwrite to regenerate.")

    meta = pd.read_csv(METADATA_PATH)
    labels = np.load(LABELS_PATH)
    if len(meta) != len(labels):
        raise ValueError("Metadata and labels have different lengths.")
    if not np.array_equal(meta["label_id"].to_numpy(), labels):
        raise ValueError("Metadata labels and labels.npy are not aligned.")

    val_speakers, split_info = choose_speaker_split(meta, args.seed, VAL_FRACTION, args.trials)
    train_speakers = sorted(set(meta.speaker) - set(val_speakers))
    overlap = set(train_speakers) & set(val_speakers)
    if overlap:
        raise RuntimeError(f"Speaker overlap found: {sorted(overlap)}")

    train = meta[meta.speaker.isin(train_speakers)].copy()
    val = meta[meta.speaker.isin(val_speakers)].copy()
    if len(train) + len(val) != len(meta):
        raise RuntimeError("Split does not cover every clip exactly once.")
    if train.label_id.nunique() < 2 or val.label_id.nunique() < 2:
        raise RuntimeError("Both splits must contain both classes.")

    train_indices = np.sort(train["row_index"].to_numpy())
    val_indices = np.sort(val["row_index"].to_numpy())
    feature_mean, feature_std = compute_train_stats(FEATURES_PATH, train_indices)

    train_counts = train.label_id.value_counts().sort_index()
    class_weights = [len(train) / (2.0 * train_counts[i]) for i in [0, 1]]

    summary = {
        **split_info,
        "speaker_overlap": len(overlap),
        "n_total": len(meta),
        "n_train": len(train),
        "n_val": len(val),
        "n_train_speakers": train.speaker.nunique(),
        "n_val_speakers": val.speaker.nunique(),
        "train_real": int((train.label_id == 0).sum()),
        "train_fake": int((train.label_id == 1).sum()),
        "val_real": int((val.label_id == 0).sum()),
        "val_fake": int((val.label_id == 1).sum()),
        "train_fake_share": float((train.label_id == 1).mean()),
        "val_fake_share": float((val.label_id == 1).mean()),
        "global_fake_share": float((meta.label_id == 1).mean()),
        "class_weights": [round(float(w), 6) for w in class_weights],
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "val_speakers": sorted(val_speakers),
        "train_speakers": train_speakers,
    }

    keep_columns = [
        "row_index",
        "file",
        "speaker",
        "label",
        "label_id",
        "class_name",
        "source_path",
        "duration_original_sec",
        "pad_sec",
        "rms_before_dbfs",
        "rms_after_dbfs",
    ]
    train[keep_columns].to_csv(SPLIT_DIR / "train.csv", index=False)
    val[keep_columns].to_csv(SPLIT_DIR / "val.csv", index=False)
    (SPLIT_DIR / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (SPLIT_DIR / "train_stats.json").write_text(
        json.dumps(
            {
                "feature_mean": feature_mean,
                "feature_std": feature_std,
                "class_weights": summary["class_weights"],
                "seed": args.seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_text_summary(SPLIT_DIR / "split_summary.txt", summary)

    train_dataset = MelSpectrogramDataset(
        FEATURES_PATH, LABELS_PATH, train_indices, feature_mean, feature_std
    )
    val_dataset = MelSpectrogramDataset(
        FEATURES_PATH, LABELS_PATH, val_indices, feature_mean, feature_std
    )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        **loader_kwargs,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    print("\nSplit summary")
    print(f"  Train: {len(train):,} clips, {train.speaker.nunique()} speakers, fake {summary['train_fake_share']:.2%}")
    print(f"  Val:   {len(val):,} clips, {val.speaker.nunique()} speakers, fake {summary['val_fake_share']:.2%}")
    print(f"  Speaker overlap: {len(overlap)}")
    print(f"  Feature mean/std (train only): {feature_mean:.4f} / {feature_std:.4f}")
    print(f"  Class weights: {summary['class_weights']}")

    smoke_test_dataloaders(train_loader, val_loader, SPLIT_DIR / "dataloader_smoke.txt")
    plot_split(train, val, FIG_DIR / "04_split_composition.png")
    print(f"\nSplit files: {SPLIT_DIR}")
    print(f"Figure: {FIG_DIR / '04_split_composition.png'}")


if __name__ == "__main__":
    main()
