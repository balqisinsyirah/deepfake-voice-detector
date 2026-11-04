"""Step 2: Exploratory data analysis (EDA).

Analyze duration, sample rate, channel count, and loudness for all 31,779 clips,
test for systematic real/fake differences, and create waveform and Mel-spectrogram comparisons.

Run: python src/02_eda.py
Outputs:
    data/processed/audio_stats.csv          per-clip statistics
    data/processed/eda_label_summary.csv    summary by class
    data/processed/eda_speaker_summary.csv  summary by speaker
    data/processed/eda_tests.csv            statistical test results
    outputs/figures/02_*.png                figures for the blog
    outputs/eda_summary.txt                 text summary
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")  # Save figures without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import soundfile as sf
from scipy.stats import ks_2samp, mannwhitneyu, wilcoxon
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = PROJECT_ROOT / "data" / "raw" / "dataset_index.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORT_PATH = PROJECT_ROOT / "outputs" / "eda_summary.txt"

SR = 16000
N_MELS = 128
N_FFT = 2048
HOP = 512
COLORS = {"real": "#2A9D8F", "fake": "#E76F51"}
CLASS_LABELS = {"real": "Real", "fake": "Fake"}
PALETTE = {CLASS_LABELS[key]: COLORS[key] for key in COLORS}

# Seaborn resets font settings, so apply the theme first and override fonts afterwards
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


def analyze_one(path: str) -> dict:
    """Read one clip (header + full decode) and compute its duration and loudness."""
    try:
        info = sf.info(path)
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        mono = audio.mean(axis=1)  # Mix to mono; this dataset is mono throughout
        rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
        peak = float(np.max(np.abs(mono))) if mono.size else 0.0
        return {
            "path": path,
            "samplerate": info.samplerate,
            "channels": info.channels,
            "subtype": info.subtype,
            "frames": info.frames,
            "duration": info.frames / info.samplerate,
            "rms_dbfs": 20 * np.log10(max(rms, 1e-8)),
            "peak_dbfs": 20 * np.log10(max(peak, 1e-8)),
            "clip_ratio": float(np.mean(np.abs(mono) >= 0.999)) if mono.size else 0.0,
            "error": "",
        }
    except Exception as exc:  # A single corrupt file must not abort the whole EDA
        return {"path": path, "error": str(exc)}


def collect_audio_stats(index: pd.DataFrame, workers: int = 12) -> pd.DataFrame:
    """Read all clips with a thread pool and return per-file statistics."""
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(analyze_one, p) for p in index["path"]]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Reading audio"):
            rows.append(fut.result())
    stats = pd.DataFrame(rows)
    bad = stats["error"].fillna("") != ""
    if bad.any():
        print(f"Warning: {bad.sum()} files could not be read and were excluded.")
        print(stats.loc[bad, ["path", "error"]].head().to_string(index=False))
    stats = stats.loc[~bad].drop(columns=["error"])
    return index.merge(stats, on="path", how="inner")


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize duration and loudness by class."""
    summary = (
        df.groupby("label")
        .agg(
            n=("file", "size"),
            hours=("duration", lambda s: s.sum() / 3600),
            duration_mean=("duration", "mean"),
            duration_median=("duration", "median"),
            duration_std=("duration", "std"),
            duration_min=("duration", "min"),
            duration_q05=("duration", lambda s: s.quantile(0.05)),
            duration_q25=("duration", lambda s: s.quantile(0.25)),
            duration_q75=("duration", lambda s: s.quantile(0.75)),
            duration_q95=("duration", lambda s: s.quantile(0.95)),
            duration_max=("duration", "max"),
            rms_median=("rms_dbfs", "median"),
            peak_median=("peak_dbfs", "median"),
            clip_ratio_mean=("clip_ratio", "mean"),
        )
        .reindex(["real", "fake"])
    )
    return summary


def run_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Run overall difference tests for duration and loudness and compute effect sizes."""
    rows = []
    for col in ["duration", "rms_dbfs"]:
        real = df.loc[df.label == "real", col].to_numpy()
        fake = df.loc[df.label == "fake", col].to_numpy()
        u, p_u = mannwhitneyu(real, fake, alternative="two-sided")
        ks_stat, p_ks = ks_2samp(real, fake)
        # Cliff's delta: >0 means real values tend to be larger, <0 means fake values do
        cliff = 2 * u / (len(real) * len(fake)) - 1
        rows.append(
            {
                "metric": col,
                "real_median": np.median(real),
                "fake_median": np.median(fake),
                "mannwhitney_p": p_u,
                "ks_stat": ks_stat,
                "ks_p": p_ks,
                "cliffs_delta": cliff,
            }
        )
    return pd.DataFrame(rows)


def shortcut_auc(df: pd.DataFrame) -> pd.DataFrame:
    """Measure how well duration or loudness alone separates real from fake audio.

    AUC=0.5 means no discrimination; 1.0 means perfect discrimination.
    The within-speaker variants subtract each speaker's own median to remove
    the effect of speaker composition.
    """
    y = (df.label == "fake").astype(int)
    duration_norm = df.duration / df.groupby("speaker").duration.transform("median")
    rms_norm = df.rms_dbfs - df.groupby("speaker").rms_dbfs.transform("median")
    rows = [
        {"feature": "duration_raw", "auc_for_fake": roc_auc_score(y, df.duration)},
        {"feature": "rms_raw", "auc_for_fake": roc_auc_score(y, -df.rms_dbfs)},
        {"feature": "duration_within_speaker", "auc_for_fake": roc_auc_score(y, duration_norm)},
        {"feature": "rms_within_speaker", "auc_for_fake": roc_auc_score(y, -rms_norm)},
    ]
    return pd.DataFrame(rows)


def per_speaker_auc(df: pd.DataFrame, min_per_class: int = 20) -> pd.DataFrame:
    """Compute single-feature AUC within each speaker to check whether shortcuts are widespread."""
    rows = []
    for speaker, g in df.groupby("speaker"):
        n_real = int((g.label == "real").sum())
        n_fake = int((g.label == "fake").sum())
        if n_real < min_per_class or n_fake < min_per_class:
            continue
        y = (g.label == "fake").astype(int)
        rows.append(
            {
                "speaker": speaker,
                "n": len(g),
                "n_real": n_real,
                "n_fake": n_fake,
                "auc_duration": roc_auc_score(y, g.duration),
                "auc_rms": roc_auc_score(y, -g.rms_dbfs),
            }
        )
    return pd.DataFrame(rows)


def plot_per_speaker_auc(auc: pd.DataFrame) -> None:
    bins = np.linspace(0.0, 1.0, 41)
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.hist(
        [auc.auc_duration, auc.auc_rms],
        bins=bins,
        label=["Duration only", "Loudness only"],
        color=["#457B9D", "#E76F51"],
        alpha=0.65,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.axvline(0.5, color="black", ls="--", lw=1, label="Chance (0.5)")
    ax.axvline(auc.auc_duration.median(), color="#457B9D", ls=":", lw=2)
    ax.axvline(auc.auc_rms.median(), color="#E76F51", ls=":", lw=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
    ax.set_xlabel("AUC (computed within each speaker)")
    ax.set_ylabel("Number of speakers")
    ax.set_title("How widespread are single-feature shortcuts?")
    ax.legend(loc="upper left")
    ax.text(
        0.5,
        0.98,
        f"Median duration AUC = {auc.auc_duration.median():.3f}\n"
        f"Median loudness AUC = {auc.auc_rms.median():.3f}\n"
        f"AUC near 0 is also strong separation, just in the opposite direction\n"
        f"({len(auc)} speakers with enough samples)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.88, edgecolor="none"),
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_shortcut_auc_by_speaker.png")
    plt.close(fig)


def speaker_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize by speaker and class to inspect speaker composition and within-speaker differences."""
    spk = (
        df.groupby(["speaker", "label"])
        .agg(
            n=("file", "size"),
            duration_median=("duration", "median"),
            duration_mean=("duration", "mean"),
            rms_median=("rms_dbfs", "median"),
        )
        .reset_index()
    )
    return spk


def paired_speaker_tests(spk: pd.DataFrame) -> pd.DataFrame:
    """Keep speakers with both classes and run paired tests to control for speaker composition."""
    rows = []
    for metric in ["duration_median", "rms_median"]:
        piv = spk.pivot(index="speaker", columns="label", values=metric).dropna()
        if len(piv) < 3:
            continue
        stat, p = wilcoxon(piv["real"], piv["fake"], alternative="two-sided")
        diff = piv["fake"] - piv["real"]
        rows.append(
            {
                "metric": metric,
                "n_speakers": len(piv),
                "median_real": piv["real"].median(),
                "median_fake": piv["fake"].median(),
                "median_diff_fake_minus_real": diff.median(),
                "n_fake_larger": int((diff > 0).sum()),
                "wilcoxon_p": p,
            }
        )
    return pd.DataFrame(rows)


def save_summary_text(df, summary, tests, spk_tests, shortcut, spk_auc, spk, pair, mel_note) -> None:
    """Write the key numbers to a text report for quick reference."""
    lines = []
    lines.append("Deepfake Voice Detector - Step 2 EDA summary\n")
    lines.append(f"Total clips: {len(df):,}")
    lines.append(f"Sample-rate distribution: {df.samplerate.value_counts().to_dict()}")
    lines.append(f"Channel distribution: {df.channels.value_counts().to_dict()}")
    lines.append(f"Total duration: {df.duration.sum()/3600:.2f} hours\n")

    lines.append("Summary by class:\n")
    lines.append(summary.round(3).to_string())
    lines.append("\n\nOverall difference tests:\n")
    lines.append(tests.round(6).to_string(index=False))
    lines.append("\n\nWithin-speaker paired tests (controls for speaker composition):\n")
    lines.append(spk_tests.round(6).to_string(index=False))
    lines.append("\n\nSingle-feature shortcut AUC (0.5=chance, 1.0=perfect):\n")
    lines.append(shortcut.round(4).to_string(index=False))
    lines.append("\n\nPer-speaker shortcut AUC summary:\n")
    lines.append(
        spk_auc[["auc_duration", "auc_rms"]]
        .describe()
        .loc[["count", "mean", "50%", "min", "max"]]
        .round(4)
        .to_string()
    )
    lines.append("\n\nNumber of speakers: " + str(spk.speaker.nunique()))
    both = spk.pivot(index="speaker", columns="label", values="n").dropna()
    lines.append(f"Speakers with both real and fake samples: {len(both)}")
    lines.append("\n\nWaveform / Mel example:\n")
    lines.append(pair.to_string(index=False))
    lines.append("\n" + mel_note)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def plot_duration(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    cap = df.duration.quantile(0.99)
    plot_df = df[df.duration <= cap]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), gridspec_kw={"width_ratios": [1.5, 1]})

    sns.histplot(
        data=plot_df,
        x="duration",
        hue="class_name",
        hue_order=["Real", "Fake"],
        palette=PALETTE,
        bins=70,
        stat="density",
        common_norm=False,
        element="step",
        fill=True,
        alpha=0.35,
        ax=axes[0],
    )
    for label, ha in [("real", "right"), ("fake", "left")]:
        med = summary.loc[label, "duration_median"]
        axes[0].axvline(med, color=COLORS[label], ls="--", lw=1.5)
        axes[0].text(
            med,
            axes[0].get_ylim()[1] * 0.92,
            f"{CLASS_LABELS[label]} median {med:.2f}s ",
            color=COLORS[label],
            fontsize=9,
            ha=ha,
            va="top",
        )
    axes[0].set_title(f"Audio duration distribution (clipped at 99th percentile, {cap:.1f}s)")
    axes[0].set_xlabel("Duration (seconds)")
    axes[0].set_ylabel("Density")
    legend = axes[0].get_legend()
    if legend is not None:
        legend.set_title("Class")

    sns.boxplot(
        data=plot_df,
        x="class_name",
        y="duration",
        order=["Real", "Fake"],
        hue="class_name",
        hue_order=["Real", "Fake"],
        palette=PALETTE,
        width=0.45,
        fliersize=0,
        ax=axes[1],
    )
    axes[1].set_title("Duration box plot")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Duration (seconds)")
    axes[1].tick_params(axis="x", labelsize=11)

    fig.suptitle("Real vs Fake Speech: Systematic Difference in Duration", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_duration_distribution.png")
    plt.close(fig)


def plot_loudness(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    low, high = -60, 0
    plot_df = df[(df.rms_dbfs >= low) & (df.rms_dbfs <= high)]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), gridspec_kw={"width_ratios": [1.5, 1]})

    sns.histplot(
        data=plot_df,
        x="rms_dbfs",
        hue="class_name",
        hue_order=["Real", "Fake"],
        palette=PALETTE,
        bins=70,
        stat="density",
        common_norm=False,
        element="step",
        fill=True,
        alpha=0.35,
        ax=axes[0],
    )
    for label in ["real", "fake"]:
        med = summary.loc[label, "rms_median"]
        axes[0].axvline(med, color=COLORS[label], ls="--", lw=1.5)
    # The two medians are only ~2.5 dB apart, so stagger the labels in the top-right corner
    axes[0].text(
        0.98,
        0.95,
        f"Real median {summary.loc['real', 'rms_median']:.1f} dBFS",
        color=COLORS["real"],
        fontsize=9,
        ha="right",
        va="top",
        transform=axes[0].transAxes,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="none"),
    )
    axes[0].text(
        0.98,
        0.87,
        f"Fake median {summary.loc['fake', 'rms_median']:.1f} dBFS",
        color=COLORS["fake"],
        fontsize=9,
        ha="right",
        va="top",
        transform=axes[0].transAxes,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="none"),
    )
    axes[0].set_title("Overall loudness distribution (RMS, dBFS)")
    axes[0].set_xlabel("RMS loudness (dBFS; closer to 0 is louder)")
    axes[0].set_ylabel("Density")
    legend = axes[0].get_legend()
    if legend is not None:
        legend.set_title("Class")

    sns.boxplot(
        data=plot_df,
        x="class_name",
        y="rms_dbfs",
        order=["Real", "Fake"],
        hue="class_name",
        hue_order=["Real", "Fake"],
        palette=PALETTE,
        width=0.45,
        fliersize=0,
        ax=axes[1],
    )
    axes[1].set_title("Loudness box plot")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("RMS (dBFS)")

    fig.suptitle("Real vs Fake Speech: Overall Loudness Comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_loudness_distribution.png")
    plt.close(fig)


def plot_speaker_composition(spk: pd.DataFrame) -> None:
    counts = spk.pivot(index="speaker", columns="label", values="n").fillna(0)
    counts["total"] = counts.sum(axis=1)
    top = counts.sort_values("total", ascending=True).tail(15)
    top = top.drop(columns=["total"])

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.barh(top.index, top["real"], color=COLORS["real"], label="Real")
    ax.barh(top.index, top["fake"], left=top["real"], color=COLORS["fake"], label="Fake")
    ax.set_xlabel("Number of clips")
    ax.set_title("Top 15 speakers by sample count: real/fake composition")
    ax.legend(loc="lower right")
    for i, (real, fake) in enumerate(zip(top["real"], top["fake"])):
        ax.text(real + fake + 20, i, f"{int(real+fake):,}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_speaker_class_composition.png")
    plt.close(fig)


def plot_speaker_duration(spk_tests: pd.DataFrame, spk: pd.DataFrame) -> None:
    piv = spk.pivot(index="speaker", columns="label", values="duration_median").dropna()
    counts = spk.pivot(index="speaker", columns="label", values="n").fillna(0)
    totals = counts.sum(axis=1).reindex(piv.index)

    fig, ax = plt.subplots(figsize=(7.5, 7))
    lim = max(piv["real"].max(), piv["fake"].max()) * 1.05
    ax.plot([0, lim], [0, lim], color="grey", ls="--", lw=1, label="y = x")
    scatter = ax.scatter(
        piv["real"],
        piv["fake"],
        s=np.clip(totals / 10, 20, 400),
        c=np.log10(totals),
        cmap="viridis",
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xlabel("Median real-speech duration (seconds)")
    ax.set_ylabel("Median fake-speech duration (seconds)")
    ax.set_title("Within each speaker: are fake clips longer?")
    ax.legend(loc="lower right")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("log10(number of clips for this speaker)")
    p = spk_tests.loc[spk_tests.metric == "duration_median", "wilcoxon_p"].iloc[0]
    diff = spk_tests.loc[spk_tests.metric == "duration_median", "median_diff_fake_minus_real"].iloc[0]
    ax.text(
        0.02,
        0.98,
        f"{len(piv)} speakers\nMedian fake - real = {diff:.2f}s\nWilcoxon p = {p:.2e}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_speaker_duration_scatter.png")
    plt.close(fig)


def pick_pair(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Pick a speaker with enough real and fake clips, then choose one typical clip per class."""
    counts = df.groupby(["speaker", "label"]).size().unstack(fill_value=0)
    counts = counts[(counts.get("real", 0) >= 100) & (counts.get("fake", 0) >= 100)]
    counts["min_count"] = counts[["real", "fake"]].min(axis=1)
    speaker = counts.sort_values("min_count", ascending=False).index[0]

    picked = []
    for label in ["real", "fake"]:
        sub = df[(df.speaker == speaker) & (df.label == label)].copy()
        # Prefer 3.5-4.5 s clips with loudness close to the class median.
        # This avoids extreme loud/quiet examples and makes the comparison fairer.
        near_4s = sub[(sub.duration >= 3.5) & (sub.duration <= 4.5)]
        if len(near_4s) >= 5:
            sub = near_4s
        med_rms = sub.rms_dbfs.median()
        sub["score"] = (sub.rms_dbfs - med_rms).abs()
        sub["dist_to_4s"] = (sub.duration - 4.0).abs()
        row = sub.sort_values(["score", "dist_to_4s", "file"]).iloc[0]
        picked.append(
            {
                "speaker": speaker,
                "label": label,
                "class_name": CLASS_LABELS[label],
                "file": row.file,
                "duration": row.duration,
                "rms_dbfs": row.rms_dbfs,
                "path": row.path,
            }
        )
    return pd.DataFrame(picked), speaker


def load_fixed_length(path: str, seconds: float = 4.0) -> np.ndarray:
    """Load audio at 16 kHz with fixed length; pad if short, truncate if long."""
    y, _ = librosa.load(path, sr=SR, mono=True, duration=seconds)
    target = int(SR * seconds)
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    return y[:target]


def plot_waveform(pair: pd.DataFrame, speaker: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True, sharey=True)
    t = np.arange(int(SR * 4.0)) / SR
    for ax, (_, row) in zip(axes, pair.iterrows()):
        y = load_fixed_length(row.path)
        ax.plot(t, y, color=COLORS[row["label"]], lw=0.7)
        ax.set_ylim(-1, 1)
        ax.set_ylabel("Amplitude")
        ax.set_title(
            f"{row['class_name']} speech  |  {row['file']}  |  original duration {row['duration']:.2f}s",
            fontsize=10,
        )
    axes[1].set_xlabel("Time (seconds; first 4 seconds shown)")
    fig.suptitle(f"Waveform comparison for the same speaker ({speaker})", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_waveform_comparison.png")
    plt.close(fig)


def mel_db(y: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, fmin=20, fmax=SR // 2
    )
    return librosa.power_to_db(mel, ref=1.0)


def plot_mel_pair(pair: pd.DataFrame, speaker: str) -> None:
    specs = {}
    for _, row in pair.iterrows():
        specs[row["class_name"]] = mel_db(load_fixed_length(row.path))
    joint_max = max(s.max() for s in specs.values())
    vmin, vmax = joint_max - 80, joint_max

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, class_name in zip(axes, ["Real", "Fake"]):
        img = librosa.display.specshow(
            specs[class_name],
            sr=SR,
            hop_length=HOP,
            x_axis="time",
            y_axis="mel",
            fmax=SR // 2,
            vmin=vmin,
            vmax=vmax,
            cmap="magma",
            ax=ax,
        )
        ax.set_title(f"{class_name} speech log-Mel")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Mel frequency")
    cbar = fig.colorbar(img, ax=axes, format="%+2.0f dB")
    cbar.set_label("Energy (dB, shared color scale)")
    fig.suptitle(f"Mel-spectrogram comparison for the same speaker ({speaker})", fontsize=14)
    fig.savefig(FIG_DIR / "02_mel_comparison.png")
    plt.close(fig)


def mean_mel_by_class(df: pd.DataFrame, n_per_class: int = 120) -> tuple[dict, np.ndarray, float]:
    """Average 4-second log-Mel over 120 random clips per class to reduce single-clip noise."""
    means = {}
    for label in ["real", "fake"]:
        paths = df[df.label == label].sample(n_per_class, random_state=42)["path"].tolist()
        acc = None
        for p in tqdm(paths, desc=f"{CLASS_LABELS[label]} Mel"):
            spec = mel_db(load_fixed_length(p))
            # Subtract each clip's mean energy to remove loudness effects and compare spectral shape
            spec = spec - spec.mean()
            acc = spec if acc is None else acc + spec
        means[label] = acc / len(paths)
    diff = means["real"] - means["fake"]
    limit = np.percentile(
        np.abs(np.concatenate([means["real"].ravel(), means["fake"].ravel(), diff.ravel()])), 99
    )
    return means, diff, limit


def plot_mean_mel(means: dict, diff: np.ndarray, limit: float) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), sharey=True)
    for ax, label in zip(axes[:2], ["real", "fake"]):
        img = librosa.display.specshow(
            means[label],
            sr=SR,
            hop_length=HOP,
            x_axis="time",
            y_axis="mel",
            fmax=SR // 2,
            vmin=-limit,
            vmax=limit,
            cmap="coolwarm",
            ax=ax,
        )
        ax.set_title(f"{CLASS_LABELS[label]}: mean log-Mel (loudness removed)")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Mel frequency")

    diff_limit = np.percentile(np.abs(diff), 99)
    img2 = librosa.display.specshow(
        diff,
        sr=SR,
        hop_length=HOP,
        x_axis="time",
        y_axis="mel",
        fmax=SR // 2,
        vmin=-diff_limit,
        vmax=diff_limit,
        cmap="coolwarm",
        ax=axes[2],
    )
    axes[2].set_title("Difference: Real - Fake")
    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_ylabel("")
    fig.colorbar(img, ax=axes[:2], format="%+2.0f dB", label="Relative energy (dB)")
    fig.colorbar(img2, ax=axes[2], label="Difference (dB)")
    fig.suptitle(
        "Mean 4-second Mel-spectrogram over 120 random clips per class "
        "(per-clip mean removed; red = higher relative energy)",
        fontsize=12,
    )
    fig.savefig(FIG_DIR / "02_mean_mel_by_class.png")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    index = pd.read_csv(INDEX_PATH)
    index["class_name"] = index["label"].map(CLASS_LABELS)

    stats_path = PROCESSED_DIR / "audio_stats.csv"
    if stats_path.exists():
        print(f"Reusing existing per-file statistics: {stats_path}")
        df = pd.read_csv(stats_path)
        # Drop legacy non-ASCII display columns and rebuild the English class column
        df = df.drop(columns=[c for c in df.columns if not str(c).isascii()], errors="ignore")
        df = df.drop(columns=["class_name"], errors="ignore").merge(
            index[["path", "class_name"]], on="path", how="left"
        )
        df.to_csv(stats_path, index=False)
    else:
        df = collect_audio_stats(index)
        df.to_csv(stats_path, index=False)
        print(f"Per-file statistics saved to: {stats_path}")

    summary = summarize(df)
    tests = run_tests(df)
    spk = speaker_summary(df)
    spk_tests = paired_speaker_tests(spk)
    shortcut = shortcut_auc(df)
    spk_auc = per_speaker_auc(df)
    pair, speaker = pick_pair(df)

    summary.to_csv(PROCESSED_DIR / "eda_label_summary.csv")
    spk.to_csv(PROCESSED_DIR / "eda_speaker_summary.csv", index=False)
    tests.to_csv(PROCESSED_DIR / "eda_tests.csv", index=False)
    spk_tests.to_csv(PROCESSED_DIR / "eda_speaker_tests.csv", index=False)
    shortcut.to_csv(PROCESSED_DIR / "eda_shortcut_auc.csv", index=False)
    spk_auc.to_csv(PROCESSED_DIR / "eda_per_speaker_auc.csv", index=False)

    print("\nSummary by class:")
    print(summary.round(3).to_string())
    print("\nOverall difference tests:")
    print(tests.round(6).to_string(index=False))
    print("\nWithin-speaker paired tests:")
    print(spk_tests.round(6).to_string(index=False))
    print("\nSingle-feature shortcut AUC:")
    print(shortcut.round(4).to_string(index=False))
    print("\nMedian per-speaker shortcut AUC:")
    print(spk_auc[["auc_duration", "auc_rms"]].median().round(4).to_string())
    print(f"\nExample speaker: {speaker}")
    print(pair[["class_name", "file", "duration", "rms_dbfs"]].to_string(index=False))

    plot_duration(df, summary)
    plot_loudness(df, summary)
    plot_speaker_composition(spk)
    plot_speaker_duration(spk_tests, spk)
    plot_per_speaker_auc(spk_auc)
    plot_waveform(pair, speaker)
    plot_mel_pair(pair, speaker)
    means, diff, limit = mean_mel_by_class(df)
    plot_mean_mel(means, diff, limit)

    mel_note = (
        "Mean Mel figure: 120 random 4-second clips per class; each clip's mean energy is "
        "subtracted before averaging. Red in the difference map means higher relative energy for real speech."
    )
    save_summary_text(df, summary, tests, spk_tests, shortcut, spk_auc, spk, pair, mel_note)
    print(f"\nFigures saved to {FIG_DIR}")
    print(f"Summary saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
