"""Step 0: Environment check.

Uses only the standard library plus optional third-party imports, so it can run
before all dependencies are installed. Run: python src/00_environment_check.py
"""

import importlib
import os
import platform
import shutil
import sys
from pathlib import Path

# Third-party packages and their purpose
PACKAGES = {
    "numpy": "numerical computing",
    "pandas": "tabular data",
    "matplotlib": "plotting",
    "seaborn": "statistical plots",
    "librosa": "audio feature extraction",
    "soundfile": "audio I/O",
    "sklearn": "data splitting and metrics",
    "torch": "deep learning",
    "torchaudio": "PyTorch audio utilities",
    "torchvision": "pretrained image models",
    "tqdm": "progress bars",
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def check_python() -> None:
    """Print Python and operating-system information."""
    print("=" * 60)
    print("Python environment")
    print("=" * 60)
    print(f"Python version : {sys.version.split()[0]}")
    print(f"Interpreter    : {sys.executable}")
    print(f"Operating system: {platform.platform()}")
    print(f"CPU cores      : {os.cpu_count()}")


def check_packages() -> None:
    """Try importing each package and report missing ones."""
    print("\n" + "=" * 60)
    print("Third-party packages")
    print("=" * 60)
    missing = []
    for name, purpose in PACKAGES.items():
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, "__version__", "unknown")
            print(f"[OK]      {name:<12} {version:<12} ({purpose})")
        except Exception:
            missing.append(name)
            print(f"[MISSING] {name:<12} {'':<12} ({purpose})")

    if missing:
        print("\nMissing packages. Install them with:")
        print(f"  pip install {' '.join(missing)}")
    else:
        print("\nAll third-party packages are available.")


def check_gpu() -> None:
    """Check whether PyTorch can see an NVIDIA GPU."""
    print("\n" + "=" * 60)
    print("GPU / CUDA")
    print("=" * 60)
    try:
        import torch
    except Exception:
        print("PyTorch is not installed; skipping GPU check.")
        return

    print(f"PyTorch CUDA build : {torch.version.cuda}")
    print(f"CUDA available     : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU name           : {torch.cuda.get_device_name(0)}")
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU memory         : {total:.1f} GB")


def check_disk_and_data() -> None:
    """Check free disk space and whether audio files are present."""
    print("\n" + "=" * 60)
    print("Disk and data")
    print("=" * 60)
    usage = shutil.disk_usage(PROJECT_ROOT)
    print(f"Free disk space: {usage.free / 1024**3:.1f} GB")

    raw_dir = PROJECT_ROOT / "data" / "raw"
    audio_files = [
        p
        for p in raw_dir.rglob("*")
        if p.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    ] if raw_dir.exists() else []
    print(f"Audio files under data/raw: {len(audio_files)}")
    if not audio_files:
        print("Dataset has not been downloaded yet (Step 1).")


if __name__ == "__main__":
    check_python()
    check_packages()
    check_gpu()
    check_disk_and_data()
    print("\nEnvironment check complete.")
