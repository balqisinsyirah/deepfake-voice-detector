"""Step 1: Organize downloaded flat audio files into real/ and fake/ folders and build an index.

Usage:
    python src/01_organize_dataset.py            # preview only; do not move files
    python src/01_organize_dataset.py --apply    # actually move files and build the index
"""

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
AUDIO_DIR = RAW_DIR / "release_in_the_wild"
META_PATH = RAW_DIR / "meta.csv"
INDEX_PATH = RAW_DIR / "dataset_index.csv"

# Original dataset labels -> folder names used in this project
LABEL_TO_FOLDER = {"bona-fide": "real", "spoof": "fake"}
LABEL_TO_ID = {"real": 0, "fake": 1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Move files for real; without this flag only preview")
    args = parser.parse_args()

    meta = pd.read_csv(META_PATH)
    meta["folder"] = meta["label"].map(LABEL_TO_FOLDER)
    if meta["folder"].isna().any():
        bad = meta.loc[meta["folder"].isna(), "label"].unique()
        raise ValueError(f"Unknown labels found: {bad}")

    plan = {"move": 0, "already_done": 0, "missing": 0, "conflict": 0}
    rows = []

    # Scan first and count each case to avoid stopping halfway through a move
    for row in tqdm(meta.itertuples(index=False), total=len(meta), desc="Scanning"):
        src = AUDIO_DIR / row.file
        dst = AUDIO_DIR / row.folder / row.file
        if src.exists() and dst.exists():
            status = "conflict"  # Both source and destination exist: do not overwrite
        elif src.exists():
            status = "move"
        elif dst.exists():
            status = "already_done"
        else:
            status = "missing"
        plan[status] += 1
        rows.append((row.file, row.speaker, row.label, row.folder, status, src, dst))

    print("\nScan summary:")
    for key, value in plan.items():
        print(f"  {key:12s}: {value}")
    if plan["conflict"] or plan["missing"]:
        raise SystemExit("Conflicts or missing files found. Stopped; please inspect first.")

    if not args.apply:
        print("\nPreview mode: no files were moved. Re-run with --apply when ready.")
        return

    # Actual move: same-disk rename is fast and uses no extra space
    for _, _, _, folder, status, src, dst in tqdm(rows, desc="Organizing"):
        if status != "move":
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.replace(dst)  # Atomic move; never leaves a half-written file

    # Build the index used by later splitting and EDA
    index = meta[["file", "speaker", "label"]].copy()
    index["label"] = index["label"].map(LABEL_TO_FOLDER)
    index["label_id"] = index["label"].map(LABEL_TO_ID)
    index["path"] = index.apply(
        lambda r: str((AUDIO_DIR / r["label"] / r["file"]).relative_to(PROJECT_ROOT)),
        axis=1,
    )
    index["size_bytes"] = index["path"].map(lambda p: (PROJECT_ROOT / p).stat().st_size)
    index.to_csv(INDEX_PATH, index=False)

    print(f"\nDone: moved {plan['move']} files; index written to {INDEX_PATH.relative_to(PROJECT_ROOT)}")
    print(index["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
