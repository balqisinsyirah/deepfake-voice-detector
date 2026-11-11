# Deepfake Voice Detector - Project Process
**procedures**

0. Environment Setup & Project Directory Structure

1. Dataset Download & Organization

2. Exploratory Data Analysis (EDA): Sample counts, audio duration, sampling rates, waveforms, and Mel-spectrogram comparisons

3. Preprocessing: Standardizing sampling rate and duration, extracting log-Mel-spectrograms, and saving features

4. Dataset Splitting (70/30) & DataLoader Setup

5. Model A: Simple CNN Baseline

6. Model B: Transfer Learning (Pre-trained ResNet / EfficientNet)

7. Model C: Hybrid CNN + LSTM/GRU Model

8. Evaluation & Comparison: Accuracy, Precision/Recall, F1-score, Confusion Matrix, ROC-AUC (optional: EER)

9. Error Analysis & Limitations Discussion

10. Blog Content Preparation & Formatting

## Project overview

- Task: binary classification of real vs. fake speech
- Dataset: [In-the-Wild](https://huggingface.co/datasets/mueller91/In-The-Wild) on Hugging Face
- Dataset paper: Müller et al., *Does Audio Deepfake Detection Generalize?* (2022)
- License: CC BY-SA 4.0
- Features: log-Mel spectrograms extracted with librosa
- Models to compare: a simple CNN baseline, pretrained ResNet/EfficientNet transfer learning, and CNN + LSTM/GRU hybrids
- Planned split: 70% training / 30% validation, with speaker-disjoint splitting
- Hardware: WSL2 on a local workstation with an NVIDIA RTX 3060 Laptop GPU (6 GB)

---

## Step 0: Environment check and project layout

**Goal:** confirm that the local machine can run the project and create a reproducible
project structure.

**Environment**

- OS: WSL2 (Linux), Python 3.12.3
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU (6 GB), CUDA driver available
- RAM: 7.6 GB; free disk space: about 204 GB
- Project folders: `data/raw`, `data/processed`, `data/splits`, `outputs/{figures,models,logs}`, `src/`
- Isolated environment: project-local `.venv/` (system Python was not modified)
- Key versions: torch 2.15.0+cu132, torchvision 0.29.2, torchaudio 2.11.1, librosa 1.0.0, numpy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1

**Verification**

- `torch.cuda.is_available() == True`; the RTX 3060 Laptop GPU is detected
- A real 1000 x 1000 matrix multiplication on the GPU completed successfully
- librosa extracted a 128-band Mel-spectrogram from a 2-second, 16 kHz clip (shape `(128, 63)`)
- `src/00_environment_check.py` can be used to repeat the environment check

---

## Step 1: Dataset download and organization

**Goal:** obtain all audio files, verify labels and speaker metadata, and organize
the data into a structure that later steps can read directly.

**Source and process**

- Dataset: [Hugging Face `mueller91/In-The-Wild`](https://huggingface.co/datasets/mueller91/In-The-Wild)
- License: CC BY-SA 4.0
- The dataset is distributed as `release_in_the_wild.zip`; the extracted files were placed in
  `data/raw/release_in_the_wild/`
- `meta.csv` was used to verify that every audio file has a matching metadata row.
  There were no missing files, no extra files, and no zero-byte files.
- Audio files were moved into `real/` (bona-fide) and `fake/` (spoof) subfolders.
  The move was done on the same disk, so no extra copy was created.
- An index table was written to `data/raw/dataset_index.csv` with the columns
  `file, speaker, label, label_id, path, size_bytes`.

**Key numbers**

| Item | Value |
|---|---:|
| Total audio clips | 31,779 |
| Total size | 8.12 GiB |
| Real (`real`) | 19,963 (62.8%) |
| Fake (`fake`) | 11,816 (37.2%) |
| Real : fake ratio | about 1.69 : 1 |
| Speakers in `meta.csv` | 54 (the dataset card lists 58) |

**Caveats**

- The classes are not strictly balanced. Later splits should be stratified, and
  evaluation should not rely on accuracy alone.
- Speaker labels are available. The validation split should therefore be
  speaker-disjoint, so the model is tested on voices it has never seen.
- Full duration, sample-rate, and loudness statistics are covered in Step 2.

**Blog takeaway**

> The dataset contains 31,779 real and fake speech clips, with roughly 1.69 times
> more real than fake samples. This is not an extreme imbalance, but it is enough
> for accuracy to be misleading; stratified splitting and Recall/F1 metrics are
> therefore necessary. Speaker annotations also make it possible to build a
> speaker-disjoint validation set and test whether the model learns spoofing cues
> rather than speaker identity.

---

## Step 2: Exploratory data analysis (EDA)

**Goal:** characterize duration, sample rate, channels, and loudness; test for
systematic real/fake differences; and visualize waveforms and Mel-spectrograms.

**Method**

- Read all 31,779 clips (header plus full decode) and computed duration, RMS
  loudness in dBFS, peak level, and clipping ratio.
- Statistical tests: Mann-Whitney U, Kolmogorov-Smirnov, Cliff's delta; Wilcoxon
  signed-rank for paired within-speaker comparisons.
- Shortcut analysis: ROC-AUC of duration-only and loudness-only classifiers,
  including speaker-centered variants and per-speaker AUC.
- Visualization sample: one real and one fake clip from the same speaker
  (Alec Guinness), each close to 4 seconds and close to its class median loudness.
- Mean Mel: 120 random 4-second clips per class; each clip's mean energy was
  subtracted before averaging so that loudness would not dominate the comparison.
- Outputs: `data/processed/audio_stats.csv`, `eda_label_summary.csv`,
  `eda_tests.csv`, `eda_speaker_summary.csv`, `eda_speaker_tests.csv`,
  `eda_shortcut_auc.csv`, and `eda_per_speaker_auc.csv`.

**Key numbers**

| Metric | Real | Fake |
|---|---:|---:|
| Clips | 19,963 | 11,816 |
| Total hours | 20.76 | 17.10 |
| Median duration | 2.99 s | 4.05 s |
| Mean duration | 3.74 s | 5.21 s |
| Median RMS loudness | -26.1 dBFS | -28.5 dBFS |
| Median peak level | -9.3 dBFS | -10.1 dBFS |

- Format: all 31,779 clips are 16 kHz, mono, and unclipped (clipping ratio approximately 0).
- Duration difference: KS = 0.188, Cliff's delta = -0.239.
- Within-speaker duration: 37 of 54 speakers have longer fake clips; median difference
  is +1.06 s; Wilcoxon p = 0.0022.
- Within-speaker loudness: median difference is -2.14 dB (fake quieter); Wilcoxon
  p = 0.096, and the direction is not consistent across speakers (21 of 54 speakers
  have louder fake clips).
- Single-feature AUC (0.5 = chance): duration 0.619, loudness 0.649,
  speaker-centered duration 0.613, speaker-centered loudness 0.706.
- Per-speaker AUC (39 speakers with at least 20 clips per class): median duration
  AUC = 0.596, median loudness AUC = 0.744. Loudness AUC is above 0.5 for 29 of 39
  speakers and below 0.5 for 10; an AUC near 0 is also strong separation, just in
  the opposite direction.

**Figures**

| Figure | What it shows |
|---|---|
| `outputs/figures/02_duration_distribution.png` | Fake clips are shifted toward longer durations. |
| `outputs/figures/02_loudness_distribution.png` | Fake clips are shifted toward lower RMS loudness. |
| `outputs/figures/02_speaker_class_composition.png` | Class ratios vary strongly by speaker; for example, Barack Obama has 3,302 real and 334 fake clips, while Alec Guinness has 1,907 real and 1,718 fake clips. |
| `outputs/figures/02_speaker_duration_scatter.png` | Most speakers lie above the y = x line: fake clips are usually longer even within the same speaker. |
| `outputs/figures/02_shortcut_auc_by_speaker.png` | Loudness is a strong shortcut for many speakers, with AUC values close to 0 or 1. |
| `outputs/figures/02_waveform_comparison.png` | Waveforms from the same speaker look different in amplitude and envelope. |
| `outputs/figures/02_mel_comparison.png` | The fake example shows less high-frequency energy and stronger harmonic stripes; this is illustrative only. |
| `outputs/figures/02_mean_mel_by_class.png` | After removing loudness, the average spectra still differ, with real speech showing relatively more energy in the upper bands; the models must confirm whether this difference is robust. |

**Caveats**

- The dataset contains clear shortcuts:
  - Fake clips are longer overall, and also tend to be longer within the same speaker.
  - Fake clips are quieter overall. Per-speaker loudness AUC has a median of 0.744,
    making loudness the strongest shortcut found in this dataset.
- Speaker composition is strongly correlated with class balance. A random file-level
  split would leak both speaker identity and the duration/loudness shortcuts.
- The single waveform and Mel examples are illustrative, not causal evidence. The
  averaged Mel comparison is more reliable but still descriptive; the models must
  confirm whether these differences generalize.
- Step 3 preprocessing should therefore consider duration standardization
  (crop/pad) and per-clip loudness normalization, while keeping a non-normalized
  control version to measure how much performance depends on shortcuts.
- Step 4 should use a speaker-disjoint split.

**Blog takeaway**

> EDA revealed two shortcuts that are easier to learn than the actual spoofing
> artifacts: fake clips are about 1.06 seconds longer within the same speaker
> (p = 0.002), and they are about 2.5 dB quieter overall. Loudness alone reaches a
> median per-speaker AUC of 0.744. Without loudness normalization and duration
> standardization, a CNN may become a "volume/length detector" that fails when the
> speaker or recording condition changes. This is also why a speaker-disjoint
> validation set is essential.

---

## Step 3: Preprocessing and feature extraction

**Goal:** standardize sample rate, duration, and loudness; extract log-Mel
spectrograms; and save a feature matrix that later training steps can load
directly.

**Pipeline**

- Sample rate: resample every clip to 16 kHz mono (the dataset already satisfies this).
- Duration: fix every clip to 4 seconds. Long clips are center-cropped; short clips
  are repeat-padded (wrap-pad) instead of zero-padded, avoiding a silence shortcut.
- Loudness: normalize each clip to -23 dBFS RMS. If the peak exceeds 0.99, scale the
  clip down to avoid clipping.
- Mel parameters: `n_mels=128`, `n_fft=2048`, `hop_length=512`, `fmin=20`,
  `fmax=8000`, `power_to_db(ref=np.max, top_db=80)`.
- Parallelism: 12 worker processes; the full run over 31,779 clips took about 24 minutes.
- Output directory: `data/processed/mel/`.

**Key numbers**

| Item | Value |
|---|---:|
| Feature matrix | `mel_spectrograms.npy` |
| Shape | `(31779, 128, 126)` |
| Data type | float32 |
| File size | about 2.0 GB |
| Labels | 0 = Real (19,963); 1 = Fake (11,816) |
| Value range | -80 to 0 dB |
| Invalid values | no NaN, no Inf |
| Metadata rows | 31,779 (no read errors) |

- Median RMS before normalization: Real -25.9 dBFS, Fake -28.5 dBFS.
- Median RMS after normalization: -23.0 dBFS for both classes.
- Loudness-only shortcut AUC: 0.649 before normalization, 0.505 after.
- 61% of clips are shorter than 4 seconds and require wrap-padding.
- Median padding: Real 1.01 s, Fake 0.00 s.
- Residual padding shortcut: AUC = 0.399 (0.5 means no class signal).
- Mean center-crop start: Real 0.43 s, Fake 0.99 s (this value is not a model input).

**Figures**

| Figure | What it shows |
|---|---|
| `outputs/figures/03_loudness_before_after.png` | RMS distributions before and after normalization; after normalization both classes overlap at -23 dBFS. |
| `outputs/figures/03_preprocessed_examples.png` | Final waveform and log-Mel input for one real and one fake clip from the same speaker. |
| `outputs/figures/03_padding_by_class.png` | Wrap-padding needed to reach 4 seconds; the residual padding AUC is 0.399. |

**Caveats**

- The strong loudness shortcut is essentially removed, but 61% of clips still need
  repeat-padding. The padding amount retains a weak class signal, and a model could
  potentially exploit repeated content.
- Training should therefore use augmentation such as SpecAugment or time masking,
  and error analysis should group mistakes by padding amount.
- Feature mean and standard deviation for neural networks must be computed on the
  training split only, never on the full dataset, to avoid validation leakage.
- A non-normalized control feature set can be generated with
  `--no-normalize --output-dir data/processed/mel_raw` for an ablation study.
- `data/processed/` is ignored by git, so generated features are not committed.

**Blog takeaway**

> Per-clip RMS normalization reduced the loudness-only AUC from 0.649 to 0.505,
> essentially removing the strongest shortcut found in EDA. Fixing the duration at
> 4 seconds also removed the direct duration cue. However, 61% of clips are shorter
> than 4 seconds and require repeat-padding, and the padding amount still carries a
> weak class signal (AUC = 0.399). Preprocessing alone is therefore not enough:
> training-time augmentation, a speaker-disjoint split, and error analysis grouped
> by padding amount are still needed to test whether the model learns genuine
> spoofing artifacts.

---

## Step 4: Dataset split (70/30) and DataLoaders

**Goal:** create a speaker-disjoint training/validation split, compute normalization
statistics on the training split only, and build PyTorch Dataset/DataLoader objects.

**Method**

- Split at the speaker level: every clip from a speaker goes to either training or
  validation, never both.
- Fixed random seed 42. Search 5,000 random speaker permutations and choose the one
  that keeps:
  - validation close to 30% of all clips;
  - validation fake share close to the global 37.18%;
  - both classes present in both splits.
- Normalization: compute the global feature mean/std on the training split only;
  validation reuses those statistics.
- Class weights: inverse-frequency CrossEntropy weights computed from the training
  split to reduce the effect of the 63/37 imbalance.
- Dataset: lazily reads rows from the `.npy` memmap and returns a `(1, 128, 126)`
  float32 tensor plus a `long` label.
- DataLoader: batch size 32, shuffled training loader, unshuffled validation loader,
  4 workers, and pinned memory when CUDA is available.

**Key numbers**

| Item | Train | Validation |
|---|---:|---:|
| Clips | 22,233 (69.96%) | 9,546 (30.04%) |
| Speakers | 40 | 14 |
| Real clips | 13,954 | 6,009 |
| Fake clips | 8,279 | 3,537 |
| Fake share | 37.24% | 37.05% |

- Global fake share: 37.18%.
- Speaker overlap: 0; row-index overlap: 0.
- Class weights (Real/Fake): `[0.7967, 1.3427]`.
- Training feature mean/std: `-50.6645 / 17.4524`.
- DataLoader: 695 training batches and 299 validation batches; batch shape
  `(32, 1, 128, 126)`.
- After standardization, the training batch has mean/std about `-0.014 / 0.956`;
  the validation batch has about `-0.107 / 1.039`.
- Validation speakers: 2Pac, Adam Driver, Alan Watts, Ayn Rand, Barack Obama,
  Bob Ross, Dwight Eisenhower, Kamala Harris, Lyndon Johnson, Mr. Rogers,
  Norm MacDonald, Ronald Reagan, The Notorious B.I.G., and William F. Buckley.

**Figure**

- `outputs/figures/04_split_composition.png`: real/fake counts per split and fake
  share versus the global 37.2% line.

**Caveats**

- The validation set contains only 14 speakers, so results can vary depending on
  which speakers happen to land in validation. Speaker-level cross-validation would
  provide a more robust estimate later.
- Overall class balance is preserved, but individual speakers have very different
  real/fake ratios (for example, Barack Obama has far more real than fake clips).
  The validation set may therefore be harder than a random file-level split.
- Normalization statistics come from the training split. The validation batch mean
  is slightly shifted (-0.107), which reflects the different speaker distribution
  and should not be corrected using validation data.
- The Dataset does not apply augmentation yet. The CNN training step will add
  SpecAugment/time masking and use the saved class weights.

**Blog takeaway**

> The speaker-disjoint 70/30 split produced 22,233 training clips and 9,546
> validation clips with zero speaker overlap, while preserving the class ratio
> (37.24% fake in training, 37.05% in validation, 37.18% globally). This is stricter
> than a random file-level split and tests generalization to unseen speakers. The
> DataLoader reliably yields `(32, 1, 128, 126)` tensors, and normalization
> statistics are computed on the training split only, avoiding validation leakage.

---

## Key concepts

**Why can accuracy be misleading with a 63/37 split?**

If 63% of the data is real, a model that always predicts "real" already achieves
63% accuracy while catching zero fake clips. In an anti-fraud setting, missed fake
clips are costly. Precision, Recall, F1, the confusion matrix, and ROC-AUC are
therefore reported alongside accuracy.

**Why split by speaker instead of by file?**

A random file-level split can place the same speaker in both training and
validation. The model may then memorize the speaker's voice or recording channel
instead of learning spoofing artifacts, inflating validation performance. A
speaker-disjoint split simulates deployment on unseen voices and gives a more
honest estimate of generalization.

---

## Repository notes

- `src/00_environment_check.py`: environment check (English output).
- `src/01_organize_dataset.py`: organize the flat audio files and build `dataset_index.csv`.
- `src/02_eda.py`: run the full EDA and regenerate all Step 2 figures.
- `src/03_preprocess.py`: build the log-Mel feature matrix and Step 3 figures
  (`--figures-only` regenerates the figures from existing features).
- `src/04_split_dataloader.py`: create the speaker-disjoint split, compute training
  statistics, and smoke-test the DataLoaders.
